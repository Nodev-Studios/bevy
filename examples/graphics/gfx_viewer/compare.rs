//! The split view: the set not in use (A/B) on the left of a divider, the live set on the right, on the same
//! view.
//!
//! The ladder holds one `GraphicsSettings` for the whole app and shapes every `GraphicsCamera` (and every light's
//! shadow maps) to it, so two live cameras with different rungs side by side cannot work without changing
//! `bevy_graphics`; each would also need its own temporal history. Instead the left side is a frame of the other
//! set taken on the same view: the viewer flips to it, lets it settle, captures it (panels hidden for that one
//! frame) and flips back. After the camera moves the left side is stale; it is taken again when the camera stops
//! (or with "Refresh left").

use crate::{actions::flip_ab, ui::UiState, AbCompare, PendingPipelines, ViewerCamera};
use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
};
use bevy_egui::egui;
use std::sync::{atomic::Ordering, Arc, Mutex};

/// Frames the other set renders before its capture: the temporal filters and the denoisers settle. to tune.
const SETTLE_FRAMES: u32 = 45;
/// Frames the panels are hidden before the capture (egui's output reaches the frame late).
const HIDE_FRAMES: u32 = 4;
/// Frames the camera stays still before a stale left side is taken again. to tune.
const STILL_FRAMES: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    /// Showing the other set, waiting for it to settle.
    Settling {
        left: u32,
    },
    /// The panels are hidden; egui's output leaves the frame a few frames later.
    Hiding {
        left: u32,
    },
    /// The capture is on its way.
    Capturing,
}

#[derive(Resource)]
pub struct Split {
    pub on: bool,
    /// Where the divider stands, from the left edge (0..1).
    pub divider: f32,
    /// Take the left side again when the camera stops after a move.
    pub auto: bool,
    pub phase: Phase,
    /// A refresh was asked for.
    pub requested: bool,
    /// The left side's frame, as RGBA8, and its size.
    image: Option<(Vec<u8>, [usize; 2])>,
    texture: Option<egui::TextureHandle>,
    texture_stale: bool,
    /// The camera when the left side was taken.
    taken_at: Option<GlobalTransform>,
    /// Which set the left side shows (true: B).
    pub left_is_b: bool,
    still: u32,
    last_camera: Option<GlobalTransform>,
    slot: Arc<Mutex<Option<Image>>>,
    /// How many left sides were taken (the self-test waits on it).
    pub taken: u32,
}

impl Default for Split {
    fn default() -> Self {
        Self {
            on: false,
            divider: 0.5,
            auto: true,
            phase: Phase::Idle,
            requested: false,
            image: None,
            texture: None,
            texture_stale: false,
            taken_at: None,
            left_is_b: false,
            still: 0,
            last_camera: None,
            slot: Arc::default(),
            taken: 0,
        }
    }
}

impl Split {
    pub fn status(&self) -> String {
        match self.phase {
            Phase::Settling { .. } | Phase::Hiding { .. } | Phase::Capturing => "taking the left side…".into(),
            Phase::Idle if self.image.is_none() => "no left side yet".into(),
            Phase::Idle if self.stale() => "the camera moved: the left side is stale".into(),
            Phase::Idle => format!(
                "left: set {}, right: set {} (live)",
                if self.left_is_b { "B" } else { "A" },
                if self.left_is_b { "A" } else { "B" }
            ),
        }
    }

    pub fn stale(&self) -> bool {
        match (self.taken_at, self.last_camera) {
            (Some(taken), Some(now)) => !same_view(&taken, &now),
            _ => true,
        }
    }

    /// The left side is ready and matches the view.
    pub fn ready(&self) -> bool {
        self.on && self.phase == Phase::Idle && self.image.is_some() && !self.stale()
    }
}

fn same_view(a: &GlobalTransform, b: &GlobalTransform) -> bool {
    a.translation().distance(b.translation()) < 1e-4
        && a.rotation().angle_between(b.rotation()) < 1e-4
}

/// Drives the left side's capture.
pub fn run_split(world: &mut World) {
    let mut split = world
        .remove_resource::<Split>()
        .expect("the viewer has a Split");
    step(world, &mut split);
    world.insert_resource(split);
}

fn step(world: &mut World, split: &mut Split) {
    let camera = world
        .query_filtered::<&GlobalTransform, With<ViewerCamera>>()
        .iter(world)
        .next()
        .copied();
    let moved = match (split.last_camera, camera) {
        (Some(last), Some(now)) => !same_view(&last, &now),
        _ => true,
    };
    split.still = if moved { 0 } else { split.still + 1 };
    split.last_camera = camera;
    if !split.on {
        split.requested = false;
        if split.phase != Phase::Idle {
            // Switched off half-way: put the live set back.
            split.phase = Phase::Idle;
            world.resource_mut::<UiState>().hidden_for_capture = false;
            flip_ab(world);
        }
        return;
    }
    match split.phase {
        Phase::Idle => {
            let auto = split.auto && split.stale() && split.still >= STILL_FRAMES;
            if split.requested || split.image.is_none() || auto {
                split.requested = false;
                split.phase = Phase::Settling {
                    left: SETTLE_FRAMES,
                };
                flip_ab(world);
            }
        }
        Phase::Settling { left } => {
            if moved {
                // The camera moved while the other set settled: back to the live set, and again later.
                split.phase = Phase::Idle;
                flip_ab(world);
                return;
            }
            let compiling = world
                .resource::<PendingPipelines>()
                .0
                .load(Ordering::Relaxed);
            if left > 0 || compiling > 0 {
                split.phase = Phase::Settling {
                    left: left.saturating_sub(1),
                };
                return;
            }
            split.phase = Phase::Hiding { left: HIDE_FRAMES };
            world.resource_mut::<UiState>().hidden_for_capture = true;
        }
        Phase::Hiding { left } => {
            if left > 0 {
                split.phase = Phase::Hiding { left: left - 1 };
                return;
            }
            split.phase = Phase::Capturing;
            let slot = split.slot.clone();
            world.spawn(Screenshot::primary_window()).observe(
                move |captured: On<ScreenshotCaptured>| {
                    *slot.lock().unwrap() = Some(captured.image.clone());
                },
            );
        }
        Phase::Capturing => {
            let Some(image) = split.slot.lock().unwrap().take() else {
                return;
            };
            if let Ok(dynamic) = image.try_into_dynamic() {
                let rgba = dynamic.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                split.image = Some((rgba.into_raw(), size));
                split.texture_stale = true;
            }
            split.taken_at = camera;
            split.taken += 1;
            split.phase = Phase::Idle;
            // The left side shows the set shown now; the flip back makes the other one live.
            split.left_is_b = world.resource::<AbCompare>().showing_b;
            world.resource_mut::<UiState>().hidden_for_capture = false;
            flip_ab(world);
        }
    }
}

/// Draws the left side and the divider under the panels.
pub fn draw_split(ctx: &egui::Context, split: &mut Split, _ab: &AbCompare) {
    if !split.on {
        return;
    }
    if split.texture_stale
        && let Some((rgba, size)) = &split.image
    {
        let image = egui::ColorImage::from_rgba_unmultiplied(*size, rgba);
        split.texture = Some(ctx.load_texture("split left", image, egui::TextureOptions::LINEAR));
        split.texture_stale = false;
    }
    let screen = ctx.content_rect();
    let x = screen.left() + split.divider * screen.width();
    let painter = ctx.layer_painter(egui::LayerId::background());
    if let Some(texture) = &split.texture
        && !matches!(split.phase, Phase::Settling { .. } | Phase::Hiding { .. } | Phase::Capturing)
    {
        let left = egui::Rect::from_min_max(screen.min, egui::pos2(x, screen.max.y));
        painter.image(
            texture.id(),
            left,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(split.divider, 1.0)),
            if split.stale() {
                egui::Color32::from_gray(150)
            } else {
                egui::Color32::WHITE
            },
        );
    }
    painter.vline(
        x,
        screen.y_range(),
        egui::Stroke::new(2.0, egui::Color32::from_rgb(240, 240, 240)),
    );
    let label = |set_b: bool| if set_b { "B" } else { "A" };
    for (pos, text) in [
        (
            egui::pos2(x - 10.0, screen.max.y - 24.0),
            label(split.left_is_b),
        ),
        (
            egui::pos2(x + 10.0, screen.max.y - 24.0),
            label(!split.left_is_b),
        ),
    ] {
        painter.text(
            pos,
            if pos.x < x {
                egui::Align2::RIGHT_CENTER
            } else {
                egui::Align2::LEFT_CENTER
            },
            text,
            egui::FontId::proportional(18.0),
            egui::Color32::WHITE,
        );
    }
}
