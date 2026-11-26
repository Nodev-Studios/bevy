//! The viewer's panels, drawn with `bevy_egui` in the spirit of gkNextEngine's renderer: a dark, compact theme,
//! a bar of panel switches on top, and movable, collapsible windows docked to the edges at start:
//!
//! - **Scene** (left): the scene list with each scene's licence ("no comercial" marked), a glTF file by path,
//!   camera bookmarks and a camera reset;
//! - **Render** (left, under Scene): every field of `GraphicsSettings`, and the view's continuous values
//!   (exposure, sun, lamp intensity);
//! - **Stats** (right): fps and frame time with a graph of the last seconds, GPU time per pass as bars, what
//!   resolved and every `Fallback` in words, the adapter, `BakedGiStatus`;
//! - **Baked light** (right, under Stats): bake, progress, load / unload, the coverage view;
//! - **Compare** (bottom): the A/B flip and the split view (see `compare.rs`);
//! - **Help** (`H`).
//!
//! `Tab` hides every panel. While the pointer is over a panel, or a text field has the keyboard, the camera
//! does not move and the viewer's keys do nothing.
//!
//! The panels edit `GraphicsSettings` and [`ViewControls`] in place and write [`Action`]s for everything
//! else, so the self-test drives the same code (and also clicks real widgets through egui's input, see
//! [`UiState::rects`]).

use crate::{
    actions::{Action, Bookmarks, Lamps},
    bake::Bakes,
    catalog::Catalog,
    compare::Split,
    stage::Stage,
    AbCompare, PendingPipelines,
};
use bevy::{
    camera_controller::free_camera::FreeCameraState,
    diagnostic::DiagnosticsStore,
    ecs::system::SystemParam,
    math::ops,
    graphics::{
        baked::BakedGiStatus, raster::RasterLevel, DenoiserChoice, GraphicsCapabilities,
        GraphicsPreset, GraphicsRung, GraphicsSettings, ResolvedGraphics, UpscaleQuality,
        UpscalerChoice,
    },
    prelude::*,
};
use bevy_egui::{egui, EguiContexts};
use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

/// The help text: every key.
pub const HELP_TEXT: &str = "\
Right mouse (hold)   look around
W A S D              move           Q / E   down / up
Shift                move faster    Wheel   change speed
M                    keep the mouse captured (toggle)
Tab                  hide / show every panel
L                    switch the scene's lamp(s) off / on
B                    bake indirect light for this scene (child process)
C                    A/B: flip between two settings sets
V                    split view: A on the left, B on the right
F12                  screenshot
H                    this help
Esc                  quit
";

/// Which panels are open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Windows {
    pub scene: bool,
    pub render: bool,
    pub stats: bool,
    pub baked: bool,
    pub compare: bool,
    pub help: bool,
}

/// The panels' state.
#[derive(Resource)]
pub struct UiState {
    /// `Tab`: every panel shown.
    pub visible: bool,
    /// The self-test and the split view hide everything for a capture.
    pub hidden_for_capture: bool,
    pub windows: Windows,
    /// The "open glTF" field.
    pub open_path: String,
    /// The pointer is over a panel or a widget holds it.
    pub pointer: bool,
    /// A text field has the keyboard.
    pub keyboard: bool,
    /// Where some widgets were drawn last frame, in logical pixels, for the self-test's clicks.
    pub rects: HashMap<&'static str, egui::Rect>,
    /// The right button went down over the scene, not over a panel: looking around goes on over panels.
    looking: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            visible: true,
            hidden_for_capture: false,
            windows: Windows {
                scene: true,
                render: true,
                stats: true,
                baked: false,
                compare: false,
                help: false,
            },
            open_path: String::new(),
            pointer: false,
            keyboard: false,
            rects: HashMap::new(),
            looking: false,
        }
    }
}

/// The sun's controls: where it stands and how bright it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunControl {
    /// Degrees from north (−Z) towards east (+X).
    pub azimuth: f32,
    /// Degrees above the horizon: the time of day.
    pub elevation: f32,
    /// Lux.
    pub lux: f32,
}

impl SunControl {
    /// The direction the light travels.
    pub fn direction(&self) -> Vec3 {
        let (az, el) = (self.azimuth.to_radians(), self.elevation.to_radians());
        -Vec3::new(
            ops::sin(az) * ops::cos(el),
            ops::sin(el),
            -ops::cos(az) * ops::cos(el),
        )
    }

    pub fn from_direction(direction: Vec3, lux: f32) -> Self {
        let towards = -direction.normalize_or(Vec3::NEG_Y);
        Self {
            azimuth: ops::atan2(towards.x, -towards.z).to_degrees().rem_euclid(360.0),
            elevation: ops::asin(towards.y.clamp(-1.0, 1.0)).to_degrees(),
            lux,
        }
    }
}

/// The view's continuous values the Render panel edits; [`crate::actions::apply_view_controls`] applies them.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct ViewControls {
    /// The camera's EV100.
    pub exposure: f32,
    /// Every point, spot and rect light's intensity times this.
    pub lamp_scale: f32,
    /// The scene's sun, when it has one.
    pub sun: Option<SunControl>,
    /// The coverage view: white matte materials and no analytic light, so only baked light shows.
    pub coverage: bool,
    /// The scene these values were read from.
    pub generation: u32,
    /// The sun as the scene opened with it (the self-test puts it back).
    pub initial_sun: Option<SunControl>,
}

impl Default for ViewControls {
    fn default() -> Self {
        Self {
            exposure: bevy::camera::Exposure::default().ev100,
            lamp_scale: 1.0,
            sun: None,
            coverage: false,
            generation: u32::MAX,
            initial_sun: None,
        }
    }
}

/// Frame times of the last seconds, for the median and the graph.
#[derive(Resource, Default)]
pub struct FrameTimes {
    times: VecDeque<(Instant, f64)>,
    last: Option<Instant>,
}

/// How long the graph looks back.
const GRAPH_SECONDS: f64 = 5.0;

impl FrameTimes {
    /// The median frame time of the last second, in ms.
    pub fn median_ms(&self) -> Option<f64> {
        let now = Instant::now();
        let mut values: Vec<f64> = self
            .times
            .iter()
            .filter(|(at, _)| now - *at <= Duration::from_secs(1))
            .map(|(_, ms)| *ms)
            .collect();
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        Some(values[values.len() / 2])
    }
}

pub fn record_frame_time(mut times: ResMut<FrameTimes>) {
    let now = Instant::now();
    if let Some(last) = times.last.replace(now) {
        times
            .times
            .push_back((now, (now - last).as_secs_f64() * 1000.0));
    }
    while times
        .times
        .front()
        .is_some_and(|(at, _)| (now - *at).as_secs_f64() > GRAPH_SECONDS)
    {
        times.times.pop_front();
    }
}

/// The GPU time of each render pass measured in the last second, in ms, longest first.
pub fn gpu_passes(diagnostics: &DiagnosticsStore) -> Vec<(String, f64)> {
    let mut passes = Vec::new();
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        let (Some(pass), Some(measurement)) = (
            path.strip_prefix("render/")
                .and_then(|path| path.strip_suffix("/elapsed_gpu")),
            diagnostic.measurement(),
        ) else {
            continue;
        };
        // A pass that stopped running keeps its last value in the store.
        if measurement.time.elapsed() > Duration::from_secs(1) {
            continue;
        }
        passes.push((
            pass.to_owned(),
            diagnostic.average().unwrap_or(measurement.value),
        ));
    }
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    passes
}

/// The rung in words, with Raster's light table when `Auto` chose it.
pub fn rung_text(resolved: &ResolvedGraphics) -> String {
    match (resolved.rung, resolved.raster) {
        (GraphicsRung::Raster, RasterLevel::Light) => "Raster (light table)".into(),
        (rung, _) => format!("{rung:?}"),
    }
}

/// The dark, compact look.
pub fn style(mut contexts: EguiContexts, mut done: Local<bool>) -> Result {
    if *done {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgba_unmultiplied(22, 23, 27, 236);
    visuals.panel_fill = visuals.window_fill;
    visuals.window_corner_radius = egui::CornerRadius::same(4);
    visuals.selection.bg_fill = egui::Color32::from_rgb(52, 92, 150);
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(6.0, 3.0);
        style.spacing.button_padding = egui::vec2(5.0, 1.0);
        style.spacing.interact_size.y = 18.0;
        for font in style.text_styles.values_mut() {
            font.size = (font.size - 1.0).max(10.0);
        }
    });
    *done = true;
    Ok(())
}

/// What the panels read.
#[derive(SystemParam)]
pub struct View<'w> {
    catalog: Res<'w, Catalog>,
    stage: Res<'w, Stage>,
    resolved: Option<Res<'w, ResolvedGraphics>>,
    capabilities: Res<'w, GraphicsCapabilities>,
    status: Option<Res<'w, BakedGiStatus>>,
    bakes: Res<'w, Bakes>,
    ab: Res<'w, AbCompare>,
    lamps: Res<'w, Lamps>,
    pending: Res<'w, PendingPipelines>,
    times: Res<'w, FrameTimes>,
    diagnostics: Res<'w, DiagnosticsStore>,
    bookmarks: Res<'w, Bookmarks>,
}

/// What the panels change.
#[derive(SystemParam)]
pub struct Edit<'w> {
    settings: ResMut<'w, GraphicsSettings>,
    controls: ResMut<'w, ViewControls>,
    split: ResMut<'w, Split>,
    actions: MessageWriter<'w, Action>,
}

/// Draws every panel.
pub fn draw(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<UiState>,
    view: View,
    mut edit: Edit,
) -> Result {
    let ctx = contexts.ctx_mut()?.clone();
    let state = &mut *ui_state;
    state.rects.clear();
    if state.hidden_for_capture {
        state.pointer = false;
        state.keyboard = false;
        return Ok(());
    }
    crate::compare::draw_split(&ctx, &mut edit.split, &view.ab);
    if state.visible {
        top_bar(&ctx, state);
        let w = state.windows;
        let screen = ctx.content_rect();
        if w.scene {
            window("Scene", egui::pos2(8.0, 34.0), 290.0)
                .show(&ctx, |ui| scene_panel(ui, state, &view, &mut edit));
        }
        if w.render {
            let top = 300.0;
            window("Render", egui::pos2(8.0, top), 290.0).show(&ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("render scroll")
                    .max_height((screen.height() - top - 50.0).max(150.0))
                    .show(ui, |ui| render_panel(ui, state, &view, &mut edit));
            });
        }
        if w.stats {
            window("Stats", egui::pos2(screen.max.x - 348.0, 34.0), 340.0)
                .show(&ctx, |ui| stats_panel(ui, &view));
        }
        if w.baked {
            window(
                "Baked light",
                egui::pos2(screen.max.x - 348.0, screen.max.y - 190.0),
                340.0,
            )
            .show(&ctx, |ui| baked_panel(ui, state, &view, &mut edit));
        }
        if w.compare {
            window(
                "Compare",
                egui::pos2(screen.center().x - 180.0, screen.max.y - 150.0),
                360.0,
            )
            .show(&ctx, |ui| compare_panel(ui, state, &view, &mut edit));
        }
        if w.help {
            window("Help", egui::pos2(screen.center().x - 220.0, 120.0), 440.0).show(&ctx, |ui| {
                ui.label(egui::RichText::new(HELP_TEXT).monospace());
            });
        }
    }
    state.pointer = ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui();
    state.keyboard = ctx.egui_wants_keyboard_input();
    Ok(())
}

fn window(title: &'static str, pos: egui::Pos2, width: f32) -> egui::Window<'static> {
    egui::Window::new(title)
        .default_pos(pos)
        .default_width(width)
        .collapsible(true)
        .resizable(true)
        .constrain(true)
}

fn top_bar(ctx: &egui::Context, state: &mut UiState) {
    egui::Area::new(egui::Id::new("top bar"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 4.0))
        .show(ctx, |ui| {
            egui::Frame::window(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("gfx_viewer").strong());
                    ui.separator();
                    let w = &mut state.windows;
                    ui.toggle_value(&mut w.scene, "Scene");
                    ui.toggle_value(&mut w.render, "Render");
                    ui.toggle_value(&mut w.stats, "Stats");
                    ui.toggle_value(&mut w.baked, "Baked light");
                    ui.toggle_value(&mut w.compare, "Compare");
                    ui.toggle_value(&mut w.help, "Help");
                    ui.separator();
                    ui.weak("Tab hides all");
                });
            });
        });
}

const NC: egui::Color32 = egui::Color32::from_rgb(240, 160, 70);

fn scene_panel(ui: &mut egui::Ui, state: &mut UiState, view: &View, edit: &mut Edit) {
    egui::ScrollArea::vertical()
        .max_height(110.0)
        .id_salt("scenes")
        .show(ui, |ui| {
            for (index, entry) in view.catalog.entries.iter().enumerate() {
                let current = index == view.stage.index;
                let mut text = egui::RichText::new(&entry.title);
                if !entry.available {
                    text = text.weak();
                }
                ui.horizontal(|ui| {
                    let response =
                        ui.add_enabled(entry.available, egui::Button::selectable(current, text));
                    if entry.non_commercial {
                        ui.label(egui::RichText::new("no comercial").small().color(NC));
                    }
                    let response = match (&entry.licence, entry.available) {
                        (_, false) => response.on_disabled_hover_text(format!(
                            "missing: run {}",
                            entry.fetch.as_deref().unwrap_or("tools/fetch-scenes.sh")
                        )),
                        (Some(licence), true) => response.on_hover_text(licence),
                        (None, true) => response.on_hover_text("this repository's test scene"),
                    };
                    if response.clicked() {
                        edit.actions.write(Action::OpenScene(index));
                    }
                    if !entry.available {
                        ui.label(
                            egui::RichText::new("run tools/fetch-scenes.sh")
                                .small()
                                .weak(),
                        );
                    }
                });
            }
        });
    let entry = &view.catalog.entries[view.stage.index];
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(&entry.title).strong());
        if view.stage.is_gltf() && !view.stage.placed {
            ui.weak("loading…");
        }
        if view.stage.failed {
            ui.colored_label(NC, "some files failed to load (see the log)");
        }
    });
    match &entry.licence {
        Some(licence) => {
            ui.label(egui::RichText::new(licence).small());
        }
        None => {
            ui.label(egui::RichText::new("test scene of this repository").small());
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        let field = ui.add(
            egui::TextEdit::singleline(&mut state.open_path)
                .hint_text("/path/to/scene.glb")
                .desired_width(190.0),
        );
        let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if ui.button("Open glTF").clicked() || enter {
            edit.actions
                .write(Action::OpenFile(state.open_path.trim().into()));
        }
    });
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("Camera:");
        if ui.button("Save view").clicked() {
            edit.actions.write(Action::SaveBookmark);
        }
        if ui.button("Reset").clicked() {
            edit.actions.write(Action::ResetCamera);
        }
        for index in 0..view.bookmarks.for_scene(&view.stage.name).len() {
            if ui
                .button(format!("{}", index + 1))
                .on_hover_text("go to this saved view")
                .clicked()
            {
                edit.actions.write(Action::GoBookmark(index));
            }
        }
    });
    ui.horizontal(|ui| {
        if ui
            .button(format!(
                "Lamps {}  [L]",
                if view.lamps.on { "on" } else { "off" }
            ))
            .clicked()
        {
            edit.actions.write(Action::ToggleLamps);
        }
        if ui.button("Screenshot  [F12]").clicked() {
            edit.actions.write(Action::Screenshot);
        }
    });
}

fn switch(ui: &mut egui::Ui, label: &str, value: &mut Option<bool>) {
    let text = |value: Option<bool>| match value {
        None => "preset",
        Some(true) => "on",
        Some(false) => "off",
    };
    ui.label(label);
    egui::ComboBox::from_id_salt(label)
        .width(70.0)
        .selected_text(text(*value))
        .show_ui(ui, |ui| {
            for option in [None, Some(true), Some(false)] {
                ui.selectable_value(value, option, text(option));
            }
        });
    ui.end_row();
}

fn render_panel(ui: &mut egui::Ui, state: &mut UiState, view: &View, edit: &mut Edit) {
    // Edited on a copy: writing the resource only when something changed keeps the ladder from resolving
    // again every frame.
    let mut s = edit.settings.clone();
    ui.horizontal(|ui| {
        ui.label("Preset");
        for (preset, name) in [
            (GraphicsPreset::Auto, "Auto"),
            (GraphicsPreset::Raster, "Raster"),
            (GraphicsPreset::RtLite, "RtLite"),
            (GraphicsPreset::RtFull, "RtFull"),
        ] {
            let response = ui.selectable_value(&mut s.preset, preset, name);
            state.rects.insert(
                match preset {
                    GraphicsPreset::Auto => "preset auto",
                    GraphicsPreset::Raster => "preset raster",
                    GraphicsPreset::RtLite => "preset rt-lite",
                    GraphicsPreset::RtFull => "preset rt-full",
                },
                response.rect,
            );
        }
    });
    egui::Grid::new("render grid")
        .num_columns(2)
        .spacing([8.0, 3.0])
        .show(ui, |ui| {
            ui.label("Upscaler");
            egui::ComboBox::from_id_salt("upscaler")
                .selected_text(format!("{:?}", s.upscaler))
                .show_ui(ui, |ui| {
                    for (value, name) in [
                        (UpscalerChoice::Auto, "Auto"),
                        (UpscalerChoice::Off, "Off (TAA)"),
                        (UpscalerChoice::Tsr, "TSR"),
                        (UpscalerChoice::Dlss, "DLSS"),
                        (UpscalerChoice::Fsr, "FSR"),
                    ] {
                        ui.selectable_value(&mut s.upscaler, value, name);
                    }
                });
            ui.end_row();
            // The ladder has no free render scale: the quality sets it (shown resolved in Stats).
            ui.label("Quality (render scale)");
            egui::ComboBox::from_id_salt("quality")
                .selected_text(format!("{:?}", s.upscale_quality))
                .show_ui(ui, |ui| {
                    for value in [
                        UpscaleQuality::Auto,
                        UpscaleQuality::NativeAa,
                        UpscaleQuality::Quality,
                        UpscaleQuality::Balanced,
                        UpscaleQuality::Performance,
                    ] {
                        ui.selectable_value(&mut s.upscale_quality, value, format!("{value:?}"));
                    }
                });
            ui.end_row();
            ui.label("Denoiser (traced)");
            let denoiser = |value: DenoiserChoice| match value {
                DenoiserChoice::Auto => "Auto",
                DenoiserChoice::DlssRayReconstruction => "DLSS-RR",
                DenoiserChoice::Nrd => "NRD",
            };
            egui::ComboBox::from_id_salt("denoiser")
                .selected_text(denoiser(s.denoiser))
                .show_ui(ui, |ui| {
                    for value in [
                        DenoiserChoice::Auto,
                        DenoiserChoice::DlssRayReconstruction,
                        DenoiserChoice::Nrd,
                    ] {
                        ui.selectable_value(&mut s.denoiser, value, denoiser(value));
                    }
                });
            ui.end_row();
        });
    egui::CollapsingHeader::new("Effects (overrides)")
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("overrides")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    let o = &mut s.overrides;
                    switch(ui, "Shadow maps", &mut o.shadows);
                    switch(ui, "SSAO (Raster)", &mut o.ssao);
                    switch(ui, "SSR (Raster)", &mut o.ssr);
                    switch(ui, "Contact shadows (Raster)", &mut o.contact_shadows);
                    switch(ui, "Bloom", &mut o.bloom);
                    switch(ui, "ReSTIR (traced)", &mut o.restir);
                    ui.label("Max bounces (traced)");
                    ui.horizontal(|ui| {
                        let mut own = o.max_bounces.is_some();
                        let response = ui.checkbox(&mut own, "");
                        state.rects.insert("bounces override", response.rect);
                        let mut bounces = o.max_bounces.unwrap_or(2);
                        ui.add_enabled(own, egui::Slider::new(&mut bounces, 0..=8));
                        o.max_bounces = own.then_some(bounces);
                    });
                    ui.end_row();
                    ui.label("Baked GI");
                    let mut shown = o.baked_gi != Some(false);
                    ui.checkbox(&mut shown, "shown when loaded");
                    o.baked_gi = (!shown).then_some(false);
                    ui.end_row();
                    ui.label("Ambient while baked");
                    ui.add(
                        egui::Slider::new(&mut s.ambient_while_baked, 0.0..=200.0)
                            .logarithmic(true)
                            .suffix(" cd/m2"),
                    );
                    ui.end_row();
                });
        });
    if s != *edit.settings {
        *edit.settings = s;
    }

    egui::CollapsingHeader::new("View")
        .default_open(true)
        .show(ui, |ui| {
            let mut c = edit.controls.clone();
            egui::Grid::new("view grid")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    ui.label("Exposure (EV100)");
                    ui.horizontal(|ui| {
                        ui.add(egui::Slider::new(&mut c.exposure, 0.0..=16.0).step_by(0.1));
                        if ui
                            .small_button("↺")
                            .on_hover_text("the scene's own")
                            .clicked()
                        {
                            c.exposure = view.stage.exposure;
                        }
                    });
                    ui.end_row();
                    ui.label("Lamp intensity ×");
                    ui.add(
                        egui::Slider::new(&mut c.lamp_scale, 0.0..=8.0)
                            .logarithmic(true)
                            .smallest_positive(0.01),
                    );
                    ui.end_row();
                    if let Some(sun) = &mut c.sun {
                        ui.label("Sun height (time of day)");
                        ui.add(egui::Slider::new(&mut sun.elevation, -10.0..=90.0).suffix("°"));
                        ui.end_row();
                        ui.label("Sun direction");
                        ui.add(egui::Slider::new(&mut sun.azimuth, 0.0..=360.0).suffix("°"));
                        ui.end_row();
                        ui.label("Sun intensity");
                        ui.add(
                            egui::Slider::new(&mut sun.lux, 0.0..=150_000.0)
                                .logarithmic(true)
                                .suffix(" lx"),
                        );
                        ui.end_row();
                    }
                });
            if c.sun.is_none() {
                ui.weak("This scene has no sun.");
            }
            if c != *edit.controls {
                *edit.controls = c;
            }
        });
}

fn stats_panel(ui: &mut egui::Ui, view: &View) {
    match view.times.median_ms() {
        Some(ms) => ui.label(
            egui::RichText::new(format!("{:.0} fps   {ms:.2} ms", 1000.0 / ms))
                .strong()
                .size(15.0),
        ),
        None => ui.label("measuring…"),
    };
    frame_graph(ui, &view.times);
    let compiling = view.pending.0.load(Ordering::Relaxed);
    if compiling > 0 && compiling != u32::MAX {
        ui.colored_label(NC, format!("compiling {compiling} pipelines…"));
    }
    egui::Grid::new("stats grid")
        .num_columns(2)
        .spacing([8.0, 1.0])
        .show(ui, |ui| {
            let c = &view.capabilities;
            ui.label("Adapter");
            ui.label(&c.adapter_name);
            ui.end_row();
            ui.label("Driver");
            ui.label(format!("{} {}", c.driver, c.driver_info));
            ui.end_row();
            if let Some(resolved) = &view.resolved {
                ui.label("Rung");
                ui.label(egui::RichText::new(rung_text(resolved)).strong());
                ui.end_row();
                ui.label("Upscaler");
                ui.label(format!("{:?} ({:?})", resolved.upscaler, resolved.quality));
                ui.end_row();
                ui.label("Denoiser");
                ui.label(format!("{:?}", resolved.denoiser));
                ui.end_row();
                ui.label("Render scale");
                ui.label(format!("{:.0} %", resolved.render_scale * 100.0));
                ui.end_row();
            }
            ui.label("Baked GI");
            ui.label(format!(
                "{:?}",
                view.status
                    .as_deref()
                    .copied()
                    .unwrap_or(BakedGiStatus::Absent)
            ));
            ui.end_row();
            ui.label("Settings");
            ui.label(if view.ab.showing_b { "set B" } else { "set A" });
            ui.end_row();
        });
    if let Some(resolved) = &view.resolved {
        if resolved.unfitted_meshes > 0 || resolved.skipped_deforming_meshes > 0 {
            ui.label(format!(
                "Not ray traced: {} meshes that cannot be fitted, {} skinned/morphed",
                resolved.unfitted_meshes, resolved.skipped_deforming_meshes
            ));
        }
        if !resolved.fallbacks.is_empty() {
            ui.label(egui::RichText::new("Why it differs from what was asked:").small());
            for fallback in &resolved.fallbacks {
                ui.label(
                    egui::RichText::new(format!("· {fallback}"))
                        .small()
                        .color(NC),
                );
            }
        }
    }
    if let Some((message, at)) = &view.bakes.message
        && at.elapsed() < Duration::from_secs(20)
    {
        ui.separator();
        ui.label(egui::RichText::new(message).small());
    }
    let passes = gpu_passes(&view.diagnostics);
    if !passes.is_empty() {
        ui.separator();
        ui.label(egui::RichText::new("GPU ms per pass").small());
        let longest = passes[0].1.max(0.01);
        for (pass, ms) in passes.iter().take(10) {
            ui.add(
                egui::ProgressBar::new((*ms / longest) as f32)
                    .desired_height(13.0)
                    .text(egui::RichText::new(format!("{pass}  {ms:.2}")).small()),
            );
        }
    }
}

/// The frame times of the last seconds as a line, with the 16.7 and 33.3 ms marks.
fn frame_graph(ui: &mut egui::Ui, times: &FrameTimes) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 56.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(14, 15, 18));
    let Some((last, _)) = times.times.back() else {
        return;
    };
    let top = times
        .times
        .iter()
        .map(|(_, ms)| *ms)
        .fold(20.0_f64, f64::max)
        .min(200.0);
    let y = |ms: f64| rect.bottom() - (ms / top).min(1.0) as f32 * rect.height();
    for mark in [1000.0 / 60.0, 1000.0 / 30.0] {
        if mark < top {
            painter.hline(
                rect.x_range(),
                y(mark),
                egui::Stroke::new(1.0, egui::Color32::from_gray(50)),
            );
        }
    }
    let points: Vec<egui::Pos2> = times
        .times
        .iter()
        .map(|(at, ms)| {
            let age = (*last - *at).as_secs_f64();
            egui::pos2(
                rect.right() - (age / GRAPH_SECONDS) as f32 * rect.width(),
                y(*ms),
            )
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.2, egui::Color32::from_rgb(110, 200, 120)),
    ));
    painter.text(
        rect.left_top() + egui::vec2(3.0, 1.0),
        egui::Align2::LEFT_TOP,
        format!("{top:.0} ms"),
        egui::FontId::monospace(9.0),
        egui::Color32::from_gray(140),
    );
}

fn baked_panel(ui: &mut egui::Ui, state: &mut UiState, view: &View, edit: &mut Edit) {
    ui.label(format!(
        "Status: {:?}",
        view.status
            .as_deref()
            .copied()
            .unwrap_or(BakedGiStatus::Absent)
    ));
    match &view.bakes.loaded {
        Some(dir) => ui.label(egui::RichText::new(dir.display().to_string()).small()),
        None => ui.weak("no bake loaded for this scene"),
    };
    ui.horizontal(|ui| {
        let running = view.bakes.running_for();
        let response = ui.add_enabled(running.is_none(), egui::Button::new("Bake  [B]"));
        state.rects.insert("bake", response.rect);
        if response.clicked() {
            edit.actions.write(Action::Bake);
        }
        if ui.button("Load newest").clicked() {
            edit.actions.write(Action::LoadBake);
        }
        if ui.button("Unload").clicked() {
            edit.actions.write(Action::UnloadBake);
        }
    });
    if let Some(seconds) = view.bakes.running_for() {
        ui.add(egui::Spinner::new());
        ui.label(format!("baking… {seconds} s"));
        if let Some(line) = view.bakes.progress_line() {
            ui.label(egui::RichText::new(line).small().monospace());
        }
    }
    let mut coverage = edit.controls.coverage;
    ui.checkbox(
        &mut coverage,
        "Coverage view (bake alone: white matte, no lamps)",
    )
    .on_hover_text("Meaningful on Raster with a bake loaded");
    if coverage != edit.controls.coverage {
        edit.controls.coverage = coverage;
    }
}

fn compare_panel(ui: &mut egui::Ui, state: &mut UiState, view: &View, edit: &mut Edit) {
    ui.horizontal(|ui| {
        ui.label(format!(
            "Showing set {}",
            if view.ab.showing_b { "B" } else { "A" }
        ));
        if ui.button("Flip A/B  [C]").clicked() {
            edit.actions.write(Action::FlipAb);
        }
        if ui
            .button("Copy to the other set")
            .on_hover_text("the other set becomes what is shown now")
            .clicked()
        {
            edit.actions.write(Action::CopyToOther);
        }
    });
    let mut on = edit.split.on;
    let response = ui.checkbox(
        &mut on,
        "Split view  [V]: the other set on the left, live on the right",
    );
    state.rects.insert("split", response.rect);
    if on != edit.split.on {
        edit.actions.write(Action::Split(on));
    }
    if edit.split.on {
        ui.add(egui::Slider::new(&mut edit.split.divider, 0.05..=0.95).text("divider"));
        ui.horizontal(|ui| {
            if ui.button("Refresh left").clicked() {
                edit.actions.write(Action::RefreshSplit);
            }
            ui.checkbox(&mut edit.split.auto, "refresh when the camera stops");
        });
        ui.label(egui::RichText::new(edit.split.status()).small());
    }
}

/// Keeps the camera and the viewer's keys off while a panel has the pointer or the keyboard. A look that
/// started over the scene goes on when the pointer crosses a panel.
pub fn gate_camera(
    mut state: ResMut<UiState>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cameras: Query<&mut FreeCameraState>,
) {
    if buttons.just_pressed(MouseButton::Right) {
        state.looking = !state.pointer;
    }
    if !buttons.pressed(MouseButton::Right) {
        state.looking = false;
    }
    let enabled = !state.keyboard && (!state.pointer || state.looking);
    for mut camera in &mut cameras {
        if camera.enabled != enabled {
            camera.enabled = enabled;
        }
    }
}
