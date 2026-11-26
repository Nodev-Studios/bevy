//! The settings panel (`Tab`), the live overlay (always shown) and the key help (`H`).

use crate::{
    actions::{Action, Lamps},
    bake::Bakes,
    catalog::Catalog,
    stage::Stage,
    AbCompare, PendingPipelines,
};
use bevy::{
    diagnostic::DiagnosticsStore,
    graphics::{
        baked::BakedGiStatus, raster::RasterLevel, DenoiserChoice, GraphicsCapabilities,
        GraphicsPreset, GraphicsRung, GraphicsSettings, ResolvedGraphics, UpscaleQuality,
        UpscalerChoice,
    },
    picking::hover::Hovered,
    prelude::*,
    ui::Pressed,
    ui_widgets::{observe, Activate, Button},
};
use std::{
    collections::VecDeque,
    fmt::Write,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

/// One setting of [`GraphicsSettings`] the panel edits. A click moves it to its next value.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Preset,
    Upscaler,
    Quality,
    Denoiser,
    Shadows,
    Ssao,
    Ssr,
    ContactShadows,
    Bloom,
    Restir,
    MaxBounces,
    BakedGi,
    Ambient,
}

impl Field {
    pub const ALL: [Field; 13] = [
        Field::Preset,
        Field::Upscaler,
        Field::Quality,
        Field::Denoiser,
        Field::Shadows,
        Field::Ssao,
        Field::Ssr,
        Field::ContactShadows,
        Field::Bloom,
        Field::Restir,
        Field::MaxBounces,
        Field::BakedGi,
        Field::Ambient,
    ];
}

/// The values `ambient_while_baked` steps through, in cd/m². The first is `bevy_graphics`'s default.
const AMBIENTS: [f32; 5] = [
    bevy::graphics::baked::DEFAULT_AMBIENT_WHILE_BAKED,
    30.0,
    100.0,
    0.0,
    3.0,
];

/// The values the bounce override steps through: the preset's, then these. to tune.
const BOUNCES: [Option<u32>; 6] = [None, Some(1), Some(2), Some(3), Some(4), Some(6)];

fn next<T: Copy + PartialEq>(values: &[T], current: T) -> T {
    let index = values.iter().position(|value| *value == current).unwrap_or(0);
    values[(index + 1) % values.len()]
}

fn next_switch(current: Option<bool>) -> Option<bool> {
    next(&[None, Some(true), Some(false)], current)
}

/// Moves `field` of `settings` to its next value.
pub fn cycle(settings: &mut GraphicsSettings, field: Field) {
    use DenoiserChoice as D;
    use GraphicsPreset as P;
    use UpscaleQuality as Q;
    use UpscalerChoice as U;
    let o = &mut settings.overrides;
    match field {
        Field::Preset => {
            settings.preset = next(&[P::Auto, P::Raster, P::RtLite, P::RtFull], settings.preset)
        }
        Field::Upscaler => {
            settings.upscaler = next(
                &[U::Auto, U::Off, U::Tsr, U::Dlss, U::Fsr],
                settings.upscaler,
            )
        }
        Field::Quality => {
            settings.upscale_quality = next(
                &[Q::Auto, Q::NativeAa, Q::Quality, Q::Balanced, Q::Performance],
                settings.upscale_quality,
            )
        }
        Field::Denoiser => {
            settings.denoiser = next(
                &[D::Auto, D::DlssRayReconstruction, D::Nrd],
                settings.denoiser,
            )
        }
        Field::Shadows => o.shadows = next_switch(o.shadows),
        Field::Ssao => o.ssao = next_switch(o.ssao),
        Field::Ssr => o.ssr = next_switch(o.ssr),
        Field::ContactShadows => o.contact_shadows = next_switch(o.contact_shadows),
        Field::Bloom => o.bloom = next_switch(o.bloom),
        Field::Restir => o.restir = next_switch(o.restir),
        Field::MaxBounces => o.max_bounces = next(&BOUNCES, o.max_bounces),
        Field::BakedGi => o.baked_gi = next_switch(o.baked_gi),
        Field::Ambient => {
            let nearest = AMBIENTS
                .iter()
                .position(|value| (*value - settings.ambient_while_baked).abs() < 1e-3);
            settings.ambient_while_baked =
                AMBIENTS[nearest.map_or(0, |index| (index + 1) % AMBIENTS.len())];
        }
    }
}

fn switch_text(value: Option<bool>) -> &'static str {
    match value {
        None => "preset",
        Some(true) => "on",
        Some(false) => "off",
    }
}

/// The panel's text for `field`.
pub fn label(settings: &GraphicsSettings, field: Field) -> String {
    let o = &settings.overrides;
    match field {
        Field::Preset => format!("Preset: {:?}", settings.preset),
        Field::Upscaler => format!("Upscaler: {:?}", settings.upscaler),
        Field::Quality => format!("Upscale quality: {:?}", settings.upscale_quality),
        Field::Denoiser => format!(
            "Denoiser: {}",
            match settings.denoiser {
                DenoiserChoice::Auto => "Auto",
                DenoiserChoice::DlssRayReconstruction => "DLSS-RR",
                DenoiserChoice::Nrd => "NRD",
            }
        ),
        Field::Shadows => format!("Shadow maps: {}", switch_text(o.shadows)),
        Field::Ssao => format!("SSAO (Raster): {}", switch_text(o.ssao)),
        Field::Ssr => format!("SSR (Raster): {}", switch_text(o.ssr)),
        Field::ContactShadows => {
            format!("Contact shadows (Raster): {}", switch_text(o.contact_shadows))
        }
        Field::Bloom => format!("Bloom: {}", switch_text(o.bloom)),
        Field::Restir => format!("ReSTIR (traced): {}", switch_text(o.restir)),
        Field::MaxBounces => format!(
            "Max bounces (traced): {}",
            o.max_bounces
                .map_or_else(|| "preset".into(), |n| n.to_string())
        ),
        Field::BakedGi => format!(
            "Baked GI: {}",
            match o.baked_gi {
                Some(false) => "hidden",
                _ => "shown when loaded",
            }
        ),
        Field::Ambient => format!(
            // The default font has no superscript two.
            "Ambient while baked: {} cd/m2",
            settings.ambient_while_baked
        ),
    }
}

/// The panel's root.
#[derive(Component)]
pub struct Panel;

/// The overlay's text.
#[derive(Component)]
pub struct OverlayText;

/// The help's root.
#[derive(Component)]
pub struct Help;

/// Every root of the viewer's UI, hidden for the self-test's captures.
#[derive(Component)]
pub struct ViewerUi;

/// A scene button: opens catalog entry `0`.
#[derive(Component)]
pub struct SceneButton(pub usize);

const FONT: f32 = 13.0;
const BUTTON: Color = Color::srgba(0.16, 0.16, 0.18, 0.95);
const BUTTON_HOVER: Color = Color::srgba(0.26, 0.26, 0.30, 0.95);
const BUTTON_PRESS: Color = Color::srgba(0.30, 0.55, 0.35, 0.95);
const BUTTON_CURRENT: Color = Color::srgba(0.22, 0.35, 0.55, 0.95);
const PANEL: Color = Color::srgba(0.05, 0.05, 0.06, 0.82);

/// The help text: every key.
pub const HELP_TEXT: &str = "\
gfx_viewer keys

  Right mouse (hold)   look around
  W A S D              move           Q / E   down / up
  Shift                move faster    Wheel   change speed
  M                    keep the mouse captured (toggle)
  Tab                  settings panel (click a row for its next value)
  L                    switch the scene's lamp(s) off / on
  B                    bake indirect light for this scene (child process)
  C                    A/B: flip between two settings sets
  F12                  screenshot
  H                    this help
  Esc                  quit
";

fn text(value: impl Into<String>) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font_size: FontSize::Px(FONT),
            ..default()
        },
        TextColor(Color::srgb(0.92, 0.92, 0.92)),
    )
}

fn header(value: &str) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font_size: FontSize::Px(FONT + 1.0),
            ..default()
        },
        TextColor(Color::srgb(0.65, 0.80, 1.0)),
        Node {
            margin: UiRect::top(px(6)),
            ..default()
        },
    )
}

fn button(label: String) -> impl Bundle {
    (
        Button,
        Hovered::default(),
        Node {
            padding: UiRect::axes(px(6), px(2)),
            margin: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(3)),
            ..default()
        },
        BackgroundColor(BUTTON),
        children![text(label)],
    )
}

/// Spawns the panel, the overlay and the help.
pub fn spawn_ui(mut commands: Commands, catalog: Res<Catalog>, settings: Res<GraphicsSettings>) {
    commands.spawn((
        ViewerUi,
        Node {
            position_type: PositionType::Absolute,
            left: px(6),
            top: px(6),
            padding: UiRect::all(px(6)),
            max_width: px(620),
            border_radius: BorderRadius::all(px(4)),
            ..default()
        },
        BackgroundColor(PANEL),
        children![(OverlayText, text(""))],
    ));

    let panel = commands
        .spawn((
            ViewerUi,
            Panel,
            Node {
                position_type: PositionType::Absolute,
                right: px(6),
                top: px(6),
                width: px(320),
                max_height: percent(98),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(6)),
                overflow: Overflow::clip(),
                border_radius: BorderRadius::all(px(4)),
                ..default()
            },
            BackgroundColor(PANEL),
        ))
        .id();
    commands
        .entity(panel)
        .with_child(header("Scene"));
    let scenes = commands
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .id();
    commands.entity(panel).add_child(scenes);
    for (index, entry) in catalog.entries.iter().enumerate() {
        commands.entity(scenes).with_child((
            button(entry.name.clone()),
            SceneButton(index),
            observe(move |_: On<Activate>, mut actions: MessageWriter<Action>| {
                actions.write(Action::OpenScene(index));
            }),
        ));
    }
    commands
        .entity(panel)
        .with_child(header("Graphics settings (Tab hides)"));
    for field in Field::ALL {
        commands.entity(panel).with_child((
            button(label(&settings, field)),
            field,
            observe(move |_: On<Activate>, mut actions: MessageWriter<Action>| {
                actions.write(Action::Cycle(field));
            }),
        ));
    }
    commands.entity(panel).with_child(header("Actions"));
    for (name, action) in [
        ("Lamp on/off  [L]", Action::ToggleLamps),
        ("Bake indirect light  [B]", Action::Bake),
        ("A/B flip  [C]", Action::FlipAb),
        ("Screenshot  [F12]", Action::Screenshot),
        ("Help  [H]", Action::ToggleHelp),
    ] {
        commands.entity(panel).with_child((
            button(name.into()),
            observe(move |_: On<Activate>, mut actions: MessageWriter<Action>| {
                actions.write(action);
            }),
        ));
    }

    commands.spawn((
        ViewerUi,
        Help,
        Node {
            position_type: PositionType::Absolute,
            left: px(6),
            bottom: px(6),
            padding: UiRect::all(px(12)),
            display: Display::None,
            border_radius: BorderRadius::all(px(6)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.88)),
        children![(
            Text::new(HELP_TEXT),
            TextFont {
                font_size: FontSize::Px(FONT + 2.0),
                ..default()
            },
        )],
    ));
}

/// Keeps every button's colour and every setting's text current.
pub fn refresh_panel(
    settings: Res<GraphicsSettings>,
    stage: Res<Stage>,
    mut buttons: Query<
        (
            &Hovered,
            Has<Pressed>,
            Option<&Field>,
            Option<&SceneButton>,
            &mut BackgroundColor,
            &Children,
        ),
        With<Button>,
    >,
    mut texts: Query<&mut Text>,
) {
    for (hovered, pressed, field, scene, mut color, children) in &mut buttons {
        let current = scene.is_some_and(|scene| scene.0 == stage.index);
        let wanted = if pressed {
            BUTTON_PRESS
        } else if hovered.get() {
            BUTTON_HOVER
        } else if current {
            BUTTON_CURRENT
        } else {
            BUTTON
        };
        if color.0 != wanted {
            color.0 = wanted;
        }
        if let Some(field) = field
            && settings.is_changed()
            && let Ok(mut text) = texts.get_mut(children[0])
        {
            **text = label(&settings, *field);
        }
    }
}

/// Frame times of the last second, for the overlay's median.
#[derive(Resource, Default)]
pub struct FrameTimes {
    times: VecDeque<(Instant, f64)>,
    last: Option<Instant>,
}

impl FrameTimes {
    /// The median frame time of the last second, in ms.
    pub fn median_ms(&self) -> Option<f64> {
        let mut values: Vec<f64> = self.times.iter().map(|(_, ms)| *ms).collect();
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
        .is_some_and(|(at, _)| now - *at > Duration::from_secs(1))
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

/// Writes the overlay four times a second.
pub fn update_overlay(
    mut last: Local<Option<Instant>>,
    times: Res<FrameTimes>,
    diagnostics: Res<DiagnosticsStore>,
    resolved: Option<Res<ResolvedGraphics>>,
    capabilities: Res<GraphicsCapabilities>,
    status: Option<Res<BakedGiStatus>>,
    bakes: Res<Bakes>,
    stage: Res<Stage>,
    ab: Res<AbCompare>,
    lamps: Res<Lamps>,
    pending: Res<PendingPipelines>,
    mut overlay: Single<&mut Text, With<OverlayText>>,
) {
    let now = Instant::now();
    if last.is_some_and(|last| now - last < Duration::from_millis(250)) {
        return;
    }
    *last = Some(now);
    let mut out = String::new();
    match times.median_ms() {
        Some(ms) => {
            let _ = write!(out, "{:.0} fps   {ms:.2} ms/frame (median, last second)", 1000.0 / ms);
        }
        None => out.push_str("measuring…"),
    }
    let compiling = pending.0.load(Ordering::Relaxed);
    if compiling > 0 && compiling != u32::MAX {
        let _ = write!(out, "   compiling {compiling} pipelines…");
    }
    let _ = write!(
        out,
        "\n{} ({}, {})",
        capabilities.adapter_name, capabilities.driver, capabilities.driver_info
    );
    let _ = write!(
        out,
        "\nScene: {}{}   settings {}   lamps {}",
        stage.name,
        if stage.is_gltf() && !stage.placed {
            " (loading…)"
        } else {
            ""
        },
        if ab.showing_b { "B" } else { "A" },
        if lamps.on { "on" } else { "off" },
    );
    if let Some(resolved) = &resolved {
        let _ = write!(
            out,
            "\nRung {}   upscaler {:?}   denoiser {:?}   {:?} = {:.0} % render scale",
            rung_text(resolved),
            resolved.upscaler,
            resolved.denoiser,
            resolved.quality,
            resolved.render_scale * 100.0
        );
        if resolved.unfitted_meshes > 0 || resolved.skipped_deforming_meshes > 0 {
            let _ = write!(
                out,
                "\nNot ray traced: {} meshes that cannot be fitted, {} skinned/morphed",
                resolved.unfitted_meshes, resolved.skipped_deforming_meshes
            );
        }
        for fallback in &resolved.fallbacks {
            let _ = write!(out, "\n  · {fallback}");
        }
    }
    let _ = write!(
        out,
        "\nBaked GI: {:?}",
        status.map_or(BakedGiStatus::Absent, |status| *status)
    );
    if let Some(seconds) = bakes.running_for() {
        let _ = write!(out, "   baking… {seconds} s");
    }
    if stage.failed {
        out.push_str("\nSome files of this scene failed to load: see the log");
    }
    if let Some((message, at)) = &bakes.message
        && at.elapsed() < Duration::from_secs(20)
    {
        let _ = write!(out, "\n{message}");
    }
    let passes = gpu_passes(&diagnostics);
    if !passes.is_empty() {
        out.push_str("\nGPU ms:");
        for (index, (pass, ms)) in passes.iter().take(6).enumerate() {
            let _ = write!(
                out,
                "{}{pass} {ms:.2}",
                if index % 3 == 0 { "\n  " } else { "   " }
            );
        }
    }
    out.push_str("\n[Tab] settings   [H] help");
    overlay.0 = out;
}
