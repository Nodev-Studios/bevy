//! `--self-test DIR`: walks through every scene × every preset, a few overrides, the lamp switch, a panel click,
//! the A/B flip and one bake + load on `interior`, through the same [`Action`]s the keys and the panel write.
//! Each step waits until its scene is loaded and no pipeline compiles, holds [`HOLD_FRAMES`] frames, captures
//! the frame (with the viewer's UI hidden, except the last step's) and checks it:
//!
//! - the frame is not black;
//! - what resolved is what the step asked for, or a [`Fallback`] of the right kind says why not;
//! - the camera carries the effects the resolved rung and the overrides call for (the apply layer did its job);
//! - the lamp step differs from the frame before it, the bake step loaded a bake and changed the frame.
//!
//! It writes `NN-<step>.png` per step and `report.json` (rewritten after each step) into DIR, and exits 0 only
//! when no step failed. A panic ends the process with a non-zero code; `report.json` then holds the steps
//! before it.

use crate::{
    actions::Action,
    bake::Bakes,
    catalog::{Catalog, Source},
    stage::Stage,
    ui::{gpu_passes, rung_text, Field, FrameTimes, ViewerUi},
    AbCompare, PendingPipelines, ViewerCamera,
};
use bevy::{
    app::AppExit,
    asset::RecursiveDependencyLoadState,
    diagnostic::DiagnosticsStore,
    graphics::{
        baked::BakedGiStatus,
        raster::{raster_with, traced_bloom},
        traced::solari_lighting,
        DenoiserChoice, Fallback, GraphicsCapabilities, GraphicsPreset, GraphicsRung,
        GraphicsSettings, ResolvedDenoiser, ResolvedGraphics, ResolvedQuality, ResolvedUpscaler,
        UpscaleQuality, UpscalerChoice,
    },
    pbr::{ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    solari::prelude::SolariLighting,
    ui_widgets::Activate,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{atomic::Ordering, Arc, Mutex},
    time::{Duration, Instant},
};

/// Frames a step is held after its scene is ready and no pipeline compiles: the temporal filters, the world
/// cache and the denoisers' history settle. to tune.
const HOLD_FRAMES: u32 = 90;
/// Frames in a row with no pipeline compiling before a step counts as ready.
const CALM_FRAMES: u32 = 3;
/// Frames after a change before the pipeline count is trusted (the new components reach the render world and
/// queue their pipelines a frame or two later).
const SWITCH_FRAMES: u32 = 5;
/// A step that is not ready after this long fails instead of hanging the run.
const STEP_LIMIT: Duration = Duration::from_secs(240);
/// How long a bake may take.
const BAKE_LIMIT: Duration = Duration::from_secs(900);
/// A frame whose mean 8-bit luma is under this, or whose lit share is under [`MIN_LIT_SHARE`], is black.
/// to tune.
const MIN_MEAN_LUMA: f64 = 1.5;
/// The share of pixels with a luma over [`LIT_LUMA`] a frame needs to count as lit. to tune.
const MIN_LIT_SHARE: f64 = 0.002;
const LIT_LUMA: f64 = 8.0;
/// The mean absolute difference (8-bit) a lamp switch or a loaded bake must make to the frame. to tune.
const MIN_CHANGE: f64 = 0.5;

/// What a step does besides showing its settings.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Plain,
    /// Switches the lamps off; the frame must differ from step `reference`'s. The next step switches them on.
    LampsOff {
        reference: String,
    },
    /// Clicks the panel's preset row; the preset must move to the next value.
    PanelClick,
    /// Shows these settings as set A, `other` as set B, and flips to B.
    AbFlip {
        other: GraphicsSettings,
    },
    /// Bakes the scene, loads the bake, and compares with step `reference`'s frame.
    Bake {
        reference: String,
    },
    /// Captures with the overlay, the panel and the help shown. Not checked for black.
    ShowUi,
}

#[derive(Clone, Debug)]
struct Step {
    name: String,
    scene: usize,
    settings: GraphicsSettings,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// The step's settings and action go in.
    Start,
    /// Waiting for a bake to finish and load.
    Baking { finished_before: u32 },
    /// Waiting for the scene and the pipelines, then holding.
    Settling {
        changed_at: u32,
        calm: u32,
        held: Option<u32>,
    },
    /// The screenshot is on its way.
    Capturing,
}

/// The self-test's flags and progress.
#[derive(Resource)]
pub struct SelfTest {
    dir: PathBuf,
    presets: Vec<GraphicsPreset>,
    scenes: Option<Vec<String>>,
    /// The scenes baked and loaded.
    bake: Vec<String>,
    steps: Vec<Step>,
    index: usize,
    state: State,
    frame: u32,
    step_started: Instant,
    started: Instant,
    capture: Arc<Mutex<Option<Image>>>,
    references: HashMap<String, Vec<u8>>,
    results: Vec<Value>,
    failures: Vec<String>,
    done: bool,
}

pub const HELP: &str = "
Self-test:
  --self-test DIR                every scene x every preset, overrides, lamps, panel, A/B and a bake on
                                 interior; screenshots and report.json in DIR; exit 0 when every step passes
  --self-test-presets auto,raster,rt-lite,rt-full
                                 the presets each scene is shown with (default all four)
  --self-test-scenes NAME,...    only these scenes (default every scene of the list)
  --self-test-bake NAME,...      the scenes baked and loaded (default interior; an empty value bakes none)
";

impl SelfTest {
    /// Takes the self-test's flags; `None` without `--self-test`.
    pub fn parse(flags: &mut crate::Flags) -> Option<Self> {
        let dir = flags.value("--self-test").map(PathBuf::from);
        let presets = flags.value("--self-test-presets");
        let scenes = flags.value("--self-test-scenes");
        let bake = flags.value("--self-test-bake");
        let dir = dir?;
        let presets = match presets {
            None => vec![
                GraphicsPreset::Auto,
                GraphicsPreset::Raster,
                GraphicsPreset::RtLite,
                GraphicsPreset::RtFull,
            ],
            Some(list) => list
                .split(',')
                .filter_map(|name| {
                    let preset = preset_from(name);
                    if preset.is_none() {
                        flags.error(format!("--self-test-presets: no preset called {name}"));
                    }
                    preset
                })
                .collect(),
        };
        Some(Self {
            dir,
            presets,
            scenes: scenes.map(|list| list.split(',').map(str::to_owned).collect()),
            bake: bake.map_or_else(
                || vec!["interior".to_owned()],
                |list| {
                    list.split(',')
                        .filter(|name| !name.is_empty())
                        .map(str::to_owned)
                        .collect()
                },
            ),
            steps: Vec::new(),
            index: 0,
            state: State::Start,
            frame: 0,
            step_started: Instant::now(),
            started: Instant::now(),
            capture: Arc::default(),
            references: HashMap::new(),
            results: Vec::new(),
            failures: Vec::new(),
            done: false,
        })
    }

    /// Where the self-test's bakes go: inside its folder, so no bake from an earlier run is loaded.
    pub fn bakes_dir(&self) -> PathBuf {
        self.dir.join("bakes")
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    /// Lays out the steps for this catalog.
    pub fn plan(&mut self, catalog: &Catalog) {
        let wanted = |name: &str| {
            self.scenes
                .as_ref()
                .is_none_or(|scenes| scenes.iter().any(|scene| scene == name))
        };
        let with = |preset: GraphicsPreset| GraphicsSettings {
            preset,
            ..default()
        };
        let mut steps = Vec::new();
        let has = |preset| self.presets.contains(&preset);
        for (index, entry) in catalog.entries.iter().enumerate() {
            if !wanted(&entry.name) {
                continue;
            }
            let mut step = |name: &str, settings: GraphicsSettings, kind: Kind| {
                steps.push(Step {
                    name: format!("{}-{name}", entry.name),
                    scene: index,
                    settings,
                    kind,
                });
            };
            for preset in &self.presets {
                step(preset_name(*preset), with(*preset), Kind::Plain);
            }
            use GraphicsPreset::{Raster, RtFull, RtLite};
            let interior = matches!(entry.source, Source::Builtin("interior"));
            if interior && has(Raster) {
                let mut s = with(Raster);
                s.overrides.ssr = Some(false);
                s.overrides.ssao = Some(false);
                step("raster-no-ssr-no-ssao", s, Kind::Plain);
                let mut s = with(Raster);
                s.overrides.bloom = Some(false);
                s.overrides.contact_shadows = Some(false);
                step("raster-no-bloom-no-contact-shadows", s, Kind::Plain);
                let mut s = with(Raster);
                s.overrides.shadows = Some(false);
                step("raster-no-shadow-maps", s, Kind::Plain);
                let mut s = with(Raster);
                s.upscaler = UpscalerChoice::Tsr;
                s.upscale_quality = UpscaleQuality::Performance;
                step("raster-tsr-performance", s, Kind::Plain);
                let mut s = with(Raster);
                s.upscaler = UpscalerChoice::Dlss;
                s.upscale_quality = UpscaleQuality::Quality;
                step("raster-dlss-quality", s, Kind::Plain);
                let mut s = with(Raster);
                s.upscaler = UpscalerChoice::Off;
                step("raster-upscaler-off", s, Kind::Plain);
                step(
                    "raster-lamps-off",
                    with(Raster),
                    Kind::LampsOff {
                        reference: format!("{}-raster", entry.name),
                    },
                );
            }
            if interior && has(RtLite) {
                let mut s = with(RtLite);
                s.denoiser = DenoiserChoice::Nrd;
                step("rt-lite-nrd", s, Kind::Plain);
                let mut s = with(RtLite);
                s.denoiser = DenoiserChoice::DlssRayReconstruction;
                step("rt-lite-dlss-rr", s, Kind::Plain);
                let mut s = with(RtLite);
                s.denoiser = DenoiserChoice::Nrd;
                s.upscaler = UpscalerChoice::Tsr;
                s.upscale_quality = UpscaleQuality::Balanced;
                step("rt-lite-nrd-tsr-balanced", s, Kind::Plain);
                step(
                    "rt-lite-lamps-off",
                    with(RtLite),
                    Kind::LampsOff {
                        reference: format!("{}-rt-lite", entry.name),
                    },
                );
            }
            if interior && has(RtFull) {
                let mut s = with(RtFull);
                s.overrides.restir = Some(false);
                s.overrides.max_bounces = Some(1);
                s.overrides.bloom = Some(false);
                step("rt-full-no-restir-1-bounce-no-bloom", s, Kind::Plain);
            }
            if interior {
                step("panel-click", with(GraphicsPreset::Auto), Kind::PanelClick);
            }
            if interior && has(Raster) && has(RtLite) {
                step(
                    "ab-flip",
                    with(Raster),
                    Kind::AbFlip {
                        other: with(RtLite),
                    },
                );
            }
            if self.bake.contains(&entry.name) && has(Raster) {
                step(
                    "raster-baked",
                    with(Raster),
                    Kind::Bake {
                        reference: format!("{}-raster", entry.name),
                    },
                );
                let mut s = with(Raster);
                s.overrides.baked_gi = Some(false);
                step("raster-baked-hidden", s, Kind::Plain);
                let mut s = with(Raster);
                s.ambient_while_baked = 0.0;
                step("raster-baked-no-ambient", s, Kind::Plain);
                if has(RtFull) {
                    step("rt-full-baked", with(RtFull), Kind::Plain);
                }
            }
            if interior {
                step("ui", with(GraphicsPreset::Auto), Kind::ShowUi);
            }
        }
        self.steps = steps;
        info!("gfx_viewer self-test: {} steps", self.steps.len());
    }
}

fn preset_from(name: &str) -> Option<GraphicsPreset> {
    Some(match name {
        "auto" => GraphicsPreset::Auto,
        "raster" => GraphicsPreset::Raster,
        "rt-lite" => GraphicsPreset::RtLite,
        "rt-full" => GraphicsPreset::RtFull,
        _ => return None,
    })
}

pub fn preset_name(preset: GraphicsPreset) -> &'static str {
    match preset {
        GraphicsPreset::Auto => "auto",
        GraphicsPreset::Raster => "raster",
        GraphicsPreset::RtLite => "rt-lite",
        GraphicsPreset::RtFull => "rt-full",
    }
}

/// Runs the self-test one frame at a time.
pub fn drive(world: &mut World) {
    let mut test = world
        .remove_resource::<SelfTest>()
        .expect("drive runs only with a self-test");
    if !test.done {
        test.frame += 1;
        step(world, &mut test);
    }
    world.insert_resource(test);
}

fn step(world: &mut World, test: &mut SelfTest) {
    let Some(current) = test.steps.get(test.index).cloned() else {
        finish(world, test);
        return;
    };
    match test.state {
        State::Start => {
            test.step_started = Instant::now();
            info!(
                "gfx_viewer self-test: step {} {}",
                test.index + 1,
                current.name
            );
            // The lamps of a lamp step are switched back on by the step after it.
            if !world.resource::<crate::actions::Lamps>().on {
                world.write_message(Action::ToggleLamps);
            }
            if world.resource::<Stage>().index != current.scene || test.index == 0 {
                world.write_message(Action::OpenScene(current.scene));
            }
            *world.resource_mut::<GraphicsSettings>() = current.settings.clone();
            match &current.kind {
                Kind::LampsOff { .. } => {
                    world.write_message(Action::ToggleLamps);
                }
                Kind::PanelClick => {
                    let button = world
                        .query::<(Entity, &Field)>()
                        .iter(world)
                        .find(|(_, field)| **field == Field::Preset)
                        .map(|(entity, _)| entity);
                    match button {
                        Some(entity) => world.trigger(Activate { entity }),
                        None => test.failures.push("panel-click: no preset row".into()),
                    }
                }
                Kind::AbFlip { other } => {
                    let mut ab = world.resource_mut::<AbCompare>();
                    ab.other = other.clone();
                    ab.showing_b = false;
                    world.write_message(Action::FlipAb);
                }
                Kind::ShowUi => {
                    world.write_message(Action::ToggleHelp);
                }
                Kind::Plain | Kind::Bake { .. } => {}
            }
            test.state = if matches!(current.kind, Kind::Bake { .. }) {
                if !world.resource::<GraphicsCapabilities>().ray_query {
                    record_skip(
                        world,
                        test,
                        &current,
                        "baking needs ray tracing; this GPU has none",
                    );
                    return;
                }
                State::Baking {
                    finished_before: world.resource::<Bakes>().finished,
                }
            } else {
                State::Settling {
                    changed_at: test.frame,
                    calm: 0,
                    held: None,
                }
            };
        }
        State::Baking { finished_before } => {
            // The scene must be open before `B` means anything; then one bake, then its load.
            let ready = scene_ready(world);
            let bakes = world.resource::<Bakes>();
            if bakes.finished == finished_before && bakes.running_for().is_none() && ready {
                world.write_message(Action::Bake);
            }
            let bakes = world.resource::<Bakes>();
            if bakes.finished > finished_before {
                test.state = State::Settling {
                    changed_at: test.frame,
                    calm: 0,
                    held: None,
                };
            } else if test.step_started.elapsed() > BAKE_LIMIT {
                fail_step(
                    world,
                    test,
                    &current,
                    "the bake did not finish in time".into(),
                );
            }
        }
        State::Settling {
            changed_at,
            calm,
            held,
        } => {
            if test.step_started.elapsed() > STEP_LIMIT {
                let compiling = world
                    .resource::<PendingPipelines>()
                    .0
                    .load(Ordering::Relaxed);
                fail_step(
                    world,
                    test,
                    &current,
                    format!(
                        "not ready after {} s (scene placed: {}, pipelines compiling: {})",
                        STEP_LIMIT.as_secs(),
                        world.resource::<Stage>().placed,
                        compiling as i32
                    ),
                );
                return;
            }
            if let Some(held) = held {
                if held + 1 >= HOLD_FRAMES {
                    request_capture(world, test, &current);
                    test.state = State::Capturing;
                } else {
                    test.state = State::Settling {
                        changed_at,
                        calm,
                        held: Some(held + 1),
                    };
                }
                return;
            }
            let compiling = world
                .resource::<PendingPipelines>()
                .0
                .load(Ordering::Relaxed);
            let calm =
                if test.frame >= changed_at + SWITCH_FRAMES && compiling == 0 && scene_ready(world)
                {
                    calm + 1
                } else {
                    0
                };
            test.state = State::Settling {
                changed_at,
                calm,
                held: (calm >= CALM_FRAMES).then_some(0),
            };
        }
        State::Capturing => {
            let image = test.capture.lock().unwrap().take();
            if let Some(image) = image {
                set_ui_visible(world, true);
                evaluate(world, test, &current, image);
                test.index += 1;
                test.state = State::Start;
            } else if test.step_started.elapsed() > STEP_LIMIT + Duration::from_secs(30) {
                set_ui_visible(world, true);
                fail_step(world, test, &current, "the screenshot never arrived".into());
            }
        }
    }
}

/// The open scene has loaded everything it waits for and no bake is running.
fn scene_ready(world: &World) -> bool {
    let stage = world.resource::<Stage>();
    if stage.requested.is_some() || !stage.placed {
        return false;
    }
    let server = world.resource::<AssetServer>();
    // A file that failed to load never will: the step goes on and fails on it (see `evaluate`).
    stage.assets.iter().all(|asset| {
        matches!(
            server.recursive_dependency_load_state(asset.id()),
            RecursiveDependencyLoadState::Loaded | RecursiveDependencyLoadState::Failed(_)
        )
    }) && world.resource::<Bakes>().running_for().is_none()
}

/// Whether a file of the open scene failed to load.
fn scene_failed(world: &World) -> bool {
    let server = world.resource::<AssetServer>();
    world.resource::<Stage>().assets.iter().any(|asset| {
        matches!(
            server.recursive_dependency_load_state(asset.id()),
            RecursiveDependencyLoadState::Failed(_)
        )
    })
}

fn set_ui_visible(world: &mut World, visible: bool) {
    let mut roots = world.query_filtered::<&mut Visibility, With<ViewerUi>>();
    for mut visibility in roots.iter_mut(world) {
        *visibility = if visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

fn request_capture(world: &mut World, test: &mut SelfTest, current: &Step) {
    if current.kind != Kind::ShowUi {
        set_ui_visible(world, false);
    }
    let slot = test.capture.clone();
    world
        .spawn(Screenshot::primary_window())
        .observe(move |captured: On<ScreenshotCaptured>| {
            *slot.lock().unwrap() = Some(captured.image.clone());
        });
}

fn record_skip(world: &mut World, test: &mut SelfTest, current: &Step, reason: &str) {
    warn!("gfx_viewer self-test: {} skipped: {reason}", current.name);
    test.results.push(json!({
        "step": test.index + 1,
        "name": current.name,
        "skipped": reason,
    }));
    write_report(world, test, None);
    test.index += 1;
    test.state = State::Start;
}

fn fail_step(world: &mut World, test: &mut SelfTest, current: &Step, failure: String) {
    error!("gfx_viewer self-test: {}: {failure}", current.name);
    test.failures.push(format!("{}: {failure}", current.name));
    test.results.push(json!({
        "step": test.index + 1,
        "name": current.name,
        "failures": [failure],
    }));
    write_report(world, test, None);
    test.index += 1;
    test.state = State::Start;
}

/// Mean 8-bit luma, lit share.
fn luma_stats(rgb: &[u8]) -> (f64, f64) {
    let pixels = (rgb.len() / 3).max(1) as f64;
    let (mut sum, mut lit) = (0.0, 0.0);
    for pixel in rgb.as_chunks::<3>().0 {
        let luma = 0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64;
        sum += luma;
        if luma > LIT_LUMA {
            lit += 1.0;
        }
    }
    (sum / pixels, lit / pixels)
}

fn mean_abs_difference(a: &[u8], b: &[u8]) -> Option<f64> {
    (a.len() == b.len() && !a.is_empty()).then(|| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    })
}

fn evaluate(world: &mut World, test: &mut SelfTest, current: &Step, image: Image) {
    let mut failures = Vec::new();
    let file = format!("{:02}-{}.png", test.index + 1, current.name);
    let rgb = match image.try_into_dynamic() {
        Ok(image) => {
            let rgb = image.to_rgb8();
            if let Err(error) = rgb.save(test.dir.join(&file)) {
                failures.push(format!("cannot save {file}: {error}"));
            }
            rgb.into_raw()
        }
        Err(error) => {
            failures.push(format!("cannot read the screenshot: {error}"));
            Vec::new()
        }
    };
    let (mean, lit) = luma_stats(&rgb);
    if current.kind != Kind::ShowUi && (mean < MIN_MEAN_LUMA || lit < MIN_LIT_SHARE) {
        failures.push(format!(
            "the frame is black: mean luma {mean:.2}, {:.3} % of pixels lit",
            lit * 100.0
        ));
    }

    if scene_failed(world) {
        failures.push("a file of the scene failed to load (the log names it)".into());
    }
    let settings = world.resource::<GraphicsSettings>().clone();
    let resolved = world.get_resource::<ResolvedGraphics>().cloned();
    let mut extra = serde_json::Map::new();
    match &resolved {
        Some(resolved) => {
            failures.extend(check_resolved(&settings, resolved));
            failures.extend(check_camera(world, &settings, resolved));
        }
        None => failures.push("nothing resolved".into()),
    }
    match &current.kind {
        Kind::LampsOff { reference } | Kind::Bake { reference } => {
            let change = test
                .references
                .get(reference)
                .and_then(|before| mean_abs_difference(before, &rgb));
            extra.insert("change_from".into(), json!(reference));
            extra.insert("mean_abs_change".into(), json!(change));
            match change {
                None => failures.push(format!("no frame of {reference} to compare with")),
                Some(change) if change < MIN_CHANGE => failures.push(format!(
                    "the frame barely changed from {reference}: mean difference {change:.3}"
                )),
                Some(_) => {}
            }
            if matches!(current.kind, Kind::Bake { .. }) {
                let status = world.get_resource::<BakedGiStatus>().copied();
                let bakes = world.resource::<Bakes>();
                extra.insert("bake_succeeded".into(), json!(bakes.succeeded > 0));
                extra.insert(
                    "bake_folder".into(),
                    json!(bakes.loaded.as_ref().map(|dir| dir.display().to_string())),
                );
                if status != Some(BakedGiStatus::Loaded) {
                    failures.push(format!("after the bake, the baked GI status is {status:?}"));
                }
            }
        }
        Kind::PanelClick => {
            // The step set Auto; one click on the preset row moves it to Raster.
            extra.insert(
                "preset_after_click".into(),
                json!(format!("{:?}", settings.preset)),
            );
            if settings.preset != GraphicsPreset::Raster {
                failures.push(format!(
                    "clicking the preset row from Auto gave {:?}, not Raster",
                    settings.preset
                ));
            }
        }
        Kind::AbFlip { other } => {
            let ab = world.resource::<AbCompare>();
            extra.insert("showing_b".into(), json!(ab.showing_b));
            if !ab.showing_b || settings != *other || ab.other != current.settings {
                failures.push("C did not swap set A for set B".into());
            }
        }
        Kind::Plain | Kind::ShowUi => {}
    }
    if matches!(current.kind, Kind::Plain) {
        test.references.insert(current.name.clone(), rgb);
    }

    let frame_ms = world.resource::<FrameTimes>().median_ms();
    let passes: Vec<Value> = gpu_passes(world.resource::<DiagnosticsStore>())
        .into_iter()
        .take(8)
        .map(|(pass, ms)| json!({"pass": pass, "ms": ms}))
        .collect();
    let mut entry = json!({
        "step": test.index + 1,
        "name": current.name,
        "scene": world.resource::<Stage>().name,
        "file": file,
        "seconds": test.step_started.elapsed().as_secs_f64(),
        "asked": settings,
        "resolved": resolved,
        "rung": resolved.as_ref().map(rung_text),
        "reasons": resolved.as_ref().map(|r| r.fallbacks.iter().map(ToString::to_string).collect::<Vec<_>>()),
        "baked_gi": format!("{:?}", world.get_resource::<BakedGiStatus>().copied()),
        "lamps_on": world.resource::<crate::actions::Lamps>().on,
        "mean_luma": mean,
        "lit_share": lit,
        "frame_ms_median": frame_ms,
        "gpu_ms_top": passes,
        "failures": failures,
    });
    if let Value::Object(entry) = &mut entry {
        entry.extend(extra);
    }
    for failure in &failures {
        error!("gfx_viewer self-test: {}: {failure}", current.name);
        test.failures.push(format!("{}: {failure}", current.name));
    }
    test.results.push(entry);
    write_report(world, test, None);
}

/// Fallbacks that explain a rung other than the one asked for.
fn explains_rung(fallback: &Fallback) -> bool {
    use Fallback::*;
    matches!(
        fallback,
        NotVulkan
            | SoftwareAdapter
            | RayTracingNotCompiled
            | NoRayQuery
            | IntegratedGpu
            | NoDenoiser
            | LowVideoMemory
            | VideoMemoryBelowRtFull
            | NotOnFastRtList
            | SlowRayTracing
            | NrdFailed
    )
}

/// Fallbacks that explain an upscaler or a quality other than the one asked for.
fn explains_upscaler(fallback: &Fallback) -> bool {
    use Fallback::*;
    matches!(
        fallback,
        TsrUnavailable
            | DlssUnavailable
            | FsrUnavailable
            | FsrNeedsRadeon
            | RayReconstructionUpscales
            | NoUpscaler
            | FsrFailed
            | TsrFsr3Failed
            | LightRaster
    )
}

/// Fallbacks that explain a denoiser other than the one asked for.
fn explains_denoiser(fallback: &Fallback) -> bool {
    use Fallback::*;
    matches!(
        fallback,
        DlssRayReconstructionUnavailable | NrdUnavailable | NrdOnOlderRtx | NrdFailed
    )
}

/// What resolved is what was asked, or a fallback of the right kind says why not.
fn check_resolved(settings: &GraphicsSettings, resolved: &ResolvedGraphics) -> Vec<String> {
    let mut failures = Vec::new();
    let explained = |test: fn(&Fallback) -> bool| resolved.fallbacks.iter().any(test);
    let rung = match settings.preset {
        GraphicsPreset::Auto => None,
        GraphicsPreset::Raster => Some(GraphicsRung::Raster),
        GraphicsPreset::RtLite => Some(GraphicsRung::RtLite),
        GraphicsPreset::RtFull => Some(GraphicsRung::RtFull),
    };
    if let Some(rung) = rung
        && resolved.rung != rung
        && !explained(explains_rung)
    {
        failures.push(format!(
            "asked {rung:?}, {:?} resolved, and no fallback says why",
            resolved.rung
        ));
    }
    let upscaler_ok = match settings.upscaler {
        UpscalerChoice::Auto => true,
        UpscalerChoice::Off => resolved.upscaler == ResolvedUpscaler::Taa,
        UpscalerChoice::Tsr => resolved.upscaler == ResolvedUpscaler::Tsr,
        UpscalerChoice::Dlss => matches!(
            resolved.upscaler,
            ResolvedUpscaler::DlssSuperResolution | ResolvedUpscaler::DlssRayReconstruction
        ),
        UpscalerChoice::Fsr => resolved.upscaler == ResolvedUpscaler::Fsr,
    };
    if !upscaler_ok && !explained(explains_upscaler) {
        failures.push(format!(
            "asked upscaler {:?}, {:?} resolved, and no fallback says why",
            settings.upscaler, resolved.upscaler
        ));
    }
    let quality = match settings.upscale_quality {
        UpscaleQuality::Auto => None,
        UpscaleQuality::NativeAa => Some(ResolvedQuality::NativeAa),
        UpscaleQuality::Quality => Some(ResolvedQuality::Quality),
        UpscaleQuality::Balanced => Some(ResolvedQuality::Balanced),
        UpscaleQuality::Performance => Some(ResolvedQuality::Performance),
    };
    if let Some(quality) = quality
        && resolved.quality != quality
        && !explained(explains_upscaler)
    {
        failures.push(format!(
            "asked quality {quality:?}, {:?} resolved, and no fallback says why",
            resolved.quality
        ));
    }
    let denoiser = match settings.denoiser {
        DenoiserChoice::Auto => None,
        DenoiserChoice::DlssRayReconstruction => Some(ResolvedDenoiser::DlssRayReconstruction),
        DenoiserChoice::Nrd => Some(ResolvedDenoiser::Nrd),
    };
    if resolved.rung.is_traced()
        && let Some(denoiser) = denoiser
        && resolved.denoiser != denoiser
        && !explained(explains_denoiser)
    {
        failures.push(format!(
            "asked denoiser {denoiser:?}, {:?} resolved, and no fallback says why",
            resolved.denoiser
        ));
    }
    failures
}

/// The camera and the lights carry what the resolved rung and the overrides call for.
fn check_camera(
    world: &mut World,
    settings: &GraphicsSettings,
    resolved: &ResolvedGraphics,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut camera = world.query_filtered::<(
        Has<ScreenSpaceReflections>,
        Has<ScreenSpaceAmbientOcclusion>,
        Has<ContactShadows>,
        Has<Bloom>,
        Option<&SolariLighting>,
    ), With<ViewerCamera>>();
    let Ok((ssr, ssao, contact, bloom, lighting)) = camera.single(world) else {
        return vec!["no viewer camera".into()];
    };
    let overrides = &settings.overrides;
    let mut expect = |what: &str, has: bool, wanted: bool| {
        if has != wanted {
            failures.push(format!(
                "the camera {} {what}, but the settings call for {}",
                if has { "has" } else { "lacks" },
                if wanted { "it" } else { "none" }
            ));
        }
    };
    match solari_lighting(resolved.rung, overrides) {
        None => {
            let table = raster_with(resolved.raster, overrides);
            expect("screen-space reflections", ssr, table.reflections.is_some());
            expect("SSAO", ssao, table.ambient_occlusion.is_some());
            expect("contact shadows", contact, table.contact_shadows.is_some());
            expect("bloom", bloom, table.bloom.is_some());
            expect("Solari lighting", lighting.is_some(), false);
        }
        Some(wanted) => {
            expect("bloom", bloom, traced_bloom(overrides).is_some());
            match lighting {
                None => failures
                    .push("a traced rung resolved but the camera has no Solari lighting".into()),
                Some(lighting) => {
                    if lighting.restir != wanted.restir
                        || lighting.max_bounces != wanted.max_bounces
                    {
                        failures.push(format!(
                            "Solari runs restir {} with {} bounces; the settings call for {} with {}",
                            lighting.restir, lighting.max_bounces, wanted.restir, wanted.max_bounces
                        ));
                    }
                }
            }
        }
    }
    if overrides.shadows == Some(false) {
        let mut lights = world.query::<(
            Option<&PointLight>,
            Option<&SpotLight>,
            Option<&DirectionalLight>,
        )>();
        let shadowed = lights.iter(world).any(|(point, spot, sun)| {
            point.is_some_and(|l| l.shadow_maps_enabled)
                || spot.is_some_and(|l| l.shadow_maps_enabled)
                || sun.is_some_and(|l| l.shadow_maps_enabled)
        });
        if shadowed {
            failures.push("shadow maps are switched off, but a light still casts them".into());
        }
    }
    failures
}

fn write_report(world: &mut World, test: &SelfTest, outcome: Option<bool>) {
    let capabilities = world.resource::<GraphicsCapabilities>();
    let report = json!({
        "schema": 1,
        "tool": "gfx_viewer --self-test",
        "outcome": outcome.map(|ok| if ok { "pass" } else { "fail" }),
        "seconds": test.started.elapsed().as_secs_f64(),
        "adapter": capabilities.adapter_name,
        "capabilities": capabilities,
        "window": world
            .query_filtered::<&Window, With<bevy::window::PrimaryWindow>>()
            .iter(world)
            .next()
            .map(|window| [window.physical_width(), window.physical_height()]),
        "steps_planned": test.steps.iter().map(|step| step.name.clone()).collect::<Vec<_>>(),
        "steps": test.results,
        "failures": test.failures,
    });
    let text = serde_json::to_string_pretty(&report).expect("a JSON value serializes");
    if let Err(error) = std::fs::write(test.dir.join("report.json"), text) {
        error!("cannot write report.json: {error}");
    }
}

fn finish(world: &mut World, test: &mut SelfTest) {
    // Exit only once nothing compiles any more: a process that ends while pipelines compile on other threads
    // aborts in its teardown instead of returning its code.
    if world
        .resource::<PendingPipelines>()
        .0
        .load(Ordering::Relaxed)
        > 0
    {
        return;
    }
    test.done = true;
    let ok = test.failures.is_empty();
    write_report(world, test, Some(ok));
    info!(
        "gfx_viewer self-test: {} ({} steps, {} failures) in {:.0} s",
        if ok { "pass" } else { "FAIL" },
        test.results.len(),
        test.failures.len(),
        test.started.elapsed().as_secs_f64()
    );
    world.write_message(if ok {
        AppExit::Success
    } else {
        AppExit::error()
    });
}
