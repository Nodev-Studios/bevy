//! `--self-test DIR`: walks through every scene × every preset, a few overrides, the lamp switch, the panels'
//! controls, the A/B flip and the split view, and one bake + load on `interior`, through the same [`Action`]s and
//! resources the keys and the panels change. Two steps click real egui widgets: the cursor and the button go in
//! as window events, through `bevy_egui`'s input, exactly as winit delivers them.
//!
//! Each step waits until its scene is loaded and no pipeline compiles, does its action, holds [`HOLD_FRAMES`]
//! frames, captures the frame (with the panels hidden, except the `ui` step's) and checks it:
//!
//! - the frame is not black;
//! - what resolved is what the step asked for, or a [`Fallback`] of the right kind says why not;
//! - the camera carries the effects the resolved rung and the overrides call for (the apply layer did its job);
//! - each control did what it says: the lamp, exposure, lamp-intensity, sun and coverage steps differ from the
//!   frame before them; a click moved the setting it clicked; a bookmark brings the camera back; the bake
//!   loaded, unloaded and loaded again; the split view's left half shows the other set and its right half the
//!   live one.
//!
//! It writes `NN-<step>.png` per step and `report.json` (rewritten after each step) into DIR, and exits 0 only
//! when no step failed. A scene whose files are not downloaded is recorded as skipped. A panic ends the process
//! with a non-zero code; `report.json` then holds the steps before it.

use crate::{
    actions::{Action, Bookmarks},
    bake::Bakes,
    catalog::{Catalog, Source},
    compare::Split,
    stage::Stage,
    ui::{gpu_passes, rung_text, FrameTimes, UiState, ViewControls},
    AbCompare, PendingPipelines, ViewerCamera,
};
use bevy::{
    app::AppExit,
    math::ops,
    asset::RecursiveDependencyLoadState,
    camera::Exposure,
    camera_controller::free_camera::FreeCameraState,
    diagnostic::DiagnosticsStore,
    graphics::{
        baked::BakedGiStatus,
        raster::{raster_with, traced_bloom},
        traced::solari_lighting,
        DenoiserChoice, Fallback, GraphicsCapabilities, GraphicsPreset, GraphicsRung,
        GraphicsSettings, ResolvedDenoiser, ResolvedGraphics, ResolvedQuality, ResolvedUpscaler,
        UpscaleQuality, UpscalerChoice,
    },
    input::{mouse::MouseButtonInput, ButtonState},
    pbr::{ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceReflections},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    solari::prelude::SolariLighting,
    window::{CursorMoved, PrimaryWindow, WindowEvent},
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
/// Frames before a capture the panels are hidden (like Tab; the split view's left side stays): egui's output
/// reaches the frame later than the frame it was drawn in.
const UI_LAG_FRAMES: u32 = 4;
/// A step that is not ready after this long fails instead of hanging the run.
const STEP_LIMIT: Duration = Duration::from_secs(300);
/// How long a bake may take.
const BAKE_LIMIT: Duration = Duration::from_secs(1800);
/// A frame whose mean 8-bit luma is under this, or whose lit share is under [`MIN_LIT_SHARE`], is black.
/// to tune.
const MIN_MEAN_LUMA: f64 = 1.5;
/// The share of pixels with a luma over [`LIT_LUMA`] a frame needs to count as lit. to tune.
const MIN_LIT_SHARE: f64 = 0.002;
const LIT_LUMA: f64 = 8.0;
/// The mean absolute difference (8-bit) a control must make to the frame. to tune.
const MIN_CHANGE: f64 = 0.5;
/// The scenes whose sun the self-test turns (when they are in the list and have a sun).
const SUN_SCENES: [&str; 2] = ["bistro-exterior", "sponza"];

/// What a step does besides showing its settings.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Plain,
    /// Switches the lamps off; the frame must differ from step `reference`'s. The next step switches them on.
    LampsOff {
        reference: String,
    },
    /// Clicks the egui widget drawn as `widget` (see `UiState::rects`); `check` says what must follow.
    Click {
        widget: &'static str,
        check: ClickCheck,
    },
    /// Shows these settings as set A, `other` as set B, and flips to B.
    AbFlip {
        other: GraphicsSettings,
    },
    /// Bakes the scene, loads the bake, and compares with step `reference`'s frame.
    Bake {
        reference: String,
    },
    /// Unloads the scene's bake: it must be gone and the frame differ from `reference`'s.
    Unload {
        reference: String,
    },
    /// Loads the newest bake again.
    Reload,
    /// Sets the exposure's EV100 this much above the scene's own (darker).
    Exposure {
        delta: f32,
        reference: String,
    },
    /// Multiplies every lamp's intensity.
    LampScale {
        scale: f32,
        reference: String,
    },
    /// Moves the sun to this height (degrees) and halves its intensity.
    Sun {
        elevation: f32,
        reference: String,
    },
    /// Switches the coverage view on.
    Coverage {
        reference: String,
    },
    /// Moves the camera, saves the view, resets the camera and goes back to the saved view.
    Bookmark,
    /// Opens the scene's file by path, as the Scene panel's field does.
    OpenFile,
    /// Shows set `other` on the left of the split view and these settings live on the right.
    Split {
        other: GraphicsSettings,
        left_reference: String,
        right_reference: String,
    },
    /// Captures with the panels shown. Not checked for black.
    ShowUi,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ClickCheck {
    /// The preset became Raster.
    PresetRaster,
    /// The bounce override went on.
    BouncesOn,
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
    /// The step's settings and scene go in.
    Start,
    /// Waiting for the scene, then the step's action.
    Opening,
    /// Clicking a widget: 0 moves the cursor there; 3 (egui has seen the cursor and the camera gate has
    /// run) checks the pointer and presses; 5 releases. The stages between wait a frame.
    Clicking {
        stage: u32,
        pointer_over: bool,
        camera_held: bool,
    },
    /// Waiting for a bake to finish and load.
    Baking { finished_before: u32 },
    /// Waiting for the split view's left side.
    Splitting { taken_before: u32 },
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
    /// Scenes of the list left out because their files are missing.
    missing: Vec<String>,
    index: usize,
    state: State,
    frame: u32,
    step_started: Instant,
    started: Instant,
    capture: Arc<Mutex<Option<Image>>>,
    references: HashMap<String, Vec<u8>>,
    /// What a step's action recorded for its check (a bookmark's view, a click's pointer state).
    noted: serde_json::Map<String, Value>,
    bookmark: Option<Transform>,
    results: Vec<Value>,
    failures: Vec<String>,
    done: bool,
}

pub const HELP: &str = "
Self-test:
  --self-test DIR                every scene x every preset, overrides, lamps, the panels' controls, A/B, the
                                 split view and a bake on interior; screenshots and report.json in DIR; exit 0
                                 when every step passes
  --self-test-presets auto,raster,rt-lite,rt-full
                                 the presets each scene is shown with (default all four)
  --self-test-scenes NAME,...    only these scenes (default every scene of the list that is downloaded)
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
            missing: Vec::new(),
            index: 0,
            state: State::Start,
            frame: 0,
            step_started: Instant::now(),
            started: Instant::now(),
            capture: Arc::default(),
            references: HashMap::new(),
            noted: serde_json::Map::new(),
            bookmark: None,
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
            if !entry.available {
                self.missing.push(entry.name.clone());
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
            use GraphicsPreset::{Auto, Raster, RtFull, RtLite};
            let reference = |preset: &str| format!("{}-{preset}", entry.name);
            let interior = matches!(entry.source, Source::Builtin("interior"));
            let gltf = matches!(entry.source, Source::Gltf(_));
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
                        reference: reference("raster"),
                    },
                );
                step(
                    "raster-ev100-plus-2",
                    with(Raster),
                    Kind::Exposure {
                        delta: 2.0,
                        reference: reference("raster"),
                    },
                );
                step(
                    "raster-lamps-x0.25",
                    with(Raster),
                    Kind::LampScale {
                        scale: 0.25,
                        reference: reference("raster"),
                    },
                );
                step("raster-bookmark", with(Raster), Kind::Bookmark);
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
                        reference: reference("rt-lite"),
                    },
                );
            }
            if interior && has(RtFull) {
                let mut s = with(RtFull);
                s.overrides.restir = Some(false);
                s.overrides.max_bounces = Some(1);
                s.overrides.bloom = Some(false);
                step("rt-full-no-restir-1-bounce-no-bloom", s, Kind::Plain);
                step(
                    "click-bounces",
                    with(RtFull),
                    Kind::Click {
                        widget: "bounces override",
                        check: ClickCheck::BouncesOn,
                    },
                );
            }
            if interior {
                step(
                    "click-preset",
                    with(Auto),
                    Kind::Click {
                        widget: "preset raster",
                        check: ClickCheck::PresetRaster,
                    },
                );
            }
            if interior && has(Raster) && has(RtLite) {
                step(
                    "ab-flip",
                    with(Raster),
                    Kind::AbFlip {
                        other: with(RtLite),
                    },
                );
                step(
                    "split",
                    with(Raster),
                    Kind::Split {
                        other: with(RtLite),
                        left_reference: reference("rt-lite"),
                        right_reference: reference("raster"),
                    },
                );
            }
            if SUN_SCENES.contains(&entry.name.as_str()) && has(Raster) {
                step(
                    "raster-sun-low",
                    with(Raster),
                    Kind::Sun {
                        elevation: 8.0,
                        reference: reference("raster"),
                    },
                );
            }
            if gltf && entry.name == "flighthelmet" && has(Raster) {
                step("raster-open-by-path", with(Raster), Kind::OpenFile);
            }
            if self.bake.contains(&entry.name) && has(Raster) {
                step(
                    "raster-baked",
                    with(Raster),
                    Kind::Bake {
                        reference: reference("raster"),
                    },
                );
                step(
                    "raster-baked-coverage",
                    with(Raster),
                    Kind::Coverage {
                        reference: reference("raster-baked"),
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
                step(
                    "raster-unloaded",
                    with(Raster),
                    Kind::Unload {
                        reference: reference("raster-baked"),
                    },
                );
                step("raster-reloaded", with(Raster), Kind::Reload);
            }
            if interior {
                step("ui", with(Auto), Kind::ShowUi);
            }
        }
        // The panels over Bistro, for the owner.
        if let Some(index) = catalog.find("bistro-exterior")
            && catalog.entries[index].available
            && wanted("bistro-exterior")
        {
            steps.push(Step {
                name: "bistro-exterior-ui".into(),
                scene: index,
                settings: with(GraphicsPreset::Auto),
                kind: Kind::ShowUi,
            });
        }
        self.steps = steps;
        info!(
            "gfx_viewer self-test: {} steps; not downloaded, skipped: {:?}",
            self.steps.len(),
            self.missing
        );
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

fn settling(test: &SelfTest) -> State {
    State::Settling {
        changed_at: test.frame,
        calm: 0,
        held: None,
    }
}

fn step(world: &mut World, test: &mut SelfTest) {
    let Some(current) = test.steps.get(test.index).cloned() else {
        finish(world, test);
        return;
    };
    match test.state {
        State::Start => {
            test.step_started = Instant::now();
            test.noted.clear();
            info!(
                "gfx_viewer self-test: step {} {}",
                test.index + 1,
                current.name
            );
            // What the step before changed goes back: lamps on, controls as the scene has them, no split.
            if !world.resource::<crate::actions::Lamps>().on {
                world.write_message(Action::ToggleLamps);
            }
            {
                let exposure = world.resource::<Stage>().exposure;
                let mut controls = world.resource_mut::<ViewControls>();
                let mut reset = controls.clone();
                reset.exposure = exposure;
                reset.lamp_scale = 1.0;
                reset.coverage = false;
                reset.sun = reset.initial_sun;
                if *controls != reset {
                    *controls = reset;
                }
            }
            if world.resource::<Split>().on {
                world.write_message(Action::Split(false));
            }
            world.resource_mut::<AbCompare>().showing_b = false;
            if current.kind != Kind::OpenFile
                && (world.resource::<Stage>().index != current.scene || test.index == 0)
            {
                world.write_message(Action::OpenScene(current.scene));
            } else {
                // Every step of a scene sees it from where it starts (a bookmark step moves the camera).
                world.write_message(Action::ResetCamera);
            }
            *world.resource_mut::<GraphicsSettings>() = current.settings.clone();
            if current.kind == Kind::OpenFile {
                let Source::Gltf(scene) =
                    &world.resource::<Catalog>().entries[current.scene].source
                else {
                    unreachable!("planned for glTF scenes only")
                };
                let path = scene.files[0].0.to_string_lossy().into_owned();
                // Opened from another scene, as a person would.
                world.write_message(Action::OpenScene(0));
                world.write_message(Action::OpenFile(path));
            }
            test.state = State::Opening;
        }
        State::Opening => {
            let generation = world.resource::<Stage>().generation;
            if !scene_ready(world) || world.resource::<ViewControls>().generation != generation {
                if test.step_started.elapsed() > STEP_LIMIT {
                    fail_step(world, test, &current, "the scene never got ready".into());
                }
                return;
            }
            act(world, test, &current);
        }
        State::Clicking {
            stage,
            pointer_over,
            camera_held,
        } => {
            let Kind::Click { widget, .. } = current.kind else {
                unreachable!("only a click step clicks")
            };
            let Some(rect) = world.resource::<UiState>().rects.get(widget).copied() else {
                if test.step_started.elapsed() > Duration::from_secs(20) {
                    fail_step(
                        world,
                        test,
                        &current,
                        format!("the panels never drew {widget}"),
                    );
                }
                return;
            };
            let window = world
                .query_filtered::<Entity, With<PrimaryWindow>>()
                .single(world)
                .expect("the viewer has a window");
            let position = Vec2::new(rect.center().x, rect.center().y);
            match stage {
                1 | 2 | 4 => {
                    test.state = State::Clicking {
                        stage: stage + 1,
                        pointer_over,
                        camera_held,
                    };
                }
                0 => {
                    world.write_message(WindowEvent::CursorMoved(CursorMoved {
                        window,
                        position,
                        delta: None,
                    }));
                    test.state = State::Clicking {
                        stage: 1,
                        pointer_over,
                        camera_held,
                    };
                }
                3 => {
                    // The cursor rests over the panel: egui has the pointer and the camera stands still.
                    let over = world.resource::<UiState>().pointer;
                    let held = world
                        .query_filtered::<&FreeCameraState, With<ViewerCamera>>()
                        .iter(world)
                        .all(|state| !state.enabled);
                    world.write_message(WindowEvent::MouseButtonInput(MouseButtonInput {
                        button: MouseButton::Left,
                        state: ButtonState::Pressed,
                        window,
                    }));
                    test.state = State::Clicking {
                        stage: 4,
                        pointer_over: over,
                        camera_held: held,
                    };
                }
                _ => {
                    world.write_message(WindowEvent::MouseButtonInput(MouseButtonInput {
                        button: MouseButton::Left,
                        state: ButtonState::Released,
                        window,
                    }));
                    test.noted
                        .insert("pointer_over_panel".into(), json!(pointer_over));
                    test.noted.insert("camera_held".into(), json!(camera_held));
                    test.noted
                        .insert("clicked_at".into(), json!([position.x, position.y]));
                    test.state = settling(test);
                }
            }
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
                test.noted.insert(
                    "bake_seconds".into(),
                    json!(test.step_started.elapsed().as_secs_f64()),
                );
                test.state = settling(test);
            } else if test.step_started.elapsed() > BAKE_LIMIT {
                fail_step(
                    world,
                    test,
                    &current,
                    "the bake did not finish in time".into(),
                );
            }
        }
        State::Splitting { taken_before } => {
            let split = world.resource::<Split>();
            if split.taken > taken_before && split.ready() {
                test.state = settling(test);
            } else if test.step_started.elapsed() > STEP_LIMIT {
                fail_step(
                    world,
                    test,
                    &current,
                    "the split view never took its left side".into(),
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
                // egui's output reaches the frame a frame or two late: the panels go a few frames early.
                if held + UI_LAG_FRAMES == HOLD_FRAMES && current.kind != Kind::ShowUi {
                    world.resource_mut::<UiState>().visible = false;
                }
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
                world.resource_mut::<UiState>().visible = true;
                evaluate(world, test, &current, image);
                test.index += 1;
                test.state = State::Start;
            } else if test.step_started.elapsed() > STEP_LIMIT + Duration::from_secs(30) {
                world.resource_mut::<UiState>().visible = true;
                fail_step(world, test, &current, "the screenshot never arrived".into());
            }
        }
    }
}

/// The step's action, once its scene is ready.
fn act(world: &mut World, test: &mut SelfTest, current: &Step) {
    test.state = settling(test);
    match &current.kind {
        Kind::Plain | Kind::OpenFile => {}
        Kind::LampsOff { .. } => {
            world.write_message(Action::ToggleLamps);
        }
        Kind::Click { .. } => {
            let mut ui = world.resource_mut::<UiState>();
            ui.visible = true;
            ui.windows.render = true;
            test.state = State::Clicking {
                stage: 0,
                pointer_over: false,
                camera_held: false,
            };
        }
        Kind::AbFlip { other } => {
            let mut ab = world.resource_mut::<AbCompare>();
            ab.other = other.clone();
            ab.showing_b = false;
            world.write_message(Action::FlipAb);
        }
        Kind::Split { other, .. } => {
            world.resource_mut::<AbCompare>().other = other.clone();
            test.state = State::Splitting {
                taken_before: world.resource::<Split>().taken,
            };
            world.write_message(Action::Split(true));
        }
        Kind::Bake { .. } => {
            if !world.resource::<GraphicsCapabilities>().ray_query {
                record_skip(
                    world,
                    test,
                    current,
                    "baking needs ray tracing; this GPU has none",
                );
                return;
            }
            test.state = State::Baking {
                finished_before: world.resource::<Bakes>().finished,
            };
        }
        Kind::Unload { .. } => {
            world.write_message(Action::UnloadBake);
        }
        Kind::Reload => {
            world.write_message(Action::LoadBake);
        }
        Kind::Exposure { delta, .. } => {
            world.resource_mut::<ViewControls>().exposure += delta;
        }
        Kind::LampScale { scale, .. } => {
            world.resource_mut::<ViewControls>().lamp_scale = *scale;
        }
        Kind::Sun { elevation, .. } => {
            let mut controls = world.resource_mut::<ViewControls>();
            match &mut controls.sun {
                Some(sun) => {
                    sun.elevation = *elevation;
                    sun.lux *= 0.5;
                }
                None => {
                    record_skip(world, test, current, "this scene has no sun");
                }
            }
        }
        Kind::Coverage { .. } => {
            world.resource_mut::<ViewControls>().coverage = true;
        }
        Kind::Bookmark => {
            let start = world.resource::<Stage>().start;
            let moved = Transform::from_translation(start.translation + Vec3::new(0.4, 0.2, -0.3))
                .with_rotation(start.rotation * Quat::from_rotation_y(0.35));
            crate::stage::place_camera(world, moved, world.resource::<Stage>().fov, 1.0);
            test.bookmark = Some(moved);
            let saved = world
                .resource::<Bookmarks>()
                .for_scene(&world.resource::<Stage>().name)
                .len();
            world.write_message(Action::SaveBookmark);
            world.write_message(Action::ResetCamera);
            world.write_message(Action::GoBookmark(saved));
        }
        Kind::ShowUi => {
            let mut ui = world.resource_mut::<UiState>();
            ui.visible = true;
            ui.windows.baked = true;
            ui.windows.compare = true;
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

fn request_capture(world: &mut World, test: &mut SelfTest, _current: &Step) {
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

/// The width of the captures (the window's).
fn image_width(world: &mut World) -> Option<usize> {
    world
        .query_filtered::<&Window, With<PrimaryWindow>>()
        .iter(world)
        .next()
        .map(|window| window.physical_width() as usize)
}

/// The mean absolute difference of two RGB8 frames over the part left (or right) of `divider`, a margin of 2 %
/// away from it.
fn half_difference(a: &[u8], b: &[u8], width: usize, divider: f32, left: bool) -> f64 {
    if a.len() != b.len() || width == 0 {
        return f64::MAX;
    }
    let split = (divider * width as f32) as usize;
    let margin = width / 50;
    let columns = if left {
        0..split.saturating_sub(margin)
    } else {
        (split + margin).min(width)..width
    };
    let (mut sum, mut count) = (0.0, 0usize);
    for (row_a, row_b) in a.chunks_exact(width * 3).zip(b.chunks_exact(width * 3)) {
        for x in columns.clone() {
            for c in 0..3 {
                sum += (row_a[x * 3 + c] as f64 - row_b[x * 3 + c] as f64).abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::MAX
    } else {
        sum / count as f64
    }
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
    for (key, value) in std::mem::take(&mut test.noted) {
        extra.insert(key, value);
    }
    let changed_from = |reference: &str,
                        failures: &mut Vec<String>,
                        extra: &mut serde_json::Map<String, Value>| {
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
    };
    let status = world.get_resource::<BakedGiStatus>().copied();
    match &current.kind {
        Kind::LampsOff { reference } | Kind::Coverage { reference } => {
            changed_from(reference, &mut failures, &mut extra);
        }
        Kind::Bake { reference } => {
            changed_from(reference, &mut failures, &mut extra);
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
        Kind::Unload { reference } => {
            changed_from(reference, &mut failures, &mut extra);
            if world.resource::<Bakes>().loaded.is_some()
                || matches!(status, Some(BakedGiStatus::Loaded))
            {
                failures.push(format!("after Unload the bake is still there ({status:?})"));
            }
        }
        Kind::Reload => {
            if status != Some(BakedGiStatus::Loaded) {
                failures.push(format!(
                    "after Load newest the baked GI status is {status:?}"
                ));
            }
        }
        Kind::Exposure { delta, reference } => {
            changed_from(reference, &mut failures, &mut extra);
            let wanted = world.resource::<Stage>().exposure + delta;
            let ev = world
                .query_filtered::<&Exposure, With<ViewerCamera>>()
                .iter(world)
                .next()
                .map(|e| e.ev100);
            extra.insert("ev100".into(), json!(ev));
            if ev.is_none_or(|ev| (ev - wanted).abs() > 1e-3) {
                failures.push(format!("the camera's EV100 is {ev:?}, not {wanted}"));
            }
            if let Some(before) = test.references.get(reference)
                && luma_stats(before).0 <= mean
            {
                failures.push("a higher EV100 did not darken the frame".into());
            }
        }
        Kind::LampScale { scale, reference } => {
            changed_from(reference, &mut failures, &mut extra);
            let mut lamps = world.query::<(&crate::actions::BaseIntensity, &PointLight)>();
            let wrong = lamps
                .iter(world)
                .filter(|(base, light)| {
                    (base.0 * scale - light.intensity).abs() > 1e-3 * base.0.max(1.0)
                })
                .count();
            if wrong > 0 {
                failures.push(format!(
                    "{wrong} point lights do not carry {scale} × their intensity"
                ));
            }
        }
        Kind::Sun {
            elevation,
            reference,
        } => {
            changed_from(reference, &mut failures, &mut extra);
            let sun = world.resource::<ViewControls>().sun;
            let mut suns = world.query::<(&DirectionalLight, &Transform)>();
            let found = suns.iter(world).any(|(_, transform)| {
                let height = ops::asin(-transform.forward().y).to_degrees();
                (height - elevation).abs() < 0.5
            });
            extra.insert(
                "sun".into(),
                json!(sun.map(|s| [s.azimuth, s.elevation, s.lux])),
            );
            if !found {
                failures.push(format!("no sun stands {elevation}° high"));
            }
        }
        Kind::Click { check, .. } => {
            if extra.get("pointer_over_panel") != Some(&json!(true)) {
                failures
                    .push("with the cursor over the panel, egui did not have the pointer".into());
            }
            if extra.get("camera_held") != Some(&json!(true)) {
                failures.push("with the cursor over the panel, the camera was not held".into());
            }
            match check {
                ClickCheck::PresetRaster => {
                    extra.insert(
                        "preset_after_click".into(),
                        json!(format!("{:?}", settings.preset)),
                    );
                    if settings.preset != GraphicsPreset::Raster {
                        failures.push(format!(
                            "clicking Raster from Auto gave {:?}",
                            settings.preset
                        ));
                    }
                }
                ClickCheck::BouncesOn => {
                    extra.insert(
                        "bounces_after_click".into(),
                        json!(settings.overrides.max_bounces),
                    );
                    if settings.overrides.max_bounces.is_none() {
                        failures.push("clicking the bounce override left it off".into());
                    }
                }
            }
        }
        Kind::AbFlip { other } => {
            let ab = world.resource::<AbCompare>();
            extra.insert("showing_b".into(), json!(ab.showing_b));
            if !ab.showing_b || settings != *other || ab.other != current.settings {
                failures.push("C did not swap set A for set B".into());
            }
        }
        Kind::Bookmark => {
            let camera = world
                .query_filtered::<&Transform, With<ViewerCamera>>()
                .iter(world)
                .next()
                .copied();
            let wanted = test.bookmark.take();
            let close = match (camera, wanted) {
                (Some(c), Some(w)) => {
                    c.translation.distance(w.translation) < 1e-3
                        && c.rotation.angle_between(w.rotation) < 1e-3
                }
                _ => false,
            };
            extra.insert(
                "camera".into(),
                json!(camera.map(|c| c.translation.to_array())),
            );
            extra.insert(
                "saved".into(),
                json!(wanted.map(|c| c.translation.to_array())),
            );
            if !close {
                failures.push("going to the saved view did not bring the camera back to it".into());
            }
        }
        Kind::OpenFile => {
            let stage = world.resource::<Stage>();
            extra.insert("opened".into(), json!(stage.name));
            if stage.index != current.scene {
                failures.push(format!("opening the file by path opened {}", stage.name));
            }
        }
        Kind::Split {
            other,
            left_reference,
            right_reference,
        } => {
            let width = image_width(world);
            let split = world.resource::<Split>();
            let ab = world.resource::<AbCompare>();
            extra.insert("divider".into(), json!(split.divider));
            if !split.ready() || settings != current.settings || ab.other != *other {
                failures.push(format!(
                    "the split view is not showing the other set on the left and these settings live ({})",
                    split.status()
                ));
            }
            // Each half is closer to the frame of its own set than to the other's.
            match (
                test.references.get(left_reference),
                test.references.get(right_reference),
                width,
            ) {
                (Some(left), Some(right), Some(width)) => {
                    let half = |a: &[u8], b: &[u8], left_half: bool| {
                        half_difference(a, b, width, split.divider, left_half)
                    };
                    let l_own = half(&rgb, left, true);
                    let l_other = half(&rgb, right, true);
                    let r_own = half(&rgb, right, false);
                    let r_other = half(&rgb, left, false);
                    extra.insert("left_vs_left_set".into(), json!(l_own));
                    extra.insert("left_vs_right_set".into(), json!(l_other));
                    extra.insert("right_vs_right_set".into(), json!(r_own));
                    extra.insert("right_vs_left_set".into(), json!(r_other));
                    if l_own >= l_other || r_own >= r_other {
                        failures.push(
                            "a half of the split view is closer to the other set's frame than to its own".into(),
                        );
                    }
                }
                _ => failures.push(format!(
                    "no frames of {left_reference} and {right_reference} to compare with"
                )),
            }
        }
        Kind::Plain | Kind::ShowUi => {}
    }
    if matches!(current.kind, Kind::Plain | Kind::Bake { .. }) {
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
            .query_filtered::<&Window, With<PrimaryWindow>>()
            .iter(world)
            .next()
            .map(|window| [window.physical_width(), window.physical_height()]),
        "steps_planned": test.steps.iter().map(|step| step.name.clone()).collect::<Vec<_>>(),
        "scenes_not_downloaded": test.missing,
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
