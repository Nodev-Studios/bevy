//! The graphics testbed: one deterministic app that renders a scene one way, measures it, and writes what it
//! saw to a folder: `report.json`, screenshots and `log.txt`. `tools/gfx-check` runs it and gates the result.
//!
//! A run waits until the scene is ready (assets loaded, no pipeline still compiling), renders `--warmup` frames,
//! measures `--frames` frames, captures the next two frames as `a.png` and `b.png`, and exits. The exit code is 0
//! only when the run has nothing to report as a failure.
//!
//! Pass `--help` for the flags. Never run it bare on a desktop someone is using: it opens a window. On this
//! project's machine it goes through `tools/gfx-headless.sh`.

mod baked;
mod contract;
mod ladder;
mod scenes;

use baked::Baked;
use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::AppExit,
    camera::{CameraMainTextureUsages, Hdr},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    diagnostic::{DiagnosticsStore, FrameCount},
    log::{
        tracing_subscriber::{self, Layer},
        BoxedLayer, LogPlugin,
    },
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{PipelineCache, TextureUsages},
        renderer::{RenderAdapterInfo, RenderDevice},
        settings::{WgpuFeatures, WgpuSettings},
        view::screenshot::{Screenshot, ScreenshotCaptured},
        Render, RenderApp, RenderPlugin, RenderSystems,
    },
    solari::{
        pathtracer::{Pathtracer, PathtracingPlugin},
        prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
    },
    time::TimeUpdateStrategy,
    window::{PresentMode, PrimaryWindow, WindowResolution},
};
use contract::Contract;
use ladder::Ladder;
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::anti_alias::dlss::{
    Dlss, DlssPerfQualityMode, DlssProjectId, DlssRayReconstructionFeature,
    DlssRayReconstructionSupported, DlssSuperResolutionFeature, DlssSuperResolutionSupported,
};

const USAGE: &str = "\
gfx_testbed: renders a scene one way, measures it, and writes report.json, screenshots and log.txt.

  --scene interior|many-lights   the scene (default interior; the baked-GI flags below add more)
  --mode raster|solari|reference how the frame is lit (default raster): the deferred rasterizer, Solari's
                                 realtime lighting, or Solari's path tracer accumulating a ground truth
  --restir                       solari: turn ReSTIR on
  --taa                          raster, solari: temporal anti-aliasing
  --dlss-rr                      solari: DLSS Ray Reconstruction
  --dlss-sr                      raster, solari: DLSS Super Resolution
  --dlss-mode dlaa|quality|balanced|performance
                                 internal resolution of either DLSS; always explicit (default dlaa = native)
  --res WxH                      window size in pixels (default 1280x720)
  --warmup N                     frames rendered after the scene is ready and before measuring (default 60)
  --frames N                     frames measured (default 120)
  --camera static|orbit          a still camera, or one that circles the scene one step per frame
                                 (default static). An orbit run also captures eight of its measured frames
  --pose K                       static: stand where the orbit camera stands at its frame K (default 0);
                                 a.png of an orbit run is taken at orbit frame warmup + frames
  --light-toggle-at N            switch one fixture off at measured frame N and capture toggle.png 5 frames later
  --force-no-rt                  ask the GPU for no ray query, as on a card without ray tracing
  --force-black                  spawn no light, so the frame is black: the harness's own failure case
  --out DIR                      folder for the results (default gfx-testbed-out)
";

/// How many frames in a row must pass with every asset loaded and no pipeline compiling before the scene counts
/// as ready. One is not enough: a pipeline is queued the first frame something needs it.
const CALM_FRAMES: u32 = 3;
/// A scene that is not ready after this long is reported as a failure instead of hanging the run.
const LOADING_LIMIT: Duration = Duration::from_secs(180);
/// How long a finished run waits for its screenshots, which arrive a few frames after they are asked for, and
/// for the renderer to stop compiling pipelines. After it the run ends without them.
const SETTLE_LIMIT: Duration = Duration::from_secs(60);
/// Captures an orbit run takes of its measured frames, for the contact sheet.
const ORBIT_CAPTURES: u32 = 8;
/// Frames between switching the fixture off and capturing `toggle.png`: a temporal filter's lag shows there.
const TOGGLE_CAPTURE_DELAY: u32 = 5;

/// The command line, with each flag removed by whoever owns it.
pub struct Flags {
    args: Vec<String>,
    errors: Vec<String>,
}

impl Flags {
    /// Takes a flag without a value; `true` when it was given.
    pub fn switch(&mut self, name: &str) -> bool {
        match self.args.iter().position(|arg| arg == name) {
            Some(index) => {
                self.args.remove(index);
                true
            }
            None => false,
        }
    }

    /// Takes a flag and its value.
    pub fn value(&mut self, name: &str) -> Option<String> {
        let index = self.args.iter().position(|arg| arg == name)?;
        self.args.remove(index);
        if index < self.args.len() {
            Some(self.args.remove(index))
        } else {
            self.error(format!("{name} needs a value"));
            None
        }
    }

    /// Takes a flag and parses its value.
    pub fn parsed<T: FromStr>(&mut self, name: &str) -> Option<T> {
        let value = self.value(name)?;
        let parsed = value.parse().ok();
        if parsed.is_none() {
            self.error(format!("{name}: cannot read {value}"));
        }
        parsed
    }

    /// Takes a flag whose value is one of `choices`.
    pub fn choice<T: Copy>(&mut self, name: &str, choices: &[(&str, T)]) -> Option<T> {
        let value = self.value(name)?;
        let choice = choices.iter().find(|(text, _)| *text == value);
        if choice.is_none() {
            let texts: Vec<&str> = choices.iter().map(|(text, _)| *text).collect();
            self.error(format!("{name} takes {}, not {value}", texts.join("|")));
        }
        choice.map(|(_, choice)| *choice)
    }

    /// Records a usage error; the run does not start.
    pub fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }
}

/// How the frame is lit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// The deferred rasterizer with shadow maps.
    Raster,
    /// Solari's realtime lighting.
    Solari,
    /// Solari's path tracer, accumulating: the ground truth of the traced modes.
    Reference,
}

/// The internal resolution of DLSS.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlssMode {
    /// Native resolution.
    Dlaa,
    /// 67 % per axis.
    Quality,
    /// 58 % per axis.
    Balanced,
    /// 50 % per axis.
    Performance,
}

/// The raw flags.
#[derive(Resource, Clone)]
pub struct Args {
    /// Every argument as given, for the report.
    pub command_line: Vec<String>,
    /// `--scene`.
    pub scene: String,
    /// `--mode`.
    pub mode: Mode,
    /// `--restir`.
    pub restir: bool,
    /// `--taa`.
    pub taa: bool,
    /// `--dlss-rr`.
    pub dlss_rr: bool,
    /// `--dlss-sr`.
    pub dlss_sr: bool,
    /// `--dlss-mode`.
    pub dlss_mode: DlssMode,
    /// `--res`.
    pub res: UVec2,
    /// `--warmup`.
    pub warmup: u32,
    /// `--frames`.
    pub frames: u32,
    /// `--camera orbit`.
    pub orbit: bool,
    /// `--pose`.
    pub pose: u32,
    /// `--light-toggle-at`.
    pub light_toggle_at: Option<u32>,
    /// `--force-no-rt`.
    pub force_no_rt: bool,
    /// `--force-black`.
    pub force_black: bool,
    /// `--out`.
    pub out: PathBuf,
}

impl Args {
    fn parse(flags: &mut Flags) -> Args {
        let command_line = flags.args.clone();
        let res = flags.value("--res").map_or(UVec2::new(1280, 720), |value| {
            let size = value
                .split_once('x')
                .and_then(|(width, height)| {
                    Some(UVec2::new(width.parse().ok()?, height.parse().ok()?))
                })
                .filter(|size| size.x > 0 && size.y > 0);
            if size.is_none() {
                flags.error(format!("--res takes WxH, not {value}"));
            }
            size.unwrap_or(UVec2::ONE)
        });
        let args = Args {
            command_line,
            scene: flags.value("--scene").unwrap_or_else(|| "interior".into()),
            mode: flags
                .choice(
                    "--mode",
                    &[
                        ("raster", Mode::Raster),
                        ("solari", Mode::Solari),
                        ("reference", Mode::Reference),
                    ],
                )
                .unwrap_or(Mode::Raster),
            restir: flags.switch("--restir"),
            taa: flags.switch("--taa"),
            dlss_rr: flags.switch("--dlss-rr"),
            dlss_sr: flags.switch("--dlss-sr"),
            dlss_mode: flags
                .choice(
                    "--dlss-mode",
                    &[
                        ("dlaa", DlssMode::Dlaa),
                        ("quality", DlssMode::Quality),
                        ("balanced", DlssMode::Balanced),
                        ("performance", DlssMode::Performance),
                    ],
                )
                .unwrap_or(DlssMode::Dlaa),
            res,
            warmup: flags.parsed("--warmup").unwrap_or(60),
            frames: flags.parsed("--frames").unwrap_or(120),
            orbit: flags
                .choice("--camera", &[("static", false), ("orbit", true)])
                .unwrap_or(false),
            pose: flags.parsed("--pose").unwrap_or(0),
            light_toggle_at: flags.parsed("--light-toggle-at"),
            force_no_rt: flags.switch("--force-no-rt"),
            force_black: flags.switch("--force-black"),
            out: flags
                .value("--out")
                .map_or_else(|| PathBuf::from("gfx-testbed-out"), PathBuf::from),
        };

        let filters = [args.taa, args.dlss_rr, args.dlss_sr];
        if filters.iter().filter(|on| **on).count() > 1 {
            flags.error("--taa, --dlss-rr and --dlss-sr are one choice: each resolves the frame its own way");
        }
        if args.mode != Mode::Solari && (args.restir || args.dlss_rr) {
            flags.error("--restir and --dlss-rr need --mode solari");
        }
        if args.mode == Mode::Reference && (args.taa || args.dlss_sr) {
            flags.error("--mode reference accumulates its own frame: no --taa, no --dlss-sr");
        }
        if args.orbit && args.pose != 0 {
            flags.error("--pose places a static camera; it has no meaning with --camera orbit");
        }
        if args.frames == 0 {
            flags.error("--frames must be at least 1");
        }
        if args
            .light_toggle_at
            .is_some_and(|at| at + TOGGLE_CAPTURE_DELAY >= args.frames)
        {
            flags.error(format!(
                "--light-toggle-at must leave {TOGGLE_CAPTURE_DELAY} measured frames after it"
            ));
        }
        args
    }
}

/// What a scene tells the run about itself.
pub struct Scene {
    /// Where the camera stands at orbit frame `k`. Frame 0 is the still camera's place.
    pub orbit: fn(u32) -> Transform,
    /// The camera's vertical field of view, in radians.
    pub fov: f32,
    /// Assets the scene loads from disk; the scene is not ready before they are.
    pub assets: Vec<UntypedHandle>,
}

/// The scene of this run.
#[derive(Resource)]
pub struct ActiveScene(pub Scene);

/// The camera every scene is seen through.
#[derive(Component)]
pub struct TestbedCamera;

/// The fixture `--light-toggle-at` switches off. A scene marks one light with it.
#[derive(Component)]
pub struct ToggleLight;

/// What this GPU and this binary can do.
#[derive(Resource, Clone, Copy)]
pub struct Capabilities {
    /// The device has every feature Solari needs.
    pub ray_tracing: bool,
    /// DLSS Super Resolution is compiled in and supported.
    pub dlss_sr: bool,
    /// DLSS Ray Reconstruction is compiled in and supported.
    pub dlss_rr: bool,
}

/// Where the run is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Waiting for the scene to be ready. No frame is counted yet.
    Loading,
    /// Warm-up and measured frames, then the two final captures.
    Running,
    /// Waiting for the screenshots that were asked for.
    Finishing,
    /// The report is written and the app is exiting.
    Done,
}

/// The run's clock and measurements. Systems that read it in `Update` order themselves `.after(drive)`.
#[derive(Resource)]
pub struct Run {
    /// Where the run is.
    pub phase: Phase,
    /// The frame being simulated, counted from the scene-ready signal. Frames `warmup..warmup + frames` are the
    /// measured ones.
    pub frame: u32,
    /// The orbit frame the camera stands at this frame.
    pub orbit_frame: u32,
    since: Instant,
    calm_frames: u32,
    ready_at: Option<u32>,
    last_tick: Option<Instant>,
    frame_ms: Vec<f64>,
    pass_ms: BTreeMap<String, Vec<f64>>,
    /// When each pass's last recorded measurement was taken. A pass that stops running (`blas_build` after the
    /// load) keeps its last value in the diagnostics store, so only measurements newer than this are recorded.
    pass_seen: BTreeMap<String, Instant>,
    peak_memory: Option<(u64, u64)>,
}

/// What ends up in `report.json` beyond the run's own measurements.
#[derive(Resource, Default)]
pub struct Report {
    /// Whether the frame is lit by ray tracing. `tools/gfx-check log-gate` fails a traced run whose log says a
    /// Solari plugin was not loaded, and forgives a raster run for it.
    pub traced: bool,
    /// Why the run failed. Anything here makes the exit code non-zero.
    pub failures: Vec<String>,
    /// Extra top-level sections of `report.json`, by key.
    pub sections: Map<String, Value>,
}

struct Capture {
    name: String,
    frame: u32,
    orbit_frame: u32,
    image: Image,
}

/// The screenshots of the run. They are written to disk when the run ends, so that encoding a PNG does not
/// stall a measured frame.
#[derive(Resource, Default)]
pub struct Captures {
    pending: u32,
    done: Vec<Capture>,
}

impl Captures {
    /// Captures the frame being simulated as `<name>.png`.
    pub fn request(&mut self, commands: &mut Commands, name: impl Into<String>, run: &Run) {
        let name = name.into();
        let (frame, orbit_frame) = (run.frame, run.orbit_frame);
        self.pending += 1;
        commands.spawn(Screenshot::primary_window()).observe(
            move |captured: On<ScreenshotCaptured>, mut captures: ResMut<Captures>| {
                captures.pending -= 1;
                captures.done.push(Capture {
                    name: name.clone(),
                    frame,
                    orbit_frame,
                    image: captured.image.clone(),
                });
            },
        );
    }
}

/// How many pipelines the render world is still compiling; `u32::MAX` until it has rendered once.
#[derive(Resource, Clone)]
pub struct PendingPipelines(Arc<AtomicU32>);

fn count_pending_pipelines(cache: Res<PipelineCache>, pending: Res<PendingPipelines>) {
    pending
        .0
        .store(cache.waiting_pipelines().count() as u32, Ordering::Relaxed);
}

/// Bevy's log, also written to `log.txt` in the output folder.
fn log_file_layer(app: &mut App) -> Option<BoxedLayer> {
    let file = std::fs::File::create(app.world().resource::<Args>().out.join("log.txt")).ok()?;
    Some(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(Mutex::new(file))
            .boxed(),
    )
}

fn main() -> AppExit {
    let mut flags = Flags {
        args: std::env::args().skip(1).collect(),
        errors: Vec::new(),
    };
    if flags.switch("--help") {
        println!("{USAGE}{}{}{}", ladder::HELP, contract::HELP, baked::HELP);
        return AppExit::Success;
    }
    let args = Args::parse(&mut flags);
    let ladder = Ladder::parse(&mut flags);
    let contract = Contract::parse(&mut flags);
    let mut baked = Baked::parse(&mut flags);
    for unknown in &flags.args {
        flags.errors.push(format!("unknown argument {unknown}"));
    }
    if let Err(error) = std::fs::create_dir_all(&args.out) {
        flags
            .errors
            .push(format!("cannot create {}: {error}", args.out.display()));
    }
    if !flags.errors.is_empty() {
        for error in &flags.errors {
            eprintln!("gfx_testbed: {error}");
        }
        eprintln!("\n{USAGE}{}{}{}", ladder::HELP, contract::HELP, baked::HELP);
        return AppExit::from_code(2);
    }
    if let Some(exit) = baked.bake(&args) {
        return exit;
    }

    // A panic reaches the console on its own; this puts it in log.txt too, for a run nobody watched.
    let panic_log = args.out.join("log.txt");
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(&panic_log) {
            let _ = writeln!(file, "panicked: {info}");
        }
        default_hook(info);
    }));

    let mut app = App::new();
    app.insert_resource(args.clone());

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy_asset::uuid::uuid!(
        "678abbd5-6054-4f81-b88d-a202595f83a0" // The testbed's own id; a game generates its own.
    )));

    let pending_pipelines = PendingPipelines(Arc::new(AtomicU32::new(u32::MAX)));
    baked.register_source(&mut app, &args);
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "gfx_testbed".into(),
                    // No decorations and a scale factor of 1: the frame is exactly the size asked for.
                    resolution: WindowResolution::new(args.res.x, args.res.y)
                        .with_scale_factor_override(1.0),
                    decorations: false,
                    resizable: false,
                    // With vsync on, every frame time would read as the display's refresh interval.
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: WgpuSettings {
                    // Only the ray query goes: a card without ray tracing still has the binding arrays
                    // Solari also asks for, and the rasterizer uses them.
                    disabled_features: args
                        .force_no_rt
                        .then_some(WgpuFeatures::EXPERIMENTAL_RAY_QUERY),
                    ..default()
                }
                .into(),
                ..default()
            })
            .set(LogPlugin {
                custom_layer: log_file_layer,
                ..default()
            }),
        // Always added, as in a game that switches presets while it runs. It makes every material deferred.
        SolariPlugins,
        PathtracingPlugin,
        RenderDiagnosticsPlugin,
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .insert_resource(GlobalAmbientLight::NONE)
    // A fixed time step: whatever moves with time moves the same on every machine.
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
        1.0 / 60.0,
    )))
    .insert_resource(Run {
        phase: Phase::Loading,
        frame: 0,
        orbit_frame: args.pose,
        since: Instant::now(),
        calm_frames: 0,
        ready_at: None,
        last_tick: None,
        frame_ms: Vec::new(),
        pass_ms: BTreeMap::new(),
        pass_seen: BTreeMap::new(),
        peak_memory: None,
    })
    .insert_resource(ladder.clone())
    .insert_resource(contract.clone())
    .insert_resource(pending_pipelines.clone())
    .init_resource::<Report>()
    .init_resource::<Captures>()
    .add_systems(Startup, setup)
    .add_systems(Update, (drive, finish).chain());

    app.sub_app_mut(RenderApp)
        .insert_resource(pending_pipelines)
        .add_systems(
            Render,
            count_pending_pipelines.in_set(RenderSystems::Cleanup),
        );

    ladder.build(&mut app, &args);
    contract.build(&mut app, &args);
    baked.build(&mut app, &args);
    app.run()
}

fn setup(world: &mut World) {
    let args = world.resource::<Args>().clone();
    let ray_tracing = world
        .resource::<RenderDevice>()
        .features()
        .contains(SolariPlugins::required_wgpu_features());
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    let (dlss_sr, dlss_rr) = (
        world.contains_resource::<DlssSuperResolutionSupported>(),
        world.contains_resource::<DlssRayReconstructionSupported>(),
    );
    #[cfg(not(all(feature = "dlss", not(feature = "force_disable_dlss"))))]
    let (dlss_sr, dlss_rr) = (false, false);
    let capabilities = Capabilities {
        ray_tracing,
        dlss_sr,
        dlss_rr,
    };
    world.insert_resource(capabilities);

    // A run that cannot be what was asked for stops here, with its reason in the report, and draws nothing.
    let stop = |world: &mut World, failure: String| {
        world.resource_mut::<Report>().failures.push(failure);
        world.resource_mut::<Run>().phase = Phase::Finishing;
    };

    let Some(scene) = scenes::spawn(&args.scene, world)
        .or_else(|| Contract::scene(&args.scene, world))
        .or_else(|| Ladder::scene(&args.scene, world))
        .or_else(|| Baked::scene(&args.scene, world))
    else {
        return stop(world, format!("there is no scene called {}", args.scene));
    };

    let ladder = world.resource::<Ladder>().clone();
    let contract = world.resource::<Contract>().clone();
    let baked = world.resource::<Baked>().clone();
    let mut camera = world.spawn((
        TestbedCamera,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: scene.fov,
            ..default()
        }),
        (scene.orbit)(args.pose),
        Msaa::Off,
    ));
    let laddered = ladder.camera(&args, &mut camera);
    let raw_failure = if laddered {
        None
    } else {
        raw_camera(&args, capabilities, &mut camera).err()
    };
    contract.camera(&args, &mut camera);
    baked.camera(&args, &mut camera);
    let scene = baked.prepare(world, scene);
    world.insert_resource(ActiveScene(scene));
    if let Some(failure) = raw_failure {
        let camera = world
            .query_filtered::<Entity, With<TestbedCamera>>()
            .single(world)
            .expect("the camera was spawned above");
        world.despawn(camera);
        return stop(world, failure);
    }

    if !laddered {
        world.resource_mut::<Report>().traced = args.mode != Mode::Raster;
        if args.mode != Mode::Raster {
            raw_traced_scene(world);
        }
    }
    if args.force_black {
        let lights: Vec<Entity> = world
            .query_filtered::<Entity, Or<(
                With<PointLight>,
                With<SpotLight>,
                With<RectLight>,
                With<DirectionalLight>,
            )>>()
            .iter(world)
            .collect();
        for light in lights {
            world.despawn(light);
        }
    }
}

/// Gives the camera the components of a raw mode, or says why this machine cannot run it.
fn raw_camera(
    args: &Args,
    capabilities: Capabilities,
    camera: &mut EntityWorldMut,
) -> Result<(), String> {
    if args.mode != Mode::Raster && !capabilities.ray_tracing {
        return Err(format!(
            "--mode {:?} needs ray tracing, which this device does not offer",
            args.mode
        )
        .to_lowercase());
    }
    if (args.dlss_rr && !capabilities.dlss_rr) || (args.dlss_sr && !capabilities.dlss_sr) {
        return Err(if cfg!(feature = "dlss") {
            "the DLSS feature asked for is not supported here: it needs an RTX card, and at run time its \
             libraries beside the binary or in $DLSS_SDK"
                .into()
        } else {
            "this binary was built without the `dlss` feature".into()
        });
    }

    let mut usages = CameraMainTextureUsages::default();
    match args.mode {
        // Solari makes every material deferred for the whole app, so the rasterizer is the deferred one. A
        // depth prepass without the deferred prepass panics.
        Mode::Raster => {
            camera.insert((Hdr, DeferredPrepass, DepthPrepass, MotionVectorPrepass));
        }
        Mode::Solari => {
            usages = usages.with(TextureUsages::STORAGE_BINDING);
            camera.insert(SolariLighting {
                restir: args.restir,
                ..default()
            });
        }
        Mode::Reference => {
            usages = usages.with(TextureUsages::STORAGE_BINDING);
            // Kept reset until the scene is ready, so that nothing half-loaded stays in the accumulation.
            camera.insert(Pathtracer { reset: true });
        }
    }
    if args.taa {
        camera.insert(TemporalAntiAliasing::default());
    }
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    {
        let perf_quality_mode = match args.dlss_mode {
            DlssMode::Dlaa => DlssPerfQualityMode::Dlaa,
            DlssMode::Quality => DlssPerfQualityMode::Quality,
            DlssMode::Balanced => DlssPerfQualityMode::Balanced,
            DlssMode::Performance => DlssPerfQualityMode::Performance,
        };
        if args.dlss_rr {
            camera.insert(Dlss::<DlssRayReconstructionFeature> {
                perf_quality_mode,
                ..default()
            });
        }
        if args.dlss_sr {
            // DLSS Super Resolution copies its output into the main texture.
            usages = usages.with(TextureUsages::COPY_DST);
            camera.insert(Dlss::<DlssSuperResolutionFeature> {
                perf_quality_mode,
                ..default()
            });
        }
    }
    camera.insert(usages);
    Ok(())
}

/// Makes a scene written for the rasterizer traceable: every mesh joins the traced scene, and shadow maps,
/// which the tracer replaces, go.
fn raw_traced_scene(world: &mut World) {
    let meshes: Vec<(Entity, Handle<Mesh>)> = world
        .query_filtered::<(Entity, &Mesh3d), Without<RaytracingMesh3d>>()
        .iter(world)
        .map(|(entity, mesh)| (entity, mesh.0.clone()))
        .collect();
    for (entity, mesh) in meshes {
        world.entity_mut(entity).insert(RaytracingMesh3d(mesh));
    }
    for mut light in world.query::<&mut PointLight>().iter_mut(world) {
        light.shadow_maps_enabled = false;
    }
    for mut light in world.query::<&mut SpotLight>().iter_mut(world) {
        light.shadow_maps_enabled = false;
    }
    for mut light in world.query::<&mut DirectionalLight>().iter_mut(world) {
        light.shadow_maps_enabled = false;
    }
}

/// The run's clock: waits for the scene, counts frames, moves the camera, measures, and asks for the captures.
pub fn drive(
    mut commands: Commands,
    args: Res<Args>,
    scene: Option<Res<ActiveScene>>,
    mut run: ResMut<Run>,
    mut report: ResMut<Report>,
    mut captures: ResMut<Captures>,
    mut camera: Query<(&mut Transform, Option<&mut Pathtracer>), With<TestbedCamera>>,
    toggle_lights: Query<Entity, With<ToggleLight>>,
    window: Single<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
    pending_pipelines: Res<PendingPipelines>,
    diagnostics: Res<DiagnosticsStore>,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
) {
    let (Some(scene), Ok((mut transform, pathtracer))) = (scene, camera.single_mut()) else {
        return;
    };
    match run.phase {
        Phase::Loading => {
            let loaded = scene
                .0
                .assets
                .iter()
                .all(|asset| asset_server.is_loaded_with_dependencies(asset.id()));
            let compiling = pending_pipelines.0.load(Ordering::Relaxed);
            run.calm_frames = if loaded && compiling == 0 {
                run.calm_frames + 1
            } else {
                0
            };
            if run.calm_frames < CALM_FRAMES {
                if run.since.elapsed() > LOADING_LIMIT {
                    report.failures.push(format!(
                        "the scene was not ready after {} s (assets loaded: {loaded}, pipelines compiling: {})",
                        LOADING_LIMIT.as_secs(),
                        compiling as i32
                    ));
                    run.phase = Phase::Finishing;
                    run.since = Instant::now();
                }
                return;
            }
            if window.physical_size() != args.res {
                report.failures.push(format!(
                    "the window is {}x{}, not the {}x{} asked for: is the screen large enough?",
                    window.physical_width(),
                    window.physical_height(),
                    args.res.x,
                    args.res.y
                ));
                run.phase = Phase::Finishing;
                run.since = Instant::now();
                return;
            }
            info!("scene ready at app frame {}", frame_count.0);
            run.ready_at = Some(frame_count.0);
            run.phase = Phase::Running;
            if let Some(mut pathtracer) = pathtracer {
                pathtracer.reset = false;
            }
        }
        Phase::Running => run.frame += 1,
        Phase::Finishing | Phase::Done => return,
    }

    let frame = run.frame;
    let measured = args.warmup..args.warmup + args.frames;
    let now = Instant::now();
    if let Some(last) = run.last_tick.replace(now)
        && measured.contains(&(frame - 1))
    {
        run.frame_ms.push((now - last).as_secs_f64() * 1000.0);
    }
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        let (Some(pass), Some(measurement)) = (
            path.strip_prefix("render/")
                .and_then(|path| path.strip_suffix("/elapsed_gpu")),
            diagnostic.measurement(),
        ) else {
            continue;
        };
        // Tracked on every frame, so a value left over from the warm-up or the load is not new when measuring.
        if run.pass_seen.get(pass) == Some(&measurement.time) {
            continue;
        }
        run.pass_seen.insert(pass.to_owned(), measurement.time);
        if measured.contains(&frame) {
            run.pass_ms
                .entry(pass.to_owned())
                .or_default()
                .push(measurement.value);
        }
    }
    // Walking the allocator's list is not free, so not every frame.
    if frame.is_multiple_of(32)
        && let Some(memory) = render_device.wgpu_device().generate_allocator_report()
    {
        let peak = run.peak_memory.get_or_insert((0, 0));
        peak.0 = peak.0.max(memory.total_allocated_bytes);
        peak.1 = peak.1.max(memory.total_reserved_bytes);
    }

    // A still camera is never written to: the path tracer restarts its accumulation whenever the camera's
    // transform changes.
    if args.orbit {
        run.orbit_frame = frame;
        *transform = (scene.0.orbit)(frame);
        let step = (args.frames / ORBIT_CAPTURES).max(1);
        if measured.contains(&frame)
            && (frame - args.warmup).is_multiple_of(step)
            && (frame - args.warmup) / step < ORBIT_CAPTURES
        {
            captures.request(&mut commands, format!("orbit-{frame:05}"), &run);
        }
    }
    if let Some(at) = args.light_toggle_at {
        if frame == args.warmup + at {
            if toggle_lights.is_empty() {
                report
                    .failures
                    .push("--light-toggle-at: the scene marks no fixture to switch off".into());
            }
            for light in &toggle_lights {
                commands.entity(light).despawn();
            }
        }
        if frame == args.warmup + at + TOGGLE_CAPTURE_DELAY {
            captures.request(&mut commands, "toggle", &run);
        }
    }
    if frame == measured.end {
        captures.request(&mut commands, "a", &run);
    }
    if frame == measured.end + 1 {
        captures.request(&mut commands, "b", &run);
        run.phase = Phase::Finishing;
        run.since = now;
    }
}

/// The value at `quantile` (0 to 1) of sorted `values`.
fn quantile(values: &[f64], quantile: f64) -> f64 {
    values[((values.len() as f64 * quantile).ceil() as usize).clamp(1, values.len()) - 1]
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    quantile(values, 0.5)
}

/// Once every screenshot has arrived: writes them and `report.json`, and ends the app.
fn finish(world: &mut World) {
    let run = world.resource::<Run>();
    let pending = world.resource::<Captures>().pending;
    // A run that stops in its first frames would exit while pipelines still compile on other threads, and the
    // process then aborts in its teardown instead of returning its exit code.
    let compiling = world
        .resource::<PendingPipelines>()
        .0
        .load(Ordering::Relaxed);
    if run.phase != Phase::Finishing
        || ((pending > 0 || compiling > 0) && run.since.elapsed() < SETTLE_LIMIT)
    {
        return;
    }
    let args = world.resource::<Args>().clone();
    let mut failures = std::mem::take(&mut world.resource_mut::<Report>().failures);
    if pending > 0 {
        failures.push(format!("{pending} screenshots never arrived"));
    }

    let mut screenshots = Vec::new();
    for capture in std::mem::take(&mut world.resource_mut::<Captures>().done) {
        let file = format!("{}.png", capture.name);
        let saved = capture
            .image
            .try_into_dynamic()
            .map_err(|error| error.to_string())
            .and_then(|image| {
                image
                    .to_rgb8()
                    .save(args.out.join(&file))
                    .map_err(|error| error.to_string())
            });
        if let Err(error) = saved {
            failures.push(format!("{file}: {error}"));
        }
        screenshots.push(json!({
            "name": capture.name,
            "file": file,
            "frame": capture.frame,
            "orbit_frame": capture.orbit_frame,
        }));
    }

    let mut run = world.resource_mut::<Run>();
    run.phase = Phase::Done;
    let frame_ms = if run.frame_ms.is_empty() {
        Value::Null
    } else {
        let frames = &mut run.frame_ms;
        frames.sort_by(f64::total_cmp);
        json!({
            "median": quantile(frames, 0.5),
            "p95": quantile(frames, 0.95),
            "p99": quantile(frames, 0.99),
            "mean": frames.iter().sum::<f64>() / frames.len() as f64,
            "count": frames.len(),
        })
    };
    let mut gpu_pass_ms = Map::new();
    // A GPU shared with another process time-slices between them, and a long dispatch that a slice lands in
    // reads the other process's time too. The 10th percentile reads the frames no slice landed in.
    let mut gpu_pass_ms_p10 = Map::new();
    for (pass, mut values) in std::mem::take(&mut run.pass_ms) {
        gpu_pass_ms.insert(pass.clone(), json!(median(&mut values)));
        gpu_pass_ms_p10.insert(pass, json!(quantile(&values, 0.1)));
    }
    let gpu_memory = run.peak_memory.map(|(allocated, reserved)| {
        json!({
            "source": "wgpu's allocator: every heap, without what a vendor SDK allocates itself",
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        })
    });
    let ready_at = run.ready_at;

    let adapter = world.resource::<RenderAdapterInfo>();
    let adapter = json!({
        "name": adapter.name,
        "vendor": adapter.vendor,
        "device": adapter.device,
        "device_type": format!("{:?}", adapter.device_type),
        "driver": adapter.driver,
        "driver_info": adapter.driver_info,
        "backend": format!("{:?}", adapter.backend),
    });
    let capabilities = *world.resource::<Capabilities>();
    let window = world
        .query_filtered::<&Window, With<PrimaryWindow>>()
        .single(world)
        .map(Window::physical_size)
        .unwrap_or_default();
    let report = world.resource::<Report>();
    let mut json = json!({
        "schema": 1,
        "args": args.command_line,
        "scene": args.scene,
        "mode": format!("{:?}", args.mode).to_lowercase(),
        "traced": report.traced,
        "restir": args.restir,
        "taa": args.taa,
        "dlss": (args.dlss_rr || args.dlss_sr).then(|| json!({
            "feature": if args.dlss_rr { "ray reconstruction" } else { "super resolution" },
            "mode": format!("{:?}", args.dlss_mode).to_lowercase(),
        })),
        "adapter": adapter,
        "capabilities": {
            "ray_tracing": capabilities.ray_tracing,
            "dlss_compiled": cfg!(all(feature = "dlss", not(feature = "force_disable_dlss"))),
            "dlss_super_resolution": capabilities.dlss_sr,
            "dlss_ray_reconstruction": capabilities.dlss_rr,
        },
        "resolution": [window.x, window.y],
        "warmup": args.warmup,
        "frames": args.frames,
        "ready_at_app_frame": ready_at,
        "camera": {
            "kind": if args.orbit { "orbit" } else { "static" },
            "pose": args.pose,
        },
        "light_toggle_at": args.light_toggle_at,
        "frame_ms": frame_ms,
        "gpu_pass_ms": gpu_pass_ms,
        "gpu_pass_ms_p10": gpu_pass_ms_p10,
        "gpu_memory": gpu_memory,
        "screenshots": screenshots,
        "failures": failures,
    });
    if let Value::Object(top) = &mut json {
        top.extend(report.sections.clone());
    }

    let text = serde_json::to_string_pretty(&json).expect("a JSON value serializes");
    if let Err(error) = std::fs::write(args.out.join("report.json"), text) {
        error!("cannot write report.json: {error}");
        failures.push(error.to_string());
    }
    for failure in &failures {
        error!("{failure}");
    }
    world.write_message(if failures.is_empty() {
        AppExit::Success
    } else {
        AppExit::error()
    });
}
