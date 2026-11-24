//! TSR's test app: renders one of `gfx_testbed`'s scenes on the deferred rasterizer, anti-aliased by TAA or
//! upscaled by TSR or by DLSS Super Resolution, and writes what `tools/gfx-check` reads: `a.png` and `b.png` (two
//! consecutive frames after the measured ones, or a longer run of still frames), the orbit captures, `toggle.png`
//! and `report.json`.
//!
//! DLSS Super Resolution is here as an anchor for the upscaler's gates: a production upscaler on the same scene,
//! showing whether a gate is reachable at all. It is never a Raster gate (the card Raster models has no DLSS). It
//! needs the `dlss` feature and `DLSS_SDK` at build and run time.
//!
//! It stands in for the testbed's `--upscaler tsr` until the testbed's ladder flags exist, and follows the
//! testbed's run: wait until the scene is ready, render `--warmup` frames, measure `--frames` frames, capture.
//! Never run it bare on a desktop someone is using: it opens a window. On this project's machine it goes
//! through `tools/gfx-headless.sh`.

#[path = "gfx_testbed/scenes.rs"]
mod scenes;

use bevy::{
    anti_alias::{
        taa::TemporalAntiAliasing,
        tsr::{Tsr, TsrSupported},
    },
    app::AppExit,
    camera::{Hdr, MainPassResolutionOverride},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    diagnostic::{DiagnosticsStore, FrameCount},
    pbr::DefaultOpaqueRendererMethod,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::PipelineCache,
        renderer::RenderAdapterInfo,
        view::screenshot::{Screenshot, ScreenshotCaptured},
        Render, RenderApp, RenderSystems,
    },
    time::TimeUpdateStrategy,
    window::{PresentMode, WindowResolution},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy::{
    anti_alias::dlss::{
        Dlss, DlssPerfQualityMode, DlssProjectId, DlssSuperResolutionFeature,
        DlssSuperResolutionSupported,
    },
    camera::CameraMainTextureUsages,
    render::render_resource::TextureUsages,
};

const USAGE: &str = "\
gfx_tsr: renders a gfx_testbed scene on the rasterizer with TAA or TSR, and writes report.json and screenshots.

  --scene interior|many-lights   the scene (default interior)
  --aa taa|tsr|dlss              TAA at the output's resolution, TSR, or DLSS Super Resolution (default taa)
  --scale F                      tsr, dlss: the render resolution as a share of the output's (default 0.5); dlss
                                 takes 0.5 (Performance), 0.58 (Balanced), 0.6667 (Quality) or 1 (DLAA)
  --res WxH                      window size in pixels (default 1280x720)
  --warmup N                     frames after the scene is ready and before measuring (default 60)
  --frames N                     frames measured (default 120)
  --camera static|orbit          a still camera, or one circling the scene one step per frame (default static)
  --pose K                       static: stand where the orbit camera is at its frame K (default 0)
  --light-toggle-at N            switch one fixture off at measured frame N, capture toggle.png 5 frames later
  --light-off                    spawn the scene without that fixture
  --still-frames N               consecutive frames captured after the measured ones (default 2: a.png and
                                 b.png); above 2 they are still-000.png on, and a.png and b.png link the first two
  --align P --phase K            start measuring on a frame whose frame count is K modulo P (default 1 and 0):
                                 with P a multiple of the jitter cycle, every run sees the same jitter phases,
                                 and K shifts them
  --out DIR                      folder for the results (default gfx-tsr-out)
";

/// Frames in a row with every asset loaded and no pipeline compiling before the scene counts as ready.
const CALM_FRAMES: u32 = 3;
/// Captures of an orbit run, for the contact sheet.
const ORBIT_CAPTURES: u32 = 8;
/// Frames between switching the fixture off and capturing `toggle.png`, as in the testbed.
const TOGGLE_CAPTURE_DELAY: u32 = 5;
/// A run that has not finished after this long stops with a failure.
const RUN_LIMIT: Duration = Duration::from_secs(240);

/// What a scene tells the run about itself (the testbed's `Scene`, which `scenes.rs` builds).
pub struct Scene {
    /// Where the camera stands at orbit frame `k`.
    pub orbit: fn(u32) -> Transform,
    /// The camera's vertical field of view, in radians.
    pub fov: f32,
    /// Assets the scene loads from disk.
    pub assets: Vec<UntypedHandle>,
}

/// The fixture `--light-toggle-at` switches off.
#[derive(Component)]
pub struct ToggleLight;

/// How the frame is anti-aliased or upscaled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Aa {
    Taa,
    Tsr,
    Dlss,
}

#[derive(Resource, Clone)]
struct Args {
    command_line: Vec<String>,
    scene: String,
    aa: Aa,
    scale: f32,
    res: UVec2,
    warmup: u32,
    frames: u32,
    orbit: bool,
    pose: u32,
    light_toggle_at: Option<u32>,
    light_off: bool,
    still_frames: u32,
    align: u32,
    phase: u32,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let command_line: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        command_line: command_line.clone(),
        scene: "interior".into(),
        aa: Aa::Taa,
        scale: 0.5,
        res: UVec2::new(1280, 720),
        warmup: 60,
        frames: 120,
        orbit: false,
        pose: 0,
        light_toggle_at: None,
        light_off: false,
        still_frames: 2,
        align: 1,
        phase: 0,
        out: "gfx-tsr-out".into(),
    };
    let mut words = command_line.into_iter();
    while let Some(flag) = words.next() {
        if flag == "--light-off" {
            args.light_off = true;
            continue;
        }
        let value = words.next().ok_or(format!("{flag} needs a value"))?;
        let number = |value: &str| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{flag}: cannot read {value}"))
        };
        match flag.as_str() {
            "--scene" => args.scene = value,
            "--aa" => {
                args.aa = match value.as_str() {
                    "taa" => Aa::Taa,
                    "tsr" => Aa::Tsr,
                    "dlss" => Aa::Dlss,
                    _ => return Err(format!("--aa takes taa|tsr|dlss, not {value}")),
                }
            }
            "--scale" => {
                args.scale = value
                    .parse()
                    .ok()
                    .filter(|scale| (0.0..=1.0).contains(scale) && *scale > 0.0)
                    .ok_or(format!("--scale takes a number in (0, 1], not {value}"))?;
            }
            "--res" => {
                let (w, h) = value
                    .split_once('x')
                    .ok_or(format!("--res takes WxH, not {value}"))?;
                args.res = UVec2::new(number(w)?, number(h)?);
            }
            "--warmup" => args.warmup = number(&value)?,
            "--frames" => args.frames = number(&value)?.max(1),
            "--camera" => {
                args.orbit = match value.as_str() {
                    "static" => false,
                    "orbit" => true,
                    _ => return Err(format!("--camera takes static|orbit, not {value}")),
                }
            }
            "--pose" => args.pose = number(&value)?,
            "--light-toggle-at" => args.light_toggle_at = Some(number(&value)?),
            "--still-frames" => args.still_frames = number(&value)?.max(2),
            "--align" => args.align = number(&value)?.max(1),
            "--phase" => args.phase = number(&value)?,
            "--out" => args.out = value.into(),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(args)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Loading,
    Running,
    Finishing,
    Done,
}

#[derive(Resource)]
struct Run {
    phase: Phase,
    frame: u32,
    orbit_frame: u32,
    calm_frames: u32,
    since: Instant,
    last_tick: Option<Instant>,
    frame_ms: Vec<f64>,
    pass_ms: BTreeMap<String, Vec<f64>>,
    pending_captures: u32,
    captures: Vec<serde_json::Value>,
    failures: Vec<String>,
}

#[derive(Resource)]
struct ActiveScene(Scene);

#[derive(Component)]
struct TestCamera;

/// Pipelines the render world is still compiling; `u32::MAX` until it has rendered once.
#[derive(Resource, Clone)]
struct PendingPipelines(Arc<AtomicU32>);

/// The main pass's resolution when an upscaler overrides it, as `width << 32 | height`; 0 when none does.
#[derive(Resource, Clone)]
struct RenderSize(Arc<AtomicU64>);

fn main() -> AppExit {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("gfx_tsr: {error}\n\n{USAGE}");
            return AppExit::from_code(2);
        }
    };
    if let Err(error) = std::fs::create_dir_all(&args.out) {
        eprintln!("gfx_tsr: cannot create {}: {error}", args.out.display());
        return AppExit::from_code(2);
    }

    let pending_pipelines = PendingPipelines(Arc::new(AtomicU32::new(u32::MAX)));
    let render_size = RenderSize(Arc::new(AtomicU64::new(0)));
    let mut app = App::new();
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "3b0f6a52-8d1e-4c0b-9f47-2e5d7c1a9b64" // gfx_tsr's own id; a game generates its own.
    )));
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "gfx_tsr".into(),
                resolution: WindowResolution::new(args.res.x, args.res.y)
                    .with_scale_factor_override(1.0),
                decorations: false,
                resizable: false,
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }),
        RenderDiagnosticsPlugin,
    ))
    // As in the testbed, where Solari makes every material deferred, and as on the ladder's Raster rung.
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .insert_resource(ClearColor(Color::BLACK))
    .insert_resource(GlobalAmbientLight::NONE)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
        1.0 / 60.0,
    )))
    .insert_resource(Run {
        phase: Phase::Loading,
        frame: 0,
        orbit_frame: args.pose,
        calm_frames: 0,
        since: Instant::now(),
        last_tick: None,
        frame_ms: Vec::new(),
        pass_ms: BTreeMap::new(),
        pending_captures: 0,
        captures: Vec::new(),
        failures: Vec::new(),
    })
    .insert_resource(args)
    .insert_resource(pending_pipelines.clone())
    .insert_resource(render_size.clone())
    .add_systems(Startup, setup)
    .add_systems(Update, (drive, finish).chain());

    app.sub_app_mut(RenderApp)
        .insert_resource(pending_pipelines)
        .insert_resource(render_size)
        .add_systems(
            Render,
            (count_pending_pipelines, record_render_size).in_set(RenderSystems::Cleanup),
        );
    app.run()
}

fn count_pending_pipelines(cache: Res<PipelineCache>, pending: Res<PendingPipelines>) {
    pending
        .0
        .store(cache.waiting_pipelines().count() as u32, Ordering::Relaxed);
}

fn record_render_size(views: Query<&MainPassResolutionOverride>, size: Res<RenderSize>) {
    if let Some(resolution) = views.iter().next() {
        size.0.store(
            u64::from(resolution.x) << 32 | u64::from(resolution.y),
            Ordering::Relaxed,
        );
    }
}

/// Why the device or this build cannot run the asked method, if it cannot.
fn unsupported(world: &World, args: &Args) -> Option<String> {
    match args.aa {
        Aa::Taa => None,
        Aa::Tsr => (!world.contains_resource::<TsrSupported>())
            .then_some("TSR is not supported on this device".into()),
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        Aa::Dlss => {
            if dlss_mode(args.scale).is_none() {
                Some(format!("--aa dlss has no mode for --scale {}", args.scale))
            } else if !world.contains_resource::<DlssSuperResolutionSupported>() {
                Some(
                    "DLSS Super Resolution is not supported here: it needs an RTX card, and at run time its \
                     libraries beside the binary or in $DLSS_SDK"
                        .into(),
                )
            } else {
                None
            }
        }
        #[cfg(not(all(feature = "dlss", not(feature = "force_disable_dlss"))))]
        Aa::Dlss => Some("this binary was built without the `dlss` feature".into()),
    }
}

/// The DLSS mode whose nominal render scale is `scale`.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
fn dlss_mode(scale: f32) -> Option<DlssPerfQualityMode> {
    [
        (0.5, DlssPerfQualityMode::Performance),
        (0.58, DlssPerfQualityMode::Balanced),
        (2.0 / 3.0, DlssPerfQualityMode::Quality),
        (1.0, DlssPerfQualityMode::Dlaa),
    ]
    .into_iter()
    .find(|(nominal, _)| (nominal - scale).abs() < 0.01)
    .map(|(_, mode)| mode)
}

fn setup(world: &mut World) {
    let args = world.resource::<Args>().clone();
    if let Some(reason) = unsupported(world, &args) {
        let mut run = world.resource_mut::<Run>();
        run.failures.push(reason);
        run.phase = Phase::Finishing;
        return;
    }
    let Some(scene) = scenes::spawn(&args.scene, world) else {
        let mut run = world.resource_mut::<Run>();
        run.failures
            .push(format!("there is no scene called {}", args.scene));
        run.phase = Phase::Finishing;
        return;
    };
    if args.light_off {
        let lights: Vec<Entity> = world
            .query_filtered::<Entity, With<ToggleLight>>()
            .iter(world)
            .collect();
        for light in lights {
            world.despawn(light);
        }
    }

    let mut camera = world.spawn((
        TestCamera,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: scene.fov,
            ..default()
        }),
        (scene.orbit)(args.pose),
        Msaa::Off,
        Hdr,
        DeferredPrepass,
        DepthPrepass,
        MotionVectorPrepass,
    ));
    match args.aa {
        Aa::Taa => {
            camera.insert(TemporalAntiAliasing::default());
        }
        Aa::Tsr => {
            camera.insert(Tsr {
                render_scale: args.scale,
                ..default()
            });
        }
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        Aa::Dlss => {
            // DLSS Super Resolution copies its output into the main texture.
            camera.insert((
                Dlss::<DlssSuperResolutionFeature> {
                    perf_quality_mode: dlss_mode(args.scale).expect("checked in `unsupported`"),
                    ..default()
                },
                CameraMainTextureUsages::default().with(TextureUsages::COPY_DST),
            ));
        }
        #[cfg(not(all(feature = "dlss", not(feature = "force_disable_dlss"))))]
        Aa::Dlss => unreachable!("refused in `unsupported`"),
    }
    world.insert_resource(ActiveScene(scene));
}

/// Captures this frame into `<name>.png`. The file is written as soon as the capture arrives, so that a long run
/// of still frames is not held in memory.
fn request_capture(
    commands: &mut Commands,
    run: &mut Run,
    args: &Args,
    frame_count: u32,
    name: String,
) {
    run.pending_captures += 1;
    let (orbit_frame, frame) = (run.orbit_frame, run.frame);
    let out = args.out.clone();
    commands.spawn(Screenshot::primary_window()).observe(
        move |captured: On<ScreenshotCaptured>, mut run: ResMut<Run>| {
            run.pending_captures -= 1;
            let file = format!("{name}.png");
            let saved = captured
                .image
                .clone()
                .try_into_dynamic()
                .map_err(|error| error.to_string())
                .and_then(|image| {
                    image
                        .to_rgb8()
                        .save(out.join(&file))
                        .map_err(|error| error.to_string())
                });
            if let Err(error) = saved {
                run.failures.push(format!("{file}: {error}"));
            }
            run.captures.push(json!({
                "name": name,
                "file": file,
                "orbit_frame": orbit_frame,
                "frame": frame,
                "frame_count": frame_count,
            }));
        },
    );
}

fn drive(
    mut commands: Commands,
    args: Res<Args>,
    scene: Option<Res<ActiveScene>>,
    mut run: ResMut<Run>,
    mut camera: Query<&mut Transform, With<TestCamera>>,
    toggle_lights: Query<Entity, With<ToggleLight>>,
    asset_server: Res<AssetServer>,
    pending_pipelines: Res<PendingPipelines>,
    diagnostics: Res<DiagnosticsStore>,
    frame_count: Res<FrameCount>,
) {
    let (Some(scene), Ok(mut transform)) = (scene, camera.single_mut()) else {
        return;
    };
    if run.since.elapsed() > RUN_LIMIT && run.phase != Phase::Done {
        run.failures
            .push(format!("the run took over {} s", RUN_LIMIT.as_secs()));
        run.phase = Phase::Finishing;
    }
    match run.phase {
        Phase::Loading => {
            let loaded = scene
                .0
                .assets
                .iter()
                .all(|asset| asset_server.is_loaded_with_dependencies(asset.id()));
            let calm = loaded && pending_pipelines.0.load(Ordering::Relaxed) == 0;
            run.calm_frames = if calm { run.calm_frames + 1 } else { 0 };
            // Jitter follows the frame count, so a run aligned on it sees the same jitter phases as another.
            if run.calm_frames < CALM_FRAMES
                || frame_count.0 % args.align != args.phase % args.align
            {
                return;
            }
            run.phase = Phase::Running;
        }
        Phase::Running => run.frame += 1,
        Phase::Finishing | Phase::Done => return,
    }

    let frame = run.frame;
    let measured = args.warmup..args.warmup + args.frames;
    let now = Instant::now();
    if let Some(last) = run.last_tick.replace(now)
        && frame > 0
        && measured.contains(&(frame - 1))
    {
        run.frame_ms.push((now - last).as_secs_f64() * 1000.0);
    }
    if measured.contains(&frame) {
        for diagnostic in diagnostics.iter() {
            let path = diagnostic.path().as_str();
            if let (Some(pass), Some(ms)) = (
                path.strip_prefix("render/")
                    .and_then(|path| path.strip_suffix("/elapsed_gpu")),
                diagnostic.value(),
            ) {
                run.pass_ms.entry(pass.to_owned()).or_default().push(ms);
            }
        }
    }

    if args.orbit {
        run.orbit_frame = frame;
        *transform = (scene.0.orbit)(frame);
        let step = (args.frames / ORBIT_CAPTURES).max(1);
        if measured.contains(&frame)
            && (frame - args.warmup).is_multiple_of(step)
            && (frame - args.warmup) / step < ORBIT_CAPTURES
        {
            request_capture(
                &mut commands,
                &mut run,
                &args,
                frame_count.0,
                format!("orbit-{frame:05}"),
            );
        }
    }
    if let Some(at) = args.light_toggle_at {
        if frame == args.warmup + at {
            for light in &toggle_lights {
                commands.entity(light).despawn();
            }
        }
        if frame == args.warmup + at + TOGGLE_CAPTURE_DELAY {
            request_capture(
                &mut commands,
                &mut run,
                &args,
                frame_count.0,
                "toggle".into(),
            );
        }
    }
    if let Some(still) = frame
        .checked_sub(measured.end)
        .filter(|still| *still < args.still_frames)
    {
        let name = match (args.still_frames, still) {
            (2, 0) => "a".into(),
            (2, _) => "b".into(),
            _ => format!("still-{still:03}"),
        };
        request_capture(&mut commands, &mut run, &args, frame_count.0, name);
        if still + 1 == args.still_frames {
            run.phase = Phase::Finishing;
        }
    }
}

fn finish(
    args: Res<Args>,
    mut run: ResMut<Run>,
    adapter: Res<RenderAdapterInfo>,
    render_size: Res<RenderSize>,
    mut exit: MessageWriter<AppExit>,
) {
    if run.phase != Phase::Finishing || run.pending_captures > 0 && run.since.elapsed() < RUN_LIMIT
    {
        return;
    }
    run.phase = Phase::Done;
    let mut failures = std::mem::take(&mut run.failures);
    let screenshots = std::mem::take(&mut run.captures);
    // gfx-check's `flicker` reads the still frames from the report; a.png and b.png stay for every other reader.
    if args.still_frames > 2 {
        for (link, target) in [("a.png", "still-000.png"), ("b.png", "still-001.png")] {
            let link = args.out.join(link);
            let _ = std::fs::remove_file(&link);
            if let Err(error) = std::fs::hard_link(args.out.join(target), &link) {
                failures.push(format!("{}: {error}", link.display()));
            }
        }
    }
    let render_size = render_size.0.load(Ordering::Relaxed);
    let render_resolution = if render_size != 0 {
        [(render_size >> 32) as u32, render_size as u32]
    } else {
        [args.res.x, args.res.y]
    };

    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values.get(values.len() / 2).copied()
    };
    let frame_ms = median(&mut run.frame_ms).map(|median| json!({ "median": median }));
    let gpu_pass_ms: serde_json::Map<String, serde_json::Value> = std::mem::take(&mut run.pass_ms)
        .into_iter()
        .filter_map(|(pass, mut values)| Some((pass, json!(median(&mut values)?))))
        .collect();
    // A pipeline that fails to compile leaves the frame unscaled without stopping the run.
    if args.aa == Aa::Tsr && !gpu_pass_ms.contains_key("tsr") {
        failures.push("TSR never ran: no GPU time was recorded for it".into());
    }
    if args.aa != Aa::Taa && args.scale < 1.0 && render_size == 0 {
        failures
            .push("the frame was never upscaled: no main-pass resolution override was seen".into());
    }
    let report = json!({
        "schema": 1,
        "args": args.command_line,
        "scene": args.scene,
        "mode": "raster",
        "aa": format!("{:?}", args.aa).to_lowercase(),
        "render_scale": if args.aa == Aa::Taa { 1.0 } else { args.scale },
        "render_resolution": render_resolution,
        "align": args.align,
        "phase": args.phase,
        "still_frames": args.still_frames,
        "adapter": { "name": adapter.name, "driver_info": adapter.driver_info },
        "resolution": [args.res.x, args.res.y],
        "warmup": args.warmup,
        "frames": args.frames,
        "camera": { "kind": if args.orbit { "orbit" } else { "static" }, "pose": args.pose },
        "light_toggle_at": args.light_toggle_at,
        "frame_ms": frame_ms,
        "gpu_pass_ms": gpu_pass_ms,
        "screenshots": screenshots,
        "failures": failures,
    });
    let text = serde_json::to_string_pretty(&report).expect("a JSON value serializes");
    if let Err(error) = std::fs::write(args.out.join("report.json"), text) {
        failures.push(format!("report.json: {error}"));
    }
    for failure in &failures {
        error!("{failure}");
    }
    exit.write(if failures.is_empty() {
        AppExit::Success
    } else {
        AppExit::error()
    });
}
