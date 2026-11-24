//! TSR's test app: renders one of `gfx_testbed`'s scenes on the deferred rasterizer, anti-aliased by TAA or
//! upscaled by TSR, and writes what `tools/gfx-check` reads: `a.png` and `b.png` (two consecutive frames after the
//! measured ones), the orbit captures, `toggle.png` and `report.json`.
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
    camera::Hdr,
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    diagnostic::DiagnosticsStore,
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
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const USAGE: &str = "\
gfx_tsr: renders a gfx_testbed scene on the rasterizer with TAA or TSR, and writes report.json and screenshots.

  --scene interior|many-lights   the scene (default interior)
  --aa taa|tsr                   TAA at the output's resolution, or TSR (default taa)
  --scale F                      tsr: the render resolution as a share of the output's (default 0.5)
  --res WxH                      window size in pixels (default 1280x720)
  --warmup N                     frames after the scene is ready and before measuring (default 60)
  --frames N                     frames measured (default 120)
  --camera static|orbit          a still camera, or one circling the scene one step per frame (default static)
  --pose K                       static: stand where the orbit camera is at its frame K (default 0)
  --light-toggle-at N            switch one fixture off at measured frame N, capture toggle.png 5 frames later
  --light-off                    spawn the scene without that fixture
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

#[derive(Resource, Clone)]
struct Args {
    command_line: Vec<String>,
    scene: String,
    tsr: bool,
    scale: f32,
    res: UVec2,
    warmup: u32,
    frames: u32,
    orbit: bool,
    pose: u32,
    light_toggle_at: Option<u32>,
    light_off: bool,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let command_line: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        command_line: command_line.clone(),
        scene: "interior".into(),
        tsr: false,
        scale: 0.5,
        res: UVec2::new(1280, 720),
        warmup: 60,
        frames: 120,
        orbit: false,
        pose: 0,
        light_toggle_at: None,
        light_off: false,
        out: "gfx-tsr-out".into(),
    };
    let mut words = command_line.into_iter();
    while let Some(flag) = words.next() {
        if flag == "--light-off" {
            args.light_off = true;
            continue;
        }
        let value = words.next().ok_or(format!("{flag} needs a value"))?;
        let number = |value: &str| value.parse::<u32>().map_err(|_| format!("{flag}: cannot read {value}"));
        match flag.as_str() {
            "--scene" => args.scene = value,
            "--aa" => {
                args.tsr = match value.as_str() {
                    "taa" => false,
                    "tsr" => true,
                    _ => return Err(format!("--aa takes taa|tsr, not {value}")),
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
                let (w, h) = value.split_once('x').ok_or(format!("--res takes WxH, not {value}"))?;
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
    captures: Vec<(String, u32, Image)>,
    failures: Vec<String>,
}

#[derive(Resource)]
struct ActiveScene(Scene);

#[derive(Component)]
struct TestCamera;

/// Pipelines the render world is still compiling; `u32::MAX` until it has rendered once.
#[derive(Resource, Clone)]
struct PendingPipelines(Arc<AtomicU32>);

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
    let mut app = App::new();
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
    .add_systems(Startup, setup)
    .add_systems(Update, (drive, finish).chain());

    app.sub_app_mut(RenderApp)
        .insert_resource(pending_pipelines)
        .add_systems(
            Render,
            count_pending_pipelines.in_set(RenderSystems::Cleanup),
        );
    app.run()
}

fn count_pending_pipelines(cache: Res<PipelineCache>, pending: Res<PendingPipelines>) {
    pending
        .0
        .store(cache.waiting_pipelines().count() as u32, Ordering::Relaxed);
}

fn setup(world: &mut World) {
    let args = world.resource::<Args>().clone();
    if args.tsr && !world.contains_resource::<TsrSupported>() {
        let mut run = world.resource_mut::<Run>();
        run.failures.push("TSR is not supported on this device".into());
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
    if args.tsr {
        camera.insert(Tsr {
            render_scale: args.scale,
            ..default()
        });
    } else {
        camera.insert(TemporalAntiAliasing::default());
    }
    world.insert_resource(ActiveScene(scene));
}

fn request_capture(commands: &mut Commands, run: &mut Run, name: String) {
    run.pending_captures += 1;
    let orbit_frame = run.orbit_frame;
    commands.spawn(Screenshot::primary_window()).observe(
        move |captured: On<ScreenshotCaptured>, mut run: ResMut<Run>| {
            run.pending_captures -= 1;
            run.captures
                .push((name.clone(), orbit_frame, captured.image.clone()));
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
            if run.calm_frames < CALM_FRAMES {
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
            request_capture(&mut commands, &mut run, format!("orbit-{frame:05}"));
        }
    }
    if let Some(at) = args.light_toggle_at {
        if frame == args.warmup + at {
            for light in &toggle_lights {
                commands.entity(light).despawn();
            }
        }
        if frame == args.warmup + at + TOGGLE_CAPTURE_DELAY {
            request_capture(&mut commands, &mut run, "toggle".into());
        }
    }
    if frame == measured.end {
        request_capture(&mut commands, &mut run, "a".into());
    }
    if frame == measured.end + 1 {
        request_capture(&mut commands, &mut run, "b".into());
        run.phase = Phase::Finishing;
    }
}

fn finish(
    args: Res<Args>,
    mut run: ResMut<Run>,
    adapter: Res<RenderAdapterInfo>,
    mut exit: MessageWriter<AppExit>,
) {
    if run.phase != Phase::Finishing || run.pending_captures > 0 && run.since.elapsed() < RUN_LIMIT
    {
        return;
    }
    run.phase = Phase::Done;
    let mut failures = std::mem::take(&mut run.failures);
    let mut screenshots = Vec::new();
    for (name, orbit_frame, image) in std::mem::take(&mut run.captures) {
        let file = format!("{name}.png");
        let saved = image
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
        screenshots.push(json!({ "name": name, "file": file, "orbit_frame": orbit_frame }));
    }

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
    if args.tsr && !gpu_pass_ms.contains_key("tsr") {
        failures.push("TSR never ran: no GPU time was recorded for it".into());
    }
    let report = json!({
        "schema": 1,
        "args": args.command_line,
        "scene": args.scene,
        "mode": "raster",
        "aa": if args.tsr { "tsr" } else { "taa" },
        "render_scale": if args.tsr { args.scale } else { 1.0 },
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
