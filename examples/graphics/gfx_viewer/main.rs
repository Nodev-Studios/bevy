//! An interactive viewer of the graphics ladder: open a scene, fly around it, and change every field of
//! `GraphicsSettings` live from a panel, with what resolved (rung, upscaler, denoiser, render scale, and why
//! anything differs from what was asked) and the GPU time of each pass always on screen.
//!
//! Scenes: the testbed's `interior`, `many-lights`, `contract`, `rooms` and `two-rooms-door` (reused from
//! `gfx_testbed/` through `#[path]`), Bevy's `FlightHelmet`, and every glTF file under `--scenes-dir` (default
//! `~/.local/share/bevy-graficos/scenes`, which `tools/fetch-scenes.sh` fills). A glTF file without lights gets
//! a default sun and an atmosphere for a sky.
//!
//! Start it with `tools/gfx-viewer.sh`, which builds it with the features the ladder ships with and points DLSS
//! at its libraries. Pass `--help` for the flags and `H` in the window for the keys. `--self-test DIR` checks it
//! without a person (see `selftest.rs`); on this project's machine it runs through `tools/gfx-headless.sh`.

#[path = "../gfx_testbed/scenes.rs"]
mod scenes;

// The testbed's baked-light scenes and its baker; most of the file serves the testbed's measurements and is
// unused here.
#[allow(dead_code, unused_imports)]
#[path = "../gfx_testbed/baked.rs"]
mod baked;

// The testbed's denoiser-contract scene; the rest of the file (its checks) is unused here.
#[allow(dead_code, unused_imports)]
#[path = "../gfx_testbed/contract.rs"]
mod contract;

mod actions;
mod bake;
mod catalog;
mod selftest;
mod shim;
mod stage;
mod ui;

// Named one by one: a glob would clash with the prelude's `Scene`.
pub use shim::{
    drive, finish, ActiveScene, Args, Flags, Mode, Phase, Report, Run, Scene, TestbedCamera,
    ToggleLight, ORBIT_CAPTURES,
};

use bevy::{
    asset::io::AssetSourceBuilder,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    graphics::{GraphicsCamera, GraphicsPlugin, GraphicsPreset, GraphicsSettings},
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin, render_resource::PipelineCache, Render, RenderApp,
        RenderSystems,
    },
    solari::prelude::SolariPlugins,
    window::{PresentMode, WindowResolution},
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

/// The viewer's own DLSS project id; a game generates its own.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
pub const DLSS_PROJECT_ID: bevy::asset::uuid::Uuid =
    bevy::asset::uuid::uuid!("3c1f6f9e-2a53-4f0c-9a52-7d6a0c1b5e47");

const USAGE: &str = "\
gfx_viewer: fly around a scene and change the graphics settings live.

  --scene NAME|FILE.gltf|FILE.glb  the scene to open first (default interior)
  --scenes-dir DIR               where to look for glTF files (default ~/.local/share/bevy-graficos/scenes)
  --preset auto|raster|rt-lite|rt-full
                                 the preset to start with (default auto)
  --res WxH                      window size (default 1600x900)
  --bakes-dir DIR                where B writes bakes (default ~/.local/share/bevy-graficos/viewer-bakes)
";

/// The viewer's camera.
#[derive(Component)]
pub struct ViewerCamera;

/// The other settings set of the A/B comparison (`C` swaps it with the settings in use).
#[derive(Resource)]
pub struct AbCompare {
    /// The set not in use.
    pub other: GraphicsSettings,
    /// The settings in use are set B.
    pub showing_b: bool,
}

/// How many pipelines the render world is still compiling; `u32::MAX` until it has rendered once.
#[derive(Resource, Clone)]
pub struct PendingPipelines(pub Arc<AtomicU32>);

fn count_pending_pipelines(cache: Res<PipelineCache>, pending: Res<PendingPipelines>) {
    pending
        .0
        .store(cache.waiting_pipelines().count() as u32, Ordering::Relaxed);
}

fn main() -> AppExit {
    let mut flags = Flags::new(std::env::args().skip(1).collect());
    if flags.switch("--help") {
        println!("{USAGE}{}", selftest::HELP);
        return AppExit::Success;
    }
    if let Some(exit) = bake::child(&mut flags) {
        return exit;
    }
    let scene = flags.value("--scene");
    let scenes_dir = flags
        .value("--scenes-dir")
        .map(PathBuf::from)
        .or_else(catalog::default_scenes_dir);
    let preset = flags.choice(
        "--preset",
        &[
            ("auto", GraphicsPreset::Auto),
            ("raster", GraphicsPreset::Raster),
            ("rt-lite", GraphicsPreset::RtLite),
            ("rt-full", GraphicsPreset::RtFull),
        ],
    );
    let res = flags.value("--res").map(|value| {
        let size = value
            .split_once('x')
            .and_then(|(w, h)| Some(UVec2::new(w.parse().ok()?, h.parse().ok()?)))
            .filter(|size| size.x > 0 && size.y > 0);
        if size.is_none() {
            flags.error(format!("--res takes WxH, not {value}"));
        }
        size.unwrap_or(UVec2::ONE)
    });
    let bakes_dir = flags.value("--bakes-dir").map(PathBuf::from);
    let mut self_test = selftest::SelfTest::parse(&mut flags);
    for unknown in std::mem::take(&mut flags.args) {
        flags.error(format!("unknown argument {unknown}"));
    }

    let extra = scene
        .as_ref()
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    let catalog = catalog::Catalog::build(scenes_dir.as_deref(), extra.as_deref());
    let start = scene.as_deref().map_or(Some(0), |name| catalog.find(name));
    if start.is_none() {
        let names: Vec<&str> = catalog.entries.iter().map(|e| e.name.as_str()).collect();
        flags.error(format!(
            "there is no scene called {}: the list is {}",
            scene.as_deref().unwrap_or_default(),
            names.join(", ")
        ));
    }
    let bakes_dir = match (&self_test, bakes_dir) {
        (Some(test), _) => test.bakes_dir(),
        (None, Some(dir)) => dir,
        (None, None) => catalog::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("viewer-bakes"),
    };
    let bakes_dir = std::path::absolute(&bakes_dir).unwrap_or(bakes_dir);
    if let Err(error) = std::fs::create_dir_all(&bakes_dir) {
        flags.error(format!("cannot create {}: {error}", bakes_dir.display()));
    }
    if let Some(test) = &self_test
        && let Err(error) = std::fs::create_dir_all(test.dir())
    {
        flags.error(format!("cannot create {}: {error}", test.dir().display()));
    }
    if !flags.errors.is_empty() {
        for error in &flags.errors {
            eprintln!("gfx_viewer: {error}");
        }
        eprintln!("\n{USAGE}{}", selftest::HELP);
        return AppExit::from_code(2);
    }
    if let Some(test) = &mut self_test {
        test.plan(&catalog);
    }

    let mut app = App::new();
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(DLSS_PROJECT_ID));
    app.register_asset_source(
        "fs",
        AssetSourceBuilder::platform_default(&catalog::fs_root(), None),
    )
    .register_asset_source(
        bake::SOURCE,
        AssetSourceBuilder::platform_default(&bakes_dir.to_string_lossy(), None),
    );

    let res = res.unwrap_or(UVec2::new(1600, 900));
    let testing = self_test.is_some();
    let mut resolution = WindowResolution::new(res.x, res.y);
    if testing {
        // The self-test's frames are exactly the size asked for.
        resolution = resolution.with_scale_factor_override(1.0);
    }
    let pending = PendingPipelines(Arc::new(AtomicU32::new(u32::MAX)));
    let settings = GraphicsSettings {
        preset: preset.unwrap_or_default(),
        ..default()
    };
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "gfx_viewer".into(),
                resolution,
                decorations: !testing,
                resizable: !testing,
                // Frame times read as the GPU's, not as the display's refresh interval.
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }),
        SolariPlugins,
        RenderDiagnosticsPlugin,
        FreeCameraPlugin,
    ))
    .insert_resource(settings.clone())
    .add_plugins(GraphicsPlugin)
    .insert_resource(AbCompare {
        other: settings,
        showing_b: false,
    })
    .insert_resource(ClearColor(Color::BLACK))
    .insert_resource(GlobalAmbientLight::NONE)
    .insert_resource(Args::for_scene("interior"))
    .insert_resource(stage::Stage {
        requested: start,
        ..default()
    })
    .insert_resource(catalog)
    .insert_resource(bake::Bakes::new(bakes_dir))
    .insert_resource(pending.clone())
    .init_resource::<actions::Lamps>()
    .init_resource::<ui::FrameTimes>()
    .add_message::<actions::Action>()
    .add_systems(Startup, (spawn_camera, ui::spawn_ui))
    .add_systems(
        Update,
        (
            actions::read_keys,
            actions::run_actions,
            open_scene,
            stage::place_gltf_scene,
            stage::watch_loading,
            bake::poll_bake,
            ui::refresh_panel,
            ui::record_frame_time,
            ui::update_overlay,
        )
            .chain(),
    );
    if let Some(test) = self_test {
        app.insert_resource(test).add_systems(
            Update,
            // The actions it writes are carried out the next frame, as a key's would be.
            selftest::drive.after(ui::update_overlay),
        );
    }
    app.sub_app_mut(RenderApp)
        .insert_resource(pending)
        .add_systems(
            Render,
            count_pending_pipelines.in_set(RenderSystems::Cleanup),
        );
    app.run()
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        ViewerCamera,
        Camera3d::default(),
        GraphicsCamera,
        Msaa::Off,
        Transform::from_xyz(0.0, 1.6, 3.0),
        FreeCamera::default(),
    ));
}

/// Opens the scene asked for, then the newest bake of it.
fn open_scene(world: &mut World) {
    if world.resource::<stage::Stage>().requested.is_none() {
        return;
    }
    stage::open_requested_scene(world);
    bake::load_latest(world);
}
