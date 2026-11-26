//! `B`: baking indirect light for the open scene in a child process (this same binary with `--bake-gi`), and
//! loading the newest bake of a scene when it opens or when a bake finishes.
//!
//! Bakes go to `<bakes>/<scene>/bake-<unix seconds>/` (`<bakes>` is `~/.local/share/bevy-graficos/viewer-bakes`
//! unless `--bakes-dir` says otherwise). Each bake has its own folder so the asset server never hands back the
//! previous bake from its cache; the older folders of a scene are deleted once a newer bake succeeds.

use crate::{
    stage::{bake_regions, SceneEntity, Stage},
    Flags,
};
use bevy::{
    graphics::baked::{BakedGi, BakedGiManifest},
    prelude::*,
};
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::Instant,
};

/// The asset source the viewer reads bakes from, rooted at the bakes folder.
pub const SOURCE: &str = "viewer-bakes";

/// Where bakes live and the bake that is running.
#[derive(Resource)]
pub struct Bakes {
    /// `<bakes>`.
    pub root: PathBuf,
    /// The child process baking, if any.
    child: Option<Running>,
    /// The last thing to tell the person about bakes, and since when.
    pub message: Option<(String, Instant)>,
    /// The folder of the bake loaded in the open scene.
    pub loaded: Option<PathBuf>,
    /// How many bakes finished, and how many of them succeeded (the self-test waits on these).
    pub finished: u32,
    pub succeeded: u32,
}

struct Running {
    process: Child,
    started: Instant,
    scene: String,
    generation: u32,
    dir: PathBuf,
}

impl Bakes {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            child: None,
            message: None,
            loaded: None,
            finished: 0,
            succeeded: 0,
        }
    }

    /// `Some(seconds)` while a bake runs.
    pub fn running_for(&self) -> Option<u64> {
        self.child
            .as_ref()
            .map(|running| running.started.elapsed().as_secs())
    }

    /// Shows `message` in the overlay for a while, and logs it.
    pub fn say(&mut self, message: impl Into<String>) {
        let message = message.into();
        info!("gfx_viewer: {message}");
        self.message = Some((message, Instant::now()));
    }

    /// The newest finished bake of `scene`.
    fn latest(&self, scene: &str) -> Option<PathBuf> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(self.root.join(scene))
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("bake-"))
                    && path.join("manifest.gi.ron").is_file()
            })
            .collect();
        dirs.sort();
        dirs.pop()
    }
}

/// Loads the newest bake of the open scene, replacing any loaded one. Called when a scene opens and when a bake
/// of it finishes.
pub fn load_latest(world: &mut World) {
    let old: Vec<Entity> = world
        .query_filtered::<Entity, With<BakedGi>>()
        .iter(world)
        .collect();
    for entity in old {
        world.entity_mut(entity).despawn();
    }
    let scene = world.resource::<Stage>().name.clone();
    let latest = world.resource::<Bakes>().latest(&scene);
    world.resource_mut::<Bakes>().loaded = latest.clone();
    let Some(dir) = latest else {
        return;
    };
    let folder = dir
        .file_name()
        .expect("a bake folder has a name")
        .to_string_lossy();
    let manifest: Handle<BakedGiManifest> = world
        .resource::<AssetServer>()
        .load(format!("{SOURCE}://{scene}/{folder}/manifest.gi.ron"));
    world.spawn((SceneEntity, Name::new("baked GI"), BakedGi(manifest)));
    info!("gfx_viewer: loading the bake {}", dir.display());
}

/// Starts baking the open scene, unless a bake is running or the scene has nothing to bake.
pub fn start_bake(world: &mut World) {
    if world.resource::<Bakes>().child.is_some() {
        world
            .resource_mut::<Bakes>()
            .say("a bake is already running");
        return;
    }
    let stage = world.resource::<Stage>();
    if stage.is_gltf() && !stage.placed {
        world
            .resource_mut::<Bakes>()
            .say("the scene is still loading: bake when it is there");
        return;
    }
    let (scene, generation, gltf_file, default_sun) = (
        stage.name.clone(),
        stage.generation,
        stage
            .is_gltf()
            .then(|| world.resource::<crate::catalog::Catalog>().entries[stage.index].clone()),
        stage.default_sun,
    );
    let regions = bake_regions(world);
    if regions.is_empty() {
        world.resource_mut::<Bakes>().say(format!(
            "{scene} has no bake region: the testbed bakes only its room scenes"
        ));
        return;
    }

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let dir = world
        .resource::<Bakes>()
        .root
        .join(&scene)
        .join(format!("bake-{stamp}"));
    let spawned = (|| -> std::io::Result<Child> {
        std::fs::create_dir_all(&dir)?;
        let log = std::fs::File::create(dir.join("bake.log"))?;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("--bake-gi").arg(&dir);
        match &gltf_file {
            Some(entry) => {
                let crate::catalog::Source::Gltf(file) = &entry.source else {
                    unreachable!("a glTF stage comes from a glTF entry")
                };
                command.arg("--bake-gltf").arg(file);
                let (transform, region) = regions[0];
                let (t, s) = (transform.translation, transform.scale);
                command.arg("--bake-region").arg(format!(
                    "{},{},{},{},{},{},{}",
                    t.x, t.y, t.z, s.x, s.y, s.z, region.spacing
                ));
                if default_sun {
                    command.arg("--bake-default-sun");
                }
            }
            None => {
                command.arg("--scene").arg(&scene);
            }
        }
        command
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()
    })();
    let mut bakes = world.resource_mut::<Bakes>();
    match spawned {
        Ok(process) => {
            bakes.child = Some(Running {
                process,
                started: Instant::now(),
                scene: scene.clone(),
                generation,
                dir: dir.clone(),
            });
            bakes.say(format!("baking {scene} into {}", dir.display()));
        }
        Err(error) => bakes.say(format!("cannot start the bake: {error}")),
    }
}

/// Watches the running bake; when it ends, loads it (if its scene is still open) or says why it failed.
pub fn poll_bake(world: &mut World) {
    let finished = {
        let mut bakes = world.resource_mut::<Bakes>();
        let Some(running) = bakes.child.as_mut() else {
            return;
        };
        match running.process.try_wait() {
            Ok(None) => return,
            Ok(Some(status)) => Some(status),
            Err(error) => {
                error!("gfx_viewer: lost the bake process: {error}");
                None
            }
        }
    };
    let running = world
        .resource_mut::<Bakes>()
        .child
        .take()
        .expect("checked above");
    let seconds = running.started.elapsed().as_secs();
    let ok = finished.is_some_and(|status| status.success())
        && running.dir.join("manifest.gi.ron").is_file();
    let generation = world.resource::<Stage>().generation;
    let mut bakes = world.resource_mut::<Bakes>();
    bakes.finished += 1;
    if !ok {
        let reason = last_error_line(&running.dir.join("bake.log"));
        let code = finished
            .and_then(|status| status.code())
            .map_or_else(|| "killed".into(), |code| format!("exit {code}"));
        bakes.say(format!(
            "the bake of {} failed ({code}) after {seconds} s: {reason} (log: {})",
            running.scene,
            running.dir.join("bake.log").display()
        ));
        return;
    }
    bakes.succeeded += 1;
    bakes.say(format!("baked {} in {seconds} s", running.scene));
    // Older bakes of this scene are superseded.
    if let Ok(read) = std::fs::read_dir(running.dir.parent().expect("a bake folder has a parent")) {
        for entry in read.flatten() {
            let path = entry.path();
            if path != running.dir
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("bake-"))
            {
                let _ = std::fs::remove_dir_all(path);
            }
        }
    }
    if running.generation == generation {
        load_latest(world);
    }
}

/// The last line of a bake's log that says what went wrong.
fn last_error_line(log: &Path) -> String {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    text.lines()
        .rev()
        .find(|line| {
            let lower = line.to_lowercase();
            lower.contains("error") || lower.contains("refus") || lower.contains("panic")
        })
        .or_else(|| text.lines().last())
        .unwrap_or("no log")
        .trim()
        .to_owned()
}

// ---------------------------------------------------------------------------------------------------------------
// The child process

/// With `--bake-gi`: bakes and returns the exit code; `None` when this process is not a bake.
pub fn child(flags: &mut Flags) -> Option<AppExit> {
    if !flags.args.iter().any(|arg| arg == "--bake-gi") {
        return None;
    }
    let gltf = flags.value("--bake-gltf").map(PathBuf::from);
    let region = flags.value("--bake-region");
    let default_sun = flags.switch("--bake-default-sun");
    let scene = flags.value("--scene");
    let exit = match gltf {
        Some(file) => {
            let dir = flags.value("--bake-gi").map(PathBuf::from);
            let region = region.as_deref().and_then(parse_region);
            match (dir, region) {
                (Some(dir), Some(region)) if flags.args.is_empty() => {
                    gltf_bake::run(file, dir, region, default_sun)
                }
                _ => {
                    eprintln!(
                        "gfx_viewer: a glTF bake takes --bake-gi DIR --bake-gltf FILE --bake-region \
                         TX,TY,TZ,SX,SY,SZ,SPACING [--bake-default-sun], not {:?}",
                        flags.args
                    );
                    AppExit::from_code(2)
                }
            }
        }
        None => {
            let baked = crate::baked::Baked::parse(flags);
            if !flags.errors.is_empty() || !flags.args.is_empty() {
                eprintln!("gfx_viewer: {:?} {:?}", flags.errors, flags.args);
                return Some(AppExit::from_code(2));
            }
            let args = crate::Args::for_scene(scene.as_deref().unwrap_or("interior"));
            baked.bake(&args).unwrap_or(AppExit::from_code(2))
        }
    };
    Some(exit)
}

fn parse_region(text: &str) -> Option<(Transform, f32)> {
    let numbers: Vec<f32> = text
        .split(',')
        .map(|n| n.trim().parse().ok())
        .collect::<Option<_>>()?;
    let [tx, ty, tz, sx, sy, sz, spacing] = numbers[..] else {
        return None;
    };
    Some((
        Transform::from_xyz(tx, ty, tz).with_scale(Vec3::new(sx, sy, sz)),
        spacing,
    ))
}

#[cfg(feature = "bevy_baked_gi_bake")]
mod gltf_bake {
    use crate::{
        catalog::{fs_asset_path, fs_root},
        stage::{default_sun, default_sun_transform},
    };
    use bevy::{
        app::ScheduleRunnerPlugin,
        asset::io::AssetSourceBuilder,
        baked_gi::{
            bake::{GiBakePlugin, GiBakeSettings},
            GiBakeRegion,
        },
        prelude::*,
        render::RenderPlugin,
        window::ExitCondition,
        winit::WinitPlugin,
        world_serialization::{WorldAssetRoot, WorldInstanceReady},
    };
    use std::{path::PathBuf, time::Duration};

    /// Stands in the level until the glTF instance is spawned: a mesh whose assets never load, so the baker
    /// keeps waiting instead of finding an empty level settled and refusing it.
    #[derive(Component)]
    struct Placeholder;

    /// Bakes the glTF `file` under one region, with the default sun when the viewer gave the scene one, as
    /// `gfx_testbed`'s baker sets its app up.
    pub fn run(file: PathBuf, out: PathBuf, region: (Transform, f32), sun: bool) -> AppExit {
        println!("baking {} to {}", file.display(), out.display());
        let settings = GiBakeSettings {
            output: out,
            ..default()
        };
        let mut app = App::new();
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
            crate::DLSS_PROJECT_ID,
        ));
        app.register_asset_source("fs", AssetSourceBuilder::platform_default(&fs_root(), None));
        app.add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .disable::<WinitPlugin>(),
            GiBakePlugin { settings },
            ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        ))
        .add_systems(
            Startup,
            move |mut commands: Commands, assets: Res<AssetServer>| {
                commands.spawn((
                    Placeholder,
                    Mesh3d(Handle::default()),
                    MeshMaterial3d::<StandardMaterial>(Handle::default()),
                ));
                commands
                    .spawn(WorldAssetRoot(assets.load(
                        GltfAssetLabel::Scene(0).from_asset(fs_asset_path(&file)),
                    )))
                    .observe(
                        |_: On<WorldInstanceReady>,
                         mut commands: Commands,
                         placeholders: Query<Entity, With<Placeholder>>| {
                            for entity in &placeholders {
                                commands.entity(entity).despawn();
                            }
                        },
                    );
                let (transform, spacing) = region;
                commands.spawn((
                    Name::new("glTF bounds"),
                    GiBakeRegion {
                        spacing,
                        ..default()
                    },
                    transform,
                ));
                if sun {
                    commands.spawn((
                        Name::new("default sun"),
                        default_sun(),
                        default_sun_transform(),
                    ));
                }
            },
        )
        .run()
    }
}

#[cfg(not(feature = "bevy_baked_gi_bake"))]
mod gltf_bake {
    use bevy::prelude::*;
    use std::path::PathBuf;

    pub fn run(_: PathBuf, _: PathBuf, _: (Transform, f32), _: bool) -> AppExit {
        eprintln!("gfx_viewer: baking needs a binary built with the bevy_baked_gi_bake feature");
        AppExit::from_code(2)
    }
}
