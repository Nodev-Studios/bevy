//! What a key or a panel button does. Keys and buttons write an [`Action`]; [`run_actions`] carries it out, so
//! the self-test drives the same code a person does. The panels' continuous values ([`ViewControls`]) are
//! applied by [`apply_view_controls`].

use crate::{
    bake::{self, Bakes},
    catalog::{data_dir, Catalog},
    compare::Split,
    stage::{place_camera, DefaultSun, SceneEntity, Stage},
    ui::{rung_text, SunControl, UiState, ViewControls},
    AbCompare, ToggleLight, ViewerCamera,
};
use bevy::{
    app::AppExit,
    camera::Exposure,
    graphics::{baked::BakedGi, GraphicsSettings, ResolvedGraphics},
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use std::{collections::HashMap, path::PathBuf};

/// Something the person asked for.
#[derive(Message, Clone, Debug, PartialEq)]
pub enum Action {
    TogglePanel,
    ToggleHelp,
    ToggleLamps,
    Bake,
    LoadBake,
    UnloadBake,
    FlipAb,
    CopyToOther,
    Split(bool),
    ToggleSplit,
    RefreshSplit,
    Screenshot,
    OpenScene(usize),
    OpenFile(String),
    SaveBookmark,
    GoBookmark(usize),
    ResetCamera,
    Quit,
}

/// Whether the scene's lamps are on.
#[derive(Resource)]
pub struct Lamps {
    pub on: bool,
    generation: u32,
}

impl Default for Lamps {
    fn default() -> Self {
        Self {
            on: true,
            generation: 0,
        }
    }
}

/// Saved camera views per scene, kept in `<data>/viewer-bookmarks.json` (none in the self-test).
#[derive(Resource, Default)]
pub struct Bookmarks {
    views: HashMap<String, Vec<Transform>>,
    file: Option<PathBuf>,
}

impl Bookmarks {
    pub fn load(file: Option<PathBuf>) -> Self {
        let views = file
            .as_ref()
            .and_then(|file| std::fs::read_to_string(file).ok())
            .and_then(|text| {
                serde_json::from_str::<HashMap<String, Vec<([f32; 3], [f32; 4])>>>(&text).ok()
            })
            .map(|views| {
                views
                    .into_iter()
                    .map(|(scene, list)| {
                        let list = list
                            .into_iter()
                            .map(|(t, r)| {
                                Transform::from_translation(t.into())
                                    .with_rotation(Quat::from_array(r))
                            })
                            .collect();
                        (scene, list)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self { views, file }
    }

    pub fn for_scene(&self, scene: &str) -> &[Transform] {
        self.views.get(scene).map_or(&[], Vec::as_slice)
    }

    fn save(&mut self, scene: &str, view: Transform) {
        self.views.entry(scene.to_owned()).or_default().push(view);
        let Some(file) = &self.file else {
            return;
        };
        let out: HashMap<&String, Vec<([f32; 3], [f32; 4])>> = self
            .views
            .iter()
            .map(|(scene, list)| {
                (
                    scene,
                    list.iter()
                        .map(|t| (t.translation.to_array(), t.rotation.to_array()))
                        .collect(),
                )
            })
            .collect();
        if let Ok(text) = serde_json::to_string_pretty(&out)
            && let Err(error) = std::fs::write(file, text)
        {
            warn!("gfx_viewer: cannot save {}: {error}", file.display());
        }
    }
}

/// Turns key presses into actions, unless a text field has the keyboard.
pub fn read_keys(
    keys: Res<ButtonInput<KeyCode>>,
    ui: Res<UiState>,
    mut actions: MessageWriter<Action>,
) {
    if ui.keyboard {
        return;
    }
    for (key, action) in [
        (KeyCode::Tab, Action::TogglePanel),
        (KeyCode::KeyH, Action::ToggleHelp),
        (KeyCode::KeyL, Action::ToggleLamps),
        (KeyCode::KeyB, Action::Bake),
        (KeyCode::KeyC, Action::FlipAb),
        (KeyCode::KeyV, Action::ToggleSplit),
        (KeyCode::F12, Action::Screenshot),
        (KeyCode::Escape, Action::Quit),
    ] {
        if keys.just_pressed(key) {
            actions.write(action);
        }
    }
}

/// Carries out this frame's actions.
pub fn run_actions(world: &mut World) {
    let actions: Vec<Action> = world.resource_mut::<Messages<Action>>().drain().collect();
    // A new scene starts with its lamps on.
    let generation = world.resource::<Stage>().generation;
    if world.resource::<Lamps>().generation != generation {
        *world.resource_mut::<Lamps>() = Lamps {
            on: true,
            generation,
        };
    }
    for action in actions {
        match action {
            Action::TogglePanel => {
                let mut ui = world.resource_mut::<UiState>();
                ui.visible = !ui.visible;
            }
            Action::ToggleHelp => {
                let mut ui = world.resource_mut::<UiState>();
                ui.windows.help = !ui.windows.help;
                ui.visible = true;
            }
            Action::ToggleLamps => toggle_lamps(world),
            Action::Bake => bake::start_bake(world),
            Action::LoadBake => bake::load_latest(world),
            Action::UnloadBake => unload_bake(world),
            Action::FlipAb => {
                if world.resource::<Split>().on {
                    world
                        .resource_mut::<Bakes>()
                        .say("the split view shows both sets: switch it off to flip");
                } else {
                    flip_ab(world);
                }
            }
            Action::CopyToOther => {
                let current = world.resource::<GraphicsSettings>().clone();
                world.resource_mut::<AbCompare>().other = current;
            }
            Action::Split(on) => world.resource_mut::<Split>().on = on,
            Action::ToggleSplit => {
                let mut split = world.resource_mut::<Split>();
                split.on = !split.on;
            }
            Action::RefreshSplit => world.resource_mut::<Split>().requested = true,
            Action::Screenshot => screenshot(world),
            Action::OpenScene(index) => open_scene(world, index),
            Action::OpenFile(path) => {
                let index = world
                    .resource_mut::<Catalog>()
                    .add_file(std::path::Path::new(&path));
                match index {
                    Some(index) => open_scene(world, index),
                    None => world
                        .resource_mut::<Bakes>()
                        .say(format!("there is no file {path}")),
                }
            }
            Action::SaveBookmark => {
                let scene = world.resource::<Stage>().name.clone();
                let view = camera_transform(world);
                let mut bookmarks = world.resource_mut::<Bookmarks>();
                bookmarks.save(&scene, view);
                let count = bookmarks.for_scene(&scene).len();
                world
                    .resource_mut::<Bakes>()
                    .say(format!("view {count} of {scene} saved"));
            }
            Action::GoBookmark(index) => {
                let scene = world.resource::<Stage>().name.clone();
                if let Some(view) = world
                    .resource::<Bookmarks>()
                    .for_scene(&scene)
                    .get(index)
                    .copied()
                {
                    move_camera(world, view);
                }
            }
            Action::ResetCamera => {
                let start = world.resource::<Stage>().start;
                move_camera(world, start);
            }
            Action::Quit => {
                world.write_message(AppExit::Success);
            }
        }
    }
}

fn open_scene(world: &mut World, index: usize) {
    let entry = &world.resource::<Catalog>().entries[index];
    if !entry.available {
        let message = format!(
            "{} is not downloaded: run {}",
            entry.title,
            entry.fetch.as_deref().unwrap_or("tools/fetch-scenes.sh")
        );
        world.resource_mut::<Bakes>().say(message);
        return;
    }
    world.resource_mut::<Stage>().requested = Some(index);
}

fn camera_transform(world: &mut World) -> Transform {
    world
        .query_filtered::<&Transform, With<ViewerCamera>>()
        .iter(world)
        .next()
        .copied()
        .unwrap_or_default()
}

/// Puts the camera at `view` (keeping its projection and speed), and restarts the free camera there.
fn move_camera(world: &mut World, view: Transform) {
    let stage = world.resource::<Stage>();
    let fov = stage.fov;
    let speed = stage.bounds.map_or(1.0, |(min, max)| {
        crate::stage::start_from_bounds(min, max).1
    });
    place_camera(world, view, fov, speed);
}

fn unload_bake(world: &mut World) {
    let loaded: Vec<Entity> = world
        .query_filtered::<Entity, With<BakedGi>>()
        .iter(world)
        .collect();
    for entity in &loaded {
        world.entity_mut(*entity).despawn();
    }
    let mut bakes = world.resource_mut::<Bakes>();
    bakes.loaded = None;
    bakes.say(if loaded.is_empty() {
        "no bake was loaded"
    } else {
        "bake unloaded"
    });
}

/// The lamps `L` switches: the ones the scene marks, else every point, spot and rect light, else its suns.
pub fn switchable_lamps(world: &mut World) -> Vec<Entity> {
    let marked: Vec<Entity> = world
        .query_filtered::<Entity, With<ToggleLight>>()
        .iter(world)
        .collect();
    if !marked.is_empty() {
        return marked;
    }
    let lamps: Vec<Entity> = world
        .query_filtered::<Entity, Or<(With<PointLight>, With<SpotLight>, With<RectLight>)>>()
        .iter(world)
        .collect();
    if !lamps.is_empty() {
        return lamps;
    }
    world
        .query_filtered::<Entity, With<DirectionalLight>>()
        .iter(world)
        .collect()
}

fn toggle_lamps(world: &mut World) {
    let on = !world.resource::<Lamps>().on;
    let lamps = switchable_lamps(world);
    // Hidden, not despawned: the ladder, Solari and the bake's lamp scaling all read a light's visibility.
    for lamp in &lamps {
        world.entity_mut(*lamp).insert(if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
    world.resource_mut::<Lamps>().on = on;
    world.resource_mut::<Bakes>().say(format!(
        "{} lamp(s) {}",
        lamps.len(),
        if on { "on" } else { "off" }
    ));
}

/// Swaps the settings in use with the other set of the A/B pair.
pub fn flip_ab(world: &mut World) {
    let other = world.resource::<AbCompare>().other.clone();
    let current = std::mem::replace(&mut *world.resource_mut::<GraphicsSettings>(), other);
    let mut ab = world.resource_mut::<AbCompare>();
    ab.other = current;
    ab.showing_b = !ab.showing_b;
}

/// The file `F12` writes: `<data>/viewer-shots/<scene>-<rung>-<unix ms>.png`.
fn screenshot(world: &mut World) {
    let scene = world.resource::<Stage>().name.clone();
    let rung = world
        .get_resource::<ResolvedGraphics>()
        .map_or_else(|| "unresolved".into(), rung_text)
        .replace([' ', '(', ')'], "");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis());
    let dir = data_dir()
        .unwrap_or_else(|| ".".into())
        .join("viewer-shots");
    if let Err(error) = std::fs::create_dir_all(&dir) {
        world
            .resource_mut::<Bakes>()
            .say(format!("cannot create {}: {error}", dir.display()));
        return;
    }
    let path = dir.join(format!("{scene}-{rung}-{stamp}.png"));
    world
        .resource_mut::<Bakes>()
        .say(format!("screenshot: {}", path.display()));
    world
        .spawn((SceneEntity, Screenshot::primary_window()))
        .observe(save_to_disk(path));
}

/// A light's intensity before the lamp multiplier.
#[derive(Component)]
pub struct BaseIntensity(pub f32);

/// A material's own values, kept while the coverage view shows white matte.
#[derive(Resource, Default)]
pub struct CoverageSaved {
    materials: Vec<(AssetId<StandardMaterial>, StandardMaterial)>,
    lights: Vec<(Entity, Visibility)>,
}

/// Reads the open scene's exposure and sun into [`ViewControls`] once it is placed, and applies the controls
/// when they change: the camera's exposure, every lamp's intensity, the sun, the coverage view.
pub fn apply_view_controls(world: &mut World) {
    let stage = world.resource::<Stage>();
    let (generation, placed, exposure) = (stage.generation, stage.placed, stage.exposure);
    if world.resource::<ViewControls>().generation != generation && placed {
        let sun = scene_sun(world).map(|(_, light, transform)| {
            SunControl::from_direction(transform.forward().as_vec3(), light.illuminance)
        });
        let mut controls = world.resource_mut::<ViewControls>();
        controls.generation = generation;
        controls.exposure = exposure;
        controls.sun = sun;
        controls.initial_sun = sun;
        // A new scene's materials and lights are not the coverage view's.
        controls.coverage = false;
        world.resource_mut::<CoverageSaved>().materials.clear();
        world.resource_mut::<CoverageSaved>().lights.clear();
    }

    // Lamps that arrived since the last frame.
    let fresh: Vec<(Entity, f32)> = world
        .query_filtered::<(
            Entity,
            Option<&PointLight>,
            Option<&SpotLight>,
            Option<&RectLight>,
        ), Without<BaseIntensity>>()
        .iter(world)
        .filter_map(|(entity, point, spot, rect)| {
            point
                .map(|l| l.intensity)
                .or(spot.map(|l| l.intensity))
                .or(rect.map(|l| l.intensity))
                .map(|intensity| (entity, intensity))
        })
        .collect();
    let controls = world.resource::<ViewControls>().clone();
    let changed = world.is_resource_changed::<ViewControls>();
    let scale = controls.lamp_scale;
    for (entity, intensity) in &fresh {
        world.entity_mut(*entity).insert(BaseIntensity(*intensity));
    }
    if changed || (!fresh.is_empty() && scale != 1.0) {
        let mut lamps = world.query::<(
            &BaseIntensity,
            Option<&mut PointLight>,
            Option<&mut SpotLight>,
            Option<&mut RectLight>,
        )>();
        for (base, point, spot, rect) in lamps.iter_mut(world) {
            let wanted = base.0 * scale;
            if let Some(mut l) = point
                && l.intensity != wanted
            {
                l.intensity = wanted;
            }
            if let Some(mut l) = spot
                && l.intensity != wanted
            {
                l.intensity = wanted;
            }
            if let Some(mut l) = rect
                && l.intensity != wanted
            {
                l.intensity = wanted;
            }
        }
    }
    if !changed {
        return;
    }
    let mut cameras = world.query_filtered::<&mut Exposure, With<ViewerCamera>>();
    for mut camera in cameras.iter_mut(world) {
        if camera.ev100 != controls.exposure {
            camera.ev100 = controls.exposure;
        }
    }
    if let (Some(sun), Some((entity, light, transform))) = (controls.sun, scene_sun(world)) {
        let now = SunControl::from_direction(transform.forward().as_vec3(), light.illuminance);
        if (now.azimuth - sun.azimuth).abs() > 1e-3
            || (now.elevation - sun.elevation).abs() > 1e-3
            || now.lux != sun.lux
        {
            let mut entity = world.entity_mut(entity);
            let up = if sun.elevation.abs() > 89.0 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            let rotation = Transform::default()
                .looking_to(sun.direction(), up)
                .rotation;
            entity
                .get_mut::<Transform>()
                .expect("a sun has a transform")
                .rotation = rotation;
            entity
                .get_mut::<DirectionalLight>()
                .expect("checked")
                .illuminance = sun.lux;
        }
    }
    set_coverage(world, controls.coverage);
}

/// The sun the Render panel turns: the one the viewer added, else the scene's first directional light.
fn scene_sun(world: &mut World) -> Option<(Entity, DirectionalLight, Transform)> {
    let mut viewer =
        world.query_filtered::<(Entity, &DirectionalLight, &Transform), With<DefaultSun>>();
    if let Some((e, l, t)) = viewer.iter(world).next() {
        return Some((e, *l, *t));
    }
    let mut any = world.query::<(Entity, &DirectionalLight, &Transform)>();
    any.iter(world).next().map(|(e, l, t)| (e, *l, *t))
}

fn set_coverage(world: &mut World, on: bool) {
    let active = !world.resource::<CoverageSaved>().materials.is_empty()
        || !world.resource::<CoverageSaved>().lights.is_empty();
    if on == active {
        return;
    }
    if on {
        let lights: Vec<(Entity, Visibility)> = world
            .query_filtered::<(Entity, &Visibility), Or<(
                With<PointLight>,
                With<SpotLight>,
                With<RectLight>,
                With<DirectionalLight>,
            )>>()
            .iter(world)
            .map(|(e, v)| (e, *v))
            .collect();
        for (light, _) in &lights {
            world.entity_mut(*light).insert(Visibility::Hidden);
        }
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        let mut saved = Vec::new();
        for (id, material) in materials.iter_mut() {
            saved.push((id, material.clone()));
            *material = StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                metallic: 0.0,
                ..default()
            };
        }
        let mut coverage = world.resource_mut::<CoverageSaved>();
        coverage.materials = saved;
        coverage.lights = lights;
    } else {
        let saved = std::mem::take(&mut *world.resource_mut::<CoverageSaved>());
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        for (id, material) in saved.materials {
            if let Some(mut slot) = materials.get_mut(id) {
                *slot = material;
            }
        }
        for (light, visibility) in saved.lights {
            if let Ok(mut entity) = world.get_entity_mut(light) {
                entity.insert(visibility);
            }
        }
    }
}
