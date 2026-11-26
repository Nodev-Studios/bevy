//! What a key or a panel button does. Keys and buttons write an [`Action`]; [`run_actions`] carries it out, so
//! the self-test drives the same code a person does.

use crate::{
    bake::{self, Bakes},
    catalog::data_dir,
    stage::{SceneEntity, Stage},
    ui::{cycle, rung_text, Field, Help, Panel},
    AbCompare, ToggleLight,
};
use bevy::{
    app::AppExit,
    graphics::{GraphicsSettings, ResolvedGraphics},
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};

/// Something the person asked for.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub enum Action {
    TogglePanel,
    ToggleHelp,
    ToggleLamps,
    Bake,
    FlipAb,
    Screenshot,
    OpenScene(usize),
    Cycle(Field),
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

/// Turns key presses into actions.
pub fn read_keys(keys: Res<ButtonInput<KeyCode>>, mut actions: MessageWriter<Action>) {
    for (key, action) in [
        (KeyCode::Tab, Action::TogglePanel),
        (KeyCode::KeyH, Action::ToggleHelp),
        (KeyCode::KeyL, Action::ToggleLamps),
        (KeyCode::KeyB, Action::Bake),
        (KeyCode::KeyC, Action::FlipAb),
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
    let actions: Vec<Action> = world
        .resource_mut::<Messages<Action>>()
        .drain()
        .collect();
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
            Action::TogglePanel => toggle_display::<Panel>(world),
            Action::ToggleHelp => toggle_display::<Help>(world),
            Action::ToggleLamps => toggle_lamps(world),
            Action::Bake => bake::start_bake(world),
            Action::FlipAb => flip_ab(world),
            Action::Screenshot => screenshot(world),
            Action::OpenScene(index) => world.resource_mut::<Stage>().requested = Some(index),
            Action::Cycle(field) => cycle(&mut world.resource_mut::<GraphicsSettings>(), field),
            Action::Quit => {
                world.write_message(AppExit::Success);
            }
        }
    }
}

fn toggle_display<T: Component>(world: &mut World) {
    let mut nodes = world.query_filtered::<&mut Node, With<T>>();
    for mut node in nodes.iter_mut(world) {
        node.display = if node.display == Display::None {
            Display::Flex
        } else {
            Display::None
        };
    }
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
fn flip_ab(world: &mut World) {
    let other = world.resource::<AbCompare>().other.clone();
    let current = std::mem::replace(&mut *world.resource_mut::<GraphicsSettings>(), other);
    let mut ab = world.resource_mut::<AbCompare>();
    ab.other = current;
    ab.showing_b = !ab.showing_b;
}

/// The file `F12` writes: `<data>/viewer-shots/<scene>-<rung>-<unix seconds>.png`.
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
    world.resource_mut::<Bakes>().say(format!("screenshot: {}", path.display()));
    world
        .spawn((SceneEntity, Screenshot::primary_window()))
        .observe(save_to_disk(path));
}
