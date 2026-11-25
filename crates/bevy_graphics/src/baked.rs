//! The ladder's part of baked indirect light (`bevy_baked_gi`).
//!
//! A game loads a bake by spawning a [`BakedGi`] that names its manifest; [`GraphicsPlugin`](crate::GraphicsPlugin)
//! adds [`BakedGiPlugin`] if the app has not, and this file does the rest:
//!
//! - **Ambient.** While a bake is loaded ([`BakedGiStatus::Loaded`] or [`BakedGiStatus::Stale`]), every
//!   [`GraphicsCamera`] gets an [`AmbientLight`] at [`GraphicsSettings::ambient_while_baked`] (default 0), on
//!   every rung: the bake holds the bounce light a game's ambient stood in for, and on a traced rung the forward
//!   materials still read the volumes. An [`AmbientLight`] the game had put on the camera is set aside and comes
//!   back when the bake goes; one the game puts there while the bake is loaded is taken as its new wish, set
//!   aside too, and overwritten.
//! - **Off.** [`GraphicsOverrides::baked_gi`](crate::GraphicsOverrides::baked_gi) `Some(false)` hides every
//!   [`BakedGi`] (Bevy then binds none of its volumes) and gives the cameras their own ambient back.
//! - **Degraded.** When the device cannot bind arrays of light probe textures the runtime loads no volume and
//!   reports [`BakedGiStatus::Degraded`]; the cameras keep the game's ambient.
//!
//! The traced rungs ignore the volumes for opaque surfaces by construction: Solari lights them itself.

use crate::{GraphicsCamera, GraphicsSettings};
use bevy_app::{App, PostUpdate};
use bevy_camera::visibility::Visibility;
use bevy_color::Color;
use bevy_ecs::prelude::*;
use bevy_light::AmbientLight;

pub use bevy_baked_gi::{
    BakedGi, BakedGiManifest, BakedGiPlugin, BakedGiStatus, GiBakeRegion, GiBakeSkipEmission,
};

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
pub(crate) fn build(app: &mut App) {
    if !app.is_plugin_added::<BakedGiPlugin>() {
        app.add_plugins(BakedGiPlugin);
    }
    app.add_systems(
        PostUpdate,
        (switch_bakes, apply_baked_ambient).in_set(crate::apply::GraphicsApplySystems::Apply),
    );
}

/// Whether the cameras are lit by a bake now.
fn bake_in_use(settings: &GraphicsSettings, status: BakedGiStatus) -> bool {
    settings.overrides.baked_gi != Some(false)
        && matches!(status, BakedGiStatus::Loaded | BakedGiStatus::Stale)
}

/// The visibility a [`BakedGi`] had before the ladder hid it.
#[derive(Component)]
struct HiddenByLadder(Visibility);

/// Hides every bake while the override switches baked light off, and puts back the visibility it had.
fn switch_bakes(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    mut roots: Query<(Entity, &mut Visibility, Option<&HiddenByLadder>), With<BakedGi>>,
) {
    let off = settings.overrides.baked_gi == Some(false);
    for (root, mut visibility, hidden) in &mut roots {
        match (off, hidden) {
            (true, None) => {
                commands.entity(root).insert(HiddenByLadder(*visibility));
                *visibility = Visibility::Hidden;
            }
            (false, Some(HiddenByLadder(previous))) => {
                *visibility = *previous;
                commands.entity(root).remove::<HiddenByLadder>();
            }
            _ => {}
        }
    }
}

/// What the ladder did to a camera's [`AmbientLight`] while a bake is in use.
#[derive(Component)]
pub struct BakedAmbient {
    /// The game's own ambient light on this camera, `None` when it had none (the camera then used
    /// [`GlobalAmbientLight`](bevy_light::GlobalAmbientLight)).
    game: Option<AmbientLight>,
    /// The brightness the ladder last wrote.
    written: f32,
}

/// The ambient light the ladder puts on a camera: the game's colour and lightmap flag at `brightness`.
fn ladder_ambient(game: Option<&AmbientLight>, brightness: f32) -> AmbientLight {
    AmbientLight {
        color: game.map_or(Color::WHITE, |ambient| ambient.color),
        brightness,
        affects_lightmapped_meshes: game.is_none_or(|ambient| ambient.affects_lightmapped_meshes),
    }
}

fn apply_baked_ambient(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    status: Res<BakedGiStatus>,
    mut cameras: Query<
        (Entity, Option<&mut AmbientLight>, Option<&mut BakedAmbient>),
        With<GraphicsCamera>,
    >,
) {
    let in_use = bake_in_use(&settings, *status);
    let brightness = settings.ambient_while_baked;
    for (camera, ambient, state) in &mut cameras {
        match (in_use, state) {
            (true, None) => {
                let game = ambient.as_deref().cloned();
                commands.entity(camera).insert((
                    ladder_ambient(game.as_ref(), brightness),
                    BakedAmbient {
                        game,
                        written: brightness,
                    },
                ));
            }
            (true, Some(mut state)) => match ambient {
                // The game removed the ambient light: its wish is now the global one.
                None => {
                    state.game = None;
                    state.written = brightness;
                    commands
                        .entity(camera)
                        .insert(ladder_ambient(None, brightness));
                }
                Some(mut ambient) => {
                    if ambient.brightness != state.written {
                        // Changed by the game since the ladder wrote it: its new wish, set aside.
                        state.game = Some(ambient.clone());
                    }
                    if ambient.brightness != brightness {
                        ambient.brightness = brightness;
                    }
                    state.written = brightness;
                }
            },
            (false, Some(state)) => {
                let mut entity = commands.entity(camera);
                match &state.game {
                    Some(game) => entity.insert(game.clone()),
                    None => entity.remove::<AmbientLight>(),
                };
                entity.remove::<BakedAmbient>();
            }
            (false, None) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphicsOverrides;
    use bevy_ecs::system::SystemId;

    struct Ladder {
        world: World,
        systems: [SystemId; 2],
    }

    impl Ladder {
        fn new() -> Self {
            let mut world = World::new();
            world.insert_resource(GraphicsSettings::default());
            world.insert_resource(BakedGiStatus::Absent);
            let systems = [
                world.register_system(switch_bakes),
                world.register_system(apply_baked_ambient),
            ];
            Self { world, systems }
        }

        fn update(&mut self) {
            for system in self.systems {
                self.world.run_system(system).unwrap();
            }
        }

        fn status(&mut self, status: BakedGiStatus) {
            *self.world.resource_mut::<BakedGiStatus>() = status;
            self.update();
        }

        fn ambient(&self, camera: Entity) -> Option<f32> {
            self.world
                .get::<AmbientLight>(camera)
                .map(|ambient| ambient.brightness)
        }
    }

    fn game_ambient(brightness: f32) -> AmbientLight {
        AmbientLight {
            color: Color::srgb(1.0, 0.9, 0.8),
            brightness,
            affects_lightmapped_meshes: false,
        }
    }

    #[test]
    fn ambient_is_set_aside_while_baked_and_restored() {
        let mut ladder = Ladder::new();
        let with_own = ladder
            .world
            .spawn((GraphicsCamera, game_ambient(120.0)))
            .id();
        let without = ladder.world.spawn(GraphicsCamera).id();
        let other = ladder.world.spawn(game_ambient(50.0)).id();

        ladder.status(BakedGiStatus::Loaded);
        assert_eq!(ladder.ambient(with_own), Some(0.0));
        assert_eq!(ladder.ambient(without), Some(0.0));
        assert_eq!(ladder.ambient(other), Some(50.0), "not a ladder camera");
        // The game's colour and flag are kept; only the brightness is the ladder's.
        let ambient = ladder.world.get::<AmbientLight>(with_own).unwrap();
        assert_eq!(ambient.color, Color::srgb(1.0, 0.9, 0.8));
        assert!(!ambient.affects_lightmapped_meshes);

        ladder
            .world
            .resource_mut::<GraphicsSettings>()
            .ambient_while_baked = 3.0;
        ladder.update();
        assert_eq!(ladder.ambient(with_own), Some(3.0));

        // A stale bake still lights the level.
        ladder.status(BakedGiStatus::Stale);
        assert_eq!(ladder.ambient(with_own), Some(3.0));

        ladder.status(BakedGiStatus::Absent);
        assert_eq!(ladder.ambient(with_own), Some(120.0));
        assert_eq!(ladder.ambient(without), None);
        assert!(ladder.world.get::<BakedAmbient>(with_own).is_none());
    }

    #[test]
    fn a_change_the_game_makes_while_baked_is_its_new_wish() {
        let mut ladder = Ladder::new();
        let camera = ladder
            .world
            .spawn((GraphicsCamera, game_ambient(120.0)))
            .id();
        ladder.status(BakedGiStatus::Loaded);
        ladder
            .world
            .get_mut::<AmbientLight>(camera)
            .unwrap()
            .brightness = 40.0;
        ladder.update();
        assert_eq!(ladder.ambient(camera), Some(0.0));
        ladder.status(BakedGiStatus::Absent);
        assert_eq!(ladder.ambient(camera), Some(40.0));

        // Removed while baked: the game wants the global ambient back afterwards.
        ladder.status(BakedGiStatus::Loaded);
        ladder.world.entity_mut(camera).remove::<AmbientLight>();
        ladder.update();
        assert_eq!(ladder.ambient(camera), Some(0.0));
        ladder.status(BakedGiStatus::Absent);
        assert_eq!(ladder.ambient(camera), None);
    }

    #[test]
    fn degraded_keeps_the_game_ambient() {
        let mut ladder = Ladder::new();
        let camera = ladder
            .world
            .spawn((GraphicsCamera, game_ambient(120.0)))
            .id();
        ladder.status(BakedGiStatus::Degraded);
        assert_eq!(ladder.ambient(camera), Some(120.0));
        assert!(ladder.world.get::<BakedAmbient>(camera).is_none());
    }

    #[test]
    fn the_override_hides_the_bake_and_restores_the_ambient() {
        let mut ladder = Ladder::new();
        let camera = ladder
            .world
            .spawn((GraphicsCamera, game_ambient(120.0)))
            .id();
        let root = ladder
            .world
            .spawn((BakedGi(Default::default()), Visibility::Visible))
            .id();
        ladder.status(BakedGiStatus::Loaded);
        assert_eq!(ladder.ambient(camera), Some(0.0));

        ladder.world.resource_mut::<GraphicsSettings>().overrides = GraphicsOverrides {
            baked_gi: Some(false),
            ..Default::default()
        };
        ladder.update();
        assert_eq!(ladder.ambient(camera), Some(120.0));
        assert_eq!(
            *ladder.world.get::<Visibility>(root).unwrap(),
            Visibility::Hidden
        );

        ladder
            .world
            .resource_mut::<GraphicsSettings>()
            .overrides
            .baked_gi = Some(true);
        ladder.update();
        assert_eq!(ladder.ambient(camera), Some(0.0));
        assert_eq!(
            *ladder.world.get::<Visibility>(root).unwrap(),
            Visibility::Visible
        );
    }

    #[test]
    fn settings_saved_before_the_bake_fields_still_load() {
        let settings: GraphicsSettings = ron::from_str(
            "(preset: Raster, upscaler: Auto, upscale_quality: Auto, denoiser: Auto, overrides: (shadows: None, \
             ssao: None, ssr: None, contact_shadows: None, bloom: None, restir: None, max_bounces: None))",
        )
        .unwrap();
        assert_eq!(settings.ambient_while_baked, 0.0);
        assert_eq!(settings.overrides.baked_gi, None);
    }
}
