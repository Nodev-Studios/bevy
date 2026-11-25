//! The run-time half of baked reflection probes: one Bevy reflection probe per baked region that has a cubemap,
//! beside the region's volume.
//!
//! The probe is a [`LightProbe`] + [`EnvironmentMapLight`] child of the same [`BakedGi`] root as the volume, with
//! the region's box and falloff and [`ParallaxCorrection::Auto`], so Bevy reprojects the cubemap onto the box. Its
//! intensity is the volume's (decision 8's lamp scaling), so a lamp switched off dims the reflections of the rooms
//! it lit by the same share as their bounce light. Hiding the root hides both, and a `Degraded` device spawns
//! neither: Bevy turns reflection probes off where it cannot bind arrays.

use crate::{BakedGi, BakedGiManifest, BakedGiVolume};
use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    name::Name,
    query::{Added, Changed},
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_light::{EnvironmentMapLight, IrradianceVolume, LightProbe, ParallaxCorrection};
use bevy_math::{Quat, Vec3};
use wgpu_types::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

/// One baked reflection probe, a child of its [`BakedGi`]: the probe of region `region` of the manifest.
#[derive(Component, Clone, Copy, Debug)]
pub struct BakedGiReflection {
    /// The index of the region in the manifest's `regions`.
    pub region: usize,
}

/// A 1×1 black cubemap: the probes' diffuse map. Bevy reads a reflection probe's diffuse map only where no
/// irradiance volume lights the fragment, and every probe sits in its region's volume, which gives the diffuse.
fn black_cubemap() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..Default::default()
    });
    image
}

/// Spawns a region's probe when its volume is spawned, and replaces the probes of a root whose bake changed.
pub(crate) fn spawn_reflection_probes(
    mut commands: Commands,
    added: Query<(&BakedGiVolume, &ChildOf, &IrradianceVolume), Added<BakedGiVolume>>,
    roots: Query<&BakedGi>,
    probes: Query<(Entity, &ChildOf), bevy_ecs::query::With<BakedGiReflection>>,
    manifests: Res<Assets<BakedGiManifest>>,
    images: Option<ResMut<Assets<Image>>>,
    mut black: Local<Option<Handle<Image>>>,
) {
    if added.is_empty() {
        return;
    }
    let Some(mut images) = images else {
        return;
    };
    for (volume, parent, irradiance) in &added {
        let root = parent.parent();
        let Ok(baked) = roots.get(root) else {
            continue;
        };
        let Some(manifest) = manifests.get(baked.0.id()) else {
            continue;
        };
        // The volume of region 0 is new only when the root's whole bake was (re)spawned.
        if volume.region == 0 {
            for (probe, probe_parent) in &probes {
                if probe_parent.parent() == root {
                    commands.entity(probe).despawn();
                }
            }
        }
        let (Some(region), Some(Some(cubemap))) = (
            manifest.regions.get(volume.region),
            manifest.reflections.get(volume.region),
        ) else {
            continue;
        };
        let diffuse = black
            .get_or_insert_with(|| images.add(black_cubemap()))
            .clone();
        commands.spawn((
            BakedGiReflection {
                region: volume.region,
            },
            LightProbe {
                falloff: Vec3::from_array(region.falloff),
            },
            EnvironmentMapLight {
                diffuse_map: diffuse,
                specular_map: cubemap.clone(),
                intensity: irradiance.intensity,
                rotation: Quat::IDENTITY,
                affects_lightmapped_mesh_diffuse: false,
            },
            // The capture's texels are directions to points of this box (see the baker), the shape Bevy's
            // parallax correction reprojects onto.
            ParallaxCorrection::Auto,
            region.transform(),
            Name::new(format!("{} reflection", region.name)),
            ChildOf(root),
        ));
    }
}

/// Keeps each probe at its region volume's intensity.
pub(crate) fn follow_volume_intensity(
    volumes: Query<(&BakedGiVolume, &ChildOf, &IrradianceVolume), Changed<IrradianceVolume>>,
    mut probes: Query<(&BakedGiReflection, &ChildOf, &mut EnvironmentMapLight)>,
) {
    for (volume, parent, irradiance) in &volumes {
        for (probe, probe_parent, mut light) in &mut probes {
            if probe.region == volume.region
                && probe_parent.parent() == parent.parent()
                && light.intensity != irradiance.intensity
            {
                light.intensity = irradiance.intensity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BakeProvenance, BakedGiPlugin, BakedLight, BakedLightKind, BakedReflection, BakedRegion,
        LightKey, MANIFEST_FORMAT,
    };
    use bevy_app::App;
    use bevy_asset::{AssetApp, AssetPlugin};
    use bevy_camera::visibility::InheritedVisibility;
    use bevy_light::SpotLight;
    use bevy_transform::components::{GlobalTransform, Transform};

    fn region(name: &str, weight: f32, reflection: bool) -> BakedRegion {
        BakedRegion {
            name: name.into(),
            file: format!("{name}.ktx2"),
            translation: [0.0, 1.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [4.0, 3.2, 4.0],
            falloff: [0.1; 3],
            resolution: [12, 10, 12],
            invalid_voxels: 0,
            samples: 1,
            pilot_noise: 0.0,
            bounce: 1.0,
            static_weight: 1.0 - weight,
            light_weights: vec![weight],
            reflection: reflection.then(|| BakedReflection {
                file: format!("{name}.refl.ktx2"),
                origin: [0.0, 1.5, 0.0],
                size: 8,
                levels: 2,
                samples: 1,
            }),
        }
    }

    #[test]
    fn a_probe_per_region_with_a_cubemap_follows_its_volume() {
        let mut app = App::new();
        app.add_plugins((AssetPlugin::default(), BakedGiPlugin))
            .init_asset::<Image>();
        let at = GlobalTransform::from_xyz(0.0, 2.9, 0.0);
        let cubemap = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        let manifest = BakedGiManifest {
            format: MANIFEST_FORMAT,
            scene_hash: "0".into(),
            bake: BakeProvenance {
                baker: "test".into(),
                adapter: "none".into(),
                driver: "none".into(),
                mode: "full".into(),
                samples: 1,
                seed: 0,
                pilot_samples: 0,
                target_noise: 0.0,
                invalid_back_face_share: 0.25,
                texel_format: "Rgba16Float".into(),
            },
            lights: vec![BakedLight {
                key: LightKey::new(BakedLightKind::Spot, &at),
                power: 1000.0,
            }],
            regions: vec![region("a", 0.6, true), region("b", 0.6, false)],
            volumes: Vec::new(),
            reflections: vec![Some(cubemap.clone()), None],
        };
        let handle = app
            .world_mut()
            .resource_mut::<Assets<BakedGiManifest>>()
            .add(manifest);
        let lamp = app
            .world_mut()
            .spawn((
                SpotLight {
                    intensity: 1000.0,
                    ..Default::default()
                },
                at,
                Transform::from_xyz(0.0, 2.9, 0.0),
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let root = app.world_mut().spawn(BakedGi(handle)).id();
        app.update();
        app.update();

        let probes = |app: &mut App| {
            app.world_mut()
                .query::<(
                    &BakedGiReflection,
                    &EnvironmentMapLight,
                    &ChildOf,
                    &Transform,
                )>()
                .iter(app.world())
                .map(|(probe, light, parent, transform)| {
                    (probe.region, light.clone(), parent.parent(), *transform)
                })
                .collect::<Vec<_>>()
        };
        let found = probes(&mut app);
        assert_eq!(found.len(), 1, "only region a has a cubemap");
        let (region_index, light, parent, transform) = &found[0];
        assert_eq!(*region_index, 0);
        assert_eq!(*parent, root);
        assert_eq!(light.specular_map, cubemap);
        assert_ne!(light.diffuse_map, cubemap);
        assert_eq!(transform.scale, Vec3::new(4.0, 3.2, 4.0));
        assert!((light.intensity - 1.0).abs() < 1e-5);

        // The lamp gone: the probe dims with its volume, to the static share.
        app.world_mut().despawn(lamp);
        app.update();
        app.update();
        assert!((probes(&mut app)[0].1.intensity - 0.4).abs() < 1e-5);
    }
}
