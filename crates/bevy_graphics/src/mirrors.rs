//! Finds the mirrors of a scene for the `Raster` rung's planar reflections.
//!
//! `bevy_pbr`'s [`planar_reflection`](bevy_pbr::planar_reflection) renders a reflection for each
//! [`PlanarMirror`] a camera with [`PlanarReflections`](bevy_pbr::planar_reflection::PlanarReflections) sees.
//! A game rarely marks its mirrors (a glTF scene cannot), so while the ladder is on `Raster` this module marks
//! them itself: every mesh whose [`StandardMaterial`] looks like a mirror ([`is_mirror_material`]) and whose
//! geometry is flat ([`measure_mirror_mesh`]). [`NoPlanarMirror`] on an entity keeps it out; a
//! `PlanarMirror` the game put on an entity is left as it is.
//!
//! Each mesh is looked at once per mesh and material it has. A material whose values change later is not
//! looked at again; a mesh still loading is looked at once it is there.

use crate::{apply::GraphicsApplySystems, GraphicsRung, ResolvedGraphics};
use bevy_app::{App, PostUpdate};
use bevy_asset::{AssetId, Assets};
use bevy_ecs::prelude::*;
use bevy_material::AlphaMode;
use bevy_mesh::{Mesh, Mesh3d};
use bevy_pbr::{
    planar_reflection::{measure_mirror_mesh, NoPlanarMirror, PlanarMirror},
    MeshMaterial3d, StandardMaterial,
};

/// The least metallic a mirror's material is. // to tune: glTF mirrors (the Bitterli bathroom's, Blender's
/// "mirror" presets) export 1.0; 0.9 leaves room for an artist's almost-metal.
pub const MIRROR_MIN_METALLIC: f32 = 0.9;

/// The roughest a mirror's material is. // to tune: above 0.1 a planar reflection, which is not blurred by
/// roughness, would look sharper than the surface is; SSR and probes blur.
pub const MIRROR_MAX_ROUGHNESS: f32 = 0.1;

/// Marks the entities this module made mirrors, and the mesh and material it looked at.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectedMirror {
    mesh: AssetId<Mesh>,
    material: AssetId<StandardMaterial>,
    /// Whether this module put a [`PlanarMirror`] on the entity.
    pub is_mirror: bool,
}

pub(crate) fn build(app: &mut App) {
    app.add_systems(
        PostUpdate,
        detect_mirrors.in_set(GraphicsApplySystems::Apply),
    );
}

/// Whether a material reflects like a mirror: metallic at least [`MIRROR_MIN_METALLIC`], roughness at most
/// [`MIRROR_MAX_ROUGHNESS`], opaque (or alpha-masked), lit, not transmissive. The factors are read; a
/// metallic-roughness texture only lowers them, which can make a mirror look rougher than it is planar
/// reflected.
pub fn is_mirror_material(material: &StandardMaterial) -> bool {
    material.metallic >= MIRROR_MIN_METALLIC
        && material.perceptual_roughness <= MIRROR_MAX_ROUGHNESS
        && matches!(material.alpha_mode, AlphaMode::Opaque | AlphaMode::Mask(_))
        && !material.unlit
        && material.specular_transmission == 0.0
        && material.diffuse_transmission == 0.0
}

fn detect_mirrors(
    resolved: Res<ResolvedGraphics>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    candidates: Query<
        (
            Entity,
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            Option<&DetectedMirror>,
            Has<PlanarMirror>,
        ),
        Without<NoPlanarMirror>,
    >,
    opted_out: Query<(Entity, &DetectedMirror), With<NoPlanarMirror>>,
    mut commands: Commands,
) {
    // A mirror marked before the game opted it out is no longer one.
    for (entity, detected) in &opted_out {
        let mut entity = commands.entity(entity);
        if detected.is_mirror {
            entity.remove::<PlanarMirror>();
        }
        entity.remove::<DetectedMirror>();
    }
    if resolved.rung != GraphicsRung::Raster {
        return;
    }
    for (entity, mesh3d, material3d, detected, has_mirror) in &candidates {
        let (mesh_id, material_id) = (mesh3d.id(), material3d.id());
        if detected.is_some_and(|detected| {
            detected.mesh == mesh_id && detected.material == material_id
        }) {
            continue;
        }
        // The game's own mirror is left as it is.
        if has_mirror && !detected.is_some_and(|detected| detected.is_mirror) {
            continue;
        }
        let (Some(mesh), Some(material)) = (meshes.get(mesh_id), materials.get(material_id)) else {
            continue;
        };
        let is_mirror = is_mirror_material(material) && measure_mirror_mesh(mesh).is_some();
        let mut entity = commands.entity(entity);
        if is_mirror {
            entity.insert(PlanarMirror);
        } else if has_mirror {
            entity.remove::<PlanarMirror>();
        }
        entity.insert(DetectedMirror {
            mesh: mesh_id,
            material: material_id,
            is_mirror,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_materials_are_smooth_opaque_metals() {
        let mirror = StandardMaterial {
            metallic: 1.0,
            perceptual_roughness: 0.0,
            ..Default::default()
        };
        assert!(is_mirror_material(&mirror));
        assert!(is_mirror_material(&StandardMaterial {
            metallic: 0.9,
            perceptual_roughness: 0.1,
            ..mirror.clone()
        }));
        // Bevy's default material: a rough dielectric.
        assert!(!is_mirror_material(&StandardMaterial::default()));
        assert!(!is_mirror_material(&StandardMaterial {
            perceptual_roughness: 0.2,
            ..mirror.clone()
        }));
        assert!(!is_mirror_material(&StandardMaterial {
            metallic: 0.5,
            ..mirror.clone()
        }));
        assert!(!is_mirror_material(&StandardMaterial {
            alpha_mode: AlphaMode::Blend,
            ..mirror.clone()
        }));
        assert!(!is_mirror_material(&StandardMaterial {
            specular_transmission: 1.0,
            ..mirror.clone()
        }));
        assert!(!is_mirror_material(&StandardMaterial {
            unlit: true,
            ..mirror
        }));
    }
}
