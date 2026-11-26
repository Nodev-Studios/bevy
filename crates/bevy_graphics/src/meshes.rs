//! Mesh fitting: gives every standard mesh a ray traceable mesh asset while a traced rung runs.
//!
//! What is traced follows `bevy_baked_gi`'s [`traceable`](bevy_baked_gi::traceable) rule, the same one the
//! baker uses, so a bake and a traced frame see one scene. While [`ResolvedGraphics::rung`] is traced, every entity
//! with a `Mesh3d` and a `MeshMaterial3d<StandardMaterial>` gets a `RaytracingMesh3d`:
//!
//! - It points at a **second, fitted mesh asset** made by [`fit`]: missing normals, UVs and tangents are
//!   generated, every other attribute is dropped, and a mesh without indices gets them. **The asset `Mesh3d`
//!   draws is never changed**, so lightmap UVs and vertex colours stay in the rasterized picture. A mesh that
//!   already fits is traced as it is, without a copy.
//! - The fitted copy declares in `Mesh::raytracing` the kind of acceleration structure its material needs: an
//!   alpha-masked material on a mesh that only declares the opaque kind is dropped by `bevy_solari`.
//! - **Blended materials are skipped** (`Blend`, `Premultiplied`, `Add`, `Multiply`: glass, curtains, particles):
//!   `bevy_solari` would trace them as opaque walls and darken every room behind a window. They are drawn, are
//!   missing from traced reflections, and are counted in [`SkippedBlendedMeshes`].
//! - **Skinned and morphed meshes are skipped**: a fitted copy would be traced frozen in its bind pose. They
//!   are drawn and lit, cast no traced shadow, and are counted in
//!   [`ResolvedGraphics::skipped_deforming_meshes`].
//! - A mesh that cannot be fitted ([`Unfit`]) is warned about once and counted in
//!   [`ResolvedGraphics::unfitted_meshes`].
//!
//! An entity that already has a `RaytracingMesh3d` is the game's and is left alone. An entity is fitted again
//! when its `Mesh3d` or its material handle changes or its mesh asset is modified; a material whose alpha
//! mode changes afterwards is not noticed. On `Raster` the ladder's `RaytracingMesh3d`s and the fitted assets
//! are removed, so nothing is kept for a renderer that is not running, and every count is 0.

#[cfg(doc)]
use crate::ResolvedGraphics;
#[cfg(test)]
use bevy_asset::RenderAssetUsages;
#[cfg(test)]
use bevy_mesh::{Indices, Mesh, MeshRaytracingFlags, PrimitiveTopology, VertexAttributeValues};

pub use bevy_baked_gi::traceable::{fit, Fitted, NotTraced, Unfit};
#[cfg(test)]
use bevy_baked_gi::traceable::{PLACEHOLDER_TANGENT, TRACED_ATTRIBUTES};

/// Called by the apply layer when [`GraphicsPlugin`](crate::GraphicsPlugin) is built.
pub(crate) fn build(app: &mut bevy_app::App) {
    #[cfg(feature = "solari")]
    systems::build(app);
    #[cfg(not(feature = "solari"))]
    let _ = app;
}

#[cfg(feature = "solari")]
pub use systems::{
    FittedMesh, SkippedBlendedMesh, SkippedBlendedMeshes, SkippedDeformingMesh, UnfittedMesh,
};

#[cfg(feature = "solari")]
mod systems {
    use super::{fit, Fitted, Unfit};
    use crate::{apply::GraphicsApplySystems, ResolvedGraphics};
    use bevy_app::{App, PostUpdate};
    use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
    use bevy_baked_gi::traceable::{is_deforming, traced_kind};
    use bevy_ecs::prelude::*;
    use bevy_mesh::{
        morph::MorphWeights, skinning::SkinnedMesh, Mesh, Mesh3d, MeshRaytracingFlags,
    };
    use bevy_pbr::{MeshMaterial3d, StandardMaterial};
    use bevy_platform::collections::HashMap;
    use bevy_solari::scene::RaytracingMesh3d;
    use tracing::warn;

    pub(super) fn build(app: &mut App) {
        app.init_resource::<FittedMeshes>()
            .init_resource::<SkippedBlendedMeshes>()
            .add_systems(PostUpdate, fit_meshes.in_set(GraphicsApplySystems::Apply));
    }

    /// On an entity whose `RaytracingMesh3d` the ladder inserted.
    #[derive(Component, Clone, Copy, Debug)]
    pub struct FittedMesh {
        source: AssetId<Mesh>,
    }

    /// On an entity whose mesh could not be made ray traceable. It is drawn, and missing from traced shadows,
    /// reflections and bounce light.
    #[derive(Component, Clone, Copy, Debug)]
    pub struct UnfittedMesh {
        source: AssetId<Mesh>,
        /// Why the mesh could not be fitted.
        pub reason: Unfit,
    }

    /// On a skinned or morphed mesh, which is never ray traced.
    #[derive(Component, Clone, Copy, Debug)]
    pub struct SkippedDeformingMesh;

    /// On a mesh whose material blends (glass, particles), which is drawn and never ray traced.
    #[derive(Component, Clone, Copy, Debug)]
    pub struct SkippedBlendedMesh;

    /// How many meshes the ladder leaves out of the traced scene because their material blends
    /// ([`SkippedBlendedMesh`]); 0 on `Raster`.
    #[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SkippedBlendedMeshes(pub u32);

    /// The fitted copy of each mesh that was looked at, per kind of acceleration structure, or why it has
    /// none. `Ok(None)` is a mesh that is traced as it is: holding its handle here would keep an asset the
    /// game has dropped alive.
    #[derive(Resource, Default)]
    pub(super) struct FittedMeshes(
        pub(super) HashMap<FittedKey, Result<Option<Handle<Mesh>>, Unfit>>,
    );

    type FittedKey = (AssetId<Mesh>, MeshRaytracingFlags);

    type Unseen = (
        Without<RaytracingMesh3d>,
        Without<UnfittedMesh>,
        Without<SkippedDeformingMesh>,
        Without<SkippedBlendedMesh>,
    );
    type Seen = Or<(
        With<FittedMesh>,
        With<UnfittedMesh>,
        With<SkippedDeformingMesh>,
        With<SkippedBlendedMesh>,
    )>;
    type Retargeted = Or<(Changed<Mesh3d>, Changed<MeshMaterial3d<StandardMaterial>>)>;

    fn forget(mut entity: EntityCommands, was_fitted: bool) {
        entity.remove::<(
            FittedMesh,
            UnfittedMesh,
            SkippedDeformingMesh,
            SkippedBlendedMesh,
        )>();
        if was_fitted {
            entity.remove::<RaytracingMesh3d>();
        }
    }

    pub(super) fn fit_meshes(
        mut resolved: ResMut<ResolvedGraphics>,
        mut fitted_meshes: ResMut<FittedMeshes>,
        mut blended_count: ResMut<SkippedBlendedMeshes>,
        mut meshes: ResMut<Assets<Mesh>>,
        materials: Res<Assets<StandardMaterial>>,
        mut mesh_events: MessageReader<AssetEvent<Mesh>>,
        unseen: Query<
            (
                Entity,
                &Mesh3d,
                &MeshMaterial3d<StandardMaterial>,
                Has<SkinnedMesh>,
                Has<MorphWeights>,
            ),
            Unseen,
        >,
        seen: Query<(Entity, Option<&FittedMesh>, Option<&UnfittedMesh>), Seen>,
        retargeted: Query<(Entity, Has<FittedMesh>), (Seen, Retargeted)>,
        unfitted: Query<(), With<UnfittedMesh>>,
        skipped: Query<(), With<SkippedDeformingMesh>>,
        blended: Query<(), With<SkippedBlendedMesh>>,
        mut commands: Commands,
    ) {
        if !resolved.rung.is_traced() {
            mesh_events.clear();
            for (entity, fitted, _) in &seen {
                forget(commands.entity(entity), fitted.is_some());
            }
            fitted_meshes.0.clear();
            set_counts(&mut resolved, 0, 0);
            blended_count.set_if_neq(SkippedBlendedMeshes(0));
            return;
        }

        // A mesh asset that changed is looked at again, and so is every entity that uses it.
        for event in mesh_events.read() {
            let (AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            | AssetEvent::Unused { id }) = *event
            else {
                continue;
            };
            let entries = fitted_meshes.0.len();
            fitted_meshes.0.retain(|(source, _), _| *source != id);
            if fitted_meshes.0.len() == entries {
                continue;
            }
            for (entity, fitted, unfitted) in &seen {
                let source = fitted
                    .map(|fitted| fitted.source)
                    .or(unfitted.map(|unfitted| unfitted.source));
                if source == Some(id) {
                    forget(commands.entity(entity), fitted.is_some());
                }
            }
        }
        for (entity, was_fitted) in &retargeted {
            forget(commands.entity(entity), was_fitted);
        }

        let (mut newly_unfitted, mut newly_skipped, mut newly_blended) = (0, 0, 0);
        for (entity, mesh_3d, material_3d, skinned, morphed) in &unseen {
            // An asset that has not loaded yet is tried again next frame.
            let (Some(mesh), Some(material)) =
                (meshes.get(&mesh_3d.0), materials.get(&material_3d.0))
            else {
                continue;
            };
            if is_deforming(mesh, skinned || morphed) {
                commands.entity(entity).insert(SkippedDeformingMesh);
                newly_skipped += 1;
                continue;
            }
            let needs = match traced_kind(material) {
                Ok(needs) => needs,
                // `traced_kind` only reads the material, so its one reason is a blended one.
                Err(_) => {
                    commands.entity(entity).insert(SkippedBlendedMesh);
                    newly_blended += 1;
                    continue;
                }
            };

            let source = mesh_3d.0.id();
            let key = (source, needs);
            let fitted = match fitted_meshes.0.get(&key) {
                Some(fitted) => fitted.clone(),
                None => {
                    let fitted = match fit(mesh, key.1) {
                        Ok(Fitted::AsIs) => Ok(None),
                        Ok(Fitted::Copy(copy)) => Ok(Some(meshes.add(*copy))),
                        Err(reason) => {
                            warn!(
                                "graphics: mesh {source} cannot be ray traced and is missing from traced \
                                 shadows, reflections and bounce light: {reason}"
                            );
                            Err(reason)
                        }
                    };
                    fitted_meshes.0.insert(key, fitted.clone());
                    fitted
                }
            };
            match fitted {
                Ok(copy) => {
                    commands.entity(entity).insert((
                        RaytracingMesh3d(copy.unwrap_or_else(|| mesh_3d.0.clone())),
                        FittedMesh { source },
                    ));
                }
                Err(reason) => {
                    commands
                        .entity(entity)
                        .insert(UnfittedMesh { source, reason });
                    newly_unfitted += 1;
                }
            }
        }

        // The markers inserted above are on their entities when this system runs next, and the ones
        // forgotten above are gone by then; until then the counts are a frame behind for those.
        set_counts(
            &mut resolved,
            unfitted.iter().len() as u32 + newly_unfitted,
            skipped.iter().len() as u32 + newly_skipped,
        );
        blended_count.set_if_neq(SkippedBlendedMeshes(
            blended.iter().len() as u32 + newly_blended,
        ));
    }

    fn set_counts(resolved: &mut ResMut<ResolvedGraphics>, unfitted: u32, skipped: u32) {
        if resolved.unfitted_meshes != unfitted || resolved.skipped_deforming_meshes != skipped {
            resolved.unfitted_meshes = unfitted;
            resolved.skipped_deforming_meshes = skipped;
        }
    }
}

#[cfg(test)]
mod tests;
