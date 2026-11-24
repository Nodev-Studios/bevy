//! Mesh fitting: gives every standard mesh a ray traceable mesh asset while a traced rung runs.
//!
//! `bevy_solari` traces a mesh only when it is an indexed triangle list with exactly a position, a normal, a
//! UV and a tangent per vertex, and drops anything else without a word. Few meshes are like that: Bevy's own
//! primitives have no tangents, and a model may carry vertex colours, a second UV set or no UVs at all.
//!
//! While [`ResolvedGraphics::rung`] is traced, every entity with a `Mesh3d` and a
//! `MeshMaterial3d<StandardMaterial>` gets a `RaytracingMesh3d`:
//!
//! - It points at a **second, fitted mesh asset** made by [`fit`]: missing normals, UVs and tangents are
//!   generated, every other attribute is dropped, and a mesh without indices gets them. **The asset `Mesh3d`
//!   draws is never changed**, so lightmap UVs and vertex colours stay in the rasterized picture. A mesh that
//!   already fits is traced as it is, without a copy.
//! - The fitted copy declares in `Mesh::raytracing` the kind of acceleration structure its material needs: an
//!   alpha-masked material on a mesh that only declares the opaque kind is dropped by `bevy_solari`.
//! - **Skinned and morphed meshes are skipped**: a fitted copy would be traced frozen in its bind pose. They
//!   are drawn and lit, cast no traced shadow, and are counted in
//!   [`ResolvedGraphics::skipped_deforming_meshes`].
//! - A mesh that cannot be fitted ([`Unfit`]) is warned about once and counted in
//!   [`ResolvedGraphics::unfitted_meshes`].
//!
//! An entity that already has a `RaytracingMesh3d` is the game's and is left alone. An entity is fitted again
//! when its `Mesh3d` or its material handle changes or its mesh asset is modified; a material whose alpha
//! mode changes afterwards is not noticed. On `Raster` the ladder's `RaytracingMesh3d`s and the fitted assets
//! are removed, so nothing is kept for a renderer that is not running, and both counts are 0.

#[cfg(doc)]
use crate::ResolvedGraphics;
use bevy_asset::RenderAssetUsages;
use bevy_mesh::{
    Indices, Mesh, MeshRaytracingFlags, MeshVertexAttribute, PrimitiveTopology,
    VertexAttributeValues,
};
use core::fmt;

/// The vertex attributes of a ray traceable mesh, in the order the renderer expects them.
const TRACED_ATTRIBUTES: [MeshVertexAttribute; 4] = [
    Mesh::ATTRIBUTE_POSITION,
    Mesh::ATTRIBUTE_NORMAL,
    Mesh::ATTRIBUTE_UV_0,
    Mesh::ATTRIBUTE_TANGENT,
];

/// A tangent for vertices that cannot have a real one. Any unit vector does: tangents are only read for
/// normal maps, which need the UVs these vertices lack.
const PLACEHOLDER_TANGENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// What [`fit`] found.
#[derive(Debug)]
pub enum Fitted {
    /// The mesh can be traced as it is.
    AsIs,
    /// The mesh needs this copy to be traced.
    Copy(Box<Mesh>),
}

/// Why a mesh cannot be made ray traceable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unfit {
    /// It is not a triangle list.
    NotATriangleList,
    /// Its vertex data lives in the render world only, where it cannot be read.
    NotInMainWorld,
    /// It has no positions.
    NoPositions,
    /// The named attribute is not in Bevy's standard format (it is compressed or quantized).
    AttributeFormat(&'static str),
    /// It has no indices and its vertex count is not a multiple of three.
    IncompleteTriangle,
    /// Its attributes differ in length, or an index points past the last vertex.
    Inconsistent,
}

impl fmt::Display for Unfit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotATriangleList => f.write_str("it is not a triangle list"),
            Self::NotInMainWorld => f.write_str(
                "its vertex data was moved to the render world (keep `RenderAssetUsages::MAIN_WORLD` on it)",
            ),
            Self::NoPositions => f.write_str("it has no positions"),
            Self::AttributeFormat(name) => {
                write!(f, "its attribute {name} is compressed or quantized")
            }
            Self::IncompleteTriangle => {
                f.write_str("it has no indices and its vertex count is not a multiple of three")
            }
            Self::Inconsistent => f.write_str(
                "its attributes differ in length, or an index points past its last vertex",
            ),
        }
    }
}

/// Makes a mesh ray traceable for a material that needs `needs` of [`Mesh::raytracing`].
///
/// Pure: `mesh` is only read.
pub fn fit(mesh: &Mesh, needs: MeshRaytracingFlags) -> Result<Fitted, Unfit> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(Unfit::NotATriangleList);
    }
    let attributes: Vec<_> = mesh
        .try_attributes()
        .map_err(|_| Unfit::NotInMainWorld)?
        .collect();
    let indices = mesh
        .try_indices_option()
        .map_err(|_| Unfit::NotInMainWorld)?;

    // The declared format is what the renderer checks, the values' format is what the buffer holds.
    let is_standard = |(attribute, values): &(&MeshVertexAttribute, &VertexAttributeValues),
                       standard: &MeshVertexAttribute| {
        attribute.id == standard.id
            && attribute.format == standard.format
            && bevy_mesh::VertexFormat::from(*values) == standard.format
    };
    let fits = attributes.len() == TRACED_ATTRIBUTES.len()
        && attributes
            .iter()
            .zip(&TRACED_ATTRIBUTES)
            .all(|(found, standard)| is_standard(found, standard));
    if fits && indices.is_some() && mesh.raytracing.contains(needs) {
        return Ok(Fitted::AsIs);
    }

    // Each attribute the copy keeps, in its standard format or not at all.
    let kept = |standard: &MeshVertexAttribute| match attributes
        .iter()
        .find(|(attribute, _)| attribute.id == standard.id)
    {
        Some(found) if is_standard(found, standard) => Ok(Some(found.1.clone())),
        Some(_) => Err(Unfit::AttributeFormat(standard.name)),
        None => Ok(None),
    };
    let positions = kept(&Mesh::ATTRIBUTE_POSITION)?.ok_or(Unfit::NoPositions)?;
    let normals = kept(&Mesh::ATTRIBUTE_NORMAL)?;
    let uvs = kept(&Mesh::ATTRIBUTE_UV_0)?;
    let tangents = kept(&Mesh::ATTRIBUTE_TANGENT)?;
    let vertex_count = positions.len();
    // Generating normals and tangents indexes the attributes by vertex and would panic on a broken mesh.
    let lengths_agree = [&normals, &uvs, &tangents]
        .into_iter()
        .flatten()
        .all(|values| values.len() == vertex_count);
    let indices_in_range =
        indices.is_none_or(|indices| indices.iter().all(|index| index < vertex_count));
    if !lengths_agree || !indices_in_range {
        return Err(Unfit::Inconsistent);
    }

    // Nothing reads the copy on the CPU once the renderer has it.
    let mut fitted = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    fitted.raytracing = needs;
    fitted.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    match indices {
        Some(indices) => fitted.insert_indices(indices.clone()),
        None if vertex_count % 3 == 0 => {
            fitted.insert_indices(Indices::U32((0..vertex_count as u32).collect()));
        }
        None => return Err(Unfit::IncompleteTriangle),
    }
    match normals {
        Some(normals) => {
            fitted.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        }
        None => fitted.compute_normals(),
    }
    let has_uvs = uvs.is_some();
    fitted.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        // Without UVs nothing on the mesh is textured, so any constant does.
        uvs.unwrap_or_else(|| VertexAttributeValues::Float32x2(vec![[0.0; 2]; vertex_count])),
    );
    match tangents {
        Some(tangents) => {
            fitted.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
        }
        // Tangents follow the UVs; degenerate UVs can make their generation fail.
        None if has_uvs && fitted.generate_tangents().is_ok() => {}
        None => {
            fitted.insert_attribute(
                Mesh::ATTRIBUTE_TANGENT,
                VertexAttributeValues::Float32x4(vec![PLACEHOLDER_TANGENT; vertex_count]),
            );
        }
    }
    Ok(Fitted::Copy(Box::new(fitted)))
}

/// Called by the apply layer when [`GraphicsPlugin`](crate::GraphicsPlugin) is built.
pub(crate) fn build(app: &mut bevy_app::App) {
    #[cfg(feature = "solari")]
    systems::build(app);
    #[cfg(not(feature = "solari"))]
    let _ = app;
}

#[cfg(feature = "solari")]
pub use systems::{FittedMesh, SkippedDeformingMesh, UnfittedMesh};

#[cfg(feature = "solari")]
mod systems {
    use super::{fit, Fitted, Unfit};
    use crate::{apply::GraphicsApplySystems, ResolvedGraphics};
    use bevy_app::{App, PostUpdate};
    use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
    use bevy_ecs::prelude::*;
    use bevy_material::AlphaMode;
    use bevy_mesh::{
        morph::MorphWeights, skinning::SkinnedMesh, Mesh, Mesh3d, MeshRaytracingFlags,
    };
    use bevy_pbr::{MeshMaterial3d, StandardMaterial};
    use bevy_platform::collections::HashMap;
    use bevy_solari::scene::RaytracingMesh3d;
    use tracing::warn;

    pub(super) fn build(app: &mut App) {
        app.init_resource::<FittedMeshes>()
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
    );
    type Seen = Or<(
        With<FittedMesh>,
        With<UnfittedMesh>,
        With<SkippedDeformingMesh>,
    )>;
    type Retargeted = Or<(Changed<Mesh3d>, Changed<MeshMaterial3d<StandardMaterial>>)>;

    /// The `BLAS` kind `bevy_solari` picks for a material.
    fn needed_flags(material: &StandardMaterial) -> MeshRaytracingFlags {
        match material.alpha_mode {
            AlphaMode::Mask(_) | AlphaMode::AlphaToCoverage => MeshRaytracingFlags::NON_OPAQUE,
            _ => MeshRaytracingFlags::OPAQUE,
        }
    }

    fn forget(mut entity: EntityCommands, was_fitted: bool) {
        entity.remove::<(FittedMesh, UnfittedMesh, SkippedDeformingMesh)>();
        if was_fitted {
            entity.remove::<RaytracingMesh3d>();
        }
    }

    pub(super) fn fit_meshes(
        mut resolved: ResMut<ResolvedGraphics>,
        mut fitted_meshes: ResMut<FittedMeshes>,
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
        mut commands: Commands,
    ) {
        if !resolved.rung.is_traced() {
            mesh_events.clear();
            for (entity, fitted, _) in &seen {
                forget(commands.entity(entity), fitted.is_some());
            }
            fitted_meshes.0.clear();
            set_counts(&mut resolved, 0, 0);
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

        let (mut newly_unfitted, mut newly_skipped) = (0, 0);
        for (entity, mesh_3d, material_3d, skinned, morphed) in &unseen {
            // An asset that has not loaded yet is tried again next frame.
            let (Some(mesh), Some(material)) =
                (meshes.get(&mesh_3d.0), materials.get(&material_3d.0))
            else {
                continue;
            };
            if skinned || morphed || mesh.try_has_morph_targets().unwrap_or(false) {
                commands.entity(entity).insert(SkippedDeformingMesh);
                newly_skipped += 1;
                continue;
            }

            let source = mesh_3d.0.id();
            let key = (source, needed_flags(material));
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
