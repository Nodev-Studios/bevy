//! What a ray tracer sees of a level: one rule shared by the baker ([`bake`](crate::bake)) and the graphics
//! ladder's traced rungs (`bevy_graphics::meshes`), so a bake and a traced frame see the same scene.
//!
//! `bevy_solari` traces a mesh only when it is an indexed triangle list with exactly a position, a normal, a UV and
//! a tangent per vertex, and drops anything else without a word. Few meshes are like that: Bevy's primitives have
//! no tangents, and a model may carry vertex colours, a second UV set or no UVs at all. The rule:
//!
//! - **Fit** ([`fit`]): a mesh that does not fit gets a traced copy: missing normals, UVs and tangents generated,
//!   every other attribute dropped, indices added. The drawn mesh is never changed.
//! - **Skip blended materials** ([`traced_kind`]): `Blend`, `Premultiplied`, `Add` and `Multiply` (curtains,
//!   particles, smoke, blended glass) are not traced. `bevy_solari` would trace them as opaque walls (its binder
//!   has no blending), which blocks the sun and the bounce behind every window. They are still drawn.
//! - **Glass is traced non-opaque** ([`traced_kind`]): a material with `specular_transmission` that does not
//!   blend (glTF `KHR_materials_transmission`) is traced hit by hit, so `bevy_solari` lets light through it,
//!   dimmed by its transmittance, and draws it with the rasterizer's transmissive pass on top of the traced
//!   image.
//! - **Skip deforming meshes** ([`is_deforming`]): a skinned or morphed mesh would be traced frozen in its bind
//!   pose.
//!
//! Only `StandardMaterial` meshes are traced; this module has no ray tracing dependency.

use alloc::{boxed::Box, vec, vec::Vec};
use bevy_asset::RenderAssetUsages;
use bevy_material::AlphaMode;
use bevy_mesh::{
    Indices, Mesh, MeshRaytracingFlags, MeshVertexAttribute, PrimitiveTopology,
    VertexAttributeValues,
};
use bevy_pbr::StandardMaterial;
use core::fmt;

/// Why the tracer leaves a mesh out whatever its geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NotTraced {
    /// Its material blends with what is behind it (`Blend`, `Premultiplied`, `Add`, `Multiply`).
    Blended,
    /// It is skinned or morphed.
    Deforming,
}

impl fmt::Display for NotTraced {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Blended => "its material blends (glass, particles): it is drawn, not traced",
            Self::Deforming => {
                "it is skinned or morphed and would be traced frozen in its bind pose"
            }
        })
    }
}

/// The kind of acceleration structure a mesh with `material` is traced with, or [`NotTraced::Blended`].
pub fn traced_kind(material: &StandardMaterial) -> Result<MeshRaytracingFlags, NotTraced> {
    match material.alpha_mode {
        // Glass: rays look at each hit to let light through.
        AlphaMode::Opaque if is_glass(material) => Ok(MeshRaytracingFlags::NON_OPAQUE),
        AlphaMode::Opaque => Ok(MeshRaytracingFlags::OPAQUE),
        AlphaMode::Mask(_) | AlphaMode::AlphaToCoverage => Ok(MeshRaytracingFlags::NON_OPAQUE),
        AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add | AlphaMode::Multiply => {
            Err(NotTraced::Blended)
        }
    }
}

/// Whether `material` lets light through as glass: some `specular_transmission` on a dielectric (a fully
/// metallic surface transmits nothing).
pub fn is_glass(material: &StandardMaterial) -> bool {
    material.specular_transmission > 0.0 && material.metallic < 1.0
}

/// Whether a mesh deforms: an entity with a `SkinnedMesh` or `MorphWeights` (`skinned_or_morphed`), or a mesh
/// asset with morph targets.
pub fn is_deforming(mesh: &Mesh, skinned_or_morphed: bool) -> bool {
    skinned_or_morphed || mesh.try_has_morph_targets().unwrap_or(false)
}

/// The vertex attributes of a ray traceable mesh, in the order the renderer expects them.
pub const TRACED_ATTRIBUTES: [MeshVertexAttribute; 4] = [
    Mesh::ATTRIBUTE_POSITION,
    Mesh::ATTRIBUTE_NORMAL,
    Mesh::ATTRIBUTE_UV_0,
    Mesh::ATTRIBUTE_TANGENT,
];

/// A tangent for vertices that cannot have a real one. Any unit vector does: tangents are only read for
/// normal maps, which need the UVs these vertices lack.
pub const PLACEHOLDER_TANGENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

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

#[cfg(test)]
mod tests {
    use super::*;

    fn material(alpha_mode: AlphaMode, specular_transmission: f32) -> StandardMaterial {
        StandardMaterial {
            alpha_mode,
            specular_transmission,
            ..Default::default()
        }
    }

    #[test]
    fn solids_are_opaque_and_masks_are_not() {
        assert_eq!(
            traced_kind(&material(AlphaMode::Opaque, 0.0)),
            Ok(MeshRaytracingFlags::OPAQUE)
        );
        assert_eq!(
            traced_kind(&material(AlphaMode::Mask(0.5), 0.0)),
            Ok(MeshRaytracingFlags::NON_OPAQUE)
        );
        assert_eq!(
            traced_kind(&material(AlphaMode::AlphaToCoverage, 0.0)),
            Ok(MeshRaytracingFlags::NON_OPAQUE)
        );
    }

    #[test]
    fn glass_is_traced_non_opaque() {
        assert_eq!(
            traced_kind(&material(AlphaMode::Opaque, 1.0)),
            Ok(MeshRaytracingFlags::NON_OPAQUE)
        );
        assert_eq!(
            traced_kind(&material(AlphaMode::Mask(0.5), 1.0)),
            Ok(MeshRaytracingFlags::NON_OPAQUE)
        );
        let metal = StandardMaterial {
            metallic: 1.0,
            ..material(AlphaMode::Opaque, 1.0)
        };
        assert_eq!(traced_kind(&metal), Ok(MeshRaytracingFlags::OPAQUE));
    }

    #[test]
    fn blended_materials_are_not_traced_even_as_glass() {
        for alpha_mode in [
            AlphaMode::Blend,
            AlphaMode::Premultiplied,
            AlphaMode::Add,
            AlphaMode::Multiply,
        ] {
            for transmission in [0.0, 1.0] {
                assert_eq!(
                    traced_kind(&material(alpha_mode, transmission)),
                    Err(NotTraced::Blended),
                    "{alpha_mode:?} {transmission}"
                );
            }
        }
    }
}
