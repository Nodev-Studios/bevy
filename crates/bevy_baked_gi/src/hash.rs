use crate::{GiBakeRegion, GiBakeSkipEmission};
use alloc::{format, vec::Vec};
use bevy_asset::{Assets, Handle};
use bevy_camera::visibility::Visibility;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::Entity,
    hierarchy::ChildOf,
    name::Name,
    query::Has,
    system::{Query, Res},
    world::World,
};
use bevy_image::Image;
use bevy_light::{DirectionalLight, PointLight, RectLight, SpotLight};
use bevy_mesh::{Indices, Mesh, Mesh3d};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_transform::components::GlobalTransform;
use core::fmt;

/// A hash of everything a bake depends on: the visible meshes (their geometry, materials and placement), the
/// lights and the regions. A bake records it; a level whose hash differs needs a new bake.
///
/// It is computed on the CPU from the main world alone, so it needs no GPU, and it does not depend on entity ids,
/// spawn order or asset ids: each entity is hashed from its content and the entity hashes are sorted before they
/// are combined. Floats are rounded first (positions to 0.1 mm, unit quantities to 1e-5), so two runs that place
/// the same level through different code paths agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneHash(pub u64);

impl fmt::Display for SceneHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Computes the [`SceneHash`] of `world`. Run it after transforms have propagated.
pub fn scene_hash(world: &mut World) -> SceneHash {
    world
        .run_system_cached(compute_scene_hash)
        .expect("the scene hash reads only components and optional resources")
}

/// FNV-1a, 64 bits: small, stable across platforms and Rust versions, which `DefaultHasher` does not promise.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.bytes(value.as_bytes());
    }

    /// Hashes `value` rounded to a multiple of `step`.
    fn f32(&mut self, value: f32, step: f32) {
        self.u64(((value / step).round() as i64) as u64);
    }

    fn color(&mut self, color: LinearRgba) {
        for c in color.to_f32_array() {
            self.f32(c, 1e-5);
        }
    }

    fn transform(&mut self, transform: &GlobalTransform) {
        let affine = transform.affine();
        for column in affine.matrix3.to_cols_array() {
            self.f32(column, 1e-5);
        }
        for t in affine.translation.to_array() {
            self.f32(t, 1e-4);
        }
    }
}

fn effectively_visible(
    entity: Entity,
    hierarchy: &Query<(Option<&Visibility>, Option<&ChildOf>)>,
) -> bool {
    let mut current = entity;
    loop {
        let Ok((visibility, parent)) = hierarchy.get(current) else {
            return true;
        };
        match visibility {
            Some(Visibility::Hidden) => return false,
            Some(Visibility::Visible) => return true,
            _ => match parent {
                Some(parent) => current = parent.parent(),
                None => return true,
            },
        }
    }
}

fn hash_texture(hasher: &mut Fnv, texture: &Option<Handle<Image>>, images: Option<&Assets<Image>>) {
    match texture {
        None => hasher.u64(0),
        Some(handle) => match handle.path() {
            Some(path) => hasher.str(&format!("{path}")),
            None => match images
                .and_then(|images| images.get(handle))
                .and_then(|i| i.data.as_ref())
            {
                Some(data) => hasher.bytes(data),
                None => hasher.str(&format!("{:?}", handle.id())),
            },
        },
    }
}

fn hash_material(
    hasher: &mut Fnv,
    material: &StandardMaterial,
    skip_emission: bool,
    images: Option<&Assets<Image>>,
) {
    hasher.color(material.base_color.into());
    hash_texture(hasher, &material.base_color_texture, images);
    // A marked entity is baked without emission, so its emission is not baked and does not count.
    if skip_emission {
        hasher.u64(1);
    } else {
        hasher.color(material.emissive);
        hash_texture(hasher, &material.emissive_texture, images);
    }
    hasher.f32(material.perceptual_roughness, 1e-5);
    hasher.f32(material.metallic, 1e-5);
    hasher.f32(material.reflectance, 1e-5);
    hash_texture(hasher, &material.metallic_roughness_texture, images);
    hash_texture(hasher, &material.normal_map_texture, images);
    hasher.u64(u64::from(material.flip_normal_map_y));
    hasher.u64(u64::from(material.double_sided));
    hasher.str(&format!("{:?}", material.alpha_mode));
    for value in material.uv_transform.to_cols_array() {
        hasher.f32(value, 1e-5);
    }
}

fn hash_mesh(hasher: &mut Fnv, mesh: &Mesh) {
    for attribute in [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
    ] {
        match mesh.try_attribute_option(attribute) {
            Ok(Some(values)) => hasher.bytes(values.get_bytes()),
            Ok(None) => hasher.u64(0),
            Err(_) => hasher.str("extracted"),
        }
    }
    match mesh.try_indices() {
        Ok(Indices::U16(indices)) => indices.iter().for_each(|i| hasher.u64(u64::from(*i))),
        Ok(Indices::U32(indices)) => indices.iter().for_each(|i| hasher.u64(u64::from(*i))),
        Err(_) => hasher.u64(0),
    }
}

/// The hash of one mesh entity: its placement, geometry and material.
pub(crate) fn mesh_entity_hash(
    transform: &GlobalTransform,
    mesh: &Mesh3d,
    material: &MeshMaterial3d<StandardMaterial>,
    skip_emission: bool,
    mesh_assets: Option<&Assets<Mesh>>,
    materials: Option<&Assets<StandardMaterial>>,
    images: Option<&Assets<Image>>,
) -> u64 {
    let mut hasher = Fnv::new();
    hasher.str("mesh");
    hasher.transform(transform);
    match mesh_assets.and_then(|assets| assets.get(&mesh.0)) {
        Some(mesh) => hash_mesh(&mut hasher, mesh),
        None => hasher.str(&format!("{:?}", mesh.0.path())),
    }
    match materials.and_then(|assets| assets.get(&material.0)) {
        Some(material) => hash_material(&mut hasher, material, skip_emission, images),
        None => hasher.str(&format!("{:?}", material.0.path())),
    }
    hasher.0
}

fn compute_scene_hash(
    meshes: Query<(
        Entity,
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
        Has<GiBakeSkipEmission>,
    )>,
    lights: Query<(
        Entity,
        &GlobalTransform,
        Option<&PointLight>,
        Option<&SpotLight>,
        Option<&RectLight>,
        Option<&DirectionalLight>,
    )>,
    regions: Query<(&GiBakeRegion, &GlobalTransform, Option<&Name>)>,
    hierarchy: Query<(Option<&Visibility>, Option<&ChildOf>)>,
    mesh_assets: Option<Res<Assets<Mesh>>>,
    materials: Option<Res<Assets<StandardMaterial>>>,
    images: Option<Res<Assets<Image>>>,
) -> SceneHash {
    let images = images.as_deref();
    let mut entity_hashes: Vec<u64> = Vec::new();

    for (entity, mesh, material, transform, skip_emission) in &meshes {
        if !effectively_visible(entity, &hierarchy) {
            continue;
        }
        entity_hashes.push(mesh_entity_hash(
            transform,
            mesh,
            material,
            skip_emission,
            mesh_assets.as_deref(),
            materials.as_deref(),
            images,
        ));
    }

    for (entity, transform, point, spot, rect, directional) in &lights {
        if !effectively_visible(entity, &hierarchy) {
            continue;
        }
        let mut hasher = Fnv::new();
        hasher.transform(transform);
        if let Some(light) = point {
            hasher.str("point");
            hasher.color(light.color.into());
            hasher.f32(light.intensity, 1e-3);
            hasher.f32(light.radius, 1e-5);
        }
        if let Some(light) = spot {
            hasher.str("spot");
            hasher.color(light.color.into());
            hasher.f32(light.intensity, 1e-3);
            hasher.f32(light.radius, 1e-5);
            hasher.f32(light.inner_angle, 1e-5);
            hasher.f32(light.outer_angle, 1e-5);
        }
        if let Some(light) = rect {
            hasher.str("rect");
            hasher.color(light.color.into());
            hasher.f32(light.intensity, 1e-3);
            hasher.f32(light.width, 1e-5);
            hasher.f32(light.height, 1e-5);
        }
        if let Some(light) = directional {
            hasher.str("directional");
            hasher.color(light.color.into());
            hasher.f32(light.illuminance, 1e-3);
        }
        if point.is_some() || spot.is_some() || rect.is_some() || directional.is_some() {
            entity_hashes.push(hasher.0);
        }
    }

    for (region, transform, name) in &regions {
        let mut hasher = Fnv::new();
        hasher.str("region");
        hasher.str(name.map(Name::as_str).unwrap_or(""));
        hasher.transform(transform);
        hasher.f32(region.spacing, 1e-5);
        for value in region.falloff.to_array() {
            hasher.f32(value, 1e-5);
        }
        entity_hashes.push(hasher.0);
    }

    entity_hashes.sort_unstable();
    let mut hasher = Fnv::new();
    hasher.u64(entity_hashes.len() as u64);
    for hash in entity_hashes {
        hasher.u64(hash);
    }
    SceneHash(hasher.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::AssetApp;
    use bevy_asset::RenderAssetUsages;
    use bevy_ecs::world::World;
    use bevy_math::Vec3;
    use bevy_mesh::PrimitiveTopology;
    use bevy_transform::components::Transform;

    fn quad(width: f32, height: f32) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [width, 0.0, 0.0],
                [width, height, 0.0],
                [0.0, height, 0.0],
            ],
        )
        .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
    }

    fn world_with_wall(offset: f32, reversed: bool) -> World {
        let mut app = bevy_app::App::new();
        app.add_plugins(bevy_asset::AssetPlugin::default())
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Image>();
        let world = app.world_mut();
        let wall = world.resource_mut::<Assets<Mesh>>().add(quad(4.0, 3.0));
        let floor = world.resource_mut::<Assets<Mesh>>().add(quad(4.0, 4.0));
        let grey = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let mut spawn = |mesh: &Handle<Mesh>, at: Vec3| {
            world.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(grey.clone()),
                GlobalTransform::from(Transform::from_translation(at)),
            ));
        };
        if reversed {
            spawn(&floor, Vec3::ZERO);
            spawn(&wall, Vec3::new(offset, 1.5, 2.0));
        } else {
            spawn(&wall, Vec3::new(offset, 1.5, 2.0));
            spawn(&floor, Vec3::ZERO);
        }
        world.spawn((
            SpotLight::default(),
            GlobalTransform::from_xyz(0.0, 2.9, 0.0),
        ));
        core::mem::take(app.world_mut())
    }

    #[test]
    fn hash_ignores_spawn_order_and_sees_a_moved_wall() {
        let a = scene_hash(&mut world_with_wall(0.0, false));
        let b = scene_hash(&mut world_with_wall(0.0, true));
        let moved = scene_hash(&mut world_with_wall(0.05, false));
        assert_eq!(a, b);
        assert_ne!(a, moved);
    }
}
