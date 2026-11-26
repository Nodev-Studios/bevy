//! Planar reflections: flat mirrors that reflect the whole scene on a rasterized, deferred view.
//!
//! Screen-space reflections only find what is on screen, and a reflection probe is a blurred cube seen from
//! one point, so neither shows a flat mirror right. A planar reflection renders the scene a second time, from
//! the camera mirrored about the mirror's plane, into a texture, and the mirror's pixels take their specular
//! reflection from it.
//!
//! # Use
//!
//! - Put [`PlanarReflections`] on a `Camera3d` with a [`DeferredPrepass`]. It says how many mirrors that
//!   camera reflects at most, and at what resolution.
//! - Put [`PlanarMirror`] on each mirror: an entity with a `Mesh3d` whose geometry is flat (every normal
//!   within [`PLANAR_NORMAL_COS`] of the others, every vertex on the plane). A mesh that is not flat is
//!   reported once and left alone. [`NoPlanarMirror`] is for code that marks mirrors by itself (such as
//!   `bevy_graphics`' detection) and must leave a mesh alone.
//!
//! # How it works
//!
//! Every frame, for each camera with [`PlanarReflections`], the mirrors in its frustum that face it are
//! ranked by the share of the screen they cover, and the largest [`PlanarReflections::max_mirrors`] get a
//! *mirror camera*: a `Camera3d` rendering into a texture at [`PlanarReflections::resolution_scale`] of the
//! camera's viewport, placed at the camera's reflection about the mirror's plane (so its culling is
//! inverted), with an oblique near plane on the mirror ([`PerspectiveProjection::near_clip_plane`]) so that
//! nothing behind the mirror shows. The mirror camera is a cheap view: no screen-space effects, no bloom, no
//! tonemapping (the texture keeps the light as it is), no anti-aliasing, and shadow maps filtered with the
//! cheapest filter. Point and spot light shadow maps are shared by every view; a directional light draws its
//! cascades for the mirror camera too. A mirror camera has no planar reflections itself: a mirror seen in a
//! mirror shows what it shows without them (screen-space reflections are off there, so its probe or
//! environment map).
//!
//! The deferred lighting pass of the main camera finds its mirror pixels geometrically (on one of the chosen
//! mirrors' planes, inside its rectangle, normal facing out) and replaces their environment specular with the
//! mirror camera's texture, sampled where the pixel's position projects in the mirror camera, shifted by the
//! normal map's bend ([`PlanarReflections::normal_distortion`]), and weighted by the same split-sum Fresnel
//! the environment map uses. The screen-space reflection pass leaves those pixels alone. A view with no
//! mirror chosen this frame compiles and runs the passes exactly as before: the planar code is behind a
//! shader definition that only such views get.
//!
//! # Limits
//!
//! - Perspective cameras only; deferred materials only (the main camera's forward materials, and
//!   transparent ones, keep their usual reflections).
//! - The mirror's roughness does not blur the reflection: it is meant for mirrors, roughness near zero.
//! - A camera's [`Atmosphere`](crate::Atmosphere) is not copied to its mirror cameras (each would compute its
//!   own); a `Skybox`, environment map, ambient light, fog and exposure are.

mod render;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetId, Assets, Handle, RenderAssetUsages};
use bevy_camera::{
    primitives::Aabb,
    visibility::{InheritedVisibility, RenderLayers, VisibilitySystems},
    Camera, Camera3d, CameraProjection, CameraUpdateSystems, Exposure, Hdr, PerspectiveProjection,
    Projection, RenderTarget, SubCameraView, Viewport,
};
use bevy_core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_light::{
    AmbientLight, EnvironmentMapLight, ShadowFilteringMethod, SimulationLightSystems, Skybox,
};
use bevy_math::{Affine3A, Mat4, URect, UVec2, Vec3, Vec3A, Vec4, Vec4Swizzles};
use bevy_mesh::{Mesh, Mesh3d, VertexAttributeValues};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    view::{DebandDither, Msaa, Tonemapping},
};
use bevy_shader::load_shader_library;
use bevy_transform::{components::GlobalTransform, components::Transform, TransformSystems};
use tracing::warn;

use crate::DistanceFog;

pub use render::{
    planar_reflections_layout, ExtractedPlanarMirrors, PlanarReflectionsBindGroup,
    PLANAR_REFLECTIONS_SHADER_DEF,
};

/// The most mirrors a camera can reflect in one frame: the number of mirror textures its passes bind.
pub const MAX_PLANAR_MIRRORS: usize = 4;

/// The cosine of the largest angle between a mirror mesh's normals and their mean. 0.9995 is about 1.8°:
/// tight enough that a gently curved surface is not taken for a mirror (its reflection would be wrong), loose
/// enough for normals stored at 8 or 16 bits. // to tune
pub const PLANAR_NORMAL_COS: f32 = 0.9995;

/// How far a mirror mesh's vertices may lie off its plane, as a share of the rectangle's diagonal. // to tune
pub const PLANAR_POSITION_TOLERANCE: f32 = 0.002;

/// Adds planar reflections. Part of [`PbrPlugin`](crate::PbrPlugin).
pub struct PlanarReflectionPlugin;

impl Plugin for PlanarReflectionPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "planar_reflection.wesl");

        app.add_systems(
            PostUpdate,
            (
                measure_mirror_meshes,
                update_mirror_cameras,
                (
                    copy_to_mirror_cameras::<Exposure>,
                    copy_to_mirror_cameras::<EnvironmentMapLight>,
                    copy_to_mirror_cameras::<Skybox>,
                    copy_to_mirror_cameras::<AmbientLight>,
                    copy_to_mirror_cameras::<DistanceFog>,
                    copy_to_mirror_cameras::<RenderLayers>,
                ),
            )
                .chain()
                // The mirror cameras follow the main camera's transform of this frame, and must be placed
                // before their projections, frusta, visibility and shadow cascades are computed.
                .after(TransformSystems::Propagate)
                .before(CameraUpdateSystems)
                .before(VisibilitySystems::UpdateFrusta)
                .before(SimulationLightSystems::UpdateDirectionalLightCascades),
        );

        render::build(app);
    }

    fn finish(&self, app: &mut App) {
        render::finish(app);
    }
}

/// Makes a mesh a flat mirror for the cameras with [`PlanarReflections`].
///
/// The entity needs a `Mesh3d` whose geometry is flat; see the [module docs](self).
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone)]
pub struct PlanarMirror;

/// Keeps code that finds mirrors by itself (such as `bevy_graphics`' detection) from making this mesh a
/// [`PlanarMirror`]. It does not remove a `PlanarMirror` the app put there.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default, Clone)]
pub struct NoPlanarMirror;

/// Gives a camera planar reflections on the [`PlanarMirror`]s it sees. See the [module docs](self).
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(DepthPrepass, DeferredPrepass)]
pub struct PlanarReflections {
    /// The most mirrors rendered for this camera in one frame, the largest on screen first; at most
    /// [`MAX_PLANAR_MIRRORS`]. The others keep the reflections they would have without this component.
    pub max_mirrors: usize,
    /// The mirror textures' size as a share of the camera's viewport on each axis.
    pub resolution_scale: f32,
    /// How far a normal map's bend moves the reflection, per unit of bend and metre of distance from the
    /// camera, in world units.
    pub normal_distortion: f32,
    /// The smallest share of the screen a mirror must cover to be rendered.
    pub min_screen_coverage: f32,
}

impl Default for PlanarReflections {
    fn default() -> Self {
        Self {
            // to tune: two mirrors a view covers a bathroom's wall mirror and a glossy counter, or a mirror
            // and its reflection in a window; each one is a second scene render.
            max_mirrors: 2,
            // to tune: half the viewport on each axis is a quarter of the pixels. A mirror is sharp; lower than
            // this the reflection of thin things shimmers.
            resolution_scale: 0.5,
            // to tune: small, so a normal map ripples the reflection without tearing it apart.
            normal_distortion: 0.02,
            // to tune: 0.2 % of the screen is a 100x40 pixel mirror at 1080p.
            min_screen_coverage: 0.002,
        }
    }
}

/// A camera that renders a mirror's reflection for a camera with [`PlanarReflections`]. Spawned, moved and
/// despawned by this module.
#[derive(Component, Clone, Copy, Debug)]
pub struct PlanarMirrorCamera {
    /// The camera this one renders a reflection for.
    pub main: Entity,
    /// Its place in the main camera's list of mirror cameras.
    pub slot: usize,
}

/// The mirror cameras of a camera with [`PlanarReflections`], reused from frame to frame.
#[derive(Component, Clone, Debug, Default)]
pub struct PlanarMirrorCameras(Vec<MirrorCameraSlot>);

#[derive(Clone, Debug)]
struct MirrorCameraSlot {
    camera: Entity,
    image: Handle<Image>,
    size: UVec2,
}

/// What a camera with [`PlanarReflections`] reflects this frame: one entry per mirror camera that renders.
/// Written every frame; read by the render world.
#[derive(Component, Clone, Debug, Default)]
pub struct ViewPlanarMirrors {
    /// The mirrors, largest on screen first.
    pub mirrors: Vec<ViewPlanarMirror>,
    /// [`PlanarReflections::normal_distortion`].
    pub normal_distortion: f32,
}

/// One mirror a camera reflects this frame.
#[derive(Clone, Debug)]
pub struct ViewPlanarMirror {
    /// The mirror's entity.
    pub mirror: Entity,
    /// The texture its mirror camera renders into.
    pub image: Handle<Image>,
    /// The mirror camera's projection times its view: where a point on the mirror lands in the texture.
    pub clip_from_world: Mat4,
    /// The part of the texture the mirror camera rendered, as `(min u, min v, max u, max v)`: the mirror's
    /// rectangle on screen and a margin ([`texture_rect`]).
    pub uv_rect: Vec4,
    /// The mirror in the world.
    pub rect: WorldMirror,
}

/// A mirror mesh's rectangle in the mesh's own space, and the mesh it was measured on.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PlanarMirrorGeometry {
    /// The mesh measured.
    pub mesh: AssetId<Mesh>,
    /// Its rectangle, or `None` if the mesh is not flat.
    pub rect: Option<MirrorRect>,
}

/// The rectangle a flat mesh covers, in the mesh's space: the smallest one, along two axes of the plane, that
/// holds every vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirrorRect {
    /// The rectangle's centre, on the plane.
    pub center: Vec3,
    /// The unit normal, the mean of the mesh's normals: the side that reflects.
    pub normal: Vec3,
    /// Half the rectangle along its first axis.
    pub half_u: Vec3,
    /// Half the rectangle along its second axis.
    pub half_v: Vec3,
}

/// A mirror's rectangle in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldMirror {
    /// The plane `n · x + d = 0`, with `n` of unit length and pointing out of the reflecting side.
    pub plane: Vec4,
    /// The rectangle's centre.
    pub center: Vec3,
    /// Half the rectangle along its first axis.
    pub half_u: Vec3,
    /// Half the rectangle along its second axis.
    pub half_v: Vec3,
}

/// Measures a mesh as a mirror: its rectangle if it is flat (every normal within [`PLANAR_NORMAL_COS`] of the
/// mean normal and every vertex within [`PLANAR_POSITION_TOLERANCE`] of the plane), `None` otherwise or if
/// the mesh has no positions or normals in the main world.
pub fn measure_mirror_mesh(mesh: &Mesh) -> Option<MirrorRect> {
    let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return None;
    };
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else {
        return None;
    };
    if positions.len() < 3 || normals.len() != positions.len() {
        return None;
    }

    let normal = normals
        .iter()
        .map(|n| Vec3::from_array(*n).normalize_or_zero())
        .sum::<Vec3>()
        .normalize_or_zero();
    if normal == Vec3::ZERO
        || normals
            .iter()
            .any(|n| Vec3::from_array(*n).normalize_or_zero().dot(normal) < PLANAR_NORMAL_COS)
    {
        return None;
    }

    let (u, v) = normal.any_orthonormal_pair();
    let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for position in positions {
        let p = Vec3::from_array(*position);
        let local = Vec3::new(p.dot(u), p.dot(v), p.dot(normal));
        min = min.min(local);
        max = max.max(local);
    }
    let extent = max - min;
    let diagonal = extent.truncate().length();
    if diagonal <= 0.0 || extent.z > PLANAR_POSITION_TOLERANCE * diagonal {
        return None;
    }
    let mid = (min + max) * 0.5;
    Some(MirrorRect {
        center: u * mid.x + v * mid.y + normal * mid.z,
        normal,
        half_u: u * (extent.x * 0.5),
        half_v: v * (extent.y * 0.5),
    })
}

impl MirrorRect {
    /// The rectangle placed in the world by `world_from_local`; `None` if the transform flattens it.
    pub fn to_world(&self, world_from_local: &Affine3A) -> Option<WorldMirror> {
        let center = world_from_local.transform_point3(self.center);
        let half_u = world_from_local.transform_vector3(self.half_u);
        let half_v = world_from_local.transform_vector3(self.half_v);
        // Normals transform by the inverse transpose; its direction only decides which side reflects.
        let normal_hint: Vec3 =
            (world_from_local.matrix3.inverse().transpose() * Vec3A::from(self.normal)).into();
        let mut normal = half_u.cross(half_v).normalize_or_zero();
        if normal == Vec3::ZERO || !normal.is_finite() {
            return None;
        }
        if normal.dot(normal_hint) < 0.0 {
            normal = -normal;
        }
        Some(WorldMirror {
            plane: normal.extend(-normal.dot(center)),
            center,
            half_u,
            half_v,
        })
    }
}

impl WorldMirror {
    /// The unit normal of the reflecting side.
    pub fn normal(&self) -> Vec3 {
        self.plane.xyz()
    }

    /// The signed distance from the plane, positive on the reflecting side.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.plane.xyz().dot(point) + self.plane.w
    }

    /// The rectangle's four corners.
    pub fn corners(&self) -> [Vec3; 4] {
        let (c, u, v) = (self.center, self.half_u, self.half_v);
        [c - u - v, c + u - v, c + u + v, c - u + v]
    }

    /// The reflection about the plane, as an affine matrix.
    pub fn reflection(&self) -> Mat4 {
        let n = self.normal();
        let d = self.plane.w;
        let linear = bevy_math::Mat3::IDENTITY
            - 2.0 * bevy_math::Mat3::from_cols(n * n.x, n * n.y, n * n.z);
        Mat4::from_cols(
            linear.x_axis.extend(0.0),
            linear.y_axis.extend(0.0),
            linear.z_axis.extend(0.0),
            (-2.0 * d * n).extend(1.0),
        )
    }

    /// The vectors that give a point's coordinates in the rectangle: `(p - center) · dual_u` is -1 on one edge
    /// and 1 on the other, also when the transform shears the rectangle.
    pub fn dual_axes(&self) -> (Vec3, Vec3) {
        let n = self.normal();
        let dual_u = self.half_v.cross(n);
        let dual_v = n.cross(self.half_u);
        (
            dual_u / self.half_u.dot(dual_u),
            dual_v / self.half_v.dot(dual_v),
        )
    }
}

/// The mirror camera for a camera at `world_from_view` with `projection`: its transform (the camera's
/// reflection about the mirror, so its handedness is flipped and it must invert culling) and its projection,
/// whose near plane lies on the mirror, moved `clip_offset` world units out of it so the mirror's own surface
/// is not drawn into its reflection. `aspect_ratio` is the mirror texture's.
pub fn mirror_camera(
    world_from_view: Mat4,
    projection: &PerspectiveProjection,
    mirror: &WorldMirror,
    clip_offset: f32,
    aspect_ratio: f32,
) -> (Mat4, PerspectiveProjection) {
    let mirror_world_from_view = mirror.reflection() * world_from_view;
    // A plane's coefficients go from world to view space through the transpose of `world_from_view`. The
    // reflecting side is the side the scene to reflect is on, and the side the near plane keeps.
    let world_plane = mirror.plane - Vec4::new(0.0, 0.0, 0.0, clip_offset);
    let view_plane = mirror_world_from_view.transpose() * world_plane;
    let view_plane = view_plane / view_plane.xyz().length();
    (
        mirror_world_from_view,
        PerspectiveProjection {
            near_clip_plane: view_plane,
            aspect_ratio,
            ..projection.clone()
        },
    )
}

/// A mirror camera's projection: the [`PerspectiveProjection`] of [`mirror_camera`], whose aspect ratio is
/// the whole mirror view's. A mirror camera renders a crop of that view (its viewport and
/// [`SubCameraView`]), and a plain perspective projection would take the crop's aspect ratio when the
/// viewport changes, and with it a wrong frustum for culling. Being a custom projection also makes the
/// shaders convert depth through the matrix, which holds for an oblique near plane, rather than through the
/// near distance of a plain perspective.
#[derive(Debug, Clone)]
pub struct MirrorProjection(pub PerspectiveProjection);

impl CameraProjection for MirrorProjection {
    fn get_clip_from_view(&self) -> Mat4 {
        self.0.get_clip_from_view()
    }

    fn get_clip_from_view_for_sub(&self, sub_view: &SubCameraView) -> Mat4 {
        self.0.get_clip_from_view_for_sub(sub_view)
    }

    fn update(&mut self, _width: f32, _height: f32) {}

    fn far(&self) -> f32 {
        self.0.far
    }

    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [Vec3A; 8] {
        self.0.get_frustum_corners(z_near, z_far)
    }
}

/// The share of the screen, 0 to 1, of the box around a mirror's corners projected by `clip_from_world`.
/// A mirror the camera stands beside, with a corner behind the camera, counts as the whole screen.
pub fn screen_coverage(clip_from_world: Mat4, corners: &[Vec3; 4]) -> f32 {
    let (mut min, mut max) = (bevy_math::Vec2::splat(1.0), bevy_math::Vec2::splat(-1.0));
    for corner in corners {
        let clip = clip_from_world * corner.extend(1.0);
        if clip.w <= 1e-5 {
            return 1.0;
        }
        let ndc = clip.xy() / clip.w;
        min = min.min(ndc);
        max = max.max(ndc);
    }
    let min = min.clamp(bevy_math::Vec2::NEG_ONE, bevy_math::Vec2::ONE);
    let max = max.clamp(bevy_math::Vec2::NEG_ONE, bevy_math::Vec2::ONE);
    let size = (max - min).max(bevy_math::Vec2::ZERO);
    size.x * size.y * 0.25
}

/// The pixels of a `size` mirror texture a mirror camera renders: the box around the mirror's corners
/// projected by the mirror camera's `clip_from_world`, grown by `margin` pixels, within the texture. The whole
/// texture if a corner is behind the mirror camera.
pub fn texture_rect(clip_from_world: Mat4, corners: &[Vec3; 4], size: UVec2, margin: u32) -> URect {
    let full = URect::from_corners(UVec2::ZERO, size);
    let (mut min, mut max) = (
        bevy_math::Vec2::splat(f32::INFINITY),
        bevy_math::Vec2::splat(f32::NEG_INFINITY),
    );
    for corner in corners {
        let clip = clip_from_world * corner.extend(1.0);
        if clip.w <= 1e-5 {
            return full;
        }
        // Texture pixels: y down.
        let uv = clip.xy() / clip.w * bevy_math::Vec2::new(0.5, -0.5) + 0.5;
        let pixel = uv * size.as_vec2();
        min = min.min(pixel);
        max = max.max(pixel);
    }
    let margin = margin as f32;
    let min = (min - margin).floor().max(bevy_math::Vec2::ZERO);
    let max = (max + margin).ceil().min(size.as_vec2());
    if min.x >= max.x || min.y >= max.y {
        return full;
    }
    URect::from_corners(min.as_uvec2(), max.as_uvec2())
}

/// Picks the mirrors to render: the `max_mirrors` with the largest coverage, at least `min_coverage` each,
/// largest first. Equal coverages keep their order.
pub fn choose_mirrors<T>(
    mut candidates: Vec<(f32, T)>,
    max_mirrors: usize,
    min_coverage: f32,
) -> Vec<(f32, T)> {
    candidates.retain(|(coverage, _)| *coverage >= min_coverage && *coverage > 0.0);
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(max_mirrors.min(MAX_PLANAR_MIRRORS));
    candidates
}

/// Measures the meshes of [`PlanarMirror`]s, once per mesh they have. A mesh still loading is measured once
/// it is there.
fn measure_mirror_meshes(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    mirrors: Query<(Entity, &Mesh3d, Option<&PlanarMirrorGeometry>), With<PlanarMirror>>,
) {
    for (entity, mesh3d, geometry) in &mirrors {
        let id = mesh3d.id();
        if geometry.is_some_and(|geometry| geometry.mesh == id) {
            continue;
        }
        let Some(mesh) = meshes.get(id) else {
            continue;
        };
        let rect = measure_mirror_mesh(mesh);
        if rect.is_none() {
            warn!(
                "{entity}: a PlanarMirror's mesh is not flat (or has no positions and normals in the main \
                 world); it reflects as it would without planar reflections"
            );
        }
        commands
            .entity(entity)
            .insert(PlanarMirrorGeometry { mesh: id, rect });
    }
}

/// How far the mirror camera's near plane sits off the mirror, in world units. // to tune: 2 mm keeps the
/// mirror's own surface (and a frame flush with it) out of its reflection, and is too thin to cut anything
/// placed against the mirror off visibly.
const CLIP_OFFSET: f32 = 0.002;

/// The pixels around a mirror's rectangle its mirror camera renders too, so that bilinear filtering and a
/// normal map's bend read rendered pixels. // to tune
const RECT_MARGIN: u32 = 8;

/// The order of the first mirror camera below its main camera's: mirror cameras render first.
const MIRROR_CAMERA_ORDER_OFFSET: isize = 64;

/// Chooses each camera's mirrors and places their mirror cameras.
fn update_mirror_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut mains: Query<
        (
            Entity,
            &Camera,
            &GlobalTransform,
            &Projection,
            &PlanarReflections,
            Option<&mut PlanarMirrorCameras>,
            Option<&RenderLayers>,
        ),
        (With<Camera3d>, Without<PlanarMirrorCamera>),
    >,
    mirrors: Query<
        (
            Entity,
            &PlanarMirrorGeometry,
            &GlobalTransform,
            &InheritedVisibility,
            Option<&Aabb>,
            Option<&RenderLayers>,
        ),
        (With<PlanarMirror>, Without<Camera>),
    >,
    mut mirror_cameras: Query<
        (
            &PlanarMirrorCamera,
            &mut Camera,
            &mut Transform,
            &mut GlobalTransform,
            &mut Projection,
        ),
        Without<PlanarReflections>,
    >,
    orphans: Query<(Entity, &PlanarMirrorCamera)>,
    released: Query<(Entity, &PlanarMirrorCameras), Without<PlanarReflections>>,
) {
    // Mirror cameras whose camera is gone or no longer reflects.
    for (entity, mirror_camera) in &orphans {
        if !mains.contains(mirror_camera.main) {
            commands.entity(entity).despawn();
        }
    }
    for (entity, _) in &released {
        commands
            .entity(entity)
            .remove::<(PlanarMirrorCameras, ViewPlanarMirrors)>();
    }

    for (main, camera, main_transform, projection, settings, pool, layers) in &mut mains {
        let mut chosen = Vec::new();
        let viewport = camera.physical_viewport_size();
        if let (true, Some(viewport), Projection::Perspective(perspective)) =
            (camera.is_active, viewport, projection)
        {
            let world_from_view = main_transform.to_matrix();
            let clip_from_world = projection.get_clip_from_view() * world_from_view.inverse();
            let frustum = projection.compute_frustum(main_transform);
            let eye = main_transform.translation();
            let layers = layers.cloned().unwrap_or_default();
            let candidates = mirrors
                .iter()
                // Hidden mirrors are left out; the frustum test below stands in for view visibility, which
                // is being recomputed while this runs.
                .filter(|(_, _, _, visibility, _, mirror_layers)| {
                    visibility.get()
                        && layers.intersects(mirror_layers.unwrap_or(&RenderLayers::default()))
                })
                .filter_map(|(entity, geometry, transform, _, aabb, _)| {
                    let world = geometry.rect?.to_world(&transform.affine())?;
                    // Facing the camera, and far enough from it for the oblique projection to hold.
                    if world.signed_distance(eye) <= perspective.near {
                        return None;
                    }
                    if let Some(aabb) = aabb
                        && !frustum.intersects_obb(aabb, &transform.affine(), true, false)
                    {
                        return None;
                    }
                    Some((
                        screen_coverage(clip_from_world, &world.corners()),
                        (entity, world),
                    ))
                })
                .collect();
            let size = (viewport.as_vec2() * settings.resolution_scale.clamp(0.05, 1.0))
                .round()
                .as_uvec2()
                .max(UVec2::ONE);
            for (_, (entity, world)) in choose_mirrors(
                candidates,
                settings.max_mirrors,
                settings.min_screen_coverage,
            ) {
                let (mirror_world_from_view, mirror_projection) = mirror_camera(
                    world_from_view,
                    perspective,
                    &world,
                    CLIP_OFFSET,
                    size.x as f32 / size.y as f32,
                );
                let clip_from_world =
                    mirror_projection.get_clip_from_view() * mirror_world_from_view.inverse();
                let rect = texture_rect(clip_from_world, &world.corners(), size, RECT_MARGIN);
                chosen.push((
                    entity,
                    world,
                    mirror_world_from_view,
                    mirror_projection,
                    clip_from_world,
                    rect,
                ));
            }
            // The pool grows to the most mirrors seen and keeps its cameras, switched off, after that.
            let mut pool_slots = pool.as_ref().map(|pool| pool.0.clone()).unwrap_or_default();
            let mut view_mirrors = ViewPlanarMirrors {
                mirrors: Vec::with_capacity(chosen.len()),
                normal_distortion: settings.normal_distortion,
            };
            for (
                slot,
                (entity, world, mirror_world_from_view, mirror_projection, clip_from_world, rect),
            ) in chosen.iter().enumerate()
            {
                // Only the mirror's part of the texture is rendered: a crop of the full mirror view.
                let viewport = Viewport {
                    physical_position: rect.min,
                    physical_size: rect.size(),
                    ..Default::default()
                };
                let sub_view = SubCameraView {
                    full_size: size,
                    offset: rect.min.as_vec2(),
                    size: rect.size(),
                };
                let transform = Transform::from_matrix(*mirror_world_from_view);
                let order = camera.order - MIRROR_CAMERA_ORDER_OFFSET + slot as isize;
                if slot >= pool_slots.len() {
                    let image = images.add(mirror_image(size));
                    let camera_entity = commands
                        .spawn((
                            spawn_mirror_camera(
                                main,
                                slot,
                                order,
                                image.clone(),
                                camera,
                                viewport,
                                sub_view,
                            ),
                            transform,
                            GlobalTransform::from(transform),
                            Projection::custom(MirrorProjection(mirror_projection.clone())),
                        ))
                        .id();
                    pool_slots.push(MirrorCameraSlot {
                        camera: camera_entity,
                        image,
                        size,
                    });
                } else {
                    let pool_slot = &mut pool_slots[slot];
                    if pool_slot.size != size {
                        pool_slot.image = images.add(mirror_image(size));
                        pool_slot.size = size;
                        commands
                            .entity(pool_slot.camera)
                            .insert(RenderTarget::Image(pool_slot.image.clone().into()));
                    }
                    if let Ok((_, mut mirror_camera, mut local, mut global, mut mirror_proj)) =
                        mirror_cameras.get_mut(pool_slot.camera)
                    {
                        mirror_camera.is_active = true;
                        mirror_camera.order = order;
                        mirror_camera.clear_color = camera.clear_color;
                        if !mirror_camera.viewport.as_ref().is_some_and(|current| {
                            current.physical_position == viewport.physical_position
                                && current.physical_size == viewport.physical_size
                        }) {
                            mirror_camera.viewport = Some(viewport);
                        }
                        if mirror_camera.sub_camera_view != Some(sub_view) {
                            mirror_camera.sub_camera_view = Some(sub_view);
                        }
                        *local = transform;
                        *global = GlobalTransform::from(transform);
                        *mirror_proj =
                            Projection::custom(MirrorProjection(mirror_projection.clone()));
                    }
                }
                view_mirrors.mirrors.push(ViewPlanarMirror {
                    mirror: *entity,
                    image: pool_slots[slot].image.clone(),
                    clip_from_world: *clip_from_world,
                    uv_rect: Vec4::new(
                        rect.min.x as f32,
                        rect.min.y as f32,
                        rect.max.x as f32,
                        rect.max.y as f32,
                    ) / Vec4::new(size.x as f32, size.y as f32, size.x as f32, size.y as f32),
                    rect: *world,
                });
            }
            for pool_slot in pool_slots.iter().skip(chosen.len()) {
                if let Ok((_, mut mirror_camera, ..)) = mirror_cameras.get_mut(pool_slot.camera) {
                    mirror_camera.is_active = false;
                }
            }
            match pool {
                Some(mut pool) => {
                    if pool.0.len() != pool_slots.len()
                        || pool.0.iter().zip(&pool_slots).any(|(a, b)| a.image != b.image)
                    {
                        pool.0 = pool_slots;
                    }
                }
                None => {
                    commands.entity(main).insert(PlanarMirrorCameras(pool_slots));
                }
            }
            commands.entity(main).insert(view_mirrors);
        } else {
            if let Some(pool) = pool {
                for pool_slot in &pool.0 {
                    if let Ok((_, mut mirror_camera, ..)) =
                        mirror_cameras.get_mut(pool_slot.camera)
                    {
                        mirror_camera.is_active = false;
                    }
                }
            }
            commands.entity(main).insert(ViewPlanarMirrors::default());
        }
    }
}

/// The components of a new mirror camera, but its transform and projection.
fn spawn_mirror_camera(
    main: Entity,
    slot: usize,
    order: isize,
    image: Handle<Image>,
    main_camera: &Camera,
    viewport: Viewport,
    sub_view: SubCameraView,
) -> impl Bundle {
    (
        Camera3d::default(),
        Camera {
            order,
            // The reflection flips the view's handedness, and with it every triangle's winding.
            invert_culling: true,
            clear_color: main_camera.clear_color,
            viewport: Some(viewport),
            sub_camera_view: Some(sub_view),
            ..Default::default()
        },
        RenderTarget::Image(image.into()),
        // Light values as they are: the main camera's lighting pass adds them to its own.
        Hdr,
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        // Every opaque material of a deferred app is drawn by the deferred passes only.
        DepthPrepass,
        DeferredPrepass,
        // No temporal filter here to smooth the temporal shadow filter's noise.
        ShadowFilteringMethod::Hardware2x2,
        PlanarMirrorCamera { main, slot },
    )
}

/// A mirror texture: 16-bit float, so the light keeps its range.
fn mirror_image(size: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::COPY_DST;
    image
}

/// Gives every mirror camera the main camera's value of `T`, which lights or frames its scene the same way.
fn copy_to_mirror_cameras<T: Component + Clone>(
    mut commands: Commands,
    mains: Query<(Ref<PlanarMirrorCameras>, Option<Ref<T>>)>,
    mirror_cameras: Query<Has<T>, With<PlanarMirrorCamera>>,
) {
    for (pool, value) in &mains {
        for slot in &pool.0 {
            let Ok(has) = mirror_cameras.get(slot.camera) else {
                // Spawned this frame: copied next frame.
                continue;
            };
            match &value {
                Some(value) if value.is_changed() || pool.is_changed() || !has => {
                    commands.entity(slot.camera).insert((**value).clone());
                }
                None if has => {
                    commands.entity(slot.camera).remove::<T>();
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests;

