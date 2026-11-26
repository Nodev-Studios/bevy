//! The apply layer: keeps [`ResolvedGraphics`] up to date and puts it onto every [`GraphicsCamera`] and onto
//! the lights, switching presets at run time with no restart.
//!
//! # What it does
//!
//! - **[`ResolvedGraphics`]** is inserted in `PreStartup`, once the capabilities are known, and recomputed in
//!   `PostUpdate` whenever [`GraphicsSettings`], [`GraphicsCapabilities`] or the output's size change. The
//!   output is the largest viewport among the active [`GraphicsCamera`]s, or the primary window before any
//!   camera has one.
//! - **Every [`GraphicsCamera`]** (a `Camera3d`) always has `Hdr`, `Msaa::Off`, the depth, motion vector and
//!   deferred prepasses, and one temporal anti-aliasing: TAA, DLSS, or an upscaler from
//!   [`backends`]. On a traced rung it also has `SolariLighting` with the rung's parameter set
//!   ([`traced`](crate::traced)) and a storage-bindable main texture; on `Raster` it has the effects of
//!   [`raster`](crate::raster). The lighting, the anti-aliasing and the denoiser on such a camera are the
//!   ladder's: it overwrites and removes them.
//! - **The effects** (ambient occlusion, reflections, contact shadows, bloom, the shadow filter) are inserted
//!   with the table's values only where the camera has none. A value the game put there itself is kept, and
//!   when a rung or an override switches that effect off the game's value is set aside and comes back with
//!   the effect.
//! - **Lights.** On a traced rung every light's shadow maps are switched off: shadows are traced, and a
//!   shadow map would only cost. On `Raster` each light has the shadow maps the game gave it, and contact
//!   shadows are switched on for the lights that cast shadow maps. What the game set is remembered on the
//!   light ([`LightShadowState`]) and restored. A flag the game changes while the ladder holds another value
//!   is taken as the game's new wish, unless it is set to the very value the ladder had written.
//! - **[`GraphicsOverrides`]**: `ssao`, `ssr`, `contact_shadows` and `bloom`
//!   switch that effect whatever the table says (the first three exist on `Raster` only); `restir` and
//!   `max_bounces` change the traced parameter set; `shadows: Some(false)` switches every shadow map off on
//!   `Raster`, while `Some(true)` is the same as `None`: a light the game left without shadows gets none.
//!
//! Removing [`GraphicsCamera`] from a camera leaves it as the ladder last shaped it.
//!
//! # What an app must know
//!
//! - **Every opaque material in the app is deferred.** `bevy_solari` sets this for the whole app, and the
//!   ladder sets it too so `Raster` looks the same in a build without ray tracing. A camera that has a depth
//!   prepass and no deferred prepass then panics when it draws a standard material; the ladder's cameras
//!   always carry both, and any other 3D camera of the game needs `DeferredPrepass` next to `DepthPrepass`.
//! - With the `solari` feature the plugin adds `SolariPlugins` unless they are already there. Add yours
//!   before `GraphicsPlugin`, or not at all.
//! - A `RectLight` is dark on `Raster` unless the app is built with the cargo feature `area_light_luts`.
//! - DLSS needs a project id: see [`dlss`](crate::dlss).
//! - Meshes are made traceable by [`meshes`](crate::meshes).

pub mod backends;

use crate::{
    raster::{raster_with, traced_bloom, RasterEffects, RasterLevel},
    resolve, GraphicsCamera, GraphicsCapabilities, GraphicsOverrides, GraphicsRung,
    GraphicsSettings, GraphicsSystems, ResolvedDenoiser, ResolvedGraphics, ResolvedQuality,
    ResolvedUpscaler,
};
use backends::{CameraBackend, GraphicsBackends};
use bevy_anti_alias::taa::TemporalAntiAliasing;
use bevy_app::{App, PostUpdate, PreStartup};
use bevy_camera::{Camera, Camera3d, CameraMainTextureUsages, CameraUpdateSystems, Hdr};
use bevy_core_pipeline::prepass::{
    DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
};
use bevy_ecs::{component::Mutable, prelude::*};
use bevy_light::{
    AtmosphereEnvironmentMapLight, DirectionalLight, PointLight, ShadowFilteringMethod, SpotLight,
};
use bevy_math::UVec2;
use bevy_pbr::{
    ContactShadows, DefaultOpaqueRendererMethod, ScreenSpaceAmbientOcclusion,
    ScreenSpaceReflections,
};
use bevy_post_process::bloom::Bloom;
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    render_resource::TextureUsages,
    view::Msaa,
};
use bevy_window::{PrimaryWindow, Window};
use tracing::{error, info};

/// The system sets of the apply layer. Both run in `PostUpdate`, in this order, after the cameras' viewports
/// are computed.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphicsApplySystems {
    /// Recomputes [`ResolvedGraphics`]. Also runs once in `PreStartup`, after
    /// [`GraphicsSystems::DetectCapabilities`].
    Resolve,
    /// Shapes the cameras, the lights and the meshes to [`ResolvedGraphics`]. The components it inserts and
    /// removes are on the entities when `PostUpdate` ends.
    Apply,
}

/// Called by [`GraphicsPlugin`](crate::GraphicsPlugin) when it is built.
pub(crate) fn build(app: &mut App) {
    app.init_resource::<GraphicsBackends>()
        // `Raster` is deferred, as the traced rungs are: screen-space reflections need it, and
        // `bevy_solari` sets the same for the whole app, so a build without it would look different.
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .configure_sets(
            PostUpdate,
            (
                GraphicsApplySystems::Resolve.after(CameraUpdateSystems),
                GraphicsApplySystems::Apply.after(GraphicsApplySystems::Resolve),
            ),
        )
        .add_systems(
            PreStartup,
            resolve_graphics
                .in_set(GraphicsApplySystems::Resolve)
                .after(GraphicsSystems::DetectCapabilities),
        )
        .add_systems(
            PostUpdate,
            (
                resolve_graphics.in_set(GraphicsApplySystems::Resolve),
                (
                    apply_to_cameras,
                    apply_to_lights::<PointLight>,
                    apply_to_lights::<SpotLight>,
                    apply_to_lights::<DirectionalLight>,
                )
                    .in_set(GraphicsApplySystems::Apply),
            ),
        );

    // Without a renderer there is nothing to trace with, and `bevy_solari`'s plugins expect one.
    #[cfg(feature = "solari")]
    if app.get_sub_app(bevy_render::RenderApp).is_some()
        && !app.is_plugin_added::<bevy_solari::scene::RaytracingScenePlugin>()
    {
        app.add_plugins(bevy_solari::SolariPlugins);
    }

    crate::meshes::build(app);
}

/// Recomputes [`ResolvedGraphics`] when something it depends on changed.
fn resolve_graphics(
    settings: Res<GraphicsSettings>,
    capabilities: Res<GraphicsCapabilities>,
    cameras: Query<&Camera, (With<GraphicsCamera>, With<Camera3d>)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    resolved: Option<ResMut<ResolvedGraphics>>,
    mut last_output: Local<Option<UVec2>>,
    mut commands: Commands,
) {
    let pixels = |size: &UVec2| u64::from(size.x) * u64::from(size.y);
    let output = cameras
        .iter()
        .filter(|camera| camera.is_active)
        .filter_map(Camera::physical_viewport_size)
        .max_by_key(pixels)
        .or_else(|| windows.iter().next().map(Window::physical_size))
        .unwrap_or(UVec2::ZERO);

    let stale = resolved.is_none()
        || settings.is_changed()
        || capabilities.is_changed()
        || *last_output != Some(output);
    if !stale {
        return;
    }
    *last_output = Some(output);

    let mut next = resolve(&settings, &capabilities, output);
    match resolved {
        Some(mut resolved) => {
            // The mesh counts are the apply layer's, not the policy's.
            next.unfitted_meshes = resolved.unfitted_meshes;
            next.skipped_deforming_meshes = resolved.skipped_deforming_meshes;
            if *resolved != next {
                log_resolved(&next, output);
                *resolved = next;
            }
        }
        None => {
            log_resolved(&next, output);
            commands.insert_resource(next);
        }
    }
}

fn log_resolved(resolved: &ResolvedGraphics, output: UVec2) {
    info!(
        "graphics: {:?}, denoiser {:?}, upscaler {:?} at {:?} for a {}x{} output",
        resolved.rung, resolved.denoiser, resolved.upscaler, resolved.quality, output.x, output.y
    );
    for fallback in &resolved.fallbacks {
        info!("graphics: {fallback}");
    }
}

/// What the apply layer last put on a [`GraphicsCamera`], so it only touches a camera when something
/// changed and can take back exactly what it added.
#[derive(Component, Default)]
pub struct AppliedGraphics {
    applied: Option<Applied>,
    ambient_occlusion: Managed<ScreenSpaceAmbientOcclusion>,
    reflections: Managed<ScreenSpaceReflections>,
    contact_shadows: Managed<ContactShadows>,
    bloom: Managed<Bloom>,
    shadow_filter: Managed<ShadowFilteringMethod>,
    /// The normal prepass came with the ladder's ambient occlusion and goes with it.
    normal_prepass_is_ours: bool,
    /// The main texture's storage usage was added for a traced rung.
    storage_binding_is_ours: bool,
    /// `bevy_solari` switched the filtering of the atmosphere's environment map off for a traced rung.
    atmosphere_was_filtered: bool,
}

/// Everything a camera's shape depends on.
#[derive(Clone, Copy, PartialEq)]
struct Applied {
    rung: GraphicsRung,
    upscaler: ResolvedUpscaler,
    denoiser: ResolvedDenoiser,
    quality: ResolvedQuality,
    raster: RasterLevel,
    overrides: GraphicsOverrides,
}

impl AppliedGraphics {
    /// The rung this camera was last shaped for. `None` until the apply layer has seen the camera.
    pub fn rung(&self) -> Option<GraphicsRung> {
        self.applied.map(|applied| applied.rung)
    }
}

/// One effect component on a camera: whether the ladder put it there, and the game's own value while the
/// ladder has the effect switched off.
struct Managed<T> {
    is_ours: bool,
    set_aside: Option<T>,
}

impl<T> Default for Managed<T> {
    fn default() -> Self {
        Self {
            is_ours: false,
            set_aside: None,
        }
    }
}

impl<T: Component + Clone> Managed<T> {
    /// Makes the camera have the effect (`wanted` is the value to use if the game gave none) or not have it.
    fn set(&mut self, camera: &mut EntityCommands, current: Option<&T>, wanted: Option<T>) {
        match (wanted, current) {
            (Some(table_value), None) => match self.set_aside.take() {
                Some(own_value) => {
                    camera.insert(own_value);
                    self.is_ours = false;
                }
                None => {
                    camera.insert(table_value);
                    self.is_ours = true;
                }
            },
            (None, Some(current)) => {
                if !self.is_ours {
                    self.set_aside = Some(current.clone());
                }
                camera.remove::<T>();
                self.is_ours = false;
            }
            (Some(_), Some(_)) | (None, None) => {}
        }
    }

    /// Removes the component if the ladder put it there, and leaves the game's alone.
    fn release(&mut self, camera: &mut EntityCommands) {
        if self.is_ours {
            camera.remove::<T>();
            self.is_ours = false;
        }
    }
}

/// The components of a camera the apply layer reads before it decides what to change.
#[derive(bevy_ecs::query::QueryData)]
struct CameraShape {
    ambient_occlusion: Option<&'static ScreenSpaceAmbientOcclusion>,
    reflections: Option<&'static ScreenSpaceReflections>,
    contact_shadows: Option<&'static ContactShadows>,
    bloom: Option<&'static Bloom>,
    shadow_filter: Option<&'static ShadowFilteringMethod>,
    normal_prepass: Has<NormalPrepass>,
    main_texture_usages: Option<&'static CameraMainTextureUsages>,
    atmosphere: Option<&'static AtmosphereEnvironmentMapLight>,
}

/// Shapes every [`GraphicsCamera`] whose shape is not the one [`ResolvedGraphics`] and the overrides ask for.
fn apply_to_cameras(
    resolved: Res<ResolvedGraphics>,
    settings: Res<GraphicsSettings>,
    backends: Res<GraphicsBackends>,
    mut cameras: Query<
        (Entity, Option<&mut AppliedGraphics>, CameraShape),
        (With<GraphicsCamera>, With<Camera3d>),
    >,
    mut commands: Commands,
) {
    let wanted = Applied {
        rung: resolved.rung,
        upscaler: resolved.upscaler,
        denoiser: resolved.denoiser,
        quality: resolved.quality,
        raster: resolved.raster,
        overrides: settings.overrides,
    };
    for (entity, state, shape) in &mut cameras {
        if state
            .as_ref()
            .is_some_and(|state| state.applied == Some(wanted))
        {
            continue;
        }
        let mut camera = commands.entity(entity);
        match state {
            Some(mut state) => {
                shape_camera(
                    &mut camera,
                    &mut state,
                    &shape,
                    &resolved,
                    wanted,
                    &backends,
                );
            }
            None => {
                let mut state = AppliedGraphics::default();
                shape_camera(
                    &mut camera,
                    &mut state,
                    &shape,
                    &resolved,
                    wanted,
                    &backends,
                );
                camera.insert(state);
            }
        }
    }
}

fn shape_camera(
    camera: &mut EntityCommands,
    state: &mut AppliedGraphics,
    shape: &CameraShapeItem,
    resolved: &ResolvedGraphics,
    wanted: Applied,
    backends: &GraphicsBackends,
) {
    let before = state.applied;
    let traced = wanted.rung.is_traced();

    // Every rung is deferred and temporally anti-aliased. `Hdr` also has to be there before the main texture
    // can be storage-bindable: the 8-bit format it has otherwise cannot be, and creating it fails.
    camera.insert((
        Hdr,
        Msaa::Off,
        DepthPrepass,
        DeferredPrepass,
        MotionVectorPrepass,
    ));
    camera.insert_if_new((TemporalJitter::default(), MipBias::default()));

    shape_lighting(camera, state, shape, before, wanted);

    // The effects.
    let effects = if traced {
        RasterEffects {
            contact_shadows: None,
            ambient_occlusion: None,
            reflections: None,
            bloom: traced_bloom(&wanted.overrides),
            ..raster_with(wanted.raster, &wanted.overrides)
        }
    } else {
        raster_with(wanted.raster, &wanted.overrides)
    };
    let ambient_occlusion = effects.ambient_occlusion.is_some();
    if ambient_occlusion && shape.ambient_occlusion.is_none() && !shape.normal_prepass {
        state.normal_prepass_is_ours = true;
    }
    state
        .ambient_occlusion
        .set(camera, shape.ambient_occlusion, effects.ambient_occlusion);
    if !ambient_occlusion && state.normal_prepass_is_ours {
        camera.remove::<NormalPrepass>();
        state.normal_prepass_is_ours = false;
    }
    state
        .reflections
        .set(camera, shape.reflections, effects.reflections);
    state
        .contact_shadows
        .set(camera, shape.contact_shadows, effects.contact_shadows);
    state.bloom.set(camera, shape.bloom, effects.bloom);
    if traced {
        // Nothing reads the filter while no shadow map is drawn.
        state.shadow_filter.release(camera);
    } else {
        state
            .shadow_filter
            .set(camera, shape.shadow_filter, Some(effects.shadow_filter));
    }

    shape_anti_aliasing(camera, resolved, before, wanted, backends);

    state.applied = Some(wanted);
}

/// The lighting of the rung and what it needs from the camera.
fn shape_lighting(
    camera: &mut EntityCommands,
    state: &mut AppliedGraphics,
    shape: &CameraShapeItem,
    before: Option<Applied>,
    wanted: Applied,
) {
    let traced = wanted.rung.is_traced();

    // `bevy_solari` writes the lit frame into the main texture from compute shaders.
    let usages = shape.main_texture_usages.copied().unwrap_or_default();
    if traced && !usages.0.contains(TextureUsages::STORAGE_BINDING) {
        camera.insert(usages.with(TextureUsages::STORAGE_BINDING));
        state.storage_binding_is_ours = true;
    } else if !traced && state.storage_binding_is_ours {
        camera.insert(CameraMainTextureUsages(
            usages.0.difference(TextureUsages::STORAGE_BINDING),
        ));
        state.storage_binding_is_ours = false;
    }

    // `bevy_solari` switches the filtering of the atmosphere's environment map off on its cameras, and
    // `Raster` lights with the filtered one.
    if let Some(atmosphere) = shape.atmosphere {
        if traced && atmosphere.filtered {
            state.atmosphere_was_filtered = true;
        } else if !traced && state.atmosphere_was_filtered {
            camera.insert(AtmosphereEnvironmentMapLight {
                filtered: true,
                ..atmosphere.clone()
            });
            state.atmosphere_was_filtered = false;
        }
    }

    #[cfg(feature = "solari")]
    {
        use bevy_core_pipeline::prepass::{DeferredPrepassDoubleBuffer, DepthPrepassDoubleBuffer};
        use bevy_solari::realtime::SolariLighting;

        // The parameter set depends on these alone; anything else changing must not cost the history.
        let lighting_inputs = |applied: Applied| {
            (
                applied.rung,
                applied.overrides.restir,
                applied.overrides.max_bounces,
            )
        };
        if before.map(lighting_inputs) != Some(lighting_inputs(wanted)) {
            match crate::traced::solari_lighting(wanted.rung, &wanted.overrides) {
                Some(lighting) => {
                    camera.insert(lighting);
                }
                None => {
                    // `bevy_solari` adds the double buffers for ReSTIR and only removes them from a
                    // camera that still has its lighting.
                    camera.remove::<(
                        SolariLighting,
                        DeferredPrepassDoubleBuffer,
                        DepthPrepassDoubleBuffer,
                    )>();
                }
            }
        }
    }
    #[cfg(not(feature = "solari"))]
    let _ = before;
}

/// The one temporal anti-aliasing or upscaler of the camera, and the denoiser.
fn shape_anti_aliasing(
    camera: &mut EntityCommands,
    resolved: &ResolvedGraphics,
    before: Option<Applied>,
    wanted: Applied,
    backends: &GraphicsBackends,
) {
    let upscaler_inputs = |applied: Applied| (applied.upscaler, applied.quality);
    if before.map(upscaler_inputs) != Some(upscaler_inputs(wanted)) {
        if wanted.upscaler == ResolvedUpscaler::Taa {
            if before.map(|applied| applied.upscaler) != Some(ResolvedUpscaler::Taa) {
                camera.insert(TemporalAntiAliasing::default());
            }
        } else {
            camera.remove::<TemporalAntiAliasing>();
        }
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        crate::dlss::apply(camera, resolved);
    }

    let was = |upscaler| before.is_some_and(|applied| applied.upscaler == upscaler);
    call_backend(
        camera,
        resolved,
        backends.tsr,
        "TSR",
        wanted.upscaler == ResolvedUpscaler::Tsr,
        was(ResolvedUpscaler::Tsr),
    );
    call_backend(
        camera,
        resolved,
        backends.fsr,
        "FSR",
        wanted.upscaler == ResolvedUpscaler::Fsr,
        was(ResolvedUpscaler::Fsr),
    );
    call_backend(
        camera,
        resolved,
        backends.nrd,
        "NRD",
        wanted.denoiser == ResolvedDenoiser::Nrd,
        before.is_some_and(|applied| applied.denoiser == ResolvedDenoiser::Nrd),
    );
}

fn call_backend(
    camera: &mut EntityCommands,
    resolved: &ResolvedGraphics,
    backend: Option<CameraBackend>,
    name: &str,
    is_wanted: bool,
    was_on: bool,
) {
    match backend {
        Some(backend) if is_wanted => (backend.insert)(camera, resolved),
        Some(backend) if was_on => (backend.remove)(camera),
        // Only a module that reports its capability without registering its backend gets here.
        None if is_wanted => error!(
            "graphics: {name} was resolved but has no backend in `GraphicsBackends`; nothing is applied for it"
        ),
        _ => {}
    }
}

/// The two shadow flags of a light.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct LightShadows {
    maps: bool,
    contact: bool,
}

/// What the game set on a light, kept while the ladder holds other values there.
#[derive(Component, Clone, Copy, Debug)]
pub struct LightShadowState {
    own: LightShadows,
    written: LightShadows,
}

impl LightShadowState {
    /// Whether the game wants this light to cast shadow maps, whatever the ladder has set on it now.
    pub fn own_shadow_maps_enabled(&self) -> bool {
        self.own.maps
    }

    /// Whether the game asked for contact shadows on this light, whatever the ladder has set on it now.
    pub fn own_contact_shadows_enabled(&self) -> bool {
        self.own.contact
    }
}

/// A light with shadow flags the ladder switches.
trait ShadowLight: Component<Mutability = Mutable> {
    fn shadows(&self) -> LightShadows;
    fn set_shadows(&mut self, shadows: LightShadows);
}

macro_rules! impl_shadow_light {
    ($($light:ty),*) => {$(
        impl ShadowLight for $light {
            fn shadows(&self) -> LightShadows {
                LightShadows {
                    maps: self.shadow_maps_enabled,
                    contact: self.contact_shadows_enabled,
                }
            }

            fn set_shadows(&mut self, shadows: LightShadows) {
                self.shadow_maps_enabled = shadows.maps;
                self.contact_shadows_enabled = shadows.contact;
            }
        }
    )*};
}

impl_shadow_light!(PointLight, SpotLight, DirectionalLight);

/// Keeps every light's shadow flags at what the rung wants, remembering what the game set.
fn apply_to_lights<L: ShadowLight>(
    resolved: Res<ResolvedGraphics>,
    settings: Res<GraphicsSettings>,
    mut lights: Query<(Entity, &mut L, Option<&mut LightShadowState>)>,
    mut commands: Commands,
) {
    let traced = resolved.rung.is_traced();
    let shadow_maps = !traced && settings.overrides.shadows != Some(false);
    let contact_shadows = !traced && raster_with(resolved.raster, &settings.overrides).contact_shadows.is_some();

    for (entity, mut light, state) in &mut lights {
        let current = light.shadows();
        // A value other than the one written here is the game's new wish.
        let own = match &state {
            Some(state) if state.written == current => state.own,
            _ => current,
        };
        let maps = own.maps && shadow_maps;
        let wanted = LightShadows {
            maps,
            // Contact shadows fill in where a shadow map lacks detail, so they follow the shadow maps.
            // The camera's `ContactShadows` decides whether they are drawn at all.
            contact: own.contact || (maps && contact_shadows),
        };
        if wanted != current {
            light.set_shadows(wanted);
        }
        match state {
            Some(mut state) => {
                if state.own != own || state.written != wanted {
                    *state = LightShadowState {
                        own,
                        written: wanted,
                    };
                }
            }
            None if wanted != own => {
                commands.entity(entity).insert(LightShadowState {
                    own,
                    written: wanted,
                });
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests;
