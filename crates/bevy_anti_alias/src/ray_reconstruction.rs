//! What a ray traced lighting pass and the denoiser of its output share: the textures the lighting
//! pass writes, the textures a denoiser hands back, and where a denoiser runs in the frame.
//!
//! The lighting pass (`bevy_solari`'s realtime lighting) fills [`ViewDenoiserGuideTextures`] and
//! [`ViewDenoiserInputTextures`]. A denoiser reads them in [`DenoiserSystems::Denoise`] and writes
//! its result to [`ViewDenoisedRadianceTextures`], which it inserts on the view itself. The
//! lighting pass's compose step runs in [`DenoiserSystems::Compose`] and turns the denoised
//! radiance back into the view's lit color. With no [`ViewDenoisedRadianceTextures`] on the view,
//! compose reads the noisy inputs instead.
//!
//! All textures have the size of the view's main pass (the render resolution, not the upscaled
//! one) and are indexed by pixel, like the prepass textures.

use bevy_ecs::{component::Component, schedule::SystemSet};
use bevy_render::texture::CachedTexture;

/// Hit distance, in world units, stored in [`ViewDenoiserInputTextures`] when the ray left the
/// scene. Longer rays are clamped to it.
pub const DENOISER_MISS_HIT_DISTANCE: f32 = 10000.0;

/// Depth stored in [`ViewDenoiserInputTextures::view_depth`] where the view sees no geometry.
pub const DENOISER_SKY_VIEW_DEPTH: f32 = 1.0e6;

/// Per-pixel surface attributes of a view, for a denoiser to tell surfaces apart.
///
/// A view rendered with DLSS Ray Reconstruction alone gets the attributes of the surface seen in a
/// flat perfect mirror in place of the mirror's own. A view that also writes
/// [`ViewDenoiserInputTextures`] always gets the attributes of the surface the view ray hit.
#[derive(Component)]
pub struct ViewDenoiserGuideTextures {
    /// `Rgba8Unorm`: diffuse reflectance in RGB (base color with the metallic part removed).
    pub diffuse_albedo: CachedTexture,
    /// `Rgba8Unorm`: specular reflectance integrated over the hemisphere in RGB.
    pub specular_albedo: CachedTexture,
    /// `Rgba16Float`: world-space normal in RGB, perceptual roughness in A.
    pub normal_roughness: CachedTexture,
    /// `R32Float`: depth as the depth prepass stores it (reversed, 0 where there is no geometry).
    pub depth: CachedTexture,
    /// `Rg16Float`: motion of what is seen reflected in the surface, in the units of the motion
    /// vector prepass. Equal to the surface's own motion where the surface is not smooth.
    pub specular_motion_vectors: CachedTexture,
}

/// Noisy lighting of a view, split by the lobe of the surface that reflects it, for a denoiser.
///
/// Both radiance textures hold light reflected by the surface the view ray hit, without the light
/// the surface emits itself. Radiance is multiplied by the view's exposure, like the color in the
/// view target, and divided by the reflectance of its lobe, so that surface detail stays out of
/// the signal a denoiser blurs. Values are finite and never negative.
///
/// The hit distance in each alpha channel is the length, in world units, of the ray the lighting
/// pass traced from the surface this frame, when it sampled that lobe: 0 for the lobe it did not
/// sample, and [`DENOISER_MISS_HIT_DISTANCE`] when the ray left the scene. It never is the
/// distance to a light chosen by light sampling.
#[derive(Component)]
pub struct ViewDenoiserInputTextures {
    /// `Rgba16Float`: radiance reflected by the diffuse lobe in RGB, its hit distance in A.
    pub diffuse_radiance_hit_distance: CachedTexture,
    /// `Rgba16Float`: radiance reflected by the specular lobe in RGB, its hit distance in A.
    pub specular_radiance_hit_distance: CachedTexture,
    /// `R32Float`: distance from the camera along its forward axis, in world units, positive.
    /// [`DENOISER_SKY_VIEW_DEPTH`] where there is no geometry.
    pub view_depth: CachedTexture,
}

/// The result of a denoiser, in the encoding of [`ViewDenoiserInputTextures`]' radiance.
///
/// A denoiser inserts this on the views it denoises and writes it in
/// [`DenoiserSystems::Denoise`]; only RGB is read back. The textures need
/// `TextureUsages::TEXTURE_BINDING` and a float format that can be read without filtering.
#[derive(Component)]
pub struct ViewDenoisedRadianceTextures {
    /// Denoised [`ViewDenoiserInputTextures::diffuse_radiance_hit_distance`].
    pub diffuse: CachedTexture,
    /// Denoised [`ViewDenoiserInputTextures::specular_radiance_hit_distance`].
    pub specular: CachedTexture,
}

/// Where the denoising of a ray traced lighting pass runs in the `Core3d` schedule.
///
/// The lighting pass orders both sets after itself and before the main opaque pass.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenoiserSystems {
    /// After the lighting pass wrote [`ViewDenoiserInputTextures`]. Denoisers run here.
    Denoise,
    /// After [`DenoiserSystems::Denoise`]. The lighting pass writes the view's lit color here.
    Compose,
}
