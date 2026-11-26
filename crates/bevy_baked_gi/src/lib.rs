//! Baked indirect light for rasterized rendering.
//!
//! A level marks the boxes to bake with [`GiBakeRegion`]. The baker (feature `bake`, [`bake::GiBakePlugin`])
//! traces the level with `bevy_solari` and writes one irradiance volume per region (KTX2) plus a manifest
//! ([`BakedGiManifest`], RON). At run time [`BakedGiPlugin`] turns the manifest into Bevy's own
//! [`LightProbe`](bevy_light::LightProbe) + [`IrradianceVolume`](bevy_light::IrradianceVolume) entities, so any
//! card that renders Bevy's irradiance volumes renders the bake; ray tracing is only needed to bake. Each region
//! also gets a baked reflection probe (a prefiltered cubemap, Bevy's `EnvironmentMapLight`), so rough and glossy
//! surfaces reflect the room where screen-space reflections cannot see.
//!
//! The volumes hold what a rasterizer does not draw: light that left a surface (every bounce of every light) and
//! the light of emissive meshes, but not the direct light of analytic lights. They are stored in Bevy's
//! convention: each texel is E/π for a surface facing that side (cd/m²), with no exposure applied.

extern crate alloc;

#[cfg(feature = "bake")]
pub mod bake;
mod hash;
mod manifest;
mod reflection;
mod region;
mod runtime;
mod volume;

pub use hash::{scene_hash, SceneHash};
pub use manifest::{
    light_power, BakeProvenance, BakedGiManifest, BakedGiManifestLoader, BakedLight,
    BakedLightKind, BakedReflection, BakedRegion, LightKey, MANIFEST_FORMAT,
};
pub use reflection::BakedGiReflection;
pub use region::{GiBakeRegion, GiBakeSkipEmission};
pub use runtime::{
    binding_arrays_are_usable, region_intensity, BakedGi, BakedGiPlugin, BakedGiStatus,
    BakedGiVolume,
};
pub use volume::{atlas_texel, encode_ktx2_rgba16f, VolumeData, FACE_NAMES};
