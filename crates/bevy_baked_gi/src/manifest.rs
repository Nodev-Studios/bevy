use alloc::{string::String, vec::Vec};
use bevy_asset::{io::Reader, Asset, AssetLoader, Handle, LoadContext};
use bevy_color::{Color, LinearRgba};
use bevy_image::Image;
use bevy_math::{Quat, Vec3};
use bevy_reflect::TypePath;
use bevy_transform::components::{GlobalTransform, Transform};
use serde::{Deserialize, Serialize};

/// The manifest format this crate writes and reads.
pub const MANIFEST_FORMAT: u32 = 1;

/// What one bake wrote: the regions, the lights it was baked with, and what it was baked from.
///
/// Stored as RON in a `.gi.ron` file beside the regions' KTX2 volumes. Loading it loads the volumes too.
#[derive(Asset, TypePath, Clone, Debug, Serialize, Deserialize)]
pub struct BakedGiManifest {
    /// [`MANIFEST_FORMAT`] when written.
    pub format: u32,
    /// The [`scene_hash`](crate::scene_hash) of the scene that was baked, as 16 hex digits.
    pub scene_hash: String,
    /// How the bake was made.
    pub bake: BakeProvenance,
    /// The analytic lights the bake recorded, sorted by key. A light whose intensity was 0 is not recorded.
    pub lights: Vec<BakedLight>,
    /// The regions, sorted by name.
    pub regions: Vec<BakedRegion>,
    /// The volume of each region, in `regions` order. Filled by the loader.
    #[serde(skip)]
    pub volumes: Vec<Handle<Image>>,
    /// The reflection probe's cubemap of each region that has one, in `regions` order. Filled by the loader.
    #[serde(skip)]
    pub reflections: Vec<Option<Handle<Image>>>,
}

/// How a bake was made: enough to tell two bakes apart and to repeat one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakeProvenance {
    /// `bevy_baked_gi`'s version.
    pub baker: String,
    /// The adapter's name.
    pub adapter: String,
    /// The adapter's driver and its version.
    pub driver: String,
    /// `"full"` (every bounce) or `"first-hit-emission"` (a test mode: the emission of the first surface hit).
    pub mode: String,
    /// Paths per voxel when every region used the same count; 0 when each region's count came from its noise
    /// (`BakedRegion::samples`).
    pub samples: u32,
    /// The seed of the direction rotations and of the paths.
    pub seed: u32,
    /// Paths per voxel of the two pilots that set each region's count; 0 when there were none.
    #[serde(default)]
    pub pilot_samples: u32,
    /// The noise between two rotations each region's count aims at; 0 when there were no pilots.
    #[serde(default)]
    pub target_noise: f32,
    /// The share of a voxel's first hits that are back faces above which it counts as inside geometry.
    pub invalid_back_face_share: f32,
    /// The texel format of the volumes.
    pub texel_format: String,
}

/// The analytic lights a manifest can record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BakedLightKind {
    /// A [`DirectionalLight`](bevy_light::DirectionalLight).
    Directional,
    /// A [`PointLight`](bevy_light::PointLight).
    Point,
    /// A [`SpotLight`](bevy_light::SpotLight).
    Spot,
    /// A [`RectLight`](bevy_light::RectLight).
    Rect,
}

/// How a live light is matched to a baked one: its kind and where it is, to the centimetre.
///
/// For a directional light the "position" is its direction, scaled by 100, so it too is matched to 1 %.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LightKey {
    /// The light's kind.
    pub kind: BakedLightKind,
    /// Its position in centimetres, rounded; a directional light's direction × 100.
    pub position_cm: [i32; 3],
}

impl LightKey {
    /// The key of a light of `kind` placed by `transform`.
    pub fn new(kind: BakedLightKind, transform: &GlobalTransform) -> Self {
        let point = match kind {
            BakedLightKind::Directional => transform.forward().as_vec3(),
            _ => transform.translation(),
        };
        let cm = (point * 100.0).round();
        Self {
            kind,
            position_cm: [cm.x as i32, cm.y as i32, cm.z as i32],
        }
    }
}

/// The one number a light's switching and flicker scale by: its intensity times the luminance of its colour.
pub fn light_power(color: Color, intensity: f32) -> f32 {
    let linear = LinearRgba::from(color);
    intensity * (0.2126 * linear.red + 0.7152 * linear.green + 0.0722 * linear.blue)
}

/// One analytic light as baked.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakedLight {
    /// How a live light is matched to it.
    pub key: LightKey,
    /// Its [`light_power`] when baked. Never 0.
    pub power: f32,
}

/// One baked region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakedRegion {
    /// The region's `Name`.
    pub name: String,
    /// Its volume, relative to the manifest.
    pub file: String,
    /// The box's translation.
    pub translation: [f32; 3],
    /// The box's rotation (x, y, z, w).
    pub rotation: [f32; 4],
    /// The box's size.
    pub scale: [f32; 3],
    /// The probe's falloff.
    pub falloff: [f32; 3],
    /// Voxels along each axis.
    pub resolution: [u32; 3],
    /// Voxels found inside geometry and filled from their neighbours.
    pub invalid_voxels: u32,
    /// Paths per voxel in this region.
    #[serde(default)]
    pub samples: u32,
    /// The relative RMS between the two pilots (gate 4's statistic), when pilots set `samples`; 0 otherwise.
    #[serde(default)]
    pub pilot_noise: f32,
    /// The luminance the bake gathered over the region's valid voxels and sides, summed over paths (arbitrary
    /// scale). 0 means no light reaches the region; it is then drawn at intensity 1 whatever the lights do.
    pub bounce: f64,
    /// The share of `bounce` that comes from emissive meshes and the environment, which do not switch.
    pub static_weight: f32,
    /// The share of `bounce` that comes from each of the manifest's `lights`, in that order.
    pub light_weights: Vec<f32>,
    /// The region's reflection probe, when the bake captured one. Manifests written before probes have none.
    #[serde(default)]
    pub reflection: Option<BakedReflection>,
}

/// A region's baked reflection probe: a cubemap of the light arriving at one point of the region, prefiltered for
/// rough surfaces, which Bevy reprojects onto the region's box (parallax correction).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakedReflection {
    /// The KTX2 cubemap, relative to the manifest: `Rgba16Float`, each mip level prefiltered for the perceptual
    /// roughness `level / (levels − 1)`, as Bevy's shader picks the level.
    pub file: String,
    /// Where the cubemap was captured: the box's centre, or the nearest valid voxel when the centre is inside
    /// geometry.
    pub origin: [f32; 3],
    /// Texels along a face's side at level 0.
    pub size: u32,
    /// Mip levels.
    pub levels: u32,
    /// Paths per texel at level 0.
    pub samples: u32,
}

impl BakedRegion {
    /// The box as a transform.
    pub fn transform(&self) -> Transform {
        Transform {
            translation: Vec3::from_array(self.translation),
            rotation: Quat::from_array(self.rotation),
            scale: Vec3::from_array(self.scale),
        }
    }
}

/// Loads a [`BakedGiManifest`] from a `.gi.ron` file and the volumes it names.
#[derive(Default, TypePath)]
pub struct BakedGiManifestLoader;

/// Why a manifest did not load.
#[derive(Debug, thiserror::Error)]
pub enum BakedGiManifestError {
    /// The file could not be read.
    #[error("could not read the manifest: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not a manifest.
    #[error("could not parse the manifest: {0}")]
    Ron(#[from] ron::error::SpannedError),
    /// The manifest is of another format.
    #[error("manifest format {0}, this build reads {MANIFEST_FORMAT}")]
    Format(u32),
}

impl AssetLoader for BakedGiManifestLoader {
    type Asset = BakedGiManifest;
    type Settings = ();
    type Error = BakedGiManifestError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<BakedGiManifest, BakedGiManifestError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let mut manifest: BakedGiManifest = ron::de::from_bytes(&bytes)?;
        if manifest.format != MANIFEST_FORMAT {
            return Err(BakedGiManifestError::Format(manifest.format));
        }
        // Volume files are named relative to the manifest, as glTF names its buffers.
        let manifest_path = load_context.path().clone();
        let mut load = |file: &str| match manifest_path.resolve_embed_str(file) {
            Ok(path) => load_context.load(path),
            Err(_) => load_context.load(file.to_owned()),
        };
        manifest.volumes = manifest
            .regions
            .iter()
            .map(|region| load(&region.file))
            .collect();
        manifest.reflections = manifest
            .regions
            .iter()
            .map(|region| region.reflection.as_ref().map(|r| load(&r.file)))
            .collect();
        Ok(manifest)
    }

    fn extensions(&self) -> &[&str] {
        &["gi.ron"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_keys_round_to_the_centimetre() {
        let a = LightKey::new(
            BakedLightKind::Spot,
            &GlobalTransform::from_xyz(1.004, 2.9, -0.0049),
        );
        let b = LightKey::new(
            BakedLightKind::Spot,
            &GlobalTransform::from_xyz(0.996, 2.9001, 0.0),
        );
        assert_eq!(a, b);
        assert_eq!(a.position_cm, [100, 290, 0]);
        let point = LightKey::new(
            BakedLightKind::Point,
            &GlobalTransform::from_xyz(1.0, 2.9, 0.0),
        );
        assert_ne!(a, point);
    }

    #[test]
    fn manifest_round_trips_through_ron() {
        let manifest = BakedGiManifest {
            format: MANIFEST_FORMAT,
            scene_hash: "0123456789abcdef".into(),
            bake: BakeProvenance {
                baker: "test".into(),
                adapter: "none".into(),
                driver: "none".into(),
                mode: "full".into(),
                samples: 4096,
                seed: 1,
                pilot_samples: 0,
                target_noise: 0.0,
                invalid_back_face_share: 0.25,
                texel_format: "Rgba16Float".into(),
            },
            lights: vec![BakedLight {
                key: LightKey {
                    kind: BakedLightKind::Spot,
                    position_cm: [0, 290, 0],
                },
                power: 1.5e6,
            }],
            regions: vec![BakedRegion {
                name: "a".into(),
                file: "a.ktx2".into(),
                translation: [0.0, 1.5, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [14.06, 3.2, 8.06],
                falloff: [0.0; 3],
                resolution: [43, 10, 25],
                invalid_voxels: 3,
                samples: 4096,
                pilot_noise: 0.0,
                bounce: 12.5,
                static_weight: 0.25,
                light_weights: vec![0.75],
                reflection: Some(BakedReflection {
                    file: "a.refl.ktx2".into(),
                    origin: [0.0, 1.5, 0.0],
                    size: 128,
                    levels: 6,
                    samples: 512,
                }),
            }],
            volumes: Vec::new(),
            reflections: Vec::new(),
        };
        let text = ron::ser::to_string_pretty(&manifest, Default::default()).unwrap();
        let back: BakedGiManifest = ron::de::from_str(&text).unwrap();
        assert_eq!(back.regions, manifest.regions);
        assert_eq!(back.lights, manifest.lights);
        assert_eq!(back.bake, manifest.bake);
        assert_eq!(back.scene_hash, manifest.scene_hash);
    }

    #[test]
    fn a_manifest_written_before_reflection_probes_still_loads() {
        let text = r#"(
            format: 1,
            scene_hash: "0123456789abcdef",
            bake: (baker: "old", adapter: "a", driver: "d", mode: "full", samples: 4096, seed: 0,
                   invalid_back_face_share: 0.25, texel_format: "Rgba16Float"),
            lights: [],
            regions: [(name: "a", file: "a.ktx2", translation: (0.0, 1.5, 0.0), rotation: (0.0, 0.0, 0.0, 1.0),
                       scale: (4.0, 3.2, 4.0), falloff: (0.0, 0.0, 0.0), resolution: (12, 10, 12),
                       invalid_voxels: 0, bounce: 1.0, static_weight: 1.0, light_weights: [])],
        )"#;
        let manifest: BakedGiManifest = ron::de::from_str(text).unwrap();
        assert_eq!(manifest.regions[0].reflection, None);
        assert_eq!(manifest.regions[0].samples, 0);
    }
}
