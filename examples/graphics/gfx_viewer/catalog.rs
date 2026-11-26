//! The scenes the viewer can open: the testbed's own (in code), the downloaded ones listed in `tools/scenes.ron`
//! (the format is documented at the top of that file), and a glTF file given with `--scene FILE` or typed into
//! the Scene panel.

use bevy::prelude::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The testbed scenes the viewer offers, in the list's order.
pub const BUILTINS: [&str; 5] = [
    "interior",
    "many-lights",
    "contract",
    "rooms",
    "two-rooms-door",
];

/// The manifest as this build knew it: used when `tools/scenes.ron` is not found next to the binary's
/// repository (`--scenes-manifest` overrides both).
const BUILTIN_MANIFEST: &str = include_str!("../../../tools/scenes.ron");

/// `tools/scenes.ron`.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub scenes: Vec<ManifestScene>,
}

/// One entry of `tools/scenes.ron`.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ManifestScene {
    pub id: String,
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub base: Base,
    pub licence: String,
    pub non_commercial: bool,
    #[serde(default)]
    pub fetch: Option<String>,
    #[serde(default)]
    pub exposure: Option<f32>,
    #[serde(default)]
    pub camera: Option<CameraStart>,
    #[serde(default)]
    pub sun: SunChoice,
    #[serde(default)]
    pub scale: Option<f32>,
    #[serde(default)]
    pub extra: Vec<ExtraFile>,
}

/// What a manifest path is relative to.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Base {
    /// `~/.local/share/bevy-graficos/scenes/`.
    #[default]
    Scenes,
    /// This repository's `assets/`.
    RepoAssets,
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(deny_unknown_fields)]
pub struct CameraStart {
    pub position: (f32, f32, f32),
    pub look_at: (f32, f32, f32),
}

/// The sun a glTF scene gets.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub enum SunChoice {
    /// The viewer's default sun and sky when the file has no light of its own.
    #[default]
    Auto,
    /// No sun and no sky.
    Off,
    /// This sun (and the sky) unless the file has a directional light.
    Sun {
        direction: (f32, f32, f32),
        illuminance: f32,
    },
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ExtraFile {
    pub path: String,
    #[serde(default)]
    pub translation: Option<(f32, f32, f32)>,
    #[serde(default)]
    pub rotation_y: Option<f32>,
    #[serde(default)]
    pub scale: Option<f32>,
}

/// A glTF scene: one or more files, each with its place.
#[derive(Clone, Debug, PartialEq)]
pub struct GltfScene {
    /// Each file and its transform.
    pub files: Vec<(PathBuf, Transform)>,
    /// EV100 when the scene asks for one.
    pub exposure: Option<f32>,
    /// Where the camera starts, when the scene says.
    pub camera: Option<Transform>,
    pub sun: SunChoice,
}

/// Where a scene comes from.
#[derive(Clone, Debug)]
pub enum Source {
    /// A testbed scene, built in code (`gfx_testbed/scenes.rs`, `contract.rs`, `baked.rs`).
    Builtin(&'static str),
    /// glTF files on disk.
    Gltf(GltfScene),
}

/// One scene of the list.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Its id: what `--scene` takes; also its bake folder's name.
    pub name: String,
    /// What the list shows.
    pub title: String,
    /// Its licence in one line (`None` for the testbed's scenes, which are this repository's).
    pub licence: Option<String>,
    /// It may not be used in a commercial product.
    pub non_commercial: bool,
    /// Every file it needs is there.
    pub available: bool,
    /// What to run when it is not.
    pub fetch: Option<String>,
    pub source: Source,
}

/// Every scene the viewer can open.
#[derive(Resource, Clone, Debug)]
pub struct Catalog {
    pub entries: Vec<Entry>,
    /// Which manifest was read.
    pub manifest: String,
}

impl Catalog {
    /// The testbed scenes, then every scene of `manifest` (missing files marked unavailable), then `extra` (a
    /// file given with `--scene`) if no entry loads it already.
    pub fn build(
        scenes_dir: Option<&Path>,
        manifest: Option<&Path>,
        extra: Option<&Path>,
    ) -> Result<Self, String> {
        let mut entries: Vec<Entry> = BUILTINS
            .iter()
            .map(|name| Entry {
                name: (*name).to_owned(),
                title: (*name).to_owned(),
                licence: None,
                non_commercial: false,
                available: true,
                fetch: None,
                source: Source::Builtin(name),
            })
            .collect();
        let (text, origin) = match manifest.map(Path::to_owned).or_else(repo_manifest) {
            Some(file) => (
                std::fs::read_to_string(&file)
                    .map_err(|error| format!("cannot read {}: {error}", file.display()))?,
                file.display().to_string(),
            ),
            None => (
                BUILTIN_MANIFEST.to_owned(),
                "tools/scenes.ron (built in)".to_owned(),
            ),
        };
        let parsed = parse_manifest(&text).map_err(|error| format!("{origin}: {error}"))?;
        let assets = asset_root();
        for scene in parsed.scenes {
            if entries.iter().any(|entry| entry.name == scene.id) {
                return Err(format!("{origin}: two scenes have the id {}", scene.id));
            }
            let base = match scene.base {
                Base::Scenes => scenes_dir.map(Path::to_owned),
                Base::RepoAssets => assets.clone(),
            };
            let resolve = |path: &str| base.as_ref().map(|base| base.join(path));
            let root = Transform::from_scale(Vec3::splat(scene.scale.unwrap_or(1.0)));
            let mut files = vec![(resolve(&scene.path), root)];
            for extra in &scene.extra {
                let (x, y, z) = extra.translation.unwrap_or_default();
                files.push((
                    resolve(&extra.path),
                    Transform::from_xyz(x, y, z)
                        .with_rotation(Quat::from_rotation_y(
                            extra.rotation_y.unwrap_or(0.0).to_radians(),
                        ))
                        .with_scale(Vec3::splat(extra.scale.unwrap_or(1.0))),
                ));
            }
            let available = files
                .iter()
                .all(|(file, _)| file.as_ref().is_some_and(|file| file.is_file()));
            let files = files
                .into_iter()
                .map(|(file, transform)| (file.unwrap_or_default(), transform))
                .collect();
            entries.push(Entry {
                name: scene.id,
                title: scene.name,
                licence: Some(scene.licence),
                non_commercial: scene.non_commercial,
                available,
                fetch: scene.fetch,
                source: Source::Gltf(GltfScene {
                    files,
                    exposure: scene.exposure,
                    camera: scene.camera.map(|camera| {
                        Transform::from_translation(camera.position.into())
                            .looking_at(camera.look_at.into(), Vec3::Y)
                    }),
                    sun: scene.sun,
                }),
            });
        }
        let mut catalog = Self {
            entries,
            manifest: origin,
        };
        if let Some(extra) = extra {
            catalog.add_file(extra);
        }
        Ok(catalog)
    }

    /// The entry that opens `file`, adding one for it when there is none. `None` when it is not a file.
    pub fn add_file(&mut self, file: &Path) -> Option<usize> {
        let file = std::path::absolute(file).unwrap_or_else(|_| file.to_owned());
        if !file.is_file() {
            return None;
        }
        if let Some(index) = self.find(&file.to_string_lossy()) {
            return Some(index);
        }
        let stem = file.file_stem().map_or_else(
            || "scene".into(),
            |stem| stem.to_string_lossy().to_lowercase(),
        );
        let mut name = stem.clone();
        let mut n = 2;
        while self.entries.iter().any(|entry| entry.name == name) {
            name = format!("{stem}-{n}");
            n += 1;
        }
        self.entries.push(Entry {
            title: file
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            name,
            licence: Some("unknown: a file opened by path".into()),
            non_commercial: false,
            available: true,
            fetch: None,
            source: Source::Gltf(GltfScene {
                files: vec![(file, Transform::IDENTITY)],
                exposure: None,
                camera: None,
                sun: SunChoice::Auto,
            }),
        });
        Some(self.entries.len() - 1)
    }

    /// The entry called `name`, or the single-file glTF entry whose file is `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        let as_path = std::path::absolute(name).ok();
        self.entries.iter().position(|entry| {
            entry.name == name
                || matches!(&entry.source, Source::Gltf(scene)
                    if scene.files.len() == 1 && Some(&scene.files[0].0) == as_path.as_ref())
        })
    }
}

/// Parses `tools/scenes.ron` (RON with implicit `Some`).
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
        .from_str(text)
        .map_err(|error| error.to_string())
}

/// `tools/scenes.ron` of the repository the viewer runs from (`$BEVY_ASSET_ROOT` or `$CARGO_MANIFEST_DIR`).
fn repo_manifest() -> Option<PathBuf> {
    let base =
        std::env::var_os("BEVY_ASSET_ROOT").or_else(|| std::env::var_os("CARGO_MANIFEST_DIR"))?;
    Some(PathBuf::from(base).join("tools/scenes.ron")).filter(|file| file.is_file())
}

/// The folder Bevy reads `assets/` from: `$BEVY_ASSET_ROOT/assets`, else `$CARGO_MANIFEST_DIR/assets`, else the
/// executable's folder's `assets`, as `bevy_asset`'s file reader looks for it.
pub fn asset_root() -> Option<PathBuf> {
    let base = std::env::var_os("BEVY_ASSET_ROOT")
        .or_else(|| std::env::var_os("CARGO_MANIFEST_DIR"))
        .map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok()?.parent().map(Path::to_owned))?;
    Some(base.join("assets"))
}

/// The default folder of `tools/fetch-scenes.sh`.
pub fn default_scenes_dir() -> Option<PathBuf> {
    data_dir().map(|dir| dir.join("scenes"))
}

/// `$BEVY_GFX_SDK_DIR`, default `~/.local/share/bevy-graficos`: where the fork's tools keep what is never
/// committed.
pub fn data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("BEVY_GFX_SDK_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    let home = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let home = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"));
    home.map(|home| home.join("bevy-graficos"))
}

/// The asset path of a file on disk, through the viewer's `fs` asset source rooted at `/`.
pub fn fs_asset_path(file: &Path) -> String {
    let text = file.to_string_lossy().replace('\\', "/");
    format!("fs://{}", text.trim_start_matches('/'))
}

/// The root of the `fs` asset source: the file system's root, so any absolute path is an asset path.
pub fn fs_root() -> String {
    #[cfg(windows)]
    {
        String::new()
    }
    #[cfg(not(windows))]
    {
        "/".into()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_repository_manifest_parses() {
        let manifest = super::parse_manifest(super::BUILTIN_MANIFEST).unwrap();
        assert!(manifest
            .scenes
            .iter()
            .any(|scene| scene.id == "bistro-exterior"));
    }
}
