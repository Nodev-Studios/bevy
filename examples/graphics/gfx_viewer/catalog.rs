//! The scenes the viewer can open: the testbed's own, and glTF files.

use bevy::prelude::*;
use std::path::{Path, PathBuf};

/// The testbed scenes the viewer offers, in the panel's order.
pub const BUILTINS: [&str; 5] = [
    "interior",
    "many-lights",
    "contract",
    "rooms",
    "two-rooms-door",
];

/// Where a scene comes from.
#[derive(Clone, Debug)]
pub enum Source {
    /// A testbed scene, built in code (`gfx_testbed/scenes.rs`, `contract.rs`, `baked.rs`).
    Builtin(&'static str),
    /// A glTF file on disk.
    Gltf(PathBuf),
}

/// One scene of the list.
#[derive(Clone, Debug)]
pub struct Entry {
    /// What the panel and the overlay call it; also its bake folder's name.
    pub name: String,
    pub source: Source,
}

/// Every scene the viewer can open.
#[derive(Resource, Clone, Debug)]
pub struct Catalog {
    pub entries: Vec<Entry>,
}

impl Catalog {
    /// The testbed scenes, then Bevy's `FlightHelmet` when the repository's `assets/` is there, then every glTF
    /// file under `scenes_dir`, then `extra` (a file given with `--scene`) if it is not already listed.
    pub fn build(scenes_dir: Option<&Path>, extra: Option<&Path>) -> Self {
        let mut entries: Vec<Entry> = BUILTINS
            .iter()
            .map(|name| Entry {
                name: (*name).to_owned(),
                source: Source::Builtin(name),
            })
            .collect();
        let mut files = Vec::new();
        if let Some(root) = asset_root() {
            let helmet = root.join("models/FlightHelmet/FlightHelmet.gltf");
            if helmet.is_file() {
                files.push(helmet);
            }
        }
        if let Some(dir) = scenes_dir {
            let mut found = Vec::new();
            find_gltf(dir, 0, &mut found);
            found.sort();
            files.extend(found);
        }
        if let Some(extra) = extra {
            let extra = std::path::absolute(extra).unwrap_or_else(|_| extra.to_owned());
            if !files.contains(&extra) {
                files.push(extra);
            }
        }
        for file in files {
            let stem = file.file_stem().map_or_else(
                || "scene".into(),
                |stem| stem.to_string_lossy().into_owned(),
            );
            let mut name = stem.to_lowercase();
            let mut n = 2;
            while entries.iter().any(|entry| entry.name == name) {
                name = format!("{}-{n}", stem.to_lowercase());
                n += 1;
            }
            entries.push(Entry {
                name,
                source: Source::Gltf(file),
            });
        }
        Self { entries }
    }

    /// The entry called `name`, or the glTF entry whose file is `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        let as_path = std::path::absolute(name).ok();
        self.entries.iter().position(|entry| {
            entry.name == name
                || matches!(&entry.source, Source::Gltf(file) if Some(file) == as_path.as_ref())
        })
    }
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

/// glTF files under `dir`, four folders deep at most.
fn find_gltf(dir: &Path, depth: u32, found: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth < 4 {
                find_gltf(&path, depth + 1, found);
            }
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gltf") || ext.eq_ignore_ascii_case("glb"))
        {
            found.push(path);
        }
    }
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
