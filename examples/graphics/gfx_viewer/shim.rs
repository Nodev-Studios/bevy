//! What the testbed files this viewer includes (`scenes.rs`, `baked.rs`, `contract.rs`, through `#[path]`) read
//! from their crate root. They were written as modules of `gfx_testbed`, whose `main.rs` holds these items; the
//! viewer only spawns their scenes and runs `baked.rs`'s baker, so most of this is never read at run time. The
//! fields are the ones those files use, under the same names, so the files compile here unchanged.
//! `gfx_testbed/main.rs` documents each of them.

#![allow(
    missing_docs,
    reason = "each item is documented in gfx_testbed/main.rs, whose stand-in this is"
)]

use bevy::prelude::*;
use serde_json::{Map, Value};
use std::{path::PathBuf, str::FromStr};

/// The command line, with each flag removed by whoever owns it. The same as the testbed's.
pub struct Flags {
    pub args: Vec<String>,
    pub errors: Vec<String>,
}

impl Flags {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            args,
            errors: Vec::new(),
        }
    }

    /// Takes a flag without a value; `true` when it was given.
    pub fn switch(&mut self, name: &str) -> bool {
        match self.args.iter().position(|arg| arg == name) {
            Some(index) => {
                self.args.remove(index);
                true
            }
            None => false,
        }
    }

    /// Takes a flag and its value.
    pub fn value(&mut self, name: &str) -> Option<String> {
        let index = self.args.iter().position(|arg| arg == name)?;
        self.args.remove(index);
        if index < self.args.len() {
            Some(self.args.remove(index))
        } else {
            self.error(format!("{name} needs a value"));
            None
        }
    }

    /// Takes a flag and parses its value.
    pub fn parsed<T: FromStr>(&mut self, name: &str) -> Option<T> {
        let value = self.value(name)?;
        let parsed = value.parse().ok();
        if parsed.is_none() {
            self.error(format!("{name}: cannot read {value}"));
        }
        parsed
    }

    /// Takes a flag whose value is one of `choices`.
    pub fn choice<T: Copy>(&mut self, name: &str, choices: &[(&str, T)]) -> Option<T> {
        let value = self.value(name)?;
        let choice = choices.iter().find(|(text, _)| *text == value);
        if choice.is_none() {
            let texts: Vec<&str> = choices.iter().map(|(text, _)| *text).collect();
            self.error(format!("{name} takes {}, not {value}", texts.join("|")));
        }
        choice.map(|(_, choice)| *choice)
    }

    /// Records a usage error; the app does not start.
    pub fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }
}

/// The testbed's raw lighting modes. The viewer always hands its camera to the ladder: [`Mode::Raster`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Raster,
    Solari,
    Reference,
}

/// The testbed's flags that the included files read. `scene` is the viewer's current scene, kept up to date:
/// `baked.rs` reads it to place the scene's bake regions.
#[derive(Resource, Clone)]
pub struct Args {
    pub scene: String,
    pub mode: Mode,
    pub res: UVec2,
    pub warmup: u32,
    pub frames: u32,
    pub orbit: bool,
    pub pose: u32,
    pub taa: bool,
    pub dlss_rr: bool,
    pub dlss_sr: bool,
    pub out: PathBuf,
}

impl Args {
    pub fn for_scene(scene: &str) -> Self {
        Self {
            scene: scene.to_owned(),
            mode: Mode::Raster,
            res: UVec2::new(1280, 720),
            warmup: 0,
            frames: 1,
            orbit: false,
            pose: 0,
            taa: false,
            dlss_rr: false,
            dlss_sr: false,
            out: PathBuf::from("."),
        }
    }
}

/// What a testbed scene tells its app about itself.
pub struct Scene {
    /// Where the camera stands at orbit frame `k`. Frame 0 is where the viewer starts.
    pub orbit: fn(u32) -> Transform,
    /// The camera's vertical field of view, in radians.
    pub fov: f32,
    /// Assets the scene loads from disk.
    pub assets: Vec<UntypedHandle>,
}

#[derive(Resource)]
pub struct ActiveScene(pub Scene);

/// The testbed's camera marker. The viewer's camera does not carry it, so the testbed's measuring systems
/// (never added here) would see no camera.
#[derive(Component)]
pub struct TestbedCamera;

/// The fixture a scene marks as its switchable lamp: the viewer's `L` switches it.
#[derive(Component)]
pub struct ToggleLight;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Loading,
    Running,
    Finishing,
    Done,
}

#[derive(Resource)]
pub struct Run {
    pub phase: Phase,
    pub frame: u32,
}

#[derive(Resource, Default)]
pub struct Report {
    pub traced: bool,
    pub failures: Vec<String>,
    pub sections: Map<String, Value>,
}

pub const ORBIT_CAPTURES: u32 = 8;

/// The testbed's frame clock, named by `baked.rs` and `contract.rs` only to order their systems, which the
/// viewer never adds.
pub fn drive() {}

/// The testbed's report writer; see [`drive`].
pub fn finish() {}
