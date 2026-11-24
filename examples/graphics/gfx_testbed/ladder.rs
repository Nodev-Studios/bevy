//! The testbed's ladder flags: `--preset`, `--upscaler`, `--quality`, `--denoiser`.
//!
//! A stub until `bevy_graphics` exists: it takes no flags and leaves the camera to the raw `--mode` flags.
//! `main.rs` already calls every function here, and is not edited again.
//!
//! What this file can use from `main.rs`: [`Flags`] to take its flags, [`Args`] for the raw ones, [`Scene`] to
//! add scenes of its own, and at run time the resources [`crate::Run`] (which frame the run is at),
//! [`crate::Captures`] (ask for a named screenshot) and [`crate::Report`] (sections of `report.json`, whether the
//! frame is traced, and failures, any of which makes the run exit non-zero).

use crate::{Args, Flags, Scene};
use bevy::prelude::*;

/// The usage text of this file's flags, printed by `--help` after the raw flags.
pub const HELP: &str = "";

/// This file's flags.
#[derive(Resource, Clone)]
pub struct Ladder;

impl Ladder {
    /// Takes this file's flags off the command line. Flags nobody takes are a usage error.
    pub fn parse(_flags: &mut Flags) -> Self {
        Ladder
    }

    /// Adds this file's plugins and systems, after the default plugins and Solari's.
    pub fn build(&self, _app: &mut App, _args: &Args) {}

    /// Spawns the scene called `name` if this file owns it.
    pub fn scene(_name: &str, _world: &mut World) -> Option<Scene> {
        None
    }

    /// Gives the camera its rendering components when a preset was asked for, and returns `true`; `main.rs` then
    /// adds nothing of the raw modes (no lighting component, no `RaytracingMesh3d`, no change to shadow maps).
    /// Returns `false` to leave the camera to the raw `--mode` flags.
    pub fn camera(&self, _args: &Args, _camera: &mut EntityWorldMut) -> bool {
        false
    }
}
