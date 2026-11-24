//! The testbed's checks of Solari's denoiser contract: the `--debug diffuse|specular|hitdist` views, the
//! comparison of the legacy colour with the composed colour in the same frame, and the `contract` scene.
//!
//! A stub until the contract exists: it takes no flags and adds nothing. `main.rs` already calls every function
//! here, and is not edited again.
//!
//! What this file can use from `main.rs`: [`Flags`] to take its flags, [`Args`] for the raw ones, [`Scene`] to
//! add scenes of its own, and at run time the resources [`crate::Run`] (which frame the run is at),
//! [`crate::Captures`] (ask for a named screenshot) and [`crate::Report`] (sections of `report.json` and
//! failures, any of which makes the run exit non-zero).

use crate::{Args, Flags, Scene};
use bevy::prelude::*;

/// The usage text of this file's flags, printed by `--help` after the raw flags.
pub const HELP: &str = "";

/// This file's flags.
#[derive(Resource, Clone)]
pub struct Contract;

impl Contract {
    /// Takes this file's flags off the command line. Flags nobody takes are a usage error.
    pub fn parse(_flags: &mut Flags) -> Self {
        Contract
    }

    /// Adds this file's plugins and systems, after the default plugins and Solari's.
    pub fn build(&self, _app: &mut App, _args: &Args) {}

    /// Spawns the scene called `name` if this file owns it.
    pub fn scene(_name: &str, _world: &mut World) -> Option<Scene> {
        None
    }

    /// Adds this file's components to the camera, after the raw flags or the ladder gave it theirs.
    pub fn camera(&self, _args: &Args, _camera: &mut EntityWorldMut) {}
}
