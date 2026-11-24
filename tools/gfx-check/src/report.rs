//! The parts of the testbed's `report.json` that the checks read.

use alloc::collections::BTreeMap;
use serde::Deserialize;
use std::path::Path;

/// One run of `gfx_testbed`, as it described itself.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Report {
    /// Whether the frame was lit by ray tracing (Solari or the reference tracer).
    pub traced: bool,
    /// The GPU the run used.
    pub adapter: Adapter,
    /// The camera of the run.
    pub camera: Camera,
    /// Wall-clock frame times over the measured frames; absent when the run failed before measuring.
    pub frame_ms: Option<FrameMs>,
    /// Median GPU time of each render pass, in milliseconds, keyed by the pass's diagnostic path.
    pub gpu_pass_ms: BTreeMap<String, f64>,
    /// Peak GPU memory, when wgpu could report it.
    pub gpu_memory: Option<GpuMemory>,
    /// The screenshots the run wrote.
    pub screenshots: Vec<Screenshot>,
    /// Why the run failed, in its own words; empty for a good run.
    pub failures: Vec<String>,
}

/// The adapter fields the checks use.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Adapter {
    /// The adapter's name as wgpu reports it.
    pub name: String,
    /// The driver's version text.
    pub driver_info: String,
}

/// How the camera moved.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Camera {
    /// `static` or `orbit`.
    pub kind: String,
    /// The orbit frame a static camera stands at.
    pub pose: u32,
}

/// Frame-time statistics in milliseconds.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct FrameMs {
    /// Median frame time.
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

/// GPU memory as wgpu's allocator counts it.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct GpuMemory {
    /// Largest number of bytes the allocator had reserved from the driver during the run.
    pub peak_reserved_bytes: u64,
}

/// One screenshot of a run.
#[derive(Deserialize, Default)]
#[serde(default)]
pub struct Screenshot {
    /// The capture's name, e.g. `a` or `orbit-0120`.
    pub name: String,
    /// File name inside the run's folder.
    pub file: String,
    /// The orbit frame the camera stood at when the frame was captured.
    pub orbit_frame: u32,
}

impl Report {
    /// Reads `report.json` from a run folder, or the file itself.
    pub fn read(path: &Path) -> Result<Report, String> {
        let file = if path.is_dir() {
            path.join("report.json")
        } else {
            path.to_owned()
        };
        let text = std::fs::read_to_string(&file)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", file.display()))
    }
}
