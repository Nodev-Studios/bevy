//! The testbed's ladder flags: `--preset`, `--upscaler`, `--quality`, `--denoiser`, and `--off` and `--cycle`
//! for the apply layer's own checks.
//!
//! Any of them hands the camera to `bevy_graphics`: the camera gets a [`GraphicsCamera`] and `main.rs` adds
//! nothing of the raw `--mode` flags. Without them [`GraphicsPlugin`] is not added at all, because it shapes
//! every light and mesh of the app (a traced rung switches shadow maps off) and would change the raw runs.
//!
//! `report.json` gets a `ladder` section: what was asked, what resolved (with the reason for every fallback),
//! the capabilities `bevy_graphics` read, and one entry per stage of a cycle. A run whose resolved preset is
//! not the one asked for by name fails.

use crate::{Args, Captures, Flags, Mode, PendingPipelines, Phase, Report, Run, Scene};
use bevy::{
    graphics::{
        DenoiserChoice, GraphicsCamera, GraphicsCapabilities, GraphicsOverrides, GraphicsPlugin,
        GraphicsPreset, GraphicsRung, GraphicsSettings, ResolvedGraphics, UpscaleQuality,
        UpscalerChoice,
    },
    prelude::*,
};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// The usage text of this file's flags, printed by `--help` after the raw flags.
pub const HELP: &str = "
Ladder flags. Any of them hands the camera to bevy_graphics; none of the raw --mode, --taa, --dlss-* flags then:
  --preset auto|raster|rt-lite|rt-full
                                 the rung asked for (default auto); the run fails when another one resolves
  --upscaler auto|off|tsr|dlss|fsr
                                 the upscaler asked for (default auto)
  --quality auto|native|quality|balanced|performance
                                 the internal resolution asked for (default auto)
  --denoiser auto|dlss-rr|nrd    the denoiser of the traced rungs (default auto)
  --off EFFECT,...               force effects off: shadows, ssao, ssr, contact-shadows, bloom
  --cycle                        switch raster, rt-lite, rt-full, raster in one run. Each stage starts when its
                                 pipelines are ready and is captured --stage-frames frames later as
                                 stage-K-<preset>.png, as a single run with warmup + frames = --stage-frames
                                 captures a.png. a.png comes after the cycle: give --frames enough to outlast it
  --stage-frames N               frames each stage of --cycle is held (default 120)
";

/// The stages of `--cycle`.
const CYCLE: [(&str, GraphicsPreset); 4] = [
    ("raster", GraphicsPreset::Raster),
    ("rt-lite", GraphicsPreset::RtLite),
    ("rt-full", GraphicsPreset::RtFull),
    ("raster", GraphicsPreset::Raster),
];

/// Frames after a switch before the pipeline count is trusted: the new components reach the render world and
/// queue their pipelines a frame or two later, so the count reads 0 before they are even asked for.
const SWITCH_FRAMES: u32 = 5;

/// This file's flags.
#[derive(Resource, Clone)]
pub struct Ladder {
    /// The settings asked for; `None` when no ladder flag was given and the raw flags drive the camera.
    settings: Option<GraphicsSettings>,
    /// `--cycle`: the frames each stage is held.
    cycle: Option<u32>,
}

impl Ladder {
    /// Takes this file's flags off the command line. Flags nobody takes are a usage error.
    pub fn parse(flags: &mut Flags) -> Self {
        let preset = flags.choice(
            "--preset",
            &[
                ("auto", GraphicsPreset::Auto),
                ("raster", GraphicsPreset::Raster),
                ("rt-lite", GraphicsPreset::RtLite),
                ("rt-full", GraphicsPreset::RtFull),
            ],
        );
        let upscaler = flags.choice(
            "--upscaler",
            &[
                ("auto", UpscalerChoice::Auto),
                ("off", UpscalerChoice::Off),
                ("tsr", UpscalerChoice::Tsr),
                ("dlss", UpscalerChoice::Dlss),
                ("fsr", UpscalerChoice::Fsr),
            ],
        );
        let quality = flags.choice(
            "--quality",
            &[
                ("auto", UpscaleQuality::Auto),
                ("native", UpscaleQuality::NativeAa),
                ("quality", UpscaleQuality::Quality),
                ("balanced", UpscaleQuality::Balanced),
                ("performance", UpscaleQuality::Performance),
            ],
        );
        let denoiser = flags.choice(
            "--denoiser",
            &[
                ("auto", DenoiserChoice::Auto),
                ("dlss-rr", DenoiserChoice::DlssRayReconstruction),
                ("nrd", DenoiserChoice::Nrd),
            ],
        );
        let off = flags.value("--off");
        let cycle = flags.switch("--cycle");
        let stage_frames: Option<u32> = flags.parsed("--stage-frames");

        let mut overrides = GraphicsOverrides::default();
        for effect in off.iter().flat_map(|off| off.split(',')) {
            let slot = match effect {
                "shadows" => &mut overrides.shadows,
                "ssao" => &mut overrides.ssao,
                "ssr" => &mut overrides.ssr,
                "contact-shadows" => &mut overrides.contact_shadows,
                "bloom" => &mut overrides.bloom,
                _ => {
                    flags.error(format!(
                        "--off takes shadows, ssao, ssr, contact-shadows and bloom, not {effect}"
                    ));
                    continue;
                }
            };
            *slot = Some(false);
        }
        if cycle && preset.is_some() {
            flags.error("--cycle chooses its own presets: no --preset with it");
        }
        if stage_frames.is_some() && !cycle {
            flags.error("--stage-frames needs --cycle");
        }
        if stage_frames == Some(0) {
            flags.error("--stage-frames must be at least 1");
        }

        let active = preset.is_some()
            || upscaler.is_some()
            || quality.is_some()
            || denoiser.is_some()
            || off.is_some()
            || cycle;
        Ladder {
            settings: active.then(|| GraphicsSettings {
                preset: if cycle {
                    CYCLE[0].1
                } else {
                    preset.unwrap_or_default()
                },
                upscaler: upscaler.unwrap_or_default(),
                upscale_quality: quality.unwrap_or_default(),
                denoiser: denoiser.unwrap_or_default(),
                overrides,
                ..default()
            }),
            cycle: cycle.then(|| stage_frames.unwrap_or(120)),
        }
    }

    /// Adds this file's plugins and systems, after the default plugins and Solari's.
    pub fn build(&self, app: &mut App, args: &Args) {
        let Some(settings) = &self.settings else {
            return;
        };
        let given = |flag: &str| args.command_line.iter().any(|arg| arg == flag);
        let mut errors = Vec::new();
        if args.mode != Mode::Raster
            || args.taa
            || args.dlss_rr
            || args.dlss_sr
            || given("--dlss-mode")
        {
            errors.push(
                "the ladder flags drive the camera: no --mode solari|reference, --taa, --dlss-rr, --dlss-sr \
                 or --dlss-mode with them",
            );
        }
        if self.cycle.is_some() && args.orbit {
            errors.push("--cycle compares still frames: no --camera orbit with it");
        }
        if !errors.is_empty() {
            // The same as a usage error of `main.rs`: nothing has started yet.
            for error in errors {
                eprintln!("gfx_testbed: {error}");
            }
            std::process::exit(2);
        }

        let mut asked = to_json(settings);
        if let (Value::Object(asked), Some(stage_frames)) = (&mut asked, self.cycle) {
            asked.insert(
                "cycle".into(),
                json!(CYCLE.iter().map(|(name, _)| *name).collect::<Vec<_>>()),
            );
            asked.insert("stage_frames".into(), json!(stage_frames));
        }
        app.world_mut()
            .resource_mut::<Report>()
            .sections
            .insert("ladder".into(), json!({ "asked": asked }));

        app.insert_resource(settings.clone())
            .add_plugins(GraphicsPlugin)
            .add_systems(Update, follow.after(crate::drive).before(crate::finish));
    }

    /// Spawns the scene called `name` if this file owns it.
    pub fn scene(_name: &str, _world: &mut World) -> Option<Scene> {
        None
    }

    /// Hands the camera to `bevy_graphics` when a ladder flag was given, and returns `true`; `main.rs` then
    /// adds nothing of the raw modes (no lighting component, no `RaytracingMesh3d`, no change to shadow maps).
    /// Returns `false` to leave the camera to the raw `--mode` flags.
    pub fn camera(&self, _args: &Args, camera: &mut EntityWorldMut) -> bool {
        if self.settings.is_none() {
            return false;
        }
        camera.insert(GraphicsCamera);
        true
    }
}

/// Where a `--cycle` run is.
#[derive(Default)]
struct Progress {
    /// The stage being held or waited for; `CYCLE.len()` once every stage is captured.
    stage: usize,
    /// The frame the current stage's preset was asked for.
    switched_at: u32,
    /// Frames in a row with no pipeline compiling, after [`SWITCH_FRAMES`].
    calm_frames: u32,
    /// The frame the current stage counts its held frames from; `None` while its pipelines compile.
    started_at: Option<u32>,
    /// The frame the next stage's preset is asked for: the one after the capture, so the capture shows the
    /// stage it is named after.
    switch_at: Option<u32>,
}

/// Records what resolved, runs the cycle, and fails the run when a preset asked for by name did not resolve.
fn follow(
    mut commands: Commands,
    ladder: Res<Ladder>,
    args: Res<Args>,
    run: Res<Run>,
    pending_pipelines: Res<PendingPipelines>,
    resolved: Res<ResolvedGraphics>,
    capabilities: Res<GraphicsCapabilities>,
    mut settings: ResMut<GraphicsSettings>,
    mut report: ResMut<Report>,
    mut captures: ResMut<Captures>,
    mut progress: Local<Progress>,
) {
    if run.phase != Phase::Running {
        return;
    }
    let report = &mut *report;
    // `log-gate` forgives Solari's "not loaded" warning only to a run that never traced.
    report.traced |= resolved.rung.is_traced();
    let frame = run.frame;

    if let Some(stage_frames) = ladder.cycle {
        if progress.switch_at == Some(frame) {
            progress.switch_at = None;
            settings.preset = CYCLE[progress.stage].1;
            progress.switched_at = frame;
            progress.calm_frames = 0;
        }
        if progress.stage < CYCLE.len() && progress.switch_at.is_none() {
            // The first stage starts where `main.rs` declared the scene ready.
            if progress.started_at.is_none() && progress.stage == 0 {
                progress.started_at = Some(frame);
            }
            if progress.started_at.is_none() && frame >= progress.switched_at + SWITCH_FRAMES {
                if pending_pipelines.0.load(Ordering::Relaxed) == 0 {
                    progress.calm_frames += 1;
                } else {
                    progress.calm_frames = 0;
                }
                if progress.calm_frames >= crate::CALM_FRAMES {
                    progress.started_at = Some(frame);
                }
            }
            if let Some(started_at) = progress.started_at
                && frame == started_at + stage_frames
            {
                let (name, preset) = CYCLE[progress.stage];
                let capture = format!("stage-{}-{name}", progress.stage + 1);
                captures.request(&mut commands, capture.clone(), &run);
                if let Some(failure) = mismatch(preset, &resolved) {
                    report
                        .failures
                        .push(format!("stage {}: {failure}", progress.stage + 1));
                }
                let stage = json!({
                    "stage": progress.stage + 1,
                    "asked": name,
                    "resolved": resolved_json(&resolved),
                    "capture": format!("{capture}.png"),
                    "frame": frame,
                    "held_from_frame": started_at,
                    "asked_at_frame": progress.switched_at,
                });
                ladder_section(report)
                    .entry("stages")
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .expect("stages is a list")
                    .push(stage);
                progress.stage += 1;
                progress.started_at = None;
                if progress.stage < CYCLE.len() {
                    progress.switch_at = Some(frame + 1);
                }
            }
        }
    }

    // The frame `main.rs` captures `a.png`: the mesh counts are settled and `finish` has not run yet.
    if frame == args.warmup + args.frames {
        if ladder.cycle.is_some() {
            if progress.stage < CYCLE.len() {
                report.failures.push(format!(
                    "a.png was taken at stage {} of {} of the cycle: give --frames enough to outlast it",
                    progress.stage + 1,
                    CYCLE.len()
                ));
            }
        } else if let Some(failure) = mismatch(settings.preset, &resolved) {
            report.failures.push(failure);
        }
        let section = ladder_section(report);
        section.insert("resolved".into(), resolved_json(&resolved));
        section.insert("capabilities".into(), to_json(&*capabilities));
    }
}

/// Why `resolved` is not the preset asked for, or `None` when it is (or when `Auto` was asked).
fn mismatch(asked: GraphicsPreset, resolved: &ResolvedGraphics) -> Option<String> {
    let wanted = match asked {
        GraphicsPreset::Auto => return None,
        GraphicsPreset::Raster => GraphicsRung::Raster,
        GraphicsPreset::RtLite => GraphicsRung::RtLite,
        GraphicsPreset::RtFull => GraphicsRung::RtFull,
    };
    (resolved.rung != wanted).then(|| {
        let reasons: Vec<String> = resolved.fallbacks.iter().map(ToString::to_string).collect();
        format!(
            "the preset asked for is {wanted:?} but {:?} resolved ({})",
            resolved.rung,
            reasons.join("; ")
        )
    })
}

/// [`ResolvedGraphics`] for the report, with each fallback's reason in words beside its name.
fn resolved_json(resolved: &ResolvedGraphics) -> Value {
    let mut value = to_json(resolved);
    if let Value::Object(object) = &mut value {
        object.insert(
            "reasons".into(),
            json!(resolved
                .fallbacks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()),
        );
    }
    value
}

fn to_json(value: &impl serde::Serialize) -> Value {
    serde_json::to_value(value).expect("the graphics types serialize to JSON")
}

fn ladder_section(report: &mut Report) -> &mut serde_json::Map<String, Value> {
    report
        .sections
        .entry("ladder")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("the ladder section is an object")
}
