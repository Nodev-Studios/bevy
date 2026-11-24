//! `gfx-check friend`: the hardware test of plan section 6, run by a person who is not a developer on a machine
//! the project does not have.
//!
//! One command runs the whole sequence on every adapter the machine lists, writes one results folder and zips it:
//!
//! 0. a short smoke run, whose log and zip are written before anything else, so a crash still leaves something to
//!    send; it also lists the machine's adapters;
//! 1. per adapter, a probe: `Auto` and each preset forced, briefly, with what resolved and why;
//! 2. every preset forced on both scenes, still, orbit and light-toggle runs, then the golden images and the
//!    frame-time table of `budgets.toml`, recorded and not gated (they are set for the project's RTX 5070);
//! 3. on a Radeon under Windows, FSR on and off and the provider the DLL reports;
//! 4. on a Radeon under Linux the traced rungs run with NRD and TSR, which step 2 already covers;
//! 5. on an NVIDIA card, `RtLite` with DLSS-RR and with NRD and DLSS SR;
//! 6. a few `--cycle` runs, then five yes/no questions in Spanish.
//!
//! It never stops on a failed run: each `run` records its failures and the sequence goes on. Every text file is
//! scrubbed of the home folder, the user name and the host name before each zip.
//!
//! Integrated GPUs and CPU rasterizers get a smaller sequence: the friend's discrete card is what the test is
//! for, and an integrated GPU at 1440p would take hours.

use crate::run::{self, Cell, RunCommand};
use argh::FromArgs;
use serde_json::{json, Value};
use std::{
    ffi::OsString,
    fmt::Write as _,
    fs,
    io::{BufRead, IsTerminal, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Runs the hardware test of plan section 6 on every adapter and zips the results.
///
/// Writes `resultados-<date>/` and `resultados-<date>.zip` beside this program (or under `--out`). On Linux the
/// testbed opens real windows unless `--headless` names the project's wrapper; on Windows it always does.
#[derive(FromArgs)]
#[argh(subcommand, name = "friend")]
pub struct FriendCommand {
    /// the testbed binary; default `gfx_testbed` beside this program
    #[argh(option)]
    testbed: Option<PathBuf>,
    /// folder the results folder and its zip go in; default the folder of this program
    #[argh(option)]
    out: Option<PathBuf>,
    /// the project machine's `tools/gfx-headless.sh`: every run goes through it and no window shows
    #[argh(option)]
    headless: Option<PathBuf>,
    /// test only adapters whose name contains this; may be repeated
    #[argh(option)]
    adapter: Vec<String>,
    /// the five answers as `s,n,?,s,n` instead of asking them
    #[argh(option)]
    answers: Option<String>,
    /// print the adapters, the plan and the time estimate, and run nothing more
    #[argh(switch)]
    dry_run: bool,
}

/// The five questions, asked in Spanish: the person running the kit reads them. The key is what `resumen.json`
/// records the answer under.
const QUESTIONS: [(&str, &str); 5] = [
    (
        "trails",
        "¿Viste estelas o «fantasmas» detrás de los objetos mientras la cámara giraba?",
    ),
    (
        "light_leaks",
        "¿Viste luz que atraviesa las paredes (zonas iluminadas adonde la luz no debería llegar)?",
    ),
    (
        "shadow_flicker",
        "¿Viste sombras que parpadean o tiemblan con la cámara quieta?",
    ),
    (
        "stray_pixels",
        "¿Viste puntitos sueltos, negros o muy brillantes, en la imagen?",
    ),
    (
        "crash_after_switching",
        "En la última prueba de cada tarjeta la imagen cambia de calidad muchas veces seguidas. \
         ¿Se cerró el programa o se quedó la pantalla en negro?",
    ),
];

/// Value of `WGPU_ADAPTER_NAME` that matches no adapter: Bevy then logs every adapter it checks (at trace level)
/// and falls back to its default choice.
const NO_ADAPTER: &str = "gfx-check-lists-every-adapter";

/// Cycle runs of `--cycle` (three switches each): twelve switches, the "ten times" of plan section 6 item 6,
/// spread over four processes until the testbed can repeat a cycle in one.
const CYCLE_RUNS: u32 = 4;

/// What a class of adapter is given.
struct Profile {
    /// `Full`, `Small` or `Cpu`, for the report.
    name: &'static str,
    /// Window sizes of step 2.
    resolutions: &'static [&'static str],
    /// Warm-up and measured frames of a still run.
    still: (u32, u32),
    /// Warm-up and measured frames of an orbit run.
    orbit: (u32, u32),
    /// Warm-up and measured frames of a light-toggle run, and the measured frame the light goes off.
    toggle: (u32, u32, u32),
    /// Window size, warm-up and measured frames of the probe.
    probe: (&'static str, u32, u32),
    /// Window size of the cycle runs and the frames each stage is held; `None`: no cycle.
    cycle: Option<(&'static str, u32)>,
    /// Seconds after which one run is stopped.
    seconds: u64,
    /// Whether the traced presets are tried at all.
    traced: bool,
    /// Whether the golden runs are made: 330 frames each, too slow on a CPU rasterizer.
    golden: bool,
}

/// A discrete (or unknown) GPU: the plan's sequence as written.
const FULL: Profile = Profile {
    name: "full",
    resolutions: &["1920x1080", "2560x1440"],
    still: (300, 600),
    // Long enough to pass the orbit's eight captures and see trails; shorter than a still run. To tune.
    orbit: (60, 240),
    // A temporal filter's lag shows 5 frames after the switch; 30 frames of history before it is plenty. To tune.
    toggle: (60, 60, 30),
    probe: ("1280x720", 30, 60),
    cycle: Some(("1280x720", 120)),
    seconds: 600,
    traced: true,
    golden: true,
};

/// An integrated GPU: the code path, not its speed (the project's integrated Radeon is about 80 times slower
/// than its RTX 5070).
const SMALL: Profile = Profile {
    name: "small",
    resolutions: &["640x360"],
    still: (30, 60),
    orbit: (10, 40),
    toggle: (10, 30, 10),
    probe: ("640x360", 10, 30),
    cycle: Some(("640x360", 30)),
    seconds: 900,
    traced: true,
    golden: true,
};

/// A CPU rasterizer: Raster only, a few frames.
const CPU: Profile = Profile {
    name: "cpu",
    resolutions: &["640x360"],
    still: (5, 10),
    orbit: (5, 16),
    toggle: (5, 16, 5),
    probe: ("640x360", 5, 10),
    cycle: None,
    seconds: 1200,
    traced: false,
    golden: false,
};

/// One adapter as Bevy listed it.
#[derive(Clone, Debug, PartialEq)]
struct Adapter {
    name: String,
    vendor: u32,
    device_type: String,
    driver: String,
    driver_info: String,
}

impl Adapter {
    fn profile(&self) -> &'static Profile {
        match self.device_type.as_str() {
            "Cpu" => &CPU,
            "IntegratedGpu" => &SMALL,
            _ => &FULL,
        }
    }

    fn is_nvidia(&self) -> bool {
        self.vendor == 0x10de || self.name.to_lowercase().contains("nvidia")
    }

    fn is_amd(&self) -> bool {
        let name = self.name.to_lowercase();
        self.vendor == 0x1002 || name.contains("radeon") || name.contains("amd")
    }

    /// The name passed as `WGPU_ADAPTER_NAME`, which `run` also uses as a folder name: the longest piece of the
    /// name without a character a file name cannot hold.
    fn match_name(&self) -> String {
        self.name
            .split(['/', '\\', ':', '*', '?', '"', '<', '>', '|'])
            .map(str::trim)
            .max_by_key(|piece| piece.len())
            .unwrap_or_default()
            .to_owned()
    }

    /// Sorts discrete cards first and CPU rasterizers last: the results that matter are written first.
    fn order(&self) -> u8 {
        match self.device_type.as_str() {
            "DiscreteGpu" => 0,
            "IntegratedGpu" => 2,
            "Cpu" => 3,
            _ => 1,
        }
    }
}

/// Reads the adapters out of a log: Bevy prints `AdapterInfo { name: "…", vendor: …, … }` for each adapter it
/// checks against `WGPU_ADAPTER_NAME` (trace level) and for the one it picks (info level).
fn parse_adapters(log: &str) -> Vec<Adapter> {
    let mut adapters: Vec<Adapter> = Vec::new();
    for line in log.lines() {
        let Some(start) = line.find("AdapterInfo {") else {
            continue;
        };
        let info = &line[start..];
        let backend = field(info, "backend").unwrap_or_default();
        let Some(name) = field(info, "name") else {
            continue;
        };
        if backend != "Vulkan" || adapters.iter().any(|adapter| adapter.name == name) {
            continue;
        }
        adapters.push(Adapter {
            name: name.to_owned(),
            vendor: field(info, "vendor")
                .and_then(|vendor| vendor.parse().ok())
                .unwrap_or(0),
            device_type: field(info, "device_type").unwrap_or_default().to_owned(),
            driver: field(info, "driver").unwrap_or_default().to_owned(),
            driver_info: field(info, "driver_info").unwrap_or_default().to_owned(),
        });
    }
    adapters.sort_by_key(Adapter::order);
    adapters
}

/// One field of a `Debug`-printed struct: a quoted string, or a bare value up to the next comma or brace.
fn field<'a>(info: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!(" {key}: ");
    let start = info.find(&pattern)? + pattern.len();
    let rest = &info[start..];
    if let Some(quoted) = rest.strip_prefix('"') {
        quoted.find('"').map(|end| &quoted[..end])
    } else {
        let end = rest.find([',', ' ', '}']).unwrap_or(rest.len());
        Some(&rest[..end])
    }
}

/// One run of a generated matrix.
struct RunSpec {
    name: String,
    args: Vec<String>,
    warmup: u32,
    frames: u32,
}

impl RunSpec {
    fn new(name: impl Into<String>, args: &[&str], (warmup, frames): (u32, u32)) -> RunSpec {
        RunSpec {
            name: name.into(),
            args: args.iter().map(|arg| (*arg).to_owned()).collect(),
            warmup,
            frames,
        }
    }
}

/// One step of an adapter's sequence: a matrix for `gfx-check run`.
struct Step {
    /// Folder name inside the adapter's folder, and the step's key in the summary.
    key: &'static str,
    /// What the console says, in Spanish.
    title: &'static str,
    resolutions: Vec<String>,
    scenes: Vec<&'static str>,
    seconds: u64,
    runs: Vec<RunSpec>,
}

impl Step {
    fn matrix(&self, adapter: &str) -> String {
        let quote = |text: &str| format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""));
        let list = |items: &mut dyn Iterator<Item = String>| items.collect::<Vec<_>>().join(", ");
        let mut text =
            String::from("#![enable(implicit_some)]\n// Written by `gfx-check friend`.\n(\n");
        let _ = writeln!(
            text,
            "    adapters: [(name: {}, resolutions: [{}])],",
            quote(adapter),
            list(&mut self.resolutions.iter().map(|res| quote(res)))
        );
        let _ = writeln!(
            text,
            "    scenes: [{}],",
            list(&mut self.scenes.iter().map(|scene| quote(scene)))
        );
        let _ = writeln!(
            text,
            "    warmup: 0,\n    frames: 1,\n    seconds: {},\n    runs: [",
            self.seconds
        );
        for run in &self.runs {
            let _ = writeln!(
                text,
                "        (name: {}, args: [{}], warmup: {}, frames: {}),",
                quote(&run.name),
                list(&mut run.args.iter().map(|arg| quote(arg))),
                run.warmup,
                run.frames
            );
        }
        text.push_str("    ],\n)\n");
        text
    }
}

/// The sequence for one adapter. `traced` says whether the traced presets are run (the probe decides).
fn plan(adapter: &Adapter, traced: bool) -> Vec<Step> {
    let profile = adapter.profile();
    let traced = traced && profile.traced;
    let resolutions: Vec<String> = profile.resolutions.iter().map(|&res| res.into()).collect();
    let scenes = vec!["interior", "many-lights"];
    let mut steps = Vec::new();

    let (probe_res, probe_warmup, probe_frames) = profile.probe;
    let mut probe = vec![
        RunSpec::new("auto", &["--preset", "auto"], (probe_warmup, probe_frames)),
        RunSpec::new(
            "raster",
            &["--preset", "raster"],
            (probe_warmup, probe_frames),
        ),
    ];
    if profile.traced {
        for preset in ["rt-lite", "rt-full"] {
            probe.push(RunSpec::new(
                preset,
                &["--preset", preset],
                (probe_warmup, probe_frames),
            ));
        }
    }
    steps.push(Step {
        key: "probe",
        title: "sonda: qué elige el modo automático y por qué",
        resolutions: vec![probe_res.into()],
        scenes: vec!["interior"],
        seconds: profile.seconds,
        runs: probe,
    });

    // The still runs carry the names of `budget.ron`, so `budget --table` finds them.
    let mut presets = vec![
        RunSpec::new(
            "raster-taa",
            &["--preset", "raster", "--upscaler", "off"],
            profile.still,
        ),
        RunSpec::new(
            "raster-tsr",
            &["--preset", "raster", "--upscaler", "tsr"],
            profile.still,
        ),
        RunSpec::new("raster-auto", &["--preset", "raster"], profile.still),
    ];
    let mut forced = vec!["raster"];
    if traced {
        forced.extend(["rt-lite", "rt-full"]);
        for preset in ["rt-lite", "rt-full"] {
            presets.push(RunSpec::new(
                format!("{preset}-auto"),
                &["--preset", preset],
                profile.still,
            ));
            presets.push(RunSpec::new(
                format!("{preset}-nrd-tsr"),
                &["--preset", preset, "--denoiser", "nrd", "--upscaler", "tsr"],
                profile.still,
            ));
        }
    }
    let (toggle_warmup, toggle_frames, toggle_at) = profile.toggle;
    let toggle_at = toggle_at.to_string();
    for preset in &forced {
        presets.push(RunSpec::new(
            format!("{preset}-auto-orbit"),
            &["--preset", preset, "--camera", "orbit"],
            profile.orbit,
        ));
        presets.push(RunSpec::new(
            format!("{preset}-auto-toggle"),
            &["--preset", preset, "--light-toggle-at", &toggle_at],
            (toggle_warmup, toggle_frames),
        ));
    }
    steps.push(Step {
        key: "presets",
        title: "cada calidad forzada, en las dos escenas",
        resolutions: resolutions.clone(),
        scenes: scenes.clone(),
        seconds: profile.seconds,
        runs: presets,
    });

    // `golden.ron`'s flags and frames, so `budget --golden` compares like with like.
    let golden_frames = (300, 30);
    let mut golden = vec![RunSpec::new(
        "raster",
        &["--preset", "raster", "--upscaler", "off"],
        golden_frames,
    )];
    if traced {
        for preset in ["rt-lite", "rt-full"] {
            golden.push(RunSpec::new(
                preset,
                &["--preset", preset, "--denoiser", "nrd", "--upscaler", "tsr"],
                golden_frames,
            ));
        }
    }
    if profile.golden {
        steps.push(Step {
            key: "golden",
            title: "imágenes de referencia",
            resolutions: vec![profile.resolutions[0].into()],
            scenes: scenes.clone(),
            seconds: profile.seconds,
            runs: golden,
        });
    }

    if cfg!(windows) && adapter.is_amd() {
        let mut fsr = vec![
            RunSpec::new(
                "raster-fsr",
                &["--preset", "raster", "--upscaler", "fsr"],
                profile.still,
            ),
            RunSpec::new(
                "raster-fsr-orbit",
                &[
                    "--preset",
                    "raster",
                    "--upscaler",
                    "fsr",
                    "--camera",
                    "orbit",
                ],
                profile.orbit,
            ),
        ];
        if traced {
            fsr.push(RunSpec::new(
                "rt-lite-nrd-fsr",
                &[
                    "--preset",
                    "rt-lite",
                    "--denoiser",
                    "nrd",
                    "--upscaler",
                    "fsr",
                ],
                profile.still,
            ));
        }
        steps.push(Step {
            key: "fsr",
            title: "FSR de AMD encendido (apagado ya se midió)",
            resolutions: resolutions.clone(),
            scenes: scenes.clone(),
            seconds: profile.seconds,
            runs: fsr,
        });
    }

    if traced && adapter.is_nvidia() {
        let mut rtx = Vec::new();
        for (name, args) in [
            (
                "rt-lite-dlss-rr",
                [
                    "--preset",
                    "rt-lite",
                    "--denoiser",
                    "dlss-rr",
                    "--upscaler",
                    "dlss",
                ],
            ),
            (
                "rt-lite-nrd-dlss",
                [
                    "--preset",
                    "rt-lite",
                    "--denoiser",
                    "nrd",
                    "--upscaler",
                    "dlss",
                ],
            ),
        ] {
            rtx.push(RunSpec::new(name, &args, profile.still));
            let mut orbit = args.to_vec();
            orbit.extend(["--camera", "orbit"]);
            rtx.push(RunSpec::new(format!("{name}-orbit"), &orbit, profile.orbit));
        }
        steps.push(Step {
            key: "rtx",
            title: "RtLite con DLSS-RR y con NRD + DLSS",
            resolutions: resolutions.clone(),
            scenes: scenes.clone(),
            seconds: profile.seconds,
            runs: rtx,
        });
    }

    if traced && let Some((cycle_res, stage_frames)) = profile.cycle {
        // Frames tick while a switched preset compiles its pipelines, so the run gets far more than four stages'
        // worth; a run too short says so in its failures. To tune.
        let frames = 4 * stage_frames + 25 * stage_frames;
        let stage = stage_frames.to_string();
        steps.push(Step {
            key: "cycle",
            title: "cambiar de calidad muchas veces seguidas (mira la ventana)",
            resolutions: vec![cycle_res.into()],
            scenes: vec!["interior"],
            seconds: profile.seconds,
            runs: (1..=CYCLE_RUNS)
                .map(|k| {
                    RunSpec::new(
                        format!("cycle@{k}"),
                        &["--cycle", "--stage-frames", &stage],
                        (10, frames),
                    )
                })
                .collect(),
        });
    }
    steps
}

/// What the probe measured on one adapter, from which the rest of its sequence is estimated.
#[derive(Clone, Copy)]
struct Pace {
    /// Seconds a run takes besides its frames: start-up, loading, pipelines, screenshots.
    startup: f64,
    /// Median frame milliseconds of Raster, `RtLite` and `RtFull` at the probe's size.
    ms: [f64; 3],
    /// Pixels of the probe's window.
    pixels: f64,
}

impl Pace {
    /// From the probe's cells and how long the probe took.
    fn measure(step: &Step, cells: &[Cell], seconds: f64) -> Pace {
        let pixels = step
            .resolutions
            .first()
            .map_or(1280.0 * 720.0, |res| pixels(res));
        let ms_of = |name: &str| {
            cells
                .iter()
                .find(|cell| cell.run == name)
                .and_then(|cell| cell.frame_ms)
                .map(|frame| frame[0])
        };
        // A preset the probe could not time costs, by guess, twice Raster. To tune.
        let raster = ms_of("raster").or_else(|| ms_of("auto")).unwrap_or(20.0);
        let ms = [
            raster,
            ms_of("rt-lite").unwrap_or(2.0 * raster),
            ms_of("rt-full").unwrap_or(2.0 * raster),
        ];
        let frames: f64 = step
            .runs
            .iter()
            .map(|run| f64::from(run.warmup + run.frames) * ms[preset_index(&run.args)] / 1000.0)
            .sum();
        let runs = step.runs.len().max(1) as f64;
        Pace {
            startup: ((seconds - frames) / runs).clamp(2.0, 120.0),
            ms,
            pixels,
        }
    }
}

fn pixels(resolution: &str) -> f64 {
    resolution
        .split_once('x')
        .and_then(|(w, h)| Some(w.parse::<f64>().ok()? * h.parse::<f64>().ok()?))
        .unwrap_or(1280.0 * 720.0)
}

/// Which of Raster, `RtLite` and `RtFull` a run's flags ask for; a cycle counts as `RtLite`, its middle stage.
fn preset_index(args: &[String]) -> usize {
    if args.iter().any(|arg| arg == "rt-full") {
        2
    } else if args.iter().any(|arg| arg == "rt-lite" || arg == "--cycle") {
        1
    } else {
        0
    }
}

/// How long a step is expected to take, in seconds: per run the probe's start-up, plus its frames at the probe's
/// frame time grown with the window. Frames do not grow with pixels when the CPU bounds them, so the exponent is
/// under 1. To tune.
fn estimate(step: &Step, pace: Pace) -> f64 {
    let mut seconds = 0.0;
    for resolution in &step.resolutions {
        let scale = (pixels(resolution) / pace.pixels).powf(0.75);
        for _ in &step.scenes {
            for run in &step.runs {
                let ms = pace.ms[preset_index(&run.args)] * scale;
                seconds += pace.startup + f64::from(run.warmup + run.frames) * ms / 1000.0;
            }
        }
    }
    seconds
}

/// Replaces what would identify the person or the machine in every text file of the results.
struct Scrub {
    /// Paths and their replacement, longest first, replaced wherever they appear.
    paths: Vec<(String, &'static str)>,
    /// Names and their replacement, replaced only as whole words.
    words: Vec<(String, &'static str)>,
}

impl Scrub {
    fn from_machine(kit: &Path) -> Scrub {
        let env = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let mut paths: Vec<(String, &'static str)> = Vec::new();
        let mut add_path = |path: String, with: &'static str| {
            // A home of `/` or `C:\` would replace every separator.
            if path.trim_end_matches(['/', '\\']).len() < 4 {
                return;
            }
            let path = path.trim_end_matches(['/', '\\']).to_owned();
            for variant in [
                path.clone(),
                path.replace('\\', "/"),
                path.replace('/', "\\"),
                path.replace('\\', "\\\\"),
            ] {
                if !paths.iter().any(|(known, _)| *known == variant) {
                    paths.push((variant, with));
                }
            }
        };
        add_path(kit.display().to_string(), "<carpeta-del-kit>");
        if let Ok(canonical) = kit.canonicalize() {
            add_path(canonical.display().to_string(), "<carpeta-del-kit>");
        }
        for home in ["HOME", "USERPROFILE"].into_iter().filter_map(env) {
            add_path(home, "<inicio>");
        }
        paths.sort_by_key(|(path, _)| core::cmp::Reverse(path.len()));

        let mut words: Vec<(String, &'static str)> = Vec::new();
        for user in ["USER", "USERNAME", "LOGNAME"].into_iter().filter_map(env) {
            words.push((user, "<usuario>"));
        }
        let hosts = ["HOSTNAME", "COMPUTERNAME"]
            .into_iter()
            .filter_map(env)
            .chain(
                ["/proc/sys/kernel/hostname", "/etc/hostname"]
                    .into_iter()
                    .filter_map(|file| fs::read_to_string(file).ok()),
            );
        for host in hosts {
            words.push((host.trim().to_owned(), "<equipo>"));
        }
        // Two letters would match inside ordinary words and adapter names.
        words.retain(|(word, _)| word.chars().count() >= 3);
        words.sort_by_key(|(word, _)| core::cmp::Reverse(word.len()));
        words.dedup_by(|a, b| a.0 == b.0);
        Scrub { paths, words }
    }

    fn apply(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for (path, with) in &self.paths {
            text = replace_ignoring_case(&text, path, with, false);
        }
        for (word, with) in &self.words {
            text = replace_ignoring_case(&text, word, with, true);
        }
        text
    }
}

/// Replaces `from` with `to` in `text`, ignoring ASCII case (Windows paths come in any case). With `whole_word`,
/// only where `from` is not part of a longer run of letters and digits.
fn replace_ignoring_case(text: &str, from: &str, to: &str, whole_word: bool) -> String {
    let lower = text.to_ascii_lowercase();
    let needle = from.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut search = 0;
    while let Some(found) = lower[search..].find(&needle) {
        let start = search + found;
        let end = start + needle.len();
        let alnum = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
        if whole_word
            && (alnum(text[..start].chars().next_back()) || alnum(text[end..].chars().next()))
        {
            search = end;
            continue;
        }
        out.push_str(&text[last..start]);
        out.push_str(to);
        last = end;
        search = end;
    }
    out.push_str(&text[last..]);
    out
}

/// Text files of the results, which are scrubbed and always zipped.
fn is_text(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ["txt", "json", "csv", "md", "log", "ron", "toml"].contains(&ext))
}

/// Whether a file goes in the zip. Every text file does. Of the images, only what a person looks at or the
/// project compares: the copies in `look-here`, the golden runs, the probe and the smoke run. The rest (a
/// discrete card writes several hundred megabytes) stays in the folder.
fn in_zip(relative: &str) -> bool {
    if is_text(Path::new(relative)) {
        return true;
    }
    let parts: Vec<&str> = relative.split('/').collect();
    parts.first() == Some(&"para-mirar")
        || parts.first() == Some(&"0-smoke")
        || parts
            .get(1)
            .is_some_and(|part| ["probe", "golden"].contains(part))
}

/// The CRC-32 of a zip entry (IEEE polynomial, reflected).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Writes a zip of `folder` with the files `keep` accepts, stored without compression (the images are already
/// compressed, and a stored zip needs nothing outside this file to write). Entry names start with the folder's
/// name. The zip is written beside and renamed into place, so an interrupted write leaves the previous one.
fn write_zip(folder: &Path, zip: &Path, keep: impl Fn(&str) -> bool) -> Result<u64, String> {
    fn walk(dir: &Path, base: &Path, files: &mut Vec<(String, PathBuf)>) -> std::io::Result<()> {
        let mut entries: Vec<_> = fs::read_dir(dir)?
            .flatten()
            .map(|entry| entry.path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, base, files)?;
            } else if let Ok(relative) = path.strip_prefix(base) {
                let relative: Vec<String> = relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect();
                files.push((relative.join("/"), path));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(folder, folder, &mut files).map_err(|error| format!("{}: {error}", folder.display()))?;
    let prefix = folder.file_name().map_or_else(
        || "resultados".into(),
        |name| name.to_string_lossy().into_owned(),
    );

    let mut data: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();
    let mut count: u16 = 0;
    // Version 2.0, bit 11 (names are UTF-8), stored, 1980-01-01 00:00.
    let header = |crc: u32, size: u32, name: &[u8]| {
        let mut bytes = Vec::with_capacity(26 + name.len());
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&0x0800u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0x21u16.to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes
    };
    for (relative, path) in files.iter().filter(|(relative, _)| keep(relative)) {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let name = format!("{prefix}/{relative}");
        let offset = u32::try_from(data.len()).map_err(|_| "the zip would pass 4 GiB")?;
        let size = u32::try_from(bytes.len()).map_err(|_| "a file passes 4 GiB")?;
        let fields = header(crc32(&bytes), size, name.as_bytes());

        data.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        data.extend_from_slice(&fields);
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&bytes);

        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&fields);
        // Comment length, disk, internal and external attributes, then the local header's offset.
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
        count = count.checked_add(1).ok_or("more than 65535 files")?;
    }
    let central_offset = u32::try_from(data.len()).map_err(|_| "the zip would pass 4 GiB")?;
    let central_size = central.len() as u32;
    data.extend_from_slice(&central);
    data.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    data.extend_from_slice(&[0; 4]);
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&central_size.to_le_bytes());
    data.extend_from_slice(&central_offset.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());

    let partial = zip.with_extension("zip.part");
    fs::write(&partial, &data).map_err(|error| format!("{}: {error}", partial.display()))?;
    fs::rename(&partial, zip).map_err(|error| format!("{}: {error}", zip.display()))?;
    Ok(data.len() as u64)
}

/// Today's date in UTC as `YYYY-MM-DD` (days to civil date, after Howard Hinnant's `civil_from_days`).
fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// The operating system in a line: its name and version, nothing else.
fn os_description() -> String {
    if cfg!(windows) {
        Command::new("cmd")
            .args(["/C", "ver"])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| "Windows".into())
    } else {
        let name = fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find_map(|line| line.strip_prefix("PRETTY_NAME="))
                    .map(|name| name.trim_matches('"').to_owned())
            })
            .unwrap_or_else(|| std::env::consts::OS.into());
        let kernel = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
        format!("{name}, kernel {}", kernel.trim())
    }
}

/// A folder-safe short name for an adapter's results: its position and its name.
fn slug(index: usize, name: &str) -> String {
    let mut slug = format!("{}-", index + 1);
    let mut dash = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            dash = false;
        } else if !dash {
            slug.push('-');
            dash = true;
        }
    }
    slug.trim_end_matches('-').chars().take(40).collect()
}

/// The whole test, with what it has written so far.
struct Friend {
    testbed: PathBuf,
    kit: PathBuf,
    results: PathBuf,
    zip: PathBuf,
    headless: Option<PathBuf>,
    scrub: Scrub,
    log: Option<fs::File>,
    summary: Value,
}

impl Friend {
    /// Prints a line for the person and keeps it in `console.txt`.
    fn say(&mut self, line: impl AsRef<str>) {
        println!("{}", line.as_ref());
        if let Some(log) = &mut self.log {
            let _ = writeln!(log, "{}", line.as_ref());
        }
    }

    /// Scrubs every text file, writes the summary and rewrites the zip. Never fails the test: a problem here is
    /// printed and the test goes on.
    fn checkpoint(&mut self) {
        let summary = serde_json::to_string_pretty(&self.summary).unwrap_or_default();
        let _ = fs::write(self.results.join("summary.json"), summary);
        let mut problems = Vec::new();
        scrub_folder(&self.results, &self.scrub, &mut problems);
        match write_zip(&self.results, &self.zip, in_zip) {
            Ok(_) => {}
            Err(error) => problems.push(error),
        }
        for problem in problems {
            self.say(format!("  (aviso: {problem})"));
        }
    }

    /// Runs one step's matrix through `gfx-check run`, recording what came back.
    fn run_step(&mut self, step: &Step, adapter: &Adapter, dir: &Path) -> Value {
        let started = Instant::now();
        let out = dir.join(step.key);
        let matrix = dir.join(format!("{}.ron", step.key));
        let mut record = json!({ "step": step.key });
        if let Err(error) = fs::create_dir_all(dir)
            .and_then(|()| fs::write(&matrix, step.matrix(&adapter.match_name())))
        {
            record["error"] = json!(format!("{}: {error}", matrix.display()));
            return record;
        }
        let mut args: Vec<String> = vec![
            matrix.display().to_string(),
            "--out".into(),
            out.display().to_string(),
            "--testbed".into(),
            self.testbed.display().to_string(),
        ];
        if self.headless.is_none() {
            args.push("--windowed".into());
        }
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let outcome = RunCommand::from_args(&["run"], &args)
            .map_err(|early| early.output)
            .and_then(run::run);
        if let Err(error) = outcome {
            record["error"] = json!(error);
        }
        let cells: Vec<Cell> = fs::read_to_string(out.join("results.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let failed: Vec<Value> = cells
            .iter()
            .filter(|cell| !cell.failures.is_empty())
            .map(|cell| {
                json!({
                    "scene": cell.scene,
                    "resolution": cell.resolution,
                    "run": cell.run,
                    "failures": cell.failures,
                })
            })
            .collect();
        if cells
            .iter()
            .flat_map(|cell| &cell.failures)
            .any(|failure| failure.contains("is the screen large enough"))
        {
            record["screen_too_small"] = json!(true);
            self.say("  La pantalla es más pequeña que alguna ventana de prueba: esas pruebas quedan anotadas como no hechas.");
        }
        record["runs"] = json!(cells.len());
        record["failed"] = json!(failed);
        record["minutes"] = json!((started.elapsed().as_secs_f64() / 6.0).round() / 10.0);
        record
    }

    /// Runs this program again as `gfx-check budget <results> <flag> <data>`, its output into `file`: the
    /// tables are recorded, never gated, on the friend's machine.
    fn budget(results: &Path, flag: &str, data: &Path, file: &Path) {
        let Ok(me) = std::env::current_exe() else {
            return;
        };
        let output = Command::new(me)
            .arg("budget")
            .arg(results)
            .arg(flag)
            .arg(data)
            .stdin(Stdio::null())
            .output();
        let text = match output {
            Ok(output) => format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) => format!("could not run gfx-check budget: {error}"),
        };
        let _ = fs::write(file, text);
    }

    /// Step 0: a short Raster run on whatever adapter Bevy picks by default, with every adapter it checks in its
    /// log. Started directly, not through `run`, so it needs no adapter name.
    fn smoke(&mut self) -> Result<Vec<Adapter>, String> {
        let dir = self.results.join("0-smoke");
        fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        let size = "1280x720";
        let frames = 60;
        let frames_text = frames.to_string();
        let testbed_args: Vec<OsString> = [
            "--preset",
            "raster",
            "--res",
            size,
            "--warmup",
            "10",
            "--frames",
            frames_text.as_str(),
            "--out",
        ]
        .into_iter()
        .map(OsString::from)
        .chain([dir.clone().into_os_string()])
        .collect();
        let seconds = 300;
        let mut command = match &self.headless {
            Some(wrapper) => {
                let mut command = Command::new(wrapper);
                command
                    .args(["--seconds", &seconds.to_string(), "--size", size])
                    .arg(&self.testbed);
                command
            }
            None => Command::new(&self.testbed),
        };
        command
            .args(&testbed_args)
            .env("WGPU_ADAPTER_NAME", NO_ADAPTER)
            .env("RUST_LOG", "bevy_render::renderer=trace");
        self.say(format!(
            "  {}",
            core::iter::once(command.get_program())
                .chain(command.get_args())
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(" ")
        ));
        let console = dir.join("console.txt");
        let file = fs::File::create(&console)
            .map_err(|error| format!("{}: {error}", console.display()))?;
        let started = Instant::now();
        let mut child = command
            .stdin(Stdio::null())
            .stdout(file.try_clone().map_err(|error| error.to_string())?)
            .stderr(file)
            .spawn()
            .map_err(|error| format!("could not start the testbed: {error}"))?;
        // The wrapper has its own limit; ours also covers its wait for the project machine's GPU lock.
        let limit = Duration::from_secs(seconds + if self.headless.is_some() { 3600 } else { 0 });
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                break Some(status);
            }
            if started.elapsed() > limit {
                // Our own child: stop it and nothing else.
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        let wall = started.elapsed().as_secs_f64();
        let report: Value = fs::read_to_string(dir.join("report.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(Value::Null);
        let median = report["frame_ms"]["median"].as_f64();
        let log = fs::read_to_string(&console).unwrap_or_default();
        let adapters = parse_adapters(&log);
        self.summary["smoke"] = json!({
            "exit": status.map_or_else(|| "stopped by the time limit".into(), |status| format!("{status}")),
            "seconds": (wall * 10.0).round() / 10.0,
            "adapter": report["adapter"],
            "frame_ms_median": median,
            "failures": report["failures"],
        });
        Ok(adapters)
    }

    /// What the probe's `Auto` run resolved, and whether the device can trace rays.
    fn read_probe(&mut self, adapter: &Adapter, dir: &Path) -> (Value, Option<bool>) {
        let profile = adapter.profile();
        let auto = dir
            .join("probe")
            .join(adapter.match_name())
            .join("interior")
            .join(profile.probe.0)
            .join("auto")
            .join("report.json");
        let report: Value = fs::read_to_string(&auto)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(Value::Null);
        let ladder = &report["ladder"];
        let ray_query = ladder["capabilities"]["ray_query"].as_bool();
        (
            json!({
                "resolved": ladder["resolved"],
                "capabilities": ladder["capabilities"],
                "backend": report["adapter"]["backend"],
            }),
            ray_query,
        )
    }

    /// Copies what the person should look at before the questions into `para-mirar/`.
    fn copy_to_look_at(&self, index: usize, adapter: &Adapter, dir: &Path) {
        let look = self.results.join("para-mirar");
        let _ = fs::create_dir_all(&look);
        let name = adapter.match_name();
        let first = adapter.profile().resolutions[0];
        let mut copies = Vec::new();
        for scene in ["interior", "many-lights"] {
            let cell = dir.join("presets").join(&name).join(scene).join(first);
            for preset in ["raster", "rt-lite", "rt-full"] {
                copies.push((
                    cell.join(format!("{preset}-auto-orbit")).join("sheet.png"),
                    format!("{}-{scene}-{preset}-girando.png", index + 1),
                ));
                copies.push((
                    cell.join(format!("{preset}-auto-toggle"))
                        .join("toggle.png"),
                    format!("{}-{scene}-{preset}-luz-apagada.png", index + 1),
                ));
            }
        }
        if let Some((cycle_res, _)) = adapter.profile().cycle {
            let cycle = dir
                .join("cycle")
                .join(&name)
                .join("interior")
                .join(cycle_res)
                .join("cycle@1");
            for (stage, preset) in [(1, "raster"), (2, "rt-lite"), (3, "rt-full"), (4, "raster")] {
                copies.push((
                    cycle.join(format!("stage-{stage}-{preset}.png")),
                    format!("{}-cambios-{stage}-{preset}.png", index + 1),
                ));
            }
        }
        for (from, to) in copies {
            if from.exists() {
                let _ = fs::copy(&from, look.join(to));
            }
        }
    }

    /// The FSR lines the testbed logged in a folder, without repeats: the provider the DLL picked is among them.
    fn fsr_lines(dir: &Path) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut stack = vec![dir.to_owned()];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name().is_some_and(|name| name == "log.txt") {
                    for line in fsr_log_lines(&fs::read_to_string(&path).unwrap_or_default()) {
                        if !lines.contains(&line) {
                            lines.push(line);
                        }
                    }
                }
            }
        }
        lines.sort();
        lines
    }

    /// Step 1 on one adapter: the probe. Returns the adapter's record, whether it traces, and its pace.
    fn probe(&mut self, index: usize, adapter: &Adapter) -> (Value, bool, Pace) {
        let profile = adapter.profile();
        let dir = self.results.join(slug(index, &adapter.name));
        self.say(format!(
            "  Sonda de {} ({}): qué elige el modo automático y por qué",
            adapter.name, adapter.driver_info
        ));
        let mut record = json!({
            "name": adapter.name,
            "vendor": format!("0x{:04x}", adapter.vendor),
            "device_type": adapter.device_type,
            "driver": adapter.driver,
            "driver_info": adapter.driver_info,
            "profile": profile.name,
            "folder": slug(index, &adapter.name),
            "steps": [],
        });
        let steps = plan(adapter, true);
        let started = Instant::now();
        let probe = self.run_step(&steps[0], adapter, &dir);
        let cells: Vec<Cell> = fs::read_to_string(dir.join("probe").join("results.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let pace = Pace::measure(&steps[0], &cells, started.elapsed().as_secs_f64());
        if let Some(steps) = record["steps"].as_array_mut() {
            steps.push(probe);
        }
        let (resolved, ray_query) = self.read_probe(adapter, &dir);
        record["probe"] = resolved;
        record["pace"] = json!({ "startup_s": pace.startup, "frame_ms": pace.ms });
        let traced = ray_query == Some(true);
        if !profile.traced {
            record["traced_skipped"] = json!("a CPU rasterizer runs Raster only");
        } else if !traced {
            record["traced_skipped"] = json!(match ray_query {
                Some(_) => "the probe's capabilities say the device has no ray query",
                None => "the probe wrote no capabilities",
            });
            self.say("    Esta tarjeta no hace trazado de rayos aquí: solo se prueba Raster.");
        }
        self.summary["adapters"][index] = record.clone();
        self.checkpoint();
        (record, traced, pace)
    }

    /// Steps 2 to 6 on one adapter.
    fn rest(
        &mut self,
        index: usize,
        total: usize,
        adapter: &Adapter,
        (mut record, traced, pace): (Value, bool, Pace),
    ) {
        let dir = self.results.join(slug(index, &adapter.name));
        let steps = plan(adapter, traced);
        let remaining: f64 = steps[1..].iter().map(|step| estimate(step, pace)).sum();
        self.say(format!(
            "\n== Tarjeta {} de {total}: {} (unos {} minutos) ==",
            index + 1,
            adapter.name,
            (remaining / 60.0).ceil()
        ));
        for (number, step) in steps.iter().enumerate().skip(1) {
            self.say(format!(
                "  Paso {}: {} ({} pruebas, unos {} min)",
                number + 1,
                step.title,
                step.runs.len() * step.scenes.len() * step.resolutions.len(),
                (estimate(step, pace) / 60.0).ceil()
            ));
            let result = self.run_step(step, adapter, &dir);
            if let Some(steps) = record["steps"].as_array_mut() {
                steps.push(result);
            }
            match step.key {
                "presets" => {
                    let table = self.kit.join("budgets.toml");
                    Self::budget(
                        &dir.join("presets"),
                        "--table",
                        &table,
                        &dir.join("budgets.md"),
                    );
                }
                "golden" => {
                    let golden = self.kit.join("golden");
                    Self::budget(
                        &dir.join("golden"),
                        "--golden",
                        &golden,
                        &dir.join("golden.md"),
                    );
                }
                _ => {}
            }
            self.summary["adapters"][index] = record.clone();
            self.checkpoint();
        }
        let fsr = Self::fsr_lines(&dir);
        if !fsr.is_empty() {
            record["fsr_log"] = json!(fsr);
            record["fsr_provider"] = json!(fsr_provider(&fsr));
        }
        self.copy_to_look_at(index, adapter, &dir);
        self.summary["adapters"][index] = record;
        self.checkpoint();
    }

    /// Asks the five questions, or takes them from `--answers`; a closed input answers "sin respuesta".
    fn ask(&mut self, given: Option<&str>) {
        let given: Option<Vec<&str>> =
            given.map(|answers| answers.split(',').map(str::trim).collect());
        let interactive = given.is_none() && std::io::stdin().is_terminal();
        if interactive {
            self.say(format!(
                "\nYa terminaron las pruebas. Mira las imágenes de la carpeta «para-mirar» dentro de\n{}\n\
                 y piensa en lo que viste en las ventanas. Responde s (sí), n (no) o ? (no sé).",
                self.results.display()
            ));
        }
        let stdin = std::io::stdin();
        let mut lines = stdin.lock().lines();
        let mut answers = serde_json::Map::new();
        for (index, (key, question)) in QUESTIONS.iter().enumerate() {
            let answer = match &given {
                Some(given) => normalise(given.get(index).copied().unwrap_or("")),
                None if interactive => loop {
                    print!("\n{}. {question} [s/n/?]: ", index + 1);
                    let _ = std::io::stdout().flush();
                    match lines.next() {
                        Some(Ok(line)) => match normalise(&line) {
                            "sin respuesta" => println!("Escribe s, n o ?"),
                            answer => break answer,
                        },
                        _ => break "sin respuesta",
                    }
                },
                None => "sin respuesta",
            };
            if let Some(log) = &mut self.log {
                let _ = writeln!(log, "{}. {question} -> {answer}", index + 1);
            }
            answers.insert(
                (*key).into(),
                json!({ "question": question, "answer": answer }),
            );
        }
        self.summary["answers"] = Value::Object(answers);
    }
}

/// An answer as recorded: "sí", "no", "no sé", or "sin respuesta" for anything else.
fn normalise(answer: &str) -> &'static str {
    match answer.trim().to_lowercase().as_str() {
        "s" | "si" | "sí" | "y" | "yes" => "sí",
        "n" | "no" => "no",
        "?" | "ns" | "no se" | "no sé" => "no sé",
        _ => "sin respuesta",
    }
}

/// The messages of the lines `bevy_anti_alias`'s FSR code logged, without Bevy's time stamp, level and target.
fn fsr_log_lines(log: &str) -> Vec<String> {
    log.lines()
        .filter_map(|line| line.find("FSR: ").map(|at| line[at..].to_owned()))
        .collect()
}

/// The FSR version the DLL picked, from `FSR: the DLL runs FSR <version> (provider id …) on this GPU`.
fn fsr_provider(lines: &[String]) -> Option<&str> {
    lines
        .iter()
        .find_map(|line| line.strip_prefix("FSR: the DLL runs FSR "))
        .and_then(|rest| rest.split_whitespace().next())
}

/// Scrubs every text file under `dir` in place.
fn scrub_folder(dir: &Path, scrub: &Scrub, problems: &mut Vec<String>) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            scrub_folder(&path, scrub, problems);
        } else if is_text(&path) {
            match fs::read(&path) {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let clean = scrub.apply(&text);
                    if clean != text
                        && let Err(error) = fs::write(&path, clean)
                    {
                        problems.push(format!("{}: {error}", path.display()));
                    }
                }
                Err(error) => problems.push(format!("{}: {error}", path.display())),
            }
        }
    }
}

/// The `friend` subcommand.
pub fn friend(command: FriendCommand) -> Result<bool, String> {
    let me =
        std::env::current_exe().map_err(|error| format!("cannot find this program: {error}"))?;
    let kit = me.parent().ok_or("this program has no folder")?.to_owned();
    let testbed = command.testbed.clone().unwrap_or_else(|| {
        kit.join("gfx_testbed")
            .with_extension(std::env::consts::EXE_EXTENSION)
    });
    let testbed = std::path::absolute(&testbed).map_err(|error| error.to_string())?;
    if !testbed.exists() {
        return Err(format!(
            "{} no está: descomprime el kit entero y ejecútalo desde su carpeta",
            testbed.display()
        ));
    }
    let assets = testbed
        .parent()
        .ok_or("the testbed has no folder")?
        .to_owned();

    // DLSS, NRD and FSR are Vulkan only, and on Windows a name match could land on DX12. The testbed reads its
    // assets beside itself. The process environment cannot be changed safely here, so this program starts
    // itself again with both set: every run inherits them through `run`.
    let vulkan = std::env::var("WGPU_BACKEND").is_ok_and(|backend| backend == "vulkan");
    let asset_root =
        std::env::var_os("BEVY_ASSET_ROOT").is_some_and(|root| Path::new(&root) == assets);
    if !vulkan || !asset_root {
        let status = Command::new(&me)
            .args(std::env::args_os().skip(1))
            .env("WGPU_BACKEND", "vulkan")
            .env("BEVY_ASSET_ROOT", &assets)
            .status()
            .map_err(|error| format!("cannot start {}: {error}", me.display()))?;
        return Ok(status.success());
    }

    let headless = match &command.headless {
        Some(wrapper) => {
            let wrapper = std::path::absolute(wrapper).map_err(|error| error.to_string())?;
            // `run` finds the wrapper as `tools/gfx-headless.sh` under its working folder.
            let root = wrapper
                .parent()
                .and_then(Path::parent)
                .filter(|root| root.join("tools/gfx-headless.sh") == wrapper && wrapper.exists())
                .ok_or_else(|| format!("{} is not a tools/gfx-headless.sh", wrapper.display()))?
                .to_owned();
            std::env::set_current_dir(&root).map_err(|error| error.to_string())?;
            Some(wrapper)
        }
        None => {
            std::env::set_current_dir(&kit).map_err(|error| error.to_string())?;
            None
        }
    };

    let parent = match &command.out {
        Some(out) => std::path::absolute(out).map_err(|error| error.to_string())?,
        None => kit.clone(),
    };
    let date = today();
    let mut results = parent.join(format!("resultados-{date}"));
    let mut attempt = 1;
    while results.exists() {
        attempt += 1;
        results = parent.join(format!("resultados-{date}-{attempt}"));
    }
    fs::create_dir_all(&results).map_err(|error| format!("{}: {error}", results.display()))?;
    let zip = results.with_extension("zip");
    let log = fs::File::create(results.join("console.txt")).ok();

    let mut friend = Friend {
        testbed,
        kit: kit.clone(),
        results,
        zip,
        headless,
        scrub: Scrub::from_machine(&kit),
        log,
        summary: json!({
            "kit": env!("CARGO_PKG_VERSION"),
            "date": date,
            "os": os_description(),
            "arch": std::env::consts::ARCH,
            "adapters": [],
        }),
    };
    let started = Instant::now();
    friend.say("Prueba de gráficos: no toques nada mientras se abren y cierran ventanas.");
    if let Some(wrapper) = &friend.headless {
        let wrapper = wrapper.display().to_string();
        friend.say(format!(
            "Cada ventana se abre dentro de {wrapper}: ninguna se ve."
        ));
    }
    friend.say("\nPaso 0: una prueba corta, para tener algo que enviar aunque algo falle después.");
    let mut adapters = match friend.smoke() {
        Ok(found) => found,
        Err(error) => {
            friend.summary["smoke"] = json!({ "error": error });
            Vec::new()
        }
    };
    if adapters.is_empty() {
        // Without the trace lines, at least the adapter the smoke run used.
        let reported = &friend.summary["smoke"]["adapter"];
        if let Some(name) = reported["name"].as_str() {
            adapters.push(Adapter {
                name: name.into(),
                vendor: reported["vendor"].as_u64().unwrap_or(0) as u32,
                device_type: reported["device_type"].as_str().unwrap_or_default().into(),
                driver: reported["driver"].as_str().unwrap_or_default().into(),
                driver_info: reported["driver_info"].as_str().unwrap_or_default().into(),
            });
            friend.summary["adapters_note"] = json!("only the smoke run's adapter could be listed");
        }
    }
    if !command.adapter.is_empty() {
        adapters.retain(|adapter| {
            command
                .adapter
                .iter()
                .any(|wanted| adapter.name.to_lowercase().contains(&wanted.to_lowercase()))
        });
    }
    friend.summary["listed"] = json!(adapters
        .iter()
        .map(|adapter| format!(
            "{} ({}, {})",
            adapter.name, adapter.device_type, adapter.driver_info
        ))
        .collect::<Vec<_>>());
    friend.summary["adapters"] = json!(vec![Value::Null; adapters.len()]);
    friend.checkpoint();
    if adapters.is_empty() {
        friend.say("No se encontró ninguna tarjeta gráfica con Vulkan. Envía el zip igualmente.");
    }

    friend.say(format!("\nTarjetas encontradas: {}", adapters.len()));
    for adapter in &adapters {
        friend.say(format!("  - {} ({})", adapter.name, adapter.profile().name));
    }
    if command.dry_run {
        for adapter in &adapters {
            for step in plan(adapter, true) {
                friend.say(format!(
                    "\n{} / {}:\n{}",
                    adapter.name,
                    step.key,
                    step.matrix(&adapter.match_name())
                ));
            }
        }
        friend.checkpoint();
        return Ok(true);
    }

    // Every probe first: they time each adapter, and the estimate rests on that.
    friend.say("\nPaso 1: una prueba corta en cada tarjeta, para calcular cuánto tardará todo.");
    let probes: Vec<_> = adapters
        .iter()
        .enumerate()
        .map(|(index, adapter)| friend.probe(index, adapter))
        .collect();
    let estimate_all: f64 = adapters
        .iter()
        .zip(&probes)
        .map(|(adapter, (_, traced, pace))| {
            plan(adapter, *traced)[1..]
                .iter()
                .map(|step| estimate(step, *pace))
                .sum::<f64>()
        })
        .sum();
    friend.say(format!(
        "\nTiempo estimado para el resto: unos {} minutos (puede tardar más).",
        (estimate_all / 60.0).ceil()
    ));
    friend.summary["estimated_minutes"] = json!((estimate_all / 60.0).ceil());
    let total = adapters.len();
    for (index, (adapter, probe)) in adapters.iter().zip(probes).enumerate() {
        friend.rest(index, total, adapter, probe);
    }
    friend.summary["minutes"] = json!((started.elapsed().as_secs_f64() / 6.0).round() / 10.0);
    // The zip is complete before the questions: a person who walks away still has it.
    friend.summary["answers"] = json!("not asked yet");
    friend.checkpoint();
    friend.ask(command.answers.as_deref());
    friend.checkpoint();
    friend.say(format!(
        "\nListo. Envía este archivo:\n  {}\n(La carpeta {} tiene lo mismo y más imágenes; no hace falta enviarla.)",
        friend.zip.display(),
        friend.results.display()
    ));
    friend.checkpoint();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn adapters_are_read_from_bevy_log_lines() {
        let log = "\
2026-11-24T10:00:00Z TRACE bevy_render::renderer: Checking adapter: AdapterInfo { name: \"llvmpipe (LLVM 23.1.5, 256 bits)\", vendor: 65541, device: 0, device_type: Cpu, device_pci_bus_id: \"\", driver: \"llvmpipe\", driver_info: \"Mesa 26.3.1 (LLVM 23.1.5)\", backend: Vulkan, transient_saves_memory: false }
2026-11-24T10:00:00Z TRACE bevy_render::renderer: Checking adapter: AdapterInfo { name: \"NVIDIA GeForce RTX 5070\", vendor: 4318, device: 12036, device_type: DiscreteGpu, device_pci_bus_id: \"0000:01:00.0\", driver: \"NVIDIA\", driver_info: \"620.73.04\", backend: Vulkan, transient_saves_memory: false }
2026-11-24T10:00:00Z TRACE bevy_render::renderer: Checking adapter: AdapterInfo { name: \"AMD Radeon Graphics (RADV)\", vendor: 4098, device: 5710, device_type: IntegratedGpu, driver: \"radv\", driver_info: \"Mesa 26.3.1\", backend: Vulkan }
2026-11-24T10:00:01Z  INFO bevy_render::renderer: AdapterInfo { name: \"NVIDIA GeForce RTX 5070\", vendor: 4318, device: 12036, device_type: DiscreteGpu, driver: \"NVIDIA\", driver_info: \"620.73.04\", backend: Vulkan }
2026-11-24T10:00:01Z  INFO bevy_render::renderer: AdapterInfo { name: \"NVIDIA GeForce RTX 5070/PCIe/SSE2\", vendor: 4318, device: 0, device_type: Other, driver: \"\", driver_info: \"\", backend: Gl }
";
        let adapters = parse_adapters(log);
        let names: Vec<&str> = adapters
            .iter()
            .map(|adapter| adapter.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "NVIDIA GeForce RTX 5070",
                "AMD Radeon Graphics (RADV)",
                "llvmpipe (LLVM 23.1.5, 256 bits)"
            ]
        );
        assert!(adapters[0].is_nvidia());
        assert_eq!(adapters[0].profile().name, "full");
        assert!(adapters[1].is_amd());
        assert_eq!(adapters[1].profile().name, "small");
        assert_eq!(adapters[2].profile().name, "cpu");
        assert_eq!(adapters[2].driver_info, "Mesa 26.3.1 (LLVM 23.1.5)");
    }

    #[test]
    fn match_name_keeps_a_file_name_safe_piece() {
        let adapter = Adapter {
            name: "AMD Radeon RX 6600/PCIe".into(),
            vendor: 0x1002,
            device_type: "DiscreteGpu".into(),
            driver: String::new(),
            driver_info: String::new(),
        };
        assert_eq!(adapter.match_name(), "AMD Radeon RX 6600");
    }

    #[test]
    fn scrub_removes_paths_and_names_in_every_form() {
        let scrub = Scrub {
            paths: vec![
                ("C:\\Users\\Juana\\kit".into(), "<carpeta-del-kit>"),
                ("C:\\\\Users\\\\Juana".into(), "<inicio>"),
                ("C:\\Users\\Juana".into(), "<inicio>"),
                ("/home/juana".into(), "<inicio>"),
            ],
            // Longest first, as `from_machine` sorts them.
            words: vec![
                ("torre-juana".into(), "<equipo>"),
                ("Juana".into(), "<usuario>"),
            ],
        };
        let text = "out c:\\users\\juana\\kit\\x and \"C:\\\\Users\\\\Juana\\\\a\" /home/juana/b \
                    host torre-juana user Juana JuanaPC";
        assert_eq!(
            scrub.apply(text),
            "out <carpeta-del-kit>\\x and \"<inicio>\\\\a\" <inicio>/b host <equipo> user <usuario> JuanaPC"
        );
    }

    #[test]
    fn plan_follows_the_adapter_and_the_probe() {
        let rtx = Adapter {
            name: "NVIDIA GeForce RTX 3060".into(),
            vendor: 0x10de,
            device_type: "DiscreteGpu".into(),
            driver: String::new(),
            driver_info: String::new(),
        };
        let keys: Vec<&str> = plan(&rtx, true).iter().map(|step| step.key).collect();
        assert_eq!(keys, ["probe", "presets", "golden", "rtx", "cycle"]);
        let keys: Vec<&str> = plan(&rtx, false).iter().map(|step| step.key).collect();
        assert_eq!(keys, ["probe", "presets", "golden"]);
        let presets = &plan(&rtx, true)[1];
        assert_eq!(presets.resolutions, ["1920x1080", "2560x1440"]);
        assert!(presets.runs.iter().any(|run| run.name == "rt-full-nrd-tsr"));
        // The generated matrix is one `run` reads.
        assert!(presets.matrix("NVIDIA GeForce RTX 3060").contains(
            "(name: \"raster-taa\", args: [\"--preset\", \"raster\", \"--upscaler\", \"off\"], warmup: 300, frames: 600)"
        ));
    }

    #[test]
    fn the_fsr_provider_is_read_from_the_testbed_log() {
        // The format of `bevy_anti_alias/src/fsr/mod.rs`, as Bevy's log writes it.
        let log = "\
2026-11-24T23:09:09.231748Z  INFO bevy_anti_alias::fsr: FSR: device prepared (VK_KHR_compute_shader_derivatives)
2026-11-24T23:09:09.531748Z  INFO bevy_anti_alias::fsr: FSR: the DLL runs FSR 4.1.2 (provider id 0x1234) on this GPU
2026-11-24T23:09:10.082039Z  INFO bevy_pbr::cluster: GPU clustering is supported on this device.
";
        let lines = fsr_log_lines(log);
        assert_eq!(lines.len(), 2);
        assert_eq!(fsr_provider(&lines), Some("4.1.2"));
        assert_eq!(fsr_provider(&lines[..1]), None);
    }

    #[test]
    fn answers_are_normalised() {
        assert_eq!(normalise(" Sí "), "sí");
        assert_eq!(normalise("n"), "no");
        assert_eq!(normalise("?"), "no sé");
        assert_eq!(normalise("quizá"), "sin respuesta");
    }

    #[test]
    fn zips_hold_text_and_the_images_to_look_at() {
        assert!(in_zip(
            "1-nvidia/presets/NVIDIA/interior/1920x1080/raster-taa/report.json"
        ));
        assert!(!in_zip(
            "1-nvidia/presets/NVIDIA/interior/1920x1080/raster-taa/a.png"
        ));
        assert!(in_zip(
            "1-nvidia/golden/NVIDIA/interior/1920x1080/raster/a.png"
        ));
        assert!(in_zip("para-mirar/1-interior-raster-girando.png"));
        assert!(in_zip("0-smoke/a.png"));
    }
}
