//! `gfx-check run`: the testbed over a matrix of adapters, scenes, flag sets and resolutions.

use crate::{log_gate, metric, report::Report, sheet};
use argh::FromArgs;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// Runs the testbed once per cell of a matrix file and gates every run.
///
/// Writes `results.csv`, `results.md` and `results.json` and one folder per run under the output folder. On Linux
/// each run goes through `tools/gfx-headless.sh`, so no window shows and one run uses the GPU at a time.
#[derive(FromArgs)]
#[argh(subcommand, name = "run")]
pub struct RunCommand {
    /// the matrix file (RON); see `tools/gfx-check/smoke.ron`
    #[argh(positional)]
    matrix: PathBuf,
    /// folder for the results; default `target/gfx-check/<matrix name>`
    #[argh(option)]
    out: Option<PathBuf>,
    /// the testbed binary; default `target/release/examples/gfx_testbed`
    #[argh(option)]
    testbed: Option<PathBuf>,
    /// run only this adapter of the matrix; may be repeated
    #[argh(option)]
    adapter: Vec<String>,
    /// open a real window instead of going through `tools/gfx-headless.sh`
    #[argh(switch)]
    windowed: bool,
}

#[derive(Deserialize)]
struct Matrix {
    /// The GPUs to run on, each with the resolutions it is given.
    adapters: Vec<AdapterEntry>,
    /// Values of the testbed's `--scene`.
    scenes: Vec<String>,
    /// Warm-up frames of a run that does not set its own.
    warmup: u32,
    /// Measured frames of a run that does not set its own.
    frames: u32,
    /// Seconds after which a run is stopped.
    seconds: u64,
    /// The flag sets.
    runs: Vec<RunEntry>,
}

#[derive(Deserialize)]
struct AdapterEntry {
    /// A substring of the adapter's name, passed as `WGPU_ADAPTER_NAME`.
    name: String,
    /// `WxH` window sizes. A slow adapter gets small ones.
    resolutions: Vec<String>,
}

#[derive(Deserialize)]
struct RunEntry {
    /// The run's name: its folder and its row in the results.
    name: String,
    /// Flags passed to the testbed after the matrix's own (`--scene`, `--res`, `--warmup`, `--frames`, `--out`).
    args: Vec<String>,
    #[serde(default)]
    warmup: Option<u32>,
    #[serde(default)]
    frames: Option<u32>,
    /// Adapter names this run is limited to, as written in `adapters`; absent means all.
    #[serde(default)]
    adapters: Option<Vec<String>>,
    /// The earlier run whose `a.png` this run's is compared with: the "error" column.
    #[serde(default)]
    reference: Option<String>,
    /// Largest error allowed, in points.
    #[serde(default)]
    max_error: Option<f64>,
    /// Largest difference allowed between two consecutive frames, in points.
    #[serde(default)]
    max_flicker: Option<f64>,
    /// Largest median frame time allowed, in milliseconds.
    #[serde(default)]
    max_frame_ms: Option<f64>,
}

/// One row of the results.
#[derive(Serialize, Deserialize)]
pub struct Cell {
    /// The adapter name the matrix asked for.
    pub adapter: String,
    /// The adapter and driver the run reported.
    pub adapter_reported: String,
    /// The scene.
    pub scene: String,
    /// The window size, `WxH`.
    pub resolution: String,
    /// The run's name in the matrix.
    pub run: String,
    /// The run's own flags.
    pub args: Vec<String>,
    /// Warm-up frames.
    pub warmup: u32,
    /// Measured frames.
    pub frames: u32,
    /// Median, 95th and 99th percentile frame time in milliseconds.
    pub frame_ms: Option<[f64; 3]>,
    /// Error against the reference run, in points.
    pub error: Option<f64>,
    /// Difference between two consecutive frames, in points.
    pub flicker: Option<f64>,
    /// Mean luma of `a.png`, 0 to 1.
    pub mean_luma: Option<f64>,
    /// Peak GPU memory in MiB.
    pub peak_memory_mib: Option<f64>,
    /// Median GPU milliseconds per render pass.
    pub gpu_pass_ms: Vec<(String, f64)>,
    /// Why the cell failed; empty when it passed.
    pub failures: Vec<String>,
}

/// The offending lines of a run folder's log: the captured console when there is one (it also holds what the
/// process wrote outside Bevy's logger), otherwise the testbed's own `log.txt`.
pub fn log_offences(dir: &Path, traced: bool) -> Result<Vec<String>, String> {
    let file = ["console.txt", "log.txt"]
        .map(|name| dir.join(name))
        .into_iter()
        .find(|file| file.exists())
        .ok_or_else(|| format!("{} holds no console.txt or log.txt", dir.display()))?;
    let bytes = fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    Ok(log_gate::gate(&String::from_utf8_lossy(&bytes), traced))
}

/// Starts one testbed run and waits for it, stopping it after `seconds`. Returns what it exited with.
fn launch(command: &mut Command, console: &Path, seconds: u64) -> Result<String, String> {
    let out =
        fs::File::create(console).map_err(|error| format!("{}: {error}", console.display()))?;
    let err = out.try_clone().map_err(|error| error.to_string())?;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|error| format!("could not start {:?}: {error}", command.get_program()))?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Ok(match status.code() {
                Some(0) => String::new(),
                Some(124) => "stopped by the time limit".to_owned(),
                Some(code) => format!("exit code {code}"),
                None => format!("{status}"),
            });
        }
        if Instant::now() > deadline {
            // The child is ours: stop it and nothing else.
            let _ = child.kill();
            let _ = child.wait();
            return Ok(format!("stopped after {seconds} s"));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn evaluate(cell: &mut Cell, dir: &Path, entry: &RunEntry, reference: Option<PathBuf>) {
    let report = match Report::read(dir) {
        Ok(report) => report,
        Err(error) => {
            cell.failures.push(error);
            return;
        }
    };
    cell.adapter_reported = format!("{} ({})", report.adapter.name, report.adapter.driver_info);
    cell.failures.extend(report.failures.iter().cloned());
    // WGPU_ADAPTER_NAME falls back to another GPU when nothing matches; a result from the wrong card is no result.
    if !report
        .adapter
        .name
        .to_lowercase()
        .contains(&cell.adapter.to_lowercase())
    {
        cell.failures.push(format!(
            "ran on {}, not on {}",
            report.adapter.name, cell.adapter
        ));
    }
    match log_offences(dir, report.traced) {
        Ok(offences) => cell.failures.extend(offences),
        Err(error) => cell.failures.push(error),
    }
    if let Some(frame) = &report.frame_ms {
        cell.frame_ms = Some([frame.median, frame.p95, frame.p99]);
        if let Some(max) = entry.max_frame_ms
            && frame.median > max
        {
            cell.failures
                .push(format!("median frame {:.3} ms over {max}", frame.median));
        }
    }
    cell.peak_memory_mib = report
        .gpu_memory
        .as_ref()
        .map(|memory| memory.peak_reserved_bytes as f64 / (1024.0 * 1024.0));
    cell.gpu_pass_ms = report
        .gpu_pass_ms
        .iter()
        .map(|(name, ms)| (name.clone(), *ms))
        .collect();

    let a = match metric::load(&dir.join("a.png")) {
        Ok(a) => a,
        Err(error) => {
            cell.failures.push(error);
            return;
        }
    };
    let stats = metric::stats(&a);
    cell.mean_luma = Some(stats.mean_luma);
    if stats.mean_luma < metric::DEFAULT_MIN_LUMA {
        cell.failures
            .push(format!("a.png is black (mean luma {:.4})", stats.mean_luma));
    }
    if report.camera.kind == "orbit" {
        if let Err(error) = sheet::write(dir, &dir.join("sheet.png")) {
            cell.failures.push(error);
        }
    } else {
        // Two consecutive frames differ by the camera's own motion on an orbit, so flicker is a still camera's.
        match metric::load(&dir.join("b.png")).and_then(|b| metric::nrmse_percent(&a, &b)) {
            Ok(points) => {
                cell.flicker = Some(points);
                if let Some(max) = entry.max_flicker
                    && points > max
                {
                    cell.failures
                        .push(format!("flicker {points:.3} % over {max}"));
                }
            }
            Err(error) => cell.failures.push(error),
        }
    }
    if let Some(reference) = reference {
        match metric::load(&reference.join("a.png")).and_then(|r| metric::nrmse_percent(&a, &r)) {
            Ok(points) => {
                cell.error = Some(points);
                if let Some(max) = entry.max_error
                    && points > max
                {
                    cell.failures
                        .push(format!("error {points:.3} % over {max}"));
                }
            }
            Err(error) => cell.failures.push(error),
        }
    }
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_owned()
    }
}

fn number(value: Option<f64>, digits: usize) -> String {
    value.map_or(String::new(), |value| format!("{value:.digits$}"))
}

fn write_results(out: &Path, cells: &[Cell]) -> Result<(), String> {
    let mut csv = String::from(
        "adapter,adapter_reported,scene,resolution,run,status,frame_ms_median,frame_ms_p95,frame_ms_p99,\
         error_pct,flicker_pct,mean_luma,peak_memory_mib,gpu_pass_ms,failures\n",
    );
    let mut md = String::from(
        "| adapter | scene | resolution | run | status | frame ms (median / p95 / p99) | error % | flicker % | \
         memory MiB |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    for cell in cells {
        let status = if cell.failures.is_empty() {
            "pass"
        } else {
            "FAIL"
        };
        let frame = |index: usize| number(cell.frame_ms.map(|frame| frame[index]), 3);
        let passes: Vec<String> = cell
            .gpu_pass_ms
            .iter()
            .map(|(name, ms)| format!("{name}={ms:.3}"))
            .collect();
        let fields = [
            cell.adapter.clone(),
            cell.adapter_reported.clone(),
            cell.scene.clone(),
            cell.resolution.clone(),
            cell.run.clone(),
            status.to_owned(),
            frame(0),
            frame(1),
            frame(2),
            number(cell.error, 3),
            number(cell.flicker, 3),
            number(cell.mean_luma, 4),
            number(cell.peak_memory_mib, 0),
            passes.join(" "),
            cell.failures.join(" | "),
        ];
        let fields: Vec<String> = fields.iter().map(|field| csv_field(field)).collect();
        let _ = writeln!(csv, "{}", fields.join(","));
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {status} | {} / {} / {} | {} | {} | {} |",
            cell.adapter,
            cell.scene,
            cell.resolution,
            cell.run,
            frame(0),
            frame(1),
            frame(2),
            number(cell.error, 2),
            number(cell.flicker, 2),
            number(cell.peak_memory_mib, 0),
        );
    }
    let failed: Vec<&Cell> = cells
        .iter()
        .filter(|cell| !cell.failures.is_empty())
        .collect();
    if !failed.is_empty() {
        md.push_str("\nFailures:\n\n");
        for cell in failed {
            for failure in &cell.failures {
                let _ = writeln!(
                    md,
                    "- {} / {} / {} / {}: {failure}",
                    cell.adapter, cell.scene, cell.resolution, cell.run
                );
            }
        }
    }
    let json = serde_json::to_string_pretty(cells).map_err(|error| error.to_string())?;
    for (name, text) in [
        ("results.csv", csv),
        ("results.md", md),
        ("results.json", json),
    ] {
        let file = out.join(name);
        fs::write(&file, text).map_err(|error| format!("{}: {error}", file.display()))?;
    }
    Ok(())
}

/// The `run` subcommand.
pub fn run(command: RunCommand) -> Result<bool, String> {
    let text = fs::read_to_string(&command.matrix)
        .map_err(|error| format!("{}: {error}", command.matrix.display()))?;
    let matrix: Matrix =
        ron::from_str(&text).map_err(|error| format!("{}: {error}", command.matrix.display()))?;
    for (index, entry) in matrix.runs.iter().enumerate() {
        if let Some(reference) = &entry.reference
            && !matrix.runs[..index]
                .iter()
                .any(|run| &run.name == reference)
        {
            return Err(format!(
                "run {} names the reference {reference}, which is not an earlier run",
                entry.name
            ));
        }
    }
    for name in &command.adapter {
        if !matrix.adapters.iter().any(|adapter| &adapter.name == name) {
            return Err(format!("the matrix has no adapter named {name}"));
        }
    }

    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let testbed = cwd.join(command.testbed.unwrap_or_else(|| {
        PathBuf::from("target/release/examples/gfx_testbed")
            .with_extension(std::env::consts::EXE_EXTENSION)
    }));
    if !testbed.exists() {
        return Err(format!(
            "{} does not exist; build it with `cargo build --release --example gfx_testbed --features \
             bevy_solari,area_light_luts` or pass --testbed",
            testbed.display()
        ));
    }
    // A run must never put a window on the desktop of whoever works at this machine by accident: off Windows
    // it is the wrapper, or a real window asked for by name.
    let headless = cwd.join("tools/gfx-headless.sh");
    if !command.windowed && (cfg!(windows) || !headless.exists()) {
        return Err(format!(
            "{} is not here: run from the repository root, or pass --windowed to open a real window",
            headless.display()
        ));
    }
    let stem = command.matrix.file_stem().map_or_else(
        || "matrix".into(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let out = cwd.join(
        command
            .out
            .unwrap_or_else(|| PathBuf::from("target/gfx-check").join(stem)),
    );

    let mut cells = Vec::new();
    for adapter in &matrix.adapters {
        if !command.adapter.is_empty() && !command.adapter.contains(&adapter.name) {
            continue;
        }
        for scene in &matrix.scenes {
            for resolution in &adapter.resolutions {
                let group = out.join(&adapter.name).join(scene).join(resolution);
                for entry in &matrix.runs {
                    if entry
                        .adapters
                        .as_ref()
                        .is_some_and(|only| !only.contains(&adapter.name))
                    {
                        continue;
                    }
                    let dir = group.join(&entry.name);
                    fs::create_dir_all(&dir)
                        .map_err(|error| format!("{}: {error}", dir.display()))?;
                    // What an earlier run left in this cell must not be read as this run's.
                    for stale in fs::read_dir(&dir)
                        .map_err(|error| error.to_string())?
                        .flatten()
                    {
                        let _ = fs::remove_file(stale.path());
                    }
                    let warmup = entry.warmup.unwrap_or(matrix.warmup);
                    let frames = entry.frames.unwrap_or(matrix.frames);

                    let mut process = if command.windowed {
                        Command::new(&testbed)
                    } else {
                        let mut process = Command::new(&headless);
                        process
                            .args([
                                "--seconds",
                                &matrix.seconds.to_string(),
                                "--size",
                                resolution,
                            ])
                            .arg(&testbed);
                        process
                    };
                    process
                        .args(["--scene", scene, "--res", resolution])
                        .args(["--warmup", &warmup.to_string()])
                        .args(["--frames", &frames.to_string()])
                        .arg("--out")
                        .arg(&dir)
                        .args(&entry.args)
                        .env("WGPU_ADAPTER_NAME", &adapter.name);
                    // A binary started by path looks for `assets/` beside itself, not in the repository.
                    if std::env::var_os("BEVY_ASSET_ROOT").is_none() && cwd.join("assets").is_dir()
                    {
                        process.env("BEVY_ASSET_ROOT", &cwd);
                    }

                    let mut cell = Cell {
                        adapter: adapter.name.clone(),
                        adapter_reported: String::new(),
                        scene: scene.clone(),
                        resolution: resolution.clone(),
                        run: entry.name.clone(),
                        args: entry.args.clone(),
                        warmup,
                        frames,
                        frame_ms: None,
                        error: None,
                        flicker: None,
                        mean_luma: None,
                        peak_memory_mib: None,
                        gpu_pass_ms: Vec::new(),
                        failures: Vec::new(),
                    };
                    // The wrapper has its own limit; ours also covers its wait for the GPU lock, generously.
                    let limit = matrix.seconds + if command.windowed { 0 } else { 3600 };
                    let exit = launch(&mut process, &dir.join("console.txt"), limit)?;
                    if !exit.is_empty() {
                        cell.failures
                            .push(format!("the testbed ended with: {exit}"));
                    }
                    evaluate(
                        &mut cell,
                        &dir,
                        entry,
                        entry.reference.as_ref().map(|name| group.join(name)),
                    );
                    println!(
                        "{} {} {} {}: {}{}",
                        cell.adapter,
                        cell.scene,
                        cell.resolution,
                        cell.run,
                        if cell.failures.is_empty() {
                            "pass"
                        } else {
                            "FAIL"
                        },
                        cell.failures
                            .iter()
                            .fold(String::new(), |all, failure| all + "\n    " + failure),
                    );
                    cells.push(cell);
                    write_results(&out, &cells)?;
                }
            }
        }
    }
    if cells.is_empty() {
        return Err("the matrix selected no run".to_owned());
    }
    let failed = cells
        .iter()
        .filter(|cell| !cell.failures.is_empty())
        .count();
    println!(
        "{} runs, {failed} failed; results in {}",
        cells.len(),
        out.join("results.md").display()
    );
    Ok(failed == 0)
}
