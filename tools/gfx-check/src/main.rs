//! Runs `gfx_testbed` over a matrix of adapters, scenes, flags and resolutions, and gates what each run wrote:
//! its images, its timings and its log.
//!
//! Every subcommand exits with 0 when its gate passes, 1 when it fails and 2 when it could not be evaluated.

#![expect(clippy::print_stdout, reason = "Allowed in tools.")]
#![expect(clippy::print_stderr, reason = "Allowed in tools.")]

extern crate alloc;

mod budgets;
mod calibration;
mod flicker;
mod friend;
mod log_gate;
mod metric;
mod report;
mod resize;
mod run;
mod sheet;

use argh::FromArgs;
use report::Report;
use std::{path::PathBuf, process::ExitCode};

/// Runs the graphics testbed and gates what it wrote.
#[derive(FromArgs)]
struct Cli {
    #[argh(subcommand)]
    command: Command,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Command {
    Run(run::RunCommand),
    Image(ImageCommand),
    Flicker(flicker::FlickerCommand),
    Motion(MotionCommand),
    Sheet(sheet::SheetCommand),
    Budget(BudgetCommand),
    LogGate(LogGateCommand),
    Calibration(calibration::CalibrationCommand),
    Friend(friend::FriendCommand),
}

/// Measures one image, or two against each other.
#[derive(FromArgs)]
#[argh(subcommand, name = "image")]
struct ImageCommand {
    #[argh(subcommand)]
    command: ImageSubcommand,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum ImageSubcommand {
    Stats(ImageStatsCommand),
    Diff(ImageDiffCommand),
    Resize(resize::ImageResizeCommand),
}

/// Prints an image's size and brightness, and fails when it is black.
#[derive(FromArgs)]
#[argh(subcommand, name = "stats")]
struct ImageStatsCommand {
    /// the image
    #[argh(positional)]
    image: PathBuf,
    /// mean luma (0 to 1) under which the image counts as black; default 0.01
    #[argh(option, default = "metric::DEFAULT_MIN_LUMA")]
    min_luma: f64,
}

/// Prints the normalised RMSE between two tonemapped images, in percent of full scale.
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
struct ImageDiffCommand {
    /// the first image
    #[argh(positional)]
    a: PathBuf,
    /// the second image
    #[argh(positional)]
    b: PathBuf,
    /// fail when the difference is above this many points
    #[argh(option)]
    max: Option<f64>,
}

/// Compares a frame of an orbit run against a still run at the same camera pose.
///
/// The still run is made with `--pose K`; the orbit run must hold a capture at orbit frame K (its `a.png` is at
/// frame `warmup + frames`).
#[derive(FromArgs)]
#[argh(subcommand, name = "motion")]
struct MotionCommand {
    /// the folder of the orbit run
    #[argh(positional)]
    orbit: PathBuf,
    /// the folder of the still run at the same pose, usually the reference
    #[argh(positional)]
    still: PathBuf,
    /// fail when the difference is above this many points
    #[argh(option)]
    max: Option<f64>,
}

/// Gates a run's frame time and GPU pass times; with `--table` or `--golden`, every run of a results folder
/// against the ladder's frame-time gates (`budgets.toml`) or its golden images (`golden/golden.toml`).
#[derive(FromArgs)]
#[argh(subcommand, name = "budget")]
struct BudgetCommand {
    /// the run's `report.json`, or its folder; with `--table` or `--golden`, the output folder of a `run`
    #[argh(positional)]
    report: PathBuf,
    /// the frame-time gates, e.g. `tools/gfx-check/budgets.toml`, checked against every run of the folder
    #[argh(option)]
    table: Option<PathBuf>,
    /// the golden images' folder, e.g. `tools/gfx-check/golden`, checked against every run of the folder
    #[argh(option)]
    golden: Option<PathBuf>,
    /// with `--golden`: write the golden images and their tolerances from the folder's runs instead
    #[argh(switch)]
    write_golden: bool,
    /// with `--write-golden`: the adapter the goldens are made and gated on; default RTX 5070
    #[argh(option, default = "String::from(\"RTX 5070\")")]
    golden_adapter: String,
    /// with `--write-golden`: the window size of the runs; default 1920x1080
    #[argh(option, default = "String::from(\"1920x1080\")")]
    golden_resolution: String,
    /// with `--write-golden`: the size the goldens are stored at; default 480x270
    #[argh(option, default = "String::from(\"480x270\")")]
    golden_size: String,
    /// largest median frame time allowed, in milliseconds
    #[argh(option)]
    max_frame_ms: Option<f64>,
    /// largest median GPU time allowed for one pass, as NAME=MS; may be repeated
    #[argh(option)]
    max_pass: Vec<String>,
}

/// Fails on panics, shader errors and wgpu errors in a run's log, and on "not loaded" when the run is traced.
#[derive(FromArgs)]
#[argh(subcommand, name = "log-gate")]
struct LogGateCommand {
    /// a run folder (its `log.txt`, `console.txt` and `report.json` are read), or a log file
    #[argh(positional)]
    log: PathBuf,
    /// treat a log file as a traced run's; a run folder says so itself in its report
    #[argh(switch)]
    traced: bool,
}

/// Prints a measured difference and applies its optional limit.
fn gate_points(label: &str, points: f64, max: Option<f64>) -> bool {
    match max {
        Some(max) if points > max => {
            println!("{label}: {points:.3} % FAIL (limit {max})");
            false
        }
        Some(max) => {
            println!("{label}: {points:.3} % ok (limit {max})");
            true
        }
        None => {
            println!("{label}: {points:.3} %");
            true
        }
    }
}

fn image_stats(command: ImageStatsCommand) -> Result<bool, String> {
    let stats = metric::stats(&metric::load(&command.image)?);
    let black = stats.mean_luma < command.min_luma;
    println!(
        "{}x{} mean luma {:.4} max {} {}",
        stats.width,
        stats.height,
        stats.mean_luma,
        stats.max,
        if black { "BLACK" } else { "ok" }
    );
    Ok(!black)
}

fn motion(command: MotionCommand) -> Result<bool, String> {
    let orbit = Report::read(&command.orbit)?;
    let still = Report::read(&command.still)?;
    if still.camera.kind != "static" {
        return Err(format!(
            "{} is not a static-camera run",
            command.still.display()
        ));
    }
    let pose = still.camera.pose;
    let capture = orbit
        .screenshots
        .iter()
        .find(|shot| shot.orbit_frame == pose)
        .ok_or_else(|| {
            let frames: Vec<String> = orbit
                .screenshots
                .iter()
                .map(|shot| shot.orbit_frame.to_string())
                .collect();
            format!(
                "the orbit run has no capture at orbit frame {pose}; it has {}",
                frames.join(", ")
            )
        })?;
    let points = metric::nrmse_percent_files(
        &command.orbit.join(&capture.file),
        &command.still.join("a.png"),
    )?;
    Ok(gate_points(
        &format!("motion at orbit frame {pose}"),
        points,
        command.max,
    ))
}

fn budget(command: BudgetCommand) -> Result<bool, String> {
    if let Some(table) = &command.table {
        return budgets::table(&command.report, table);
    }
    if let Some(golden) = &command.golden {
        return if command.write_golden {
            budgets::write_golden(
                &command.report,
                golden,
                &command.golden_adapter,
                &command.golden_resolution,
                &command.golden_size,
            )
        } else {
            budgets::golden(&command.report, golden)
        };
    }
    let report = Report::read(&command.report)?;
    let mut pass = true;
    let mut check = |label: &str, value: f64, max: f64| {
        let ok = value <= max;
        println!(
            "{label}: {value:.3} ms {} (limit {max})",
            if ok { "ok" } else { "FAIL" }
        );
        pass &= ok;
    };
    if let Some(max) = command.max_frame_ms {
        let frame = report
            .frame_ms
            .as_ref()
            .ok_or("the report has no frame times")?;
        check("median frame", frame.median, max);
    }
    for limit in &command.max_pass {
        let (name, max) = limit
            .split_once('=')
            .and_then(|(name, max)| Some((name, max.parse::<f64>().ok()?)))
            .ok_or_else(|| format!("--max-pass takes NAME=MS, got {limit}"))?;
        let value = report.gpu_pass_ms.get(name).ok_or_else(|| {
            let known: Vec<&str> = report.gpu_pass_ms.keys().map(String::as_str).collect();
            format!(
                "the report has no pass named {name}; it has: {}",
                known.join(", ")
            )
        })?;
        check(name, *value, max);
    }
    Ok(pass)
}

fn log_gate(command: LogGateCommand) -> Result<bool, String> {
    let offences = if command.log.is_dir() {
        run::log_offences(&command.log, Report::read(&command.log)?.traced)?
    } else {
        let text = std::fs::read_to_string(&command.log)
            .map_err(|error| format!("{}: {error}", command.log.display()))?;
        log_gate::gate(&text, command.traced)
    };
    for offence in &offences {
        println!("{offence}");
    }
    println!(
        "log gate: {}",
        if offences.is_empty() { "ok" } else { "FAIL" }
    );
    Ok(offences.is_empty())
}

fn main() -> ExitCode {
    let cli: Cli = argh::from_env();
    let result = match cli.command {
        Command::Run(command) => run::run(command),
        Command::Image(ImageCommand {
            command: ImageSubcommand::Stats(command),
        }) => image_stats(command),
        Command::Image(ImageCommand {
            command: ImageSubcommand::Diff(command),
        }) => metric::nrmse_percent_files(&command.a, &command.b)
            .map(|points| gate_points("difference", points, command.max)),
        Command::Image(ImageCommand {
            command: ImageSubcommand::Resize(command),
        }) => resize::image_resize(command),
        Command::Flicker(command) => flicker::flicker(command),
        Command::Motion(command) => motion(command),
        Command::Sheet(command) => sheet::sheet(command),
        Command::Budget(command) => budget(command),
        Command::LogGate(command) => log_gate(command),
        Command::Calibration(command) => calibration::calibration(command),
        Command::Friend(command) => friend::friend(command),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("gfx-check: {error}");
            ExitCode::from(2)
        }
    }
}
