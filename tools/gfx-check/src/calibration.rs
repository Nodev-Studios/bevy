//! `gfx-check calibration`: turns the results of a calibration matrix into `calibration.toml`.

use crate::run::Cell;
use alloc::collections::BTreeMap;
use argh::FromArgs;
use std::{
    fmt::Write as _,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

/// Writes `calibration.toml` from the results of `gfx-check run tools/gfx-check/calibration.ron`.
///
/// Runs named `config@x` are repeats of one configuration with different warm-up lengths: Solari's noise depends
/// only on the frame number, so identical runs would measure nothing. Their spread is the noise floor.
#[derive(FromArgs)]
#[argh(subcommand, name = "calibration")]
pub struct CalibrationCommand {
    /// the output folder of the calibration run (it holds `results.json`)
    #[argh(positional)]
    results: PathBuf,
    /// where to write the file; default `tools/gfx-check/calibration.toml`
    #[argh(
        option,
        default = "PathBuf::from(\"tools/gfx-check/calibration.toml\")"
    )]
    out: PathBuf,
}

/// A difference smaller than this is not told apart from noise, whatever the repeats say.
const MIN_NOISE_FLOOR: f64 = 0.1;

/// Today's date in UTC as `YYYY-MM-DD` (days since 1970 to a civil date, after Howard Hinnant).
fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

fn list(values: &[f64]) -> String {
    let values: Vec<String> = values.iter().map(|value| format!("{value:.3}")).collect();
    format!("[{}]", values.join(", "))
}

/// Writes the mean, the runs and the noise floor of one metric of one configuration.
fn metric_lines(toml: &mut String, name: &str, values: &[f64]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let spread = values.iter().copied().fold(f64::MIN, f64::max)
        - values.iter().copied().fold(f64::MAX, f64::min);
    let _ = writeln!(toml, "{name} = {mean:.3}");
    let _ = writeln!(toml, "{name}_runs = {}", list(values));
    let _ = writeln!(
        toml,
        "{name}_noise_floor = {:.3}",
        spread.max(MIN_NOISE_FLOOR)
    );
}

/// The `calibration` subcommand.
pub fn calibration(command: CalibrationCommand) -> Result<bool, String> {
    let file = command.results.join("results.json");
    let text = fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    let cells: Vec<Cell> =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", file.display()))?;
    let first = cells.first().ok_or("the results are empty")?;
    if let Some(failed) = cells.iter().find(|cell| !cell.failures.is_empty()) {
        return Err(format!(
            "run {} failed; a calibration is written only from a clean matrix",
            failed.run
        ));
    }
    if let Some(other) = cells.iter().find(|cell| {
        (&cell.adapter_reported, &cell.scene, &cell.resolution)
            != (&first.adapter_reported, &first.scene, &first.resolution)
    }) {
        return Err(format!(
            "run {} is on another adapter, scene or resolution; a calibration is one of each",
            other.run
        ));
    }

    let mut groups: BTreeMap<&str, Vec<&Cell>> = BTreeMap::new();
    for cell in &cells {
        let config = cell
            .run
            .split_once('@')
            .map_or(cell.run.as_str(), |(config, _)| config);
        groups.entry(config).or_default().push(cell);
    }

    let mut toml = String::new();
    let _ = write!(
        toml,
        "# Written by `gfx-check calibration` from a run of tools/gfx-check/calibration.ron; do not edit by hand.\n\
         #\n\
         # How noisy each way of showing Solari's frame is on the testbed's own scene. Later gates are relative to\n\
         # these numbers and hold only on the same scene, adapter class and resolution.\n\
         #\n\
         # error:   normalised RMSE of the tonemapped 8-bit frame against the `reference` run, percent of full scale.\n\
         # flicker: the same metric between two consecutive frames of the still camera.\n\
         # *_runs:  one value per repeat; the repeats differ only in warm-up length (`warmup`).\n\
         # *_noise_floor: the larger of the repeats' spread (max - min) and {MIN_NOISE_FLOOR}. A difference under it is noise.\n\
         #\n\
         # No timings here: the frame and pass times of these runs are in their report.json.\n\n\
         [meta]\n\
         date = \"{}\"\n\
         adapter = \"{}\"\n\
         scene = \"{}\"\n\
         resolution = \"{}\"\n",
        today(),
        first.adapter_reported,
        first.scene,
        first.resolution
    );
    for (config, runs) in &groups {
        let _ = writeln!(toml, "\n[{config}]");
        let _ = writeln!(toml, "args = \"{}\"", runs[0].args.join(" "));
        let warmups: Vec<String> = runs.iter().map(|run| run.warmup.to_string()).collect();
        let _ = writeln!(toml, "warmup = [{}]", warmups.join(", "));
        let _ = writeln!(toml, "frames = {}", runs[0].frames);
        let collect = |value: fn(&Cell) -> Option<f64>| -> Vec<f64> {
            runs.iter().filter_map(|run| value(run)).collect()
        };
        metric_lines(&mut toml, "error", &collect(|run| run.error));
        metric_lines(&mut toml, "flicker", &collect(|run| run.flicker));
    }
    fs::write(&command.out, toml).map_err(|error| format!("{}: {error}", command.out.display()))?;
    println!(
        "{} configurations in {}",
        groups.len(),
        command.out.display()
    );
    Ok(true)
}
