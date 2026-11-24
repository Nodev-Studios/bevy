//! `gfx-check budget --table` and `--golden`: the ladder's gates as data, checked against the results of a `run`.
//!
//! Both read a results folder whose runs are named `<config>@<k>`: the repeats of one configuration, which differ
//! in warm-up length only (Solari's noise and the jitter of the temporal filters depend on the frame number, so
//! identical runs would measure nothing). A gate reads the median over the repeats.
//!
//! - `budgets.toml` holds the frame-time gates: for a run and a resolution, the largest median frame time on the
//!   adapter the gates were set for. Other adapters' cells are printed and not gated.
//! - `golden/golden.toml` holds one small image per configuration and scene, with its tolerance. A run's `a.png`
//!   is reduced to the golden's size with the area filter and compared with the golden; `--write-golden` makes
//!   the images and the tolerances from a run instead.

use crate::{
    metric,
    resize::{self, Filter},
    run::Cell,
};
use alloc::collections::BTreeMap;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

/// `budgets.toml`.
#[derive(Deserialize)]
struct Budgets {
    /// Gates apply to cells whose reported adapter name contains this.
    adapter: String,
    /// The gates.
    gate: Vec<Gate>,
}

/// One frame-time gate.
#[derive(Deserialize)]
struct Gate {
    /// The run's name in the matrix, without its `@k`.
    run: String,
    /// The window size, `WxH`.
    resolution: String,
    /// Largest median frame time allowed, in milliseconds: the median over the repeats of each run's median.
    max_frame_ms: f64,
}

/// `golden/golden.toml`.
#[derive(Serialize, Deserialize)]
struct Goldens {
    /// Images are compared only for cells whose reported adapter name contains this, at `resolution`.
    adapter: String,
    /// The window size the goldens were rendered at, `WxH`.
    resolution: String,
    /// The size the goldens are stored at, `[width, height]`; every frame is reduced to it.
    size: [u32; 2],
    /// One per configuration and scene.
    image: Vec<Golden>,
}

/// One golden image.
#[derive(Serialize, Deserialize)]
struct Golden {
    /// The run's name in the matrix, without its `@k`.
    run: String,
    /// The scene.
    scene: String,
    /// The PNG, beside `golden.toml`.
    file: String,
    /// Largest median difference allowed, in points: twice the largest difference of one repeat from the golden
    /// when it was written, and at least [`MIN_TOLERANCE`].
    tolerance: f64,
    /// The repeats' differences from the golden when it was written, in points.
    written_from: Vec<f64>,
}

/// A difference smaller than this is not told apart from noise (the noise floor rule of the calibration).
const MIN_TOLERANCE: f64 = 0.1;

fn read_cells(results: &Path) -> Result<Vec<Cell>, String> {
    let file = results.join("results.json");
    let text = fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", file.display()))
}

fn config(cell: &Cell) -> &str {
    cell.run.split('@').next().unwrap_or(&cell.run)
}

fn run_dir(results: &Path, cell: &Cell) -> PathBuf {
    results
        .join(&cell.adapter)
        .join(&cell.scene)
        .join(&cell.resolution)
        .join(&cell.run)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    })
}

/// The GPU time of the frame: the top-level passes summed. `blas_build` runs only when new meshes arrive, and the
/// testbed keeps reading the diagnostic's last value after it, so it is left out.
fn gpu_sum(cell: &Cell) -> f64 {
    let names: Vec<&str> = cell
        .gpu_pass_ms
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    cell.gpu_pass_ms
        .iter()
        .filter(|(name, _)| {
            name != "blas_build"
                && !names
                    .iter()
                    .any(|other| name.len() > other.len() && name.starts_with(&format!("{other}/")))
        })
        .map(|(_, ms)| ms)
        .sum()
}

/// Groups cells by (adapter, scene, resolution, config), in the order they ran.
fn groups(cells: &[Cell]) -> Vec<((String, String, String, String), Vec<&Cell>)> {
    let mut groups: Vec<((String, String, String, String), Vec<&Cell>)> = Vec::new();
    for cell in cells {
        let key = (
            cell.adapter_reported.clone(),
            cell.scene.clone(),
            cell.resolution.clone(),
            config(cell).to_owned(),
        );
        match groups.iter_mut().find(|(other, _)| *other == key) {
            Some((_, members)) => members.push(cell),
            None => groups.push((key, vec![cell])),
        }
    }
    groups
}

/// `budget --table`: every gate of `budgets.toml` against a results folder.
pub fn table(results: &Path, table: &Path) -> Result<bool, String> {
    let text =
        fs::read_to_string(table).map_err(|error| format!("{}: {error}", table.display()))?;
    let budgets: Budgets =
        toml::from_str(&text).map_err(|error| format!("{}: {error}", table.display()))?;
    let cells = read_cells(results)?;
    let mut pass = true;
    let mut gated = 0;
    println!("| adapter | scene | resolution | run | repeats | median frame ms (min–max) | GPU ms | limit | result |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for ((adapter, scene, resolution, run), members) in groups(&cells) {
        let frames: Vec<f64> = members
            .iter()
            .filter_map(|cell| cell.frame_ms.map(|frame| frame[0]))
            .collect();
        let gpu = median(members.iter().map(|cell| gpu_sum(cell)).collect());
        let range = match (
            frames.iter().copied().reduce(f64::min),
            frames.iter().copied().reduce(f64::max),
        ) {
            (Some(low), Some(high)) => format!("({low:.3}–{high:.3})"),
            _ => String::new(),
        };
        let frame = median(frames);
        let gate = budgets.gate.iter().find(|gate| {
            gate.run == run && gate.resolution == resolution && adapter.contains(&budgets.adapter)
        });
        let failed_runs = members
            .iter()
            .filter(|cell| !cell.failures.is_empty())
            .count();
        let (limit, verdict) = match (gate, frame) {
            (None, _) => (String::new(), "recorded".to_owned()),
            (Some(gate), Some(frame)) => {
                gated += 1;
                let ok = frame <= gate.max_frame_ms && failed_runs == 0;
                pass &= ok;
                let verdict = if failed_runs > 0 {
                    format!("FAIL ({failed_runs} runs failed)")
                } else if ok {
                    "pass".to_owned()
                } else {
                    "FAIL".to_owned()
                };
                (format!("{}", gate.max_frame_ms), verdict)
            }
            (Some(gate), None) => {
                gated += 1;
                pass = false;
                (
                    format!("{}", gate.max_frame_ms),
                    "FAIL (no frame times)".to_owned(),
                )
            }
        };
        println!(
            "| {adapter} | {scene} | {resolution} | {run} | {} | {} {range} | {} | {limit} | {verdict} |",
            members.len(),
            frame.map_or(String::new(), |frame| format!("{frame:.3}")),
            gpu.map_or(String::new(), |gpu| format!("{gpu:.3}")),
        );
    }
    // A gate whose run is absent from the gated adapter's results was not checked: that is a failure, not a pass.
    for gate in &budgets.gate {
        let present = cells.iter().any(|cell| {
            config(cell) == gate.run
                && cell.resolution == gate.resolution
                && cell.adapter_reported.contains(&budgets.adapter)
        });
        if !present
            && cells
                .iter()
                .any(|cell| cell.adapter_reported.contains(&budgets.adapter))
        {
            println!(
                "gate {} at {}: no such run in the results: FAIL",
                gate.run, gate.resolution
            );
            pass = false;
        }
    }
    println!(
        "{gated} gated cells; {}",
        if pass { "pass" } else { "FAIL" }
    );
    Ok(pass)
}

fn parse_size(text: &str) -> Result<(u32, u32), String> {
    text.split_once('x')
        .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
        .ok_or_else(|| format!("not a WxH size: {text}"))
}

fn reduced(dir: &Path, size: [u32; 2]) -> Result<RgbImage, String> {
    let frame = metric::load(&dir.join("a.png"))?;
    Ok(resize::resize(&frame, size[0], size[1], Filter::Area))
}

/// The per-pixel mean of images of one size, rounded.
fn mean_image(images: &[RgbImage]) -> RgbImage {
    let (width, height) = images[0].dimensions();
    let mut sums = vec![0u32; images[0].as_raw().len()];
    for image in images {
        for (sum, &value) in sums.iter_mut().zip(image.as_raw()) {
            *sum += u32::from(value);
        }
    }
    let count = images.len() as u32;
    let raw = sums
        .into_iter()
        .map(|sum| ((sum + count / 2) / count) as u8)
        .collect();
    RgbImage::from_raw(width, height, raw).expect("same size as its sources")
}

/// `budget --golden DIR`: every golden image against a results folder.
pub fn golden(results: &Path, dir: &Path) -> Result<bool, String> {
    let file = dir.join("golden.toml");
    let text = fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    let goldens: Goldens =
        toml::from_str(&text).map_err(|error| format!("{}: {error}", file.display()))?;
    let cells = read_cells(results)?;
    let mut pass = true;
    let mut gated = 0;
    println!("| adapter | scene | resolution | run | repeats | difference from golden (median, min–max) | tolerance | result |");
    println!("|---|---|---|---|---|---|---|---|");
    for ((adapter, scene, resolution, run), members) in groups(&cells) {
        let Some(golden) = goldens
            .image
            .iter()
            .find(|golden| golden.run == run && golden.scene == scene)
        else {
            println!(
                "| {adapter} | {scene} | {resolution} | {run} | {} | no golden | | recorded |",
                members.len()
            );
            continue;
        };
        let reference = metric::load(&dir.join(&golden.file))?;
        let mut differences = Vec::new();
        for cell in &members {
            differences.push(metric::nrmse_percent(
                &reduced(&run_dir(results, cell), goldens.size)?,
                &reference,
            )?);
        }
        let low = differences.iter().copied().fold(f64::INFINITY, f64::min);
        let high = differences.iter().copied().fold(0.0, f64::max);
        let difference = median(differences).unwrap_or(f64::NAN);
        let gate = adapter.contains(&goldens.adapter) && resolution == goldens.resolution;
        let failed_runs = members
            .iter()
            .filter(|cell| !cell.failures.is_empty())
            .count();
        let verdict = if !gate {
            "recorded".to_owned()
        } else {
            gated += 1;
            let ok = difference <= golden.tolerance && failed_runs == 0;
            pass &= ok;
            match (ok, failed_runs) {
                (_, 1..) => format!("FAIL ({failed_runs} runs failed)"),
                (true, _) => "pass".to_owned(),
                (false, _) => "FAIL".to_owned(),
            }
        };
        println!(
            "| {adapter} | {scene} | {resolution} | {run} | {} | {difference:.3} ({low:.3}–{high:.3}) | {} | {verdict} |",
            members.len(),
            golden.tolerance
        );
    }
    let missing: Vec<String> = goldens
        .image
        .iter()
        .filter(|golden| {
            !cells.iter().any(|cell| {
                config(cell) == golden.run
                    && cell.scene == golden.scene
                    && cell.resolution == goldens.resolution
                    && cell.adapter_reported.contains(&goldens.adapter)
            })
        })
        .map(|golden| format!("{} on {}", golden.run, golden.scene))
        .collect();
    if !missing.is_empty()
        && cells
            .iter()
            .any(|cell| cell.adapter_reported.contains(&goldens.adapter))
    {
        println!("not in the results: {}: FAIL", missing.join(", "));
        pass = false;
    }
    println!(
        "{gated} gated cells; {}",
        if pass { "pass" } else { "FAIL" }
    );
    Ok(pass)
}

/// `budget --golden DIR --write-golden`: makes the golden images and `golden.toml` from a results folder. Each
/// golden is the per-pixel mean of a configuration's repeats on the gated adapter at `resolution`, reduced to
/// `size`; its tolerance is twice the largest difference of one repeat from it, and at least [`MIN_TOLERANCE`].
pub fn write_golden(
    results: &Path,
    dir: &Path,
    adapter: &str,
    resolution: &str,
    size: &str,
) -> Result<bool, String> {
    let (width, height) = parse_size(size)?;
    let cells = read_cells(results)?;
    fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut goldens = Goldens {
        adapter: adapter.to_owned(),
        resolution: resolution.to_owned(),
        size: [width, height],
        image: Vec::new(),
    };
    let mut by_config: BTreeMap<(String, String), Vec<&Cell>> = BTreeMap::new();
    for cell in &cells {
        if cell.adapter_reported.contains(adapter) && cell.resolution == resolution {
            if !cell.failures.is_empty() {
                return Err(format!(
                    "{} on {} failed; a golden is made from passing runs only",
                    cell.run, cell.scene
                ));
            }
            by_config
                .entry((config(cell).to_owned(), cell.scene.clone()))
                .or_default()
                .push(cell);
        }
    }
    if by_config.is_empty() {
        return Err(format!(
            "no run on {adapter} at {resolution} in the results"
        ));
    }
    for ((run, scene), members) in by_config {
        let images = members
            .iter()
            .map(|cell| reduced(&run_dir(results, cell), [width, height]))
            .collect::<Result<Vec<_>, _>>()?;
        let mean = mean_image(&images);
        let differences = images
            .iter()
            .map(|image| metric::nrmse_percent(image, &mean))
            .collect::<Result<Vec<_>, _>>()?;
        let largest = differences.iter().copied().fold(0.0, f64::max);
        // Rounded up to the hundredth so the file reads cleanly and never tightens the rule.
        let tolerance = ((2.0 * largest).max(MIN_TOLERANCE) * 100.0).ceil() / 100.0;
        let file = format!("{run}-{scene}.png");
        mean.save(dir.join(&file))
            .map_err(|error| format!("{file}: {error}"))?;
        println!(
            "{file}: {} repeats, largest difference from the mean {largest:.3}, tolerance {tolerance}",
            members.len()
        );
        goldens.image.push(Golden {
            run,
            scene,
            file,
            tolerance,
            written_from: differences
                .iter()
                .map(|difference| (difference * 1000.0).round() / 1000.0)
                .collect(),
        });
    }
    let mut text = String::from(
        "# Written by `gfx-check budget --golden tools/gfx-check/golden --write-golden <results>` from a run of\n\
         # tools/gfx-check/golden.ron; do not edit by hand. Each image is the per-pixel mean of one configuration's\n\
         # repeats, reduced to `size` with the area filter; `tolerance` is twice the largest difference of one\n\
         # repeat from it (at least 0.1 points), and a later run passes when the median of its repeats' differences\n\
         # is under it. Only cells on `adapter` at `resolution` are gated; others are recorded.\n\n",
    );
    let body = toml::to_string(&goldens).map_err(|error| error.to_string())?;
    let _ = write!(text, "{body}");
    let file = dir.join("golden.toml");
    fs::write(&file, text).map_err(|error| format!("{}: {error}", file.display()))?;
    Ok(true)
}
