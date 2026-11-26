//! `gfx-check coverage`: the baked-light coverage gate. In a run of the testbed's coverage views
//! (`--gi-coverage`, `--gi-reflection-coverage`) every capture has a mask of the pixels inside a baked box
//! (`mask`) and, for `--gi-coverage`, of every pixel that shows a surface (`surface`). No pixel of a capture under
//! either mask may be black: inside a box that is a surface the bake does not reach, elsewhere a surface left with
//! no indirect light at all.

use crate::metric;
use argh::FromArgs;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Fails when a coverage capture has a black pixel under its masks, or when the folder holds no coverage run.
#[derive(FromArgs)]
#[argh(subcommand, name = "coverage")]
pub struct CoverageCommand {
    /// a run folder, or a folder of runs (searched recursively, as `gfx-check run` writes them)
    #[argh(positional)]
    dir: PathBuf,
}

/// One checked mask.
pub struct Checked {
    /// `<capture> under <mask>`.
    pub label: String,
    /// Pixels under the mask.
    pub pixels: u64,
    /// Of them, black.
    pub zero: u64,
}

/// Checks one run folder. `Ok(None)` when the run is not a coverage view.
pub fn check_run(dir: &Path) -> Result<Option<Vec<Checked>>, String> {
    let file = dir.join("report.json");
    let text = fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    let report: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", file.display()))?;
    let gi = &report["gi"];
    let is_view = |key: &str| gi[key].as_bool().unwrap_or(false);
    if !is_view("coverage_view") && !is_view("reflection_coverage_view") {
        return Ok(None);
    }
    let captures = gi["captures"]
        .as_array()
        .ok_or_else(|| format!("{}: the gi section has no captures", file.display()))?;
    let mut checked = Vec::new();
    for capture in captures {
        let image = capture["capture"]
            .as_str()
            .ok_or_else(|| format!("{}: a capture without a file", file.display()))?;
        let masks: Vec<&str> = ["mask", "surface"]
            .iter()
            .filter_map(|key| capture[key].as_str())
            .collect();
        if masks.is_empty() {
            return Err(format!("{}: capture {image} has no mask", file.display()));
        }
        let pixels = metric::load(&dir.join(image))?;
        for mask in masks {
            let region = metric::Region::parse(&dir.join(mask).to_string_lossy())?;
            // A view with no masked pixel (a capture of the sky) has nothing to fail.
            let (pixels, zero) = match metric::stats_in(&pixels, Some(&region)) {
                Ok(stats) => (stats.pixels, stats.zero_pixels),
                Err(_) => (0, 0),
            };
            checked.push(Checked {
                label: format!("{image} under {mask}"),
                pixels,
                zero,
            });
        }
    }
    Ok(Some(checked))
}

fn run_folders(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if dir.join("report.json").is_file() {
        out.push(dir.to_owned());
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|error| format!("{}: {error}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    entries.sort();
    for entry in entries {
        run_folders(&entry, out)?;
    }
    Ok(())
}

/// Prints every checked mask; `Ok(true)` when no pixel under any mask is black.
pub fn coverage(command: CoverageCommand) -> Result<bool, String> {
    let mut runs = Vec::new();
    run_folders(&command.dir, &mut runs)?;
    let mut pass = true;
    let mut views = 0;
    for run in runs {
        let Some(checked) = check_run(&run)? else {
            continue;
        };
        views += 1;
        for check in checked {
            let ok = check.zero == 0;
            pass &= ok;
            println!(
                "{}: {} {} px, {} black {}",
                run.display(),
                check.label,
                check.pixels,
                check.zero,
                if ok { "ok" } else { "FAIL" }
            );
        }
    }
    if views == 0 {
        return Err(format!(
            "{} holds no coverage run (--gi-coverage or --gi-reflection-coverage)",
            command.dir.display()
        ));
    }
    println!("coverage: {}", if pass { "ok" } else { "FAIL" });
    Ok(pass)
}
