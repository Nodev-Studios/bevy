//! The contact sheet of an orbit run: its captures side by side, for a reader to look for trails and smears.

use crate::{metric, report::Report};
use argh::FromArgs;
use image::{imageops, RgbImage};
use std::path::PathBuf;

/// Tiles the captures of an orbit run into one image, in orbit order, four to a row.
#[derive(FromArgs)]
#[argh(subcommand, name = "sheet")]
pub struct SheetCommand {
    /// the folder of the orbit run
    #[argh(positional)]
    run: PathBuf,
    /// where to write the sheet; default `sheet.png` in the run folder
    #[argh(option)]
    out: Option<PathBuf>,
}

const COLUMNS: u32 = 4;
/// Width of one tile: four of them stay readable on one screen.
const TILE_WIDTH: u32 = 640;

/// Writes the contact sheet of `run` to `out`, and returns how many captures it holds.
pub fn write(run: &std::path::Path, out: &std::path::Path) -> Result<usize, String> {
    let report = Report::read(run)?;
    let mut captures: Vec<_> = report
        .screenshots
        .iter()
        .filter(|shot| shot.name.starts_with("orbit-"))
        .collect();
    captures.sort_by_key(|shot| shot.orbit_frame);
    if captures.is_empty() {
        return Err(format!("{} holds no orbit captures", run.display()));
    }

    let mut sheet: Option<RgbImage> = None;
    for (index, capture) in captures.iter().enumerate() {
        let image = metric::load(&run.join(&capture.file))?;
        let tile_height = TILE_WIDTH * image.height() / image.width();
        let tile = imageops::resize(
            &image,
            TILE_WIDTH,
            tile_height,
            imageops::FilterType::Triangle,
        );
        let rows = (captures.len() as u32).div_ceil(COLUMNS);
        let sheet =
            sheet.get_or_insert_with(|| RgbImage::new(TILE_WIDTH * COLUMNS, tile_height * rows));
        let index = index as u32;
        imageops::replace(
            sheet,
            &tile,
            i64::from(index % COLUMNS * TILE_WIDTH),
            i64::from(index / COLUMNS * tile_height),
        );
    }
    sheet
        .expect("there is at least one capture")
        .save(out)
        .map_err(|error| format!("{}: {error}", out.display()))?;
    Ok(captures.len())
}

/// The `sheet` subcommand.
pub fn sheet(command: SheetCommand) -> Result<bool, String> {
    let out = command.out.unwrap_or_else(|| command.run.join("sheet.png"));
    let count = write(&command.run, &out)?;
    println!("{count} captures in {}", out.display());
    Ok(true)
}
