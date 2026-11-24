//! Flicker: the metric between consecutive frames of a still camera, over one pair or over a sequence.
//!
//! One pair is a noisy measure: a temporal method jitters its camera over a cycle of 8 to 32 frames, and how much two
//! consecutive frames differ depends on where in that cycle they fall (D1 read 0.27 to 0.37 points on reruns of one
//! configuration). A sequence covering the whole cycle gives a mean that does not depend on the phase.

use argh::FromArgs;
use std::path::{Path, PathBuf};

use crate::{gate_points, metric, report::Report};

/// The flicker of a sequence of frames.
#[derive(Debug)]
pub struct Flicker {
    /// The metric between each frame and the next, in points.
    pub pairs: Vec<f64>,
}

impl Flicker {
    /// Mean over the pairs.
    pub fn mean(&self) -> f64 {
        self.pairs.iter().sum::<f64>() / self.pairs.len() as f64
    }

    /// Largest pair.
    pub fn max(&self) -> f64 {
        self.pairs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// The flicker of consecutive frames, in the order given.
pub fn sequence(frames: &[PathBuf]) -> Result<Flicker, String> {
    if frames.len() < 2 {
        return Err("flicker needs at least two frames".into());
    }
    let mut previous = metric::load(&frames[0])?;
    let mut pairs = Vec::with_capacity(frames.len() - 1);
    for (index, path) in frames.iter().enumerate().skip(1) {
        let frame = metric::load(path)?;
        let points = metric::nrmse_percent(&previous, &frame)?;
        // Every method gated here jitters its camera, so two renders of consecutive frames always differ.
        if points == 0.0 {
            eprintln!(
                "gfx-check: warning: frames {} and {} are identical; were they rendered on consecutive frames?",
                index - 1,
                index
            );
        }
        pairs.push(points);
        previous = frame;
    }
    Ok(Flicker { pairs })
}

/// The still frames of a run folder, in order: the captures named `still-*` in its `report.json`, which must be on
/// consecutive frames, or else `a.png` and `b.png`.
pub fn run_frames(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let report = Report::read(dir).ok();
    let mut stills: Vec<(u32, &str)> = Vec::new();
    for shot in report.iter().flat_map(|report| &report.screenshots) {
        if shot.name.starts_with("still-") {
            let frame = shot
                .frame
                .ok_or_else(|| format!("{}: capture {} has no frame", dir.display(), shot.name))?;
            stills.push((frame, &shot.file));
        }
    }
    if stills.is_empty() {
        return Ok(vec![dir.join("a.png"), dir.join("b.png")]);
    }
    stills.sort_by_key(|&(frame, _)| frame);
    for pair in stills.windows(2) {
        if pair[1].0 != pair[0].0 + 1 {
            return Err(format!(
                "{}: the still captures are not on consecutive frames ({} then {})",
                dir.display(),
                pair[0].0,
                pair[1].0
            ));
        }
    }
    Ok(stills.iter().map(|(_, file)| dir.join(file)).collect())
}

/// Prints the metric between consecutive frames of a static camera: their mean, and the largest pair.
///
/// Given a run folder it reads the run's still captures (`still-*` in its `report.json`, on consecutive frames) or
/// else its `a.png` and `b.png`. Given two or more images it takes them as consecutive, in order.
#[derive(FromArgs)]
#[argh(subcommand, name = "flicker")]
pub struct FlickerCommand {
    /// a run folder, or two or more consecutive frames
    #[argh(positional)]
    frames: Vec<PathBuf>,
    /// fail when the mean over the pairs is above this many points
    #[argh(option)]
    max: Option<f64>,
    /// print each pair's value too
    #[argh(switch)]
    pairs: bool,
}

/// Runs `flicker`.
pub fn flicker(command: FlickerCommand) -> Result<bool, String> {
    let frames = match command.frames.as_slice() {
        [dir] if dir.is_dir() => run_frames(dir)?,
        frames if frames.len() >= 2 => frames.to_vec(),
        _ => return Err("flicker takes a run folder or two or more images".into()),
    };
    let flicker = sequence(&frames)?;
    if command.pairs {
        for (index, points) in flicker.pairs.iter().enumerate() {
            println!("pair {index}: {points:.3} %");
        }
    }
    let pass = gate_points("flicker", flicker.mean(), command.max);
    println!(
        "pairs: {} mean {:.3} % max {:.3} %",
        flicker.pairs.len(),
        flicker.mean(),
        flicker.max()
    );
    Ok(pass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn a_sequence_has_a_known_mean_and_max() {
        let dir = std::env::temp_dir().join(format!("gfx-check-flicker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Every channel off by 51 is 20 points; one channel off by 51 is 11.547 points (see the metric's tests).
        let levels = [[0, 0, 0], [51, 51, 51], [51, 51, 51], [51, 0, 0]];
        let frames: Vec<PathBuf> = levels
            .iter()
            .enumerate()
            .map(|(index, &level)| {
                let path = dir.join(format!("{index}.png"));
                RgbImage::from_pixel(4, 4, Rgb(level)).save(&path).unwrap();
                path
            })
            .collect();
        let flicker = sequence(&frames).unwrap();
        // Pairs: 20 (all channels), 0, and 16.330 (two channels off by 51: sqrt(2 * 0.2^2 / 3)).
        assert_eq!(flicker.pairs.len(), 3);
        assert!((flicker.pairs[0] - 20.0).abs() < 1e-9);
        assert_eq!(flicker.pairs[1], 0.0);
        assert!((flicker.pairs[2] - 16.330).abs() < 1e-3);
        assert!((flicker.mean() - 12.110).abs() < 1e-3);
        assert!((flicker.max() - 20.0).abs() < 1e-9);
        // One pair is the old a/b flicker.
        assert!((sequence(&frames[..2]).unwrap().mean() - 20.0).abs() < 1e-9);
        assert!(sequence(&frames[..1]).is_err());

        // A run folder: still captures out of order in report.json are sorted by frame; a gap is refused.
        let report = |frames: [u32; 3]| {
            let shots: Vec<String> = frames
                .iter()
                .enumerate()
                .map(|(index, frame)| {
                    format!(
                        r#"{{"name": "still-{index}", "file": "{index}.png", "frame": {frame}}}"#
                    )
                })
                .collect();
            std::fs::write(
                dir.join("report.json"),
                format!(r#"{{"screenshots": [{}]}}"#, shots.join(",")),
            )
            .unwrap();
        };
        report([11, 10, 12]);
        let ordered = run_frames(&dir).unwrap();
        assert_eq!(
            ordered,
            [dir.join("1.png"), dir.join("0.png"), dir.join("2.png")]
        );
        report([10, 11, 13]);
        assert!(run_frames(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
