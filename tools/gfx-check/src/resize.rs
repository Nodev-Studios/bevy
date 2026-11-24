//! Resizing a frame to another frame's size, so that a run at a lower or a higher resolution can be set against a
//! run at the output's: the bilinear and supersampled baselines of an upscaler's gates.
//!
//! Both filters work on the encoded (tonemapped, sRGB) 8-bit values, as `magick -resize` does on a PNG and as every
//! gate's metric does. The upscaler gates of D1 were first measured with `magick -filter Triangle` (enlarge) and
//! `magick -filter Box` (reduce by 2); these filters reproduce them (see the README).

use argh::FromArgs;
use image::{Rgb, RgbImage};
use std::path::PathBuf;

use crate::metric;

/// How the image is resampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Linear interpolation between the four nearest pixel centres, with the edge pixels repeated: what a GPU's
    /// linear sampler does when a frame rendered small is stretched to the screen. For enlarging; it aliases when
    /// reducing.
    Bilinear,
    /// Each output pixel is the mean of the input area it covers, partly covered pixels weighted by their share:
    /// the box filter of supersampling, exact for any ratio. For reducing.
    Area,
}

impl core::str::FromStr for Filter {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "bilinear" => Ok(Filter::Bilinear),
            "area" => Ok(Filter::Area),
            _ => Err(format!("--filter takes bilinear|area, not {text}")),
        }
    }
}

/// Resizes an image to `width` x `height`.
pub fn resize(image: &RgbImage, width: u32, height: u32, filter: Filter) -> RgbImage {
    let weights = |from: u32, to: u32| match filter {
        Filter::Bilinear => bilinear_weights(from, to),
        Filter::Area => area_weights(from, to),
    };
    let columns = weights(image.width(), width);
    let rows = weights(image.height(), height);
    // Horizontal pass into floats, then vertical pass, rounding once at the end.
    let mut horizontal = vec![[0.0f64; 3]; (width * image.height()) as usize];
    for y in 0..image.height() {
        for (x, taps) in columns.iter().enumerate() {
            let sum = &mut horizontal[(y * width) as usize + x];
            for &(source, weight) in taps {
                let pixel = image.get_pixel(source, y).0;
                for channel in 0..3 {
                    sum[channel] += weight * f64::from(pixel[channel]);
                }
            }
        }
    }
    RgbImage::from_fn(width, height, |x, y| {
        let mut sum = [0.0f64; 3];
        for &(source, weight) in &rows[y as usize] {
            let pixel = horizontal[(source * width + x) as usize];
            for channel in 0..3 {
                sum[channel] += weight * pixel[channel];
            }
        }
        Rgb(sum.map(|value| value.round().clamp(0.0, 255.0) as u8))
    })
}

/// The source pixels and weights of each output pixel along one axis, for [`Filter::Bilinear`].
fn bilinear_weights(from: u32, to: u32) -> Vec<Vec<(u32, f64)>> {
    let scale = f64::from(from) / f64::from(to);
    let last = f64::from(from - 1);
    (0..to)
        .map(|i| {
            // Pixel centres line up: output centre i + 0.5 sits at input coordinate (i + 0.5) * scale.
            let position = ((f64::from(i) + 0.5) * scale - 0.5).clamp(0.0, last);
            let left = position.floor();
            let fraction = position - left;
            let left = left as u32;
            if fraction == 0.0 {
                vec![(left, 1.0)]
            } else {
                vec![(left, 1.0 - fraction), (left + 1, fraction)]
            }
        })
        .collect()
}

/// The source pixels and weights of each output pixel along one axis, for [`Filter::Area`].
fn area_weights(from: u32, to: u32) -> Vec<Vec<(u32, f64)>> {
    let scale = f64::from(from) / f64::from(to);
    (0..to)
        .map(|i| {
            let start = f64::from(i) * scale;
            let end = f64::from(i + 1) * scale;
            let mut taps = Vec::new();
            let mut source = start.floor() as u32;
            while f64::from(source) < end && source < from {
                let covered = end.min(f64::from(source + 1)) - start.max(f64::from(source));
                if covered > 0.0 {
                    taps.push((source, covered / scale));
                }
                source += 1;
            }
            taps
        })
        .collect()
}

/// Resizes an image: the bilinear baseline of an upscaler (enlarge a lower-resolution run) or the supersampled one
/// (reduce a higher-resolution run).
#[derive(FromArgs)]
#[argh(subcommand, name = "resize")]
pub struct ImageResizeCommand {
    /// the image to resize
    #[argh(positional)]
    input: PathBuf,
    /// where to write the result (PNG)
    #[argh(positional)]
    output: PathBuf,
    /// the size to resize to, as `WxH`; or the path of an image of that size
    #[argh(option)]
    size: String,
    /// bilinear (to enlarge) or area (to reduce)
    #[argh(option)]
    filter: Filter,
}

/// Runs `image resize`.
pub fn image_resize(command: ImageResizeCommand) -> Result<bool, String> {
    let (width, height) = match command
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
    {
        Some(size) => size,
        None => image::image_dimensions(&command.size)
            .map_err(|error| format!("--size takes WxH or an image: {}: {error}", command.size))?,
    };
    if width == 0 || height == 0 {
        return Err("--size must not be zero".into());
    }
    let input = metric::load(&command.input)?;
    resize(&input, width, height, command.filter)
        .save(&command.output)
        .map_err(|error| format!("{}: {error}", command.output.display()))?;
    println!(
        "{}x{} -> {width}x{height} ({:?}): {}",
        input.width(),
        input.height(),
        command.filter,
        command.output.display()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[u8]) -> RgbImage {
        RgbImage::from_fn(values.len() as u32, 1, |x, _| {
            let v = values[x as usize];
            Rgb([v, v, v])
        })
    }

    fn values(image: &RgbImage) -> Vec<u8> {
        image.pixels().map(|pixel| pixel.0[0]).collect()
    }

    #[test]
    fn a_constant_image_stays_constant() {
        let image = RgbImage::from_pixel(6, 4, Rgb([17, 128, 250]));
        for filter in [Filter::Bilinear, Filter::Area] {
            for (w, h) in [(12, 8), (3, 2), (5, 3), (7, 9)] {
                let resized = resize(&image, w, h, filter);
                assert_eq!(resized.dimensions(), (w, h));
                assert!(resized.pixels().all(|pixel| pixel.0 == [17, 128, 250]));
            }
        }
    }

    #[test]
    fn area_halves_a_checker_to_mid_grey() {
        let checker = RgbImage::from_fn(4, 4, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        });
        let reduced = resize(&checker, 2, 2, Filter::Area);
        // 127.5 rounds away from zero.
        assert!(reduced.pixels().all(|pixel| pixel.0 == [128, 128, 128]));
    }

    #[test]
    fn area_weighs_partly_covered_pixels() {
        // Three pixels to two: the first output covers pixel 0 and half of pixel 1, over an area of 1.5.
        let reduced = resize(&row(&[0, 90, 180]), 2, 1, Filter::Area);
        assert_eq!(values(&reduced), [30, 150]);
    }

    #[test]
    fn bilinear_doubles_a_ramp_at_pixel_centres() {
        // Output centres 0.5, 1.5, 2.5, 3.5 fall on input coordinates -0.25, 0.25, 0.75, 1.25; the ends clamp.
        let enlarged = resize(&row(&[0, 255]), 4, 1, Filter::Bilinear);
        assert_eq!(values(&enlarged), [0, 64, 191, 255]);
    }

    #[test]
    fn bilinear_at_the_same_size_is_the_identity() {
        let image = row(&[3, 200, 7, 90, 255]);
        assert_eq!(resize(&image, 5, 1, Filter::Bilinear), image);
        assert_eq!(resize(&image, 5, 1, Filter::Area), image);
    }
}
