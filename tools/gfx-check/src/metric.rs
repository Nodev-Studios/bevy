//! The one image metric of the graphics checks, and the black-frame test.

use image::RgbImage;
use std::path::Path;

/// Loads a screenshot as 8-bit RGB. Alpha is dropped: the testbed's screenshots are opaque.
pub fn load(path: &Path) -> Result<RgbImage, String> {
    image::open(path)
        .map(|image| image.to_rgb8())
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Normalised root-mean-square error between two tonemapped 8-bit images, in percent of full scale ("points").
///
/// It is the square root of the mean, over every pixel and every colour channel, of the squared difference, divided
/// by 255: the number `magick compare -metric RMSE` prints in parentheses, times 100. Every gate of the graphics
/// plan uses it, so a number from one run can be set against a number from another.
pub fn nrmse_percent(a: &RgbImage, b: &RgbImage) -> Result<f64, String> {
    if a.dimensions() != b.dimensions() {
        return Err(format!(
            "image sizes differ: {}x{} against {}x{}",
            a.width(),
            a.height(),
            b.width(),
            b.height()
        ));
    }
    let sum: u64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(&x, &y)| {
            let difference = u64::from(x.abs_diff(y));
            difference * difference
        })
        .sum();
    let mean = sum as f64 / a.as_raw().len() as f64;
    Ok(mean.sqrt() / 255.0 * 100.0)
}

/// [`nrmse_percent`] of two files.
pub fn nrmse_percent_files(a: &Path, b: &Path) -> Result<f64, String> {
    nrmse_percent(&load(a)?, &load(b)?)
}

/// What `image stats` prints.
pub struct Stats {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Mean Rec. 709 luma of the encoded (tonemapped, sRGB) values, 0 to 1.
    pub mean_luma: f64,
    /// Largest channel value in the image, 0 to 255.
    pub max: u8,
}

/// Measures how bright an image is.
pub fn stats(image: &RgbImage) -> Stats {
    let mut luma = 0.0;
    let mut max = 0;
    for pixel in image.pixels() {
        let [r, g, b] = pixel.0;
        luma += 0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b);
        max = max.max(r).max(g).max(b);
    }
    Stats {
        width: image.width(),
        height: image.height(),
        mean_luma: luma / 255.0 / (f64::from(image.width()) * f64::from(image.height())),
        max,
    }
}

/// A frame darker than this mean luma counts as black: the renderer drew nothing.
///
/// The darkest frame the harness makes on purpose (`interior` with one fixture off) is far above it.
pub const DEFAULT_MIN_LUMA: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn identical_images_have_no_error() {
        let image = RgbImage::from_pixel(4, 4, Rgb([10, 200, 30]));
        assert_eq!(nrmse_percent(&image, &image).unwrap(), 0.0);
    }

    #[test]
    fn black_against_white_is_full_scale() {
        let black = RgbImage::from_pixel(4, 4, Rgb([0, 0, 0]));
        let white = RgbImage::from_pixel(4, 4, Rgb([255, 255, 255]));
        assert!((nrmse_percent(&black, &white).unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn one_channel_off_by_51_is_a_known_value() {
        // One channel of three differs by 51 / 255 = 0.2 everywhere: sqrt(0.2^2 / 3) = 11.547 %.
        let a = RgbImage::from_pixel(4, 4, Rgb([0, 0, 0]));
        let b = RgbImage::from_pixel(4, 4, Rgb([51, 0, 0]));
        assert!((nrmse_percent(&a, &b).unwrap() - 11.547).abs() < 1e-3);
    }

    #[test]
    fn sizes_must_match() {
        let a = RgbImage::new(4, 4);
        let b = RgbImage::new(4, 5);
        assert!(nrmse_percent(&a, &b).is_err());
    }

    #[test]
    fn a_black_image_is_below_the_floor() {
        assert!(stats(&RgbImage::new(8, 8)).mean_luma < DEFAULT_MIN_LUMA);
        let grey = RgbImage::from_pixel(8, 8, Rgb([128, 128, 128]));
        assert!((stats(&grey).mean_luma - 128.0 / 255.0).abs() < 1e-9);
    }
}
