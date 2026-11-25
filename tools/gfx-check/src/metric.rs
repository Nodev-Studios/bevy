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
    /// Pixels measured: the whole image, or the region's.
    pub pixels: u64,
    /// Measured pixels whose three channels are all 0.
    pub zero_pixels: u64,
    /// The smallest Rec. 709 luma of a measured pixel, 0 to 1.
    pub min_luma: f64,
}

/// The part of an image `image stats` measures.
pub enum Region {
    /// Pixels `x0..x1` × `y0..y1`.
    Rect { x0: u32, y0: u32, x1: u32, y1: u32 },
    /// The pixels where a mask image of the same size is not black.
    Mask(RgbImage),
}

impl Region {
    /// Reads `x0,y0,x1,y1` (pixels, ends excluded) or the path of a mask PNG.
    pub fn parse(text: &str) -> Result<Self, String> {
        let numbers: Vec<Option<u32>> = text.split(',').map(|n| n.trim().parse().ok()).collect();
        if numbers.len() == 4 && numbers.iter().all(Option::is_some) {
            let n: Vec<u32> = numbers.into_iter().flatten().collect();
            if n[0] >= n[2] || n[1] >= n[3] {
                return Err(format!("--region {text}: an empty rectangle"));
            }
            return Ok(Self::Rect {
                x0: n[0],
                y0: n[1],
                x1: n[2],
                y1: n[3],
            });
        }
        Ok(Self::Mask(load(Path::new(text))?))
    }

    fn contains(&self, x: u32, y: u32) -> bool {
        match self {
            Self::Rect { x0, y0, x1, y1 } => (*x0..*x1).contains(&x) && (*y0..*y1).contains(&y),
            Self::Mask(mask) => mask.get_pixel(x, y).0 != [0, 0, 0],
        }
    }
}

/// Measures how bright an image is.
pub fn stats(image: &RgbImage) -> Stats {
    stats_in(image, None).expect("the whole image has pixels")
}

/// Measures how bright `region` of an image is (the whole image without one).
pub fn stats_in(image: &RgbImage, region: Option<&Region>) -> Result<Stats, String> {
    if let Some(Region::Mask(mask)) = region
        && mask.dimensions() != image.dimensions()
    {
        return Err(format!(
            "the mask is {}x{}, the image {}x{}",
            mask.width(),
            mask.height(),
            image.width(),
            image.height()
        ));
    }
    let (mut luma, mut min_luma) = (0.0, f64::INFINITY);
    let (mut max, mut pixels, mut zero_pixels) = (0, 0, 0);
    for (x, y, pixel) in image.enumerate_pixels() {
        if region.is_some_and(|region| !region.contains(x, y)) {
            continue;
        }
        let [r, g, b] = pixel.0;
        let l = 0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b);
        luma += l;
        min_luma = min_luma.min(l / 255.0);
        max = max.max(r).max(g).max(b);
        pixels += 1;
        zero_pixels += u64::from(pixel.0 == [0, 0, 0]);
    }
    if pixels == 0 {
        return Err("the region holds no pixel of the image".into());
    }
    Ok(Stats {
        width: image.width(),
        height: image.height(),
        mean_luma: luma / 255.0 / pixels as f64,
        max,
        pixels,
        zero_pixels,
        min_luma,
    })
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

    #[test]
    fn a_region_measures_only_its_pixels() {
        let mut image = RgbImage::new(4, 2);
        image.put_pixel(1, 0, Rgb([255, 255, 255]));
        image.put_pixel(3, 1, Rgb([51, 51, 51]));
        let rect = Region::parse("1,0,2,1").unwrap();
        let stats = stats_in(&image, Some(&rect)).unwrap();
        assert_eq!((stats.pixels, stats.zero_pixels), (1, 0));
        assert!((stats.mean_luma - 1.0).abs() < 1e-9);
        let mut mask = RgbImage::new(4, 2);
        mask.put_pixel(3, 1, Rgb([255, 255, 255]));
        mask.put_pixel(0, 0, Rgb([255, 255, 255]));
        let stats = stats_in(&image, Some(&Region::Mask(mask))).unwrap();
        assert_eq!((stats.pixels, stats.zero_pixels), (2, 1));
        assert!((stats.mean_luma - 0.1).abs() < 1e-9);
        assert_eq!(stats.min_luma, 0.0);
        assert!(Region::parse("2,0,1,1").is_err());
        assert!(stats_in(&image, Some(&Region::Mask(RgbImage::new(2, 2)))).is_err());
    }
}
