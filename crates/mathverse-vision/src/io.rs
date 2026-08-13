//! Image file I/O: `imread` / `imwrite` in the style of OpenCV.
//!
//! Supports the formats provided by the [`image`] crate: PNG, JPEG, BMP and
//! PNM (PPM/PGM/PBM). The file format is chosen from the file extension.
//!
//! # Examples
//!
//! ```no_run
//! use mathverse_vision::{Image, io::{imread, imwrite}};
//!
//! let img = imread("input.png").unwrap();
//! imwrite("output.png", &img).unwrap();
//! ```
//!
//! Color images are represented using the crate's interleaved-RGB convention:
//! an [`Image`] whose width is `3 × w`, with data laid out `[r, g, b, r, g, b, …]`
//! (the same layout produced by [`crate::color::gray_to_jet`]).

use crate::Image;
use image::imageops::FilterType;
use std::path::Path;

/// Loads an image file as a single-channel grayscale image.
///
/// Pixels are converted to luminance and normalized to `[0.0, 1.0]`.
/// Equivalent to `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read or decoded.
pub fn imread(path: impl AsRef<Path>) -> Result<Image, String> {
    let img = image::open(path).map_err(|e| format!("imread failed: {e}"))?;
    let gray = img.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let data = gray
        .as_raw()
        .iter()
        .map(|&p| p as f64 / 255.0)
        .collect::<Vec<_>>();
    Ok(Image::from_data(w, h, data))
}

/// Loads an image file as an interleaved RGB image.
///
/// The returned [`Image`] has width `3 × w`, height `h`, and data laid out
/// `[r, g, b, r, g, b, …]`. Equivalent to `cv2.imread(path, cv2.IMREAD_COLOR)`
/// followed by `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read or decoded.
pub fn imread_color(path: impl AsRef<Path>) -> Result<Image, String> {
    let img = image::open(path).map_err(|e| format!("imread_color failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let data = rgb
        .as_raw()
        .iter()
        .map(|&p| p as f64 / 255.0)
        .collect::<Vec<_>>();
    Ok(Image::from_data(w * 3, h, data))
}

/// Saves a grayscale image to a file.
///
/// The format is inferred from the extension (`.png`, `.jpg`, `.jpeg`,
/// `.bmp`, `.ppm`, …). Equivalent to `cv2.imwrite(path, img)` for a
/// single-channel image.
///
/// # Errors
///
/// Returns `Err` if the file cannot be created or encoded.
pub fn imwrite(path: impl AsRef<Path>, img: &Image) -> Result<(), String> {
    let (w, h) = (img.w as u32, img.h as u32);
    let raw = img
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let gray: image::GrayImage =
        image::ImageBuffer::from_raw(w, h, raw).ok_or_else(|| "imwrite: invalid dimensions".to_string())?;
    gray.save(path).map_err(|e| format!("imwrite failed: {e}"))
}

/// Saves an interleaved-RGB image to a file.
///
/// The input must follow the crate's color convention: width `3 × w`,
/// data laid out `[r, g, b, …]` (see [`imread_color`]).
///
/// # Errors
///
/// Returns `Err` if the file cannot be created or encoded, or if the width
/// is not divisible by 3.
pub fn imwrite_color(path: impl AsRef<Path>, img: &Image) -> Result<(), String> {
    if img.w % 3 != 0 {
        return Err(format!(
            "imwrite_color: width {} is not a multiple of 3 (expected interleaved RGB)",
            img.w
        ));
    }
    let w = (img.w / 3) as u32;
    let h = img.h as u32;
    let raw = img
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let rgb: image::RgbImage =
        image::ImageBuffer::from_raw(w, h, raw).ok_or_else(|| "imwrite_color: invalid dimensions".to_string())?;
    rgb.save(path).map_err(|e| format!("imwrite_color failed: {e}"))
}

/// Resizes an image to `new_w × new_h` using bilinear interpolation.
///
/// Provided here as an OpenCV-style helper (`cv2.resize` with
/// `INTER_LINEAR`); the pure-Rust [`crate::transform::resize`] uses
/// nearest-neighbour. When the target size is a multiple of the source size,
/// this is equivalent to downsampling/upsampling without visible artifacts.
///
/// # Errors
///
/// Returns `Err` if the new dimensions are zero.
pub fn resize(img: &Image, new_w: usize, new_h: usize) -> Result<Image, String> {
    if new_w == 0 || new_h == 0 {
        return Err("resize: target dimensions must be non-zero".to_string());
    }
    let w = img.w as u32;
    let h = img.h as u32;
    let raw = img
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let gray: image::GrayImage =
        image::ImageBuffer::from_raw(w, h, raw).ok_or_else(|| "resize: invalid dimensions".to_string())?;
    let scaled = image::imageops::resize(&gray, new_w as u32, new_h as u32, FilterType::Triangle);
    let data = scaled
        .as_raw()
        .iter()
        .map(|&p| p as f64 / 255.0)
        .collect::<Vec<_>>();
    Ok(Image::from_data(new_w, new_h, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_roundtrip() {
        let mut img = Image::new(8, 6);
        for i in 0..img.data.len() {
            img.data[i] = i as f64 / (img.data.len() - 1) as f64;
        }
        let path = std::env::temp_dir().join("mathverse_io_roundtrip.png");
        imwrite(&path, &img).unwrap();
        let loaded = imread(&path).unwrap();
        assert_eq!((loaded.w, loaded.h), (8, 6));
        for (a, b) in img.data.iter().zip(&loaded.data) {
            assert!((a - b).abs() < 0.02, "pixel diff {}", (a - b).abs());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn color_roundtrip() {
        // Build a small interleaved RGB image: 2x2 checkerboard of red/blue.
        let mut rgb = Image::new(3 * 2, 2);
        for y in 0..2 {
            for x in 0..2 {
                let i = y * 6 + x * 3;
                if (x + y) % 2 == 0 {
                    rgb.data[i] = 1.0; // R
                    rgb.data[i + 1] = 0.0;
                    rgb.data[i + 2] = 0.0;
                } else {
                    rgb.data[i] = 0.0;
                    rgb.data[i + 1] = 0.0;
                    rgb.data[i + 2] = 1.0; // B
                }
            }
        }
        let path = std::env::temp_dir().join("mathverse_io_color.png");
        imwrite_color(&path, &rgb).unwrap();
        let loaded = imread_color(&path).unwrap();
        assert_eq!(loaded.w, 6);
        assert_eq!(loaded.h, 2);
        for (a, b) in rgb.data.iter().zip(&loaded.data) {
            assert!((a - b).abs() < 0.02, "pixel diff {}", (a - b).abs());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn imread_missing_file() {
        assert!(imread("definitely_missing_file_xyz.png").is_err());
    }
}
