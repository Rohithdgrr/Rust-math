//! Image I/O operations for loading and saving grayscale images.

use crate::{error::Result, GrayImage, ImageError};
use image::{DynamicImage, ImageBuffer, Luma};
use std::path::Path;

/// Load an image from a file and convert to grayscale `GrayImage`.
///
/// Supports common image formats (PNG, JPEG, BMP, etc.) via the `image` crate.
///
/// # Errors
///
/// Returns an error if the file cannot be read or the image format is not supported.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use mathverse_image::io::load;
///
/// let img = load("input.png")?;
/// # Ok(())
/// # }
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> Result<GrayImage> {
    let dyn_img = image::open(path)?;
    convert_to_grayimage(dyn_img)
}

/// Save a `GrayImage` to a file.
///
/// The format is determined by the file extension (e.g., `.png`, `.jpg`, `.bmp`).
///
/// # Errors
///
/// Returns an error if the file cannot be written or the format is not supported.
///
/// # Example
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use mathverse_image::{GrayImage, io::save};
///
/// let img = GrayImage::new(64, 64);
/// save(&img, "output.png")?;
/// # Ok(())
/// # }
/// ```
pub fn save<P: AsRef<Path>>(img: &GrayImage, path: P) -> Result<()> {
    let gray_img = convert_from_grayimage(img)?;
    gray_img.save(path)?;
    Ok(())
}

/// Convert a `DynamicImage` to our `GrayImage`.
fn convert_to_grayimage(dyn_img: DynamicImage) -> Result<GrayImage> {
    let gray_img = dyn_img.to_luma8();
    let (width, height) = gray_img.dimensions();
    let w = width as usize;
    let h = height as usize;
    
    let data: Vec<f64> = gray_img
        .pixels()
        .map(|p| p[0] as f64 / 255.0)
        .collect();
    
    GrayImage::from_data(w, h, data)
}

/// Convert our `GrayImage` to an `image::GrayImage` for saving.
fn convert_from_grayimage(img: &GrayImage) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
    let pixels: Vec<u8> = img
        .data
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    ImageBuffer::from_raw(img.w as u32, img.h as u32, pixels)
        .ok_or_else(|| ImageError::InvalidImageData(format!(
            "dimensions {}x{} do not match pixel buffer length {}",
            img.w, img.h, img.data.len()
        )))
}

/// Load from raw bytes (PNG, JPEG, etc.) and convert to grayscale.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded as a valid image.
pub fn load_from_bytes(bytes: &[u8]) -> Result<GrayImage> {
    let dyn_img = image::load_from_memory(bytes)?;
    convert_to_grayimage(dyn_img)
}

/// Save to bytes in the specified format (PNG, JPEG, etc.).
///
/// # Errors
///
/// Returns an error if the image cannot be encoded in the specified format.
pub fn save_to_bytes(img: &GrayImage, format: image::ImageFormat) -> Result<Vec<u8>> {
    let gray_img = convert_from_grayimage(img)?;
    let mut bytes = Vec::new();
    gray_img.write_to(&mut std::io::Cursor::new(&mut bytes), format)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_roundtrip() {
        let mut original = GrayImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                original.set(x, y, ((x + y) % 32) as f64 / 32.0);
            }
        }

    let gray_img = convert_from_grayimage(&original).unwrap();
    let converted = convert_to_grayimage(DynamicImage::ImageLuma8(gray_img)).unwrap();

        assert_eq!(original.w, converted.w);
        assert_eq!(original.h, converted.h);
        
        for i in 0..original.data.len() {
            // Allow small rounding differences due to 8-bit conversion
            assert!((original.data[i] - converted.data[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_save_to_bytes() {
        let img = GrayImage::new(16, 16);
        let bytes = save_to_bytes(&img, image::ImageFormat::Png).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_load_from_bytes() {
        let img = GrayImage::new(16, 16);
        let bytes = save_to_bytes(&img, image::ImageFormat::Png).unwrap();
        let loaded = load_from_bytes(&bytes).unwrap();
        assert_eq!(img.w, loaded.w);
        assert_eq!(img.h, loaded.h);
    }
}
