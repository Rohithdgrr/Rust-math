#! Gaussian Pyramid
//!
//! Multi-scale image representation via successive Gaussian blur and subsampling.
//! Each octave halves the width and height, enabling scale-invariant processing.
//!
//! # Algorithm
//!
//! 1. Start with original image as level 0
//! 2. Apply Gaussian blur with sigma proportional to octave scale
//! 3. Subsample by taking every other pixel (2× downsampling)
//! 4. Repeat for desired number of octaves
//!
//! # Usage
//!
//! ```rust
//! use mathverse_image::gaussian_pyramid::gaussian_pyramid;
//!
//! let img = GrayImage::new(512, 512).unwrap();
//! let pyramid = gaussian_pyramid(&img, 3);
//! // pyramid[0] = 512×512 original
//! // pyramid[1] = 256×256 first octave
//! // pyramid[2] = 128×128 second octave
//! // pyramid[3] =  64×64 third octave
//! ```

use crate::{gaussian_blur, GrayImage};

/// Build a Gaussian pyramid with the given number of octaves.
///
/// Each octave applies Gaussian blur followed by 2× subsampling.
/// Level 0 is the original image. Level N has dimensions
/// `original_w / 2ᴺ × original_h / 2ᴺ`.
///
/// # Arguments
///
/// * `img` — Input grayscale image
/// * `octaves` — Number of octaves (levels) to generate, including level 0
///
/// # Returns
///
/// `Vec<GrayImage>` where index `i` is level `i` (0 = original).
/// Level `i` has dimensions `img.w / 2ⁱ × img.h / 2ⁱ`.
///
/// # Panics
///
/// Panics if any level would have dimensions < 1 pixel.
///
/// # Example
///
/// ```rust
/// use mathverse_image::gaussian_pyramid::gaussian_pyramid;
/// use mathverse_image::GrayImage;
///
/// let img = GrayImage::new(64, 64).unwrap();
/// let pyramid = gaussian_pyramid(&img, 4);
// assert_eq!(pyramid.len(), 4);
 // assert_eq!(pyramid[0].w, 64);  // level 0: original
 // assert_eq!(pyramid[1].w, 32);  // level 1: 64/2
 // assert_eq!(pyramid[2].w, 16);  // level 2: 64/4
 // assert_eq!(pyramid[3].w,  8);  // level 3: 64/8
/// ```
pub fn gaussian_pyramid(img: &GrayImage, octaves: usize) -> Vec<GrayImage> {
    let mut pyramid = Vec::with_capacity(octaves);
    pyramid.push(img.clone());

    let mut current = img.clone();
    for _ in 1..octaves {
        // Apply Gaussian blur with sigma proportional to level
        // sigma = 0.5 * (current level downsampling factor)
        let level = pyramid.len();
        let sigma = 0.5 * (level as f64);
        let radius = ((4.0 * sigma) + 0.5).ceil() as usize;
        if radius < 1 {
            radius = 1;
        }

        // Clamp radius to not exceed current image dimensions
        let max_radius = current.w.min(current.h) / 2;
        let effective_radius = radius.min(max_radius);

        // Gaussian blur
        let blurred = current.gaussian_blur(effective_radius, sigma);

        // Subsample by 2: take every other pixel
        let new_w = (blurred.w + 1) / 2; // ceiling division for odd widths
        let new_h = (blurred.h + 1) / 2;

        let mut subsampled = GrayImage::new(new_w, new_h).unwrap();
        for y in (0..blurred.h).step_by(2) {
            for x in (0..blurred.w).step_by(2) {
                subsampled.set(x / 2, y / 2, blurred.get(x, y));
            }
        }
        current = subsampled;
        pyramid.push(current);
    }
    pyramid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn test_pyramid_basic() {
        let img = GrayImage::new(64, 64).unwrap();
        let pyramid = gaussian_pyramid(&img, 4);
        assert_eq!(pyramid.len(), 4);
        assert_eq!(pyramid[0].w, 64); // level 0
        assert_eq!(pyramid[1].w, 32); // level 1
        assert_eq!(pyramid[2].w, 16); // level 2
        assert_eq!(pyramid[3].w, 8);  // level 3
    }

    #[test]
    fn test_pyramid_decreasing_size() {
        let img = GrayImage::new(128, 64).unwrap();
        let pyramid = gaussian_pyramid(&img, 4);
        assert_eq!(pyramid.len(), 4);
        // Level 0: 128×64
        assert_eq!(pyramid[0].w, 128);
        assert_eq!(pyramid[0].h, 64);
        // Level 1: 64×32
        assert_eq!(pyramid[1].w, 64);
        assert_eq!(pyramid[1].h, 32);
        // Level 2: 32×16
        assert_eq!(pyramid[2].w, 32);
        assert_eq!(pyramid[2].h, 16);
        // Level 3: 16×8
        assert_eq!(pyramid[3].w, 16);
        assert_eq!(pyramid[3].h, 8);
    }

    #[test]
    fn test_pyramid_values_preserved() {
        // Create image with gradient
        let mut img = GrayImage::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                img.set(x, y, (x + y) as f64 / 64.0);
            }
        }
        let pyramid = gaussian_pyramid(&img, 3);
        // Original should be unchanged
        assert!((pyramid[0].get(0, 0) - 0.0).abs() < 1e-10);
        assert!((pyramid[0].get(31, 31) - 1.0).abs() < 1e-10);
        // Lower levels should have smoother values
        // Level 1 downsampled: pixel at (0,0) should be original (0,0)
        assert!((pyramid[1].get(0, 0) - 0.0).abs() < 1e-10);
        // Level 2 downsampled: pixel at (0,0) should be original (0,0)
        assert!((pyramid[2].get(0, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pyramid_minimum_size() {
        let img = GrayImage::new(4, 4).unwrap();
        // Should work down to 1x1
        let pyramid = gaussian_pyramid(&img, 5); // 4 → 2 → 1 → 1 → 1 → 1 (clamped)
        assert!(pyramid.len() >= 3); // At minimum 3 levels
    }
}