//! # Canny Edge Detection
//!
//! This module implements the Canny edge detection algorithm, a multi-stage
//! algorithm for detecting edges in images. The algorithm consists of:
//!
//! 1. **Gaussian Blur**: Noise reduction using Gaussian smoothing
//! 2. **Sobel Gradients**: Computing gradient magnitude and direction
//! 3. **Non-Maximum Suppression**: Thinning edges to single-pixel width
//! 4. **Double Thresholding**: Classifying edges as strong or weak
//! 5. **Hysteresis**: Connecting weak edges to strong edges
//!
//! The Canny algorithm is known for producing thin, well-localized edges
//! with good noise robustness.
//!
//! # Example
//!
//! ```rust
//! use mathverse_image::{canny::canny, GrayImage};
//!
//! let mut img = GrayImage::new(32, 16);
//! for y in 0..16 {
//!     for x in 0..32 {
//!         img.set(x, y, if x < 16 { 0.0 } else { 1.0 });
//!     }
//! }
//! let edges = canny(&img, 1.5, 0.05, 0.15);
//! ```

use crate::GrayImage;

/// Performs Canny edge detection on a grayscale image.
///
/// This function implements the complete Canny edge detection pipeline:
/// 1. Gaussian blur with the specified sigma for noise reduction
/// 2. Sobel gradient computation for edge strength and direction
/// 3. Non-maximum suppression to thin edges to single-pixel width
/// 4. Double thresholding to classify strong and weak edges
/// 5. Hysteresis to connect weak edges to strong edges
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `sigma` - Standard deviation for Gaussian blur (typical values: 1.0-2.0)
/// * `low` - Lower threshold for weak edges (typical values: 0.01-0.1)
/// * `high` - Upper threshold for strong edges (typical values: 0.1-0.3)
///
/// # Returns
///
/// A binary edge map where edge pixels have value 1.0 and non-edge pixels have value 0.0.
///
/// # Algorithm Details
///
/// The algorithm uses a fixed blur radius of 3 pixels. The gradient direction is
/// quantized into 4 directions (0°, 45°, 90°, 135°) for non-maximum suppression.
/// Hysteresis uses 8-connectivity to link weak edges to strong edges.
///
/// # Examples
///
/// ```rust
/// use mathverse_image::{canny::canny, GrayImage};
///
/// let mut img = GrayImage::new(64, 64);
/// // Create a step edge
/// for y in 0..64 {
///     for x in 0..64 {
///         img.set(x, y, if x < 32 { 0.0 } else { 1.0 });
///     }
/// }
/// let edges = canny(&img, 1.5, 0.05, 0.15);
/// // edges should detect the vertical line at x=32
/// ```
pub fn canny(img: &GrayImage, sigma: f64, low: f64, high: f64) -> GrayImage {
    if img.w < 3 || img.h < 3 {
        return GrayImage::new(img.w, img.h);
    }
    let blurred = img.gaussian_blur(3, sigma);
    let (mag, dir) = blurred.sobel();

    // non-maximum suppression: keep gradient peaks along the normal
    let mut nms = GrayImage::new(img.w, img.h);
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let a = dir[y * img.w + x];
            let (q, r) = if !(-3.0 * core::f64::consts::FRAC_PI_8..3.0 * core::f64::consts::FRAC_PI_8).contains(&a) {
                (mag.get(x, y - 1), mag.get(x, y + 1))
            } else if a < -core::f64::consts::FRAC_PI_8 {
                (mag.get(x - 1, y - 1), mag.get(x + 1, y + 1))
            } else if a < core::f64::consts::FRAC_PI_8 {
                (mag.get(x - 1, y), mag.get(x + 1, y))
            } else {
                (mag.get(x - 1, y + 1), mag.get(x + 1, y - 1))
            };
            let m = mag.get(x, y);
            if m >= q && m >= r {
                nms.set(x, y, m);
            }
        }
    }

    // hysteresis: strong seeds, weak kept if touching a strong neighbor
    let strong = |v: f64| v >= high;
    let weak = |v: f64| v >= low && v < high;
    let mut out = GrayImage::new(img.w, img.h);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for y in 0..img.h {
        for x in 0..img.w {
            if strong(nms.get(x, y)) {
                out.set(x, y, 1.0);
                stack.push((x, y));
            }
        }
    }
    while let Some((x, y)) = stack.pop() {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx < 0 || ny < 0 || nx >= img.w as i64 || ny >= img.h as i64 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if out.get(nx, ny) == 0.0 && weak(nms.get(nx, ny)) {
                    out.set(nx, ny, 1.0);
                    stack.push((nx, ny));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_step_edge() {
        let mut img = GrayImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                img.set(x, y, if x < 32 { 0.0 } else { 1.0 });
            }
        }
        let e = canny(&img, 1.5, 0.05, 0.15);
        // exactly one vertical line of edges
        assert!(e.get(32, 16) > 0.0);
        assert!(e.get(33, 16) > 0.0);
        assert!(e.get(10, 16) < 0.5 && e.get(50, 16) < 0.5);
        // rows far from any edge are empty
        let count: usize = e.data.iter().filter(|v| **v > 0.5).count();
        assert!(count < 200, "edge pixels: {}", count);
    }
}
