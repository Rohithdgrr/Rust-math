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
//! let mut img = GrayImage::new(32, 16).unwrap();
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
/// * `img` — Input grayscale image
/// * `sigma` — Standard deviation for Gaussian blur (typical values: 1.0–2.0)
/// * `low` — Lower threshold for weak edges (typical values: 0.01–0.1)
/// * `high` — Upper threshold for strong edges (typical values: 0.1–0.3)
///
/// # Returns
///
/// A binary edge map where edge pixels have value 1.0 and non-edge pixels
/// have value 0.0.
///
/// # Algorithm Details
///
/// The algorithm uses a fixed blur radius of 3 pixels. The gradient direction
/// is quantised into 4 sectors for non-maximum suppression:
///
/// - `[0°, 22.5°)` ∪ `[157.5°, 180°)` — horizontal edge, compare left/right
/// - `[22.5°, 67.5°)` — diagonal `\`, compare top-left/bottom-right
/// - `[67.5°, 112.5°)` — vertical edge, compare top/bottom
/// - `[112.5°, 157.5°)` — diagonal `/`, compare top-right/bottom-left
///
/// Hysteresis uses 8-connectivity to link weak edges to strong edges.
///
/// # Examples
///
/// ```rust
/// use mathverse_image::{canny::canny, GrayImage};
///
/// let mut img = GrayImage::new(64, 64).unwrap();
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
        return GrayImage::new(img.w, img.h).unwrap();
    }

    let blurred = img.gaussian_blur(3, sigma);
    let (mag, dir) = blurred.sobel();

    // Non-maximum suppression: keep gradient peaks along the edge tangent.
    // The quantisation uses the standard 4-direction scheme where:
    //   - |θ| ∈ [0, π/8) ∪ [7π/8, π):   horizontal edge → compare left/right
    //   - |θ| ∈ [π/8,  3π/8):             diagonal \     → compare TL/BR
    //   - |θ| ∈ [3π/8, 5π/8):             vertical edge → compare top/bottom
    //   - |θ| ∈ [5π/8, 7π/8):             diagonal /     → compare TR/BL
    let mut nms = GrayImage::new(img.w, img.h).unwrap();
    let pi = std::f64::consts::PI;
    let pi_8 = pi / 8.0;

    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let a = dir[y * img.w + x].abs(); // normalise to [0, π)

            let (qx, qy, rx, ry) = if a < pi_8 || a >= 7.0 * pi_8 {
                // Horizontal edge: neighbours to left and right
                (x - 1, y, x + 1, y)
            } else if a < 3.0 * pi_8 {
                // Diagonal \ : neighbours (x-1, y-1) and (x+1, y+1)
                (x - 1, y - 1, x + 1, y + 1)
            } else if a < 5.0 * pi_8 {
                // Vertical edge: neighbours above and below
                (x, y - 1, x, y + 1)
            } else {
                // Diagonal / : neighbours (x-1, y+1) and (x+1, y-1)
                (x - 1, y + 1, x + 1, y - 1)
            };

            let m = mag.get(x, y);
            let q = mag.get(qx, qy);
            let r = mag.get(rx, ry);

            if m >= q && m >= r {
                nms.set(x, y, m);
            }
        }
    }

    // Hysteresis: strong seeds, weak kept if touching a strong neighbour
    let strong = |v: f64| v >= high;
    let weak = |v: f64| v >= low && v < high;
    let mut out = GrayImage::new(img.w, img.h).unwrap();
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
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
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
        let mut img = GrayImage::new(64, 32).unwrap();
        for y in 0..32 {
            for x in 0..64 {
                img.set(x, y, if x < 32 { 0.0 } else { 1.0 });
            }
        }
        let e = canny(&img, 1.5, 0.05, 0.15);
        assert!(e.get(32, 16) > 0.0);
        assert!(e.get(33, 16) > 0.0);
        assert!(e.get(10, 16) < 0.5 && e.get(50, 16) < 0.5);
        let count: usize = e.data.iter().filter(|v| **v > 0.5).count();
        assert!(count < 200, "edge pixels: {}", count);
    }
}
