//! Harris Corner Detector
//!
//! Detects corners in a grayscale image using the Harris corner detection algorithm.
//! The algorithm computes the cornerness measure R = det(M) - k·trace(M)²
//! where M is the second-moment matrix of image gradients.
//!
//! # Algorithm
//!
//! 1. Compute image gradients (Gx, Gy) using Sobel operator
//! 2. Compute second-moment matrix components:
//!    - Mxx = Gx² (Gaussian smoothed)
//!    - Myy = Gy² (Gaussian smoothed)
//!    - Mxy = Gx·Gy (Gaussian smoothed)
//! 3. Compute Harris response: R = det(M) - k·trace(M)²
//!    - det(M) = Mxx·Myy - Mxy²
//!    - trace(M) = Mxx + Myy
//! 4. Return corner coordinates sorted by response magnitude
//!
//! # Arguments
//!
//! * `img` — Input grayscale image
//! * `k` — Harris sensitivity parameter (typical: 0.04-0.06, default 0.04)
//! * `threshold` — Minimum response magnitude to consider a corner
//!
//! # Returns
//!
//! `Vec<(f64, f64, f64)>` — corners as (x, y, response) tuples,
//! sorted by response magnitude descending.
//!
//! # Example
//!
//! ```rust
//! use mathverse_image::harris::harris;
//! use mathverse_image::GrayImage;
//!
//! let mut img = GrayImage::new(128, 128).unwrap();
//! // Create a checkerboard-like pattern with corners
//! for y in 0..128 {
//!     for x in 0..128 {
//!         let v = if (x/32 + y/32) % 2 == 0 { 0.0 } else { 1.0 };
//!         img.set(x, y, v);
//!     }
//! }
//! let corners = harris(&img, 0.04, 0.01);
//! // corners[0] = (x, y, response) — strongest corner
//! assert!(!corners.is_empty());
//! ```
//!
//! # Notes
//!
//! - k parameter typical range: 0.04–0.06 (0.04 is common)
//! - threshold parameter filters corners by response magnitude
//! - Returns corners as (x, y, R) tuples sorted by response (descending)
//! - Non-maximum suppression is applied internally

use crate::GrayImage;

/// Detects corners in a grayscale image using the Harris algorithm.
///
/// # Algorithm
///
/// 1. Compute gradients (Gx, Gy) via Sobel
/// 2. Accumulate a per-pixel structure tensor over a `(2r+1)²` window
/// 3. Compute the local Harris response R = det(S) − k·trace(S)²
/// 4. Filter by threshold
/// 5. Apply 3×3 non-maximum suppression
/// 6. Sort by response descending
///
/// # Precision
///
/// All calculations use f64 precision in [0, 1] range.
/// Image gradients are computed assuming pixel values in [0, 1].
///
/// # Returns
///
/// `Vec<(f64, f64, f64)>` — corners as (x, y, response) tuples,
/// sorted by response magnitude descending. Returns empty vector
/// if no corners exceed the threshold.
pub fn harris(img: &GrayImage, k: f64, threshold: f64) -> Vec<(f64, f64, f64)> {
    // Compute raw Sobel gradients (not magnitude)
    const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let mut gx_img = GrayImage::new(img.w, img.h).unwrap();
    let mut gy_img = GrayImage::new(img.w, img.h).unwrap();

    for y in 0..img.h {
        for x in 0..img.w {
            let mut gx_val = 0.0;
            let mut gy_val = 0.0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = (x as i64 + kx as i64 - 1).clamp(0, img.w as i64 - 1) as usize;
                    let py = (y as i64 + ky as i64 - 1).clamp(0, img.h as i64 - 1) as usize;
                    let p = img.get(px, py);
                    gx_val += GX[ky * 3 + kx] * p;
                    gy_val += GY[ky * 3 + kx] * p;
                }
            }
            gx_img.set(x, y, gx_val);
            gy_img.set(x, y, gy_val);
        }
    }

    // Structure-tensor accumulation window (5×5) around each pixel.
    const WIN_RADIUS: i64 = 2;

    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for y in 0..img.h {
        for x in 0..img.w {
            let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
            for dy in -WIN_RADIUS..=WIN_RADIUS {
                for dx in -WIN_RADIUS..=WIN_RADIUS {
                    let sx = (x as i64 + dx).clamp(0, img.w as i64 - 1) as usize;
                    let sy = (y as i64 + dy).clamp(0, img.h as i64 - 1) as usize;
                    let gx = gx_img.get(sx, sy);
                    let gy = gy_img.get(sx, sy);
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            // Harris response R = det(S) − k·trace(S)²
            let det_s = sxx * syy - sxy * sxy;
            let trace_s = sxx + syy;
            let response = det_s - k * trace_s * trace_s;

            if response > threshold {
                candidates.push((x, y, response));
            }
        }
    }

    // Non-maximum suppression: keep only pixels whose response is the strict
    // maximum within their 3×3 neighbourhood (ties keep the first scanned).
    let resp_at = |cx: usize, cy: usize| -> Option<f64> {
        candidates
            .iter()
            .find(|(x, y, _)| *x == cx && *y == cy)
            .map(|(_, _, r)| *r)
    };

    let mut corners: Vec<(usize, usize, f64)> = Vec::new();
    for &(x, y, r) in &candidates {
        let mut is_max = true;
        'outer: for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || ny < 0 || nx >= img.w as i64 || ny >= img.h as i64 {
                    continue;
                }
                if let Some(nr) = resp_at(nx as usize, ny as usize) {
                    if nr > r {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
        }
        if is_max {
            corners.push((x, y, r));
        }
    }

    // Sort by response descending. `total_cmp` gives a total order, so the
    // sort never panics even if a response is NaN (NaN sorts deterministically
    // instead of aborting with a `partial_cmp().unwrap()` panic).
    corners.sort_by(|a, b| b.2.total_cmp(&a.2));

    corners.into_iter().map(|(x, y, r)| (x as f64, y as f64, r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn test_harris_basic() {
        let mut img = GrayImage::new(16, 16).unwrap();
        // Create a strong corner at (8, 8) with a black background
        for y in 0..16 {
            for x in 0..16 {
                img.set(x, y, 0.0);
            }
        }
        // Add a white pixel to create edges/corners
        img.set(8, 8, 1.0);

        let corners = harris(&img, 0.04, 0.001);
        // Should detect at least some corners
        assert!(!corners.is_empty(), "Should detect corners in 16x16 image");
    }

    #[test]
    fn test_harris_k_parameter() {
        let mut img = GrayImage::new(32, 32).unwrap();
        // Create diagonal line - should have corners where line changes direction
        for y in 0..32 {
            for x in 0..32 {
                let v = if x < 16 { 0.0 } else { 1.0 };
                img.set(x, y, v);
            }
        }

        // Different k values should give different results
        let corners_tight = harris(&img, 0.04, 0.001);
        let corners_loose = harris(&img, 0.01, 0.001);

        // Tighter k should be more selective
        assert!(
            corners_tight.len() <= corners_loose.len(),
            "Tighter k (0.04) should find fewer corners than looser k (0.01)"
        );
    }

    #[test]
    fn test_harris_threshold() {
        let mut img = GrayImage::new(16, 16).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                img.set(x, y, 0.0);
            }
        }

        // Very high threshold should find no corners
        let high_threshold = harris(&img, 0.04, 10.0);
        assert!(
            high_threshold.is_empty(),
            "High threshold should find no corners"
        );

        // Very low threshold should find many corners
        let low_threshold = harris(&img, 0.04, -1.0);
        // Should find some corners
        assert!(
            !low_threshold.is_empty(),
            "Low threshold should find corners"
        );
    }

    #[test]
    fn test_harris_uniform_image() {
        let mut img = GrayImage::new(16, 16).unwrap();
        // Uniform image - no corners
        for y in 0..16 {
            for x in 0..16 {
                img.set(x, y, 0.5);
            }
        }

        let corners = harris(&img, 0.04, 0.001);
        // Uniform image should have very few or no corners
        // (response will be near zero everywhere)
        let response_sum: f64 = corners.iter().map(|(_, _, r)| r).sum();
        assert!(
            response_sum < 0.1,
            "Uniform image should have minimal corner response"
        );
    }
}