//! Feature detection: Harris corner detector using Sobel gradients.

use crate::Image;

/// Computes the Harris corner response map for an image.
///
/// `sigma` is the Gaussian smoothing parameter, `k` is the Harris detector free parameter (typically 0.04 - 0.06).
pub fn harris(img: &Image, _sigma: f64, k: f64) -> Image {
    const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let dx = img.convolve3(&GX);
    let dy = img.convolve3(&GY);
    let mut ixx = Image::new(img.w, img.h);
    let mut iyy = Image::new(img.w, img.h);
    let mut ixy = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        ixx.data[i] = dx.data[i] * dx.data[i];
        iyy.data[i] = dy.data[i] * dy.data[i];
        ixy.data[i] = dx.data[i] * dy.data[i];
    }
    let ixx = ixx.gaussian_blur(2, 1.0);
    let iyy = iyy.gaussian_blur(2, 1.0);
    let ixy = ixy.gaussian_blur(2, 1.0);
    let mut r = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        let (a, b, c) = (ixx.data[i], iyy.data[i], ixy.data[i]);
        r.data[i] = (a * b - c * c) - k * (a + b).powi(2);
    }
    r
}

/// Computes the Shi–Tomasi corner response (`min(λ₁, λ₂)` of the structure
/// tensor) — the criterion used by `cv2.goodFeaturesToTrack`.
///
/// The gradient window is blurred with a 5×5 Gaussian (`sigma = 1.0`) exactly
/// as in [`harris`]. Higher values indicate stronger corners.
pub fn shi_tomasi(img: &Image) -> Image {
    const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let dx = img.convolve3(&GX);
    let dy = img.convolve3(&GY);
    let mut ixx = Image::new(img.w, img.h);
    let mut iyy = Image::new(img.w, img.h);
    let mut ixy = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        ixx.data[i] = dx.data[i] * dx.data[i];
        iyy.data[i] = dy.data[i] * dy.data[i];
        ixy.data[i] = dx.data[i] * dy.data[i];
    }
    let ixx = ixx.gaussian_blur(2, 1.0);
    let iyy = iyy.gaussian_blur(2, 1.0);
    let ixy = ixy.gaussian_blur(2, 1.0);
    let mut r = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        let (a, b, c) = (ixx.data[i], iyy.data[i], ixy.data[i]);
        // Eigenvalues of [[a, c], [c, b]].
        let trace = a + b;
        let det = a * b - c * c;
        let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let l1 = (trace + disc) / 2.0;
        let l2 = (trace - disc) / 2.0;
        r.data[i] = l2.min(l1);
    }
    r
}

/// FAST corner detector (`cv2.FastFeatureDetector` equivalent): a pixel is a
/// corner when at least 9 of the 16 pixels on the radius-3 Bresenham circle
/// are all brighter or all darker than the center by `threshold`.
///
/// Returns a binary response image (`1.0` at corners, else `0.0`). Border
/// pixels within 3 of the edge are never corners.
pub fn fast(img: &Image, threshold: f64, non_max_suppression: bool) -> Image {
    // Bresenham circle of radius 3, clockwise starting at (0, -3).
    const CIRCLE: [(i64, i64); 16] = [
        (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
    ];
    const ARC: usize = 9;

    let (w, h) = (img.w, img.h);
    let mut response = Image::new(w, h);
    let mut scores = Image::new(w, h);

    let is_corner = |x: usize, y: usize| -> bool {
        if x < 3 || y < 3 || x + 3 >= w || y + 3 >= h {
            return false;
        }
        let center = img.data[y * w + x];
        // Classify each circle pixel: +1 brighter, -1 darker, 0 similar.
        let mut states = [0i8; 16];
        for (i, (dx, dy)) in CIRCLE.iter().enumerate() {
            let v = img.data[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize];
            if v > center + threshold {
                states[i] = 1;
            } else if v < center - threshold {
                states[i] = -1;
            }
        }
        // Look for ARC contiguous +1s or −1s (wrapping around the ring).
        for start in 0..16 {
            for &target in &[1i8, -1] {
                let mut run = 0;
                for k in 0..16 {
                    if states[(start + k) % 16] == target {
                        run += 1;
                        if run >= ARC {
                            return true;
                        }
                    } else {
                        run = 0;
                    }
                }
            }
        }
        false
    };

    let score = |x: usize, y: usize| -> f64 {
        let center = img.data[y * w + x];
        CIRCLE
            .iter()
            .map(|(dx, dy)| (img.data[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize] - center).abs())
            .fold(0.0, f64::max)
    };

    for y in 3..h.saturating_sub(3) {
        for x in 3..w.saturating_sub(3) {
            if is_corner(x, y) {
                response.data[y * w + x] = 1.0;
                scores.data[y * w + x] = score(x, y);
            }
        }
    }

    if non_max_suppression {
        let mut kept = Image::new(w, h);
        for y in 3..h.saturating_sub(3) {
            for x in 3..w.saturating_sub(3) {
                if response.data[y * w + x] == 0.0 {
                    continue;
                }
                let s = scores.data[y * w + x];
                let is_max = (0..3).all(|dy| {
                    (0..3).all(|dx| {
                        let (nx, ny) = (x + dx - 1, y + dy - 1);
                        scores.data[ny * w + nx] <= s
                    })
                });
                if is_max {
                    kept.data[y * w + x] = 1.0;
                }
            }
        }
        kept
    } else {
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_vs_flat() {
        let mut img = Image::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img.set(x, y, if (x / 6 + y / 6) % 2 == 0 { 1.0 } else { 0.0 });
            }
        }
        let r = harris(&img, 1.0, 0.04);
        let at_corner = r.get(6, 6);
        let at_flat = r.get(2, 2);
        assert!(at_corner > at_flat + 1e-4, "corner {} flat {}", at_corner, at_flat);
    }

    #[test]
    fn edge_low_response() {
        let mut img = Image::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img.set(x, y, if x < 12 { 0.0 } else { 1.0 });
            }
        }
        let r = harris(&img, 1.0, 0.04);
        let mx = r.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(mx < 1e-4, "edge max {}", mx);
    }

    #[test]
    fn shi_tomasi_finds_checks_corners() {
        let mut img = Image::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img.set(x, y, if (x / 6 + y / 6) % 2 == 0 { 1.0 } else { 0.0 });
            }
        }
        let r = shi_tomasi(&img);
        // Corners of the checker squares get high response.
        let max = r.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 1e-3, "max response {max}");
        let at_corner = r.get(6, 6);
        let at_flat = r.get(2, 2);
        assert!(at_corner > at_flat + 1e-4, "corner {} flat {}", at_corner, at_flat);
    }

    #[test]
    fn fast_detects_corner_not_flat() {
        let mut img = Image::new(24, 24);
        // Four isolated 5×5 bright squares: each corner is pointy enough for
        // FAST-9 (a checkerboard junction yields arcs of exactly 8 pixels,
        // which is below the required contiguous run of 9).
        for &(x0, y0) in &[(3, 3), (16, 3), (3, 16), (16, 16)] {
            for y in y0..y0 + 5 {
                for x in x0..x0 + 5 {
                    img.set(x, y, 1.0);
                }
            }
        }
        let corners = fast(&img, 0.2, true);
        let count: usize = corners.data.iter().filter(|&&v| v > 0.5).count();
        assert!(count >= 4, "corner count {count}");
        // A flat image has no corners.
        let flat = Image::from_data(24, 24, vec![0.5; 576]);
        let flat_corners = fast(&flat, 0.2, true);
        assert!(flat_corners.data.iter().all(|&v| v == 0.0));
    }
}
