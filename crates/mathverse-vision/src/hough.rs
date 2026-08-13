//! Hough transforms: `cv2.HoughLines` and `cv2.HoughCircles` equivalents for
//! detecting lines and circles in edge images.
//!
//! Inputs are treated as binary edge maps: pixels with value `> 0.5` vote in
//! the accumulator.

use crate::Image;

/// Detects straight lines in an edge image using the Hough transform.
///
/// Each pixel `(x, y)` with value `> 0.5` votes for every line through it in
/// the `(θ, ρ)` parameter space (`θ` sampled at 1° resolution, `ρ` quantized
/// to whole pixels). Peaks with more than `threshold` votes are returned as
/// `(rho, theta)` pairs with `theta` in radians — the same representation as
/// `cv2.HoughLines`. A local-maximum (non-maximum suppression) filter is
/// applied over the accumulator.
pub fn hough_lines(img: &Image, threshold: usize) -> Vec<(f64, f64)> {
    let (w, h) = (img.w, img.h);
    let diag = ((w * w + h * h) as f64).sqrt().ceil() as usize;
    let rho_bins = 2 * diag + 1;
    let theta_bins = 180;
    let mut acc = vec![0usize; theta_bins * rho_bins];

    // Precompute cos/sin tables.
    let cos_t: Vec<f64> = (0..theta_bins).map(|t| (t as f64).to_radians().cos()).collect();
    let sin_t: Vec<f64> = (0..theta_bins).map(|t| (t as f64).to_radians().sin()).collect();

    for y in 0..h {
        for x in 0..w {
            if img.data[y * w + x] <= 0.5 {
                continue;
            }
            for t in 0..theta_bins {
                let rho = (x as f64 * cos_t[t] + y as f64 * sin_t[t]).round() as i64 + diag as i64;
                if (0..rho_bins as i64).contains(&rho) {
                    acc[t * rho_bins + rho as usize] += 1;
                }
            }
        }
    }

    // Peak extraction with 3×3 non-maximum suppression.
    let mut lines: Vec<(usize, f64, f64)> = Vec::new();
    for t in 1..theta_bins - 1 {
        for r in 1..rho_bins - 1 {
            let v = acc[t * rho_bins + r];
            if v < threshold {
                continue;
            }
            let is_max = (-1i64..=1).all(|dt| {
                (-1i64..=1).all(|dr| acc[(t as i64 + dt) as usize * rho_bins + (r as i64 + dr) as usize] <= v)
            });
            if is_max {
                let rho = (r as i64 - diag as i64) as f64;
                let theta = (t as f64).to_radians();
                lines.push((v, rho, theta));
            }
        }
    }
    // Sort by vote strength, strongest first.
    lines.sort_by(|a, b| b.0.cmp(&a.0));
    lines.into_iter().map(|(_, rho, theta)| (rho, theta)).collect()
}

/// Detects circles in an edge image using the Hough transform.
///
/// For each edge pixel and each radius in `min_radius..=max_radius`, candidate
/// centers are voted on over 360 angle steps. Peaks with more than `threshold`
/// votes are returned as `(center_x, center_y, radius)` — the same
/// representation as `cv2.HoughCircles` (without the gradient/2-1 Hough
/// refinement).
pub fn hough_circles(
    img: &Image,
    min_radius: usize,
    max_radius: usize,
    threshold: usize,
) -> Vec<(f64, f64, f64)> {
    let (w, h) = (img.w, img.h);
    assert!(min_radius >= 1 && max_radius >= min_radius, "invalid radius range");
    let mut centers: Vec<Vec<Vec<usize>>> = vec![vec![vec![0usize; w]; h]; max_radius + 1];

    let mut edge_pixels = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if img.data[y * w + x] > 0.5 {
                edge_pixels.push((x as i64, y as i64));
            }
        }
    }

    for (ex, ey) in &edge_pixels {
        for r in min_radius..=max_radius {
            let rf = r as f64;
            for deg in 0..360 {
                let th = (deg as f64).to_radians();
                let a = (*ex as f64 - rf * th.cos()).round() as i64;
                let b = (*ey as f64 - rf * th.sin()).round() as i64;
                if a >= 0 && b >= 0 && (a as usize) < w && (b as usize) < h {
                    centers[r][b as usize][a as usize] += 1;
                }
            }
        }
    }

    let mut circles: Vec<(usize, usize, usize, usize)> = Vec::new();
    for r in min_radius..=max_radius {
        for y in 0..h {
            for x in 0..w {
                let v = centers[r][y][x];
                if v < threshold {
                    continue;
                }
                // Non-maximum suppression over the center plane.
                let is_max = |x0: usize, y0: usize| {
                    let mut max_v = 0;
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let (nx, ny) = (x0 as i64 + dx, y0 as i64 + dy);
                            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                                max_v = max_v.max(centers[r][ny as usize][nx as usize]);
                            }
                        }
                    }
                    max_v <= v
                };
                if is_max(x, y) {
                    circles.push((v, x, y, r));
                }
            }
        }
    }
    // Rank by vote strength, strongest first.
    circles.sort_by(|a, b| b.0.cmp(&a.0));
    circles.into_iter().map(|(_, x, y, r)| (x as f64, y as f64, r as f64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawing::line;

    #[test]
    fn detects_horizontal_line() {
        let mut img = Image::new(30, 30);
        // A horizontal line at y = 15.
        line(&mut img, (2, 15), (27, 15), 1.0, 1);
        let lines = hough_lines(&img, 20);
        assert!(!lines.is_empty(), "no lines found");
        let (_, theta) = lines[0];
        // Horizontal line: theta ≈ 90° (pi/2).
        let deg = theta.to_degrees();
        assert!(
            (deg - 90.0).abs() < 10.0 || (deg - 0.0).abs() < 10.0 || (deg - 180.0).abs() < 10.0,
            "theta {deg}"
        );
    }

    #[test]
    fn detects_circle() {
        let mut img = Image::new(30, 30);
        // A clean single-pixel-wide circle of radius 8 centered at (15, 15),
        // generated parametrically so every edge pixel lies on the ideal ring.
        for deg in 0..360 {
            let th = (deg as f64).to_radians();
            let x = (15.0 + 8.0 * th.cos()).round() as i64;
            let y = (15.0 + 8.0 * th.sin()).round() as i64;
            img.data[y as usize * 30 + x as usize] = 1.0;
        }
        let circles = hough_circles(&img, 5, 12, 200);
        assert!(!circles.is_empty(), "no circles found");
        let (x, y, r) = circles[0];
        assert!((x - 15.0).abs() < 2.0, "x {x}");
        assert!((y - 15.0).abs() < 2.0, "y {y}");
        assert!((r - 8.0).abs() < 2.0, "r {r}");
    }

    #[test]
    fn no_lines_in_blank() {
        let img = Image::new(20, 20);
        assert!(hough_lines(&img, 5).is_empty());
        assert!(hough_circles(&img, 3, 6, 10).is_empty());
    }
}
