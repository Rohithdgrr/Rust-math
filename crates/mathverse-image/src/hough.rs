//! Hough transform for line detection.

use crate::GrayImage;

/// Performs a Hough transform for straight line detection.
///
/// Accumulates votes in a `(theta, rho)` parameter space and returns line
/// parameters whose accumulator count exceeds `peak_threshold`.
///
/// # Arguments
///
/// * `img` — Edge image (high values = edge pixels)
/// * `peak_threshold` — Minimum accumulator votes to count as a detection
/// * `theta_res` — Number of discretised θ bins in `[0, π)`
/// * `rho_res` — Number of discretised ρ bins across `[−max_rho, +max_rho]`
///
/// # Returns
///
/// A `Vec` of `(theta, rho)` pairs where `theta` is in radians and `rho` is
/// in pixels.
pub fn hough_lines(
    img: &GrayImage,
    peak_threshold: usize,
    theta_res: usize,
    rho_res: usize,
) -> Vec<(f64, f64)> {
    let max_rho = ((img.w as f64).powi(2) + (img.h as f64).powi(2)).sqrt();
    let mut accumulator = vec![vec![0usize; rho_res]; theta_res];

    for y in 0..img.h {
        for x in 0..img.w {
            if img.get(x, y) < 0.1 {
                continue;
            }
            for t in 0..theta_res {
                let theta = t as f64 * std::f64::consts::PI / theta_res as f64;
                let rho = x as f64 * theta.cos() + y as f64 * theta.sin();
                let r = (((rho + max_rho) / (2.0 * max_rho)) * rho_res as f64) as usize;
                let r = r.min(rho_res - 1);
                accumulator[t][r] += 1;
            }
        }
    }

    let mut lines = Vec::new();
    for t in 0..theta_res {
        for r in 0..rho_res {
            if accumulator[t][r] >= peak_threshold {
                let theta = t as f64 * std::f64::consts::PI / theta_res as f64;
                let rho = r as f64 * 2.0 * max_rho / rho_res as f64 - max_rho;
                lines.push((theta, rho));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hough_vertical_line() {
        let mut img = GrayImage::new(32, 32).unwrap();
        for y in 0..32 {
            img.set(10, y, 1.0);
        }
        let lines = hough_lines(&img, 5, 180, 200);
        assert!(!lines.is_empty(), "should detect at least one line");
        // With the ρ = x·cos θ + y·sin θ convention, a vertical line at
        // x = 10 peaks at θ ≈ 0 with ρ ≈ 10.
        let has_vertical = lines.iter().any(|&(theta, rho)| {
            let dtheta = (theta - 0.0).abs().min((theta - std::f64::consts::PI).abs());
            dtheta < 0.1 && (rho - 10.0).abs() < 2.0
        });
        assert!(has_vertical, "expected a near-vertical line at ρ≈10, got: {:?}", &lines[..lines.len().min(5)]);
    }
}
