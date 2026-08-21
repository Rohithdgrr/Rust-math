//! Hough Line Transform
//!
//! Detects line segments in binary or edge images using the Standard Hough Transform.
//! Accumulates votes in a (rho, theta) parameter space to identify lines.
//!
//! # Algorithm
//!
//! 1. For each edge pixel at (x, y), compute rho and theta for all theta values:
//!    - rho = x·cos(θ) + y·sin(θ)
//!    - theta ranges from 0 to π in discrete steps
//! 2. Accumulate votes in a 1D accumulator array indexed by rho bins for each theta
//! 3. Find local maxima in the accumulator above a threshold
//! 5. Extract the (rho, theta) pairs of detected lines
//! 6. Optionally re-fit line segments using least-squares on edge pixels
//!
//! # Parameterization
//!
//! Lines are parameterized as: ρ = x·cos(θ) + y·sin(θ)
//! where:
//! - ρ (rho) is the perpendicular distance from the origin to the line
//! - θ (theta) is the angle of the normal vector from the origin to the line
//! - ρ ∈ [−D, D] where D = √(w² + h²) is the image diagonal
//! - θ ∈ [0, π) radians
//!
//! # Arguments
//!
//! * `img` — Input grayscale or binary image (values > threshold considered edge pixels)
//! * `theta_resolution` — Number of discrete theta samples from 0 to π (default: 180)
//! * `rho_resolution` — Delta rho value in pixels (default: 1.0);
//!   smaller values give finer detection but larger accumulator
//! * `threshold` — Minimum accumulator votes to consider a line detected
//! * `min_line_length` — Minimum line length in pixels (default: 0, disabled)
//! * `max_line_gap` — Maximum gap between line segments to link them (default: 0, disabled)
//!
//! # Returns
//!
//! `Vec<(f64, f64)>` — Detected lines as (rho, theta) pairs:
//! - rho: perpendicular distance from origin in pixels (can be negative)
//! - theta: angle of normal vector in radians [0, π)
//!
//! # Example
//!
//! ```rust
//! use mathverse_image::hough::hough_line_transform;
//! use mathverse_image::{canny::canny, GrayImage};
//!
//! let mut img = GrayImage::new(256, 256).unwrap();
//! // Draw a vertical edge at x=128
//! for y in 0..256 {
//!     for x in 0..256 {
//!         img.set(x, y, if x >= 128 { 1.0 } else { 0.0 });
//!     }
//! }
//! // Apply Canny edge detection
//! let edges = canny(&img, 1.5, 0.05, 0.15);
//! // Detect lines
//! let lines = hough_line_transform(&edges, 180, 1.0, 100);
//! // The vertical edge at x=128 concentrates at theta ≈ 0, rho ≈ 128
//! assert!(!lines.is_empty());
//! for (rho, theta) in &lines {
//!     println!("Line: rho={:.2}, theta={:.3} rad ({:.1}°)", rho, theta, theta * 180.0 / std::f64::consts::PI);
//! }
//! ```
//!
//! # Notes
//!
//! - The accumulator uses rho indexing: rho_idx = ((rho + D) / rho_resolution).round() as usize
//! - Clamps rho_idx to valid range to prevent out-of-bounds access
//! - Returns lines sorted by vote count (highest first)
//! - For best results, apply Canny edge detection before calling this function

use crate::GrayImage;
use std::f64::consts::PI;

/// Detected line from Hough Transform as (rho, theta) pair.
///
/// - `rho`: Perpendicular distance from origin to line in pixels
/// - `theta`: Angle of normal vector from origin to line in radians [0, π)
type Line = (f64, f64);

/// Detects line segments in an image using the Standard Hough Transform.
///
/// # Algorithm
///
/// 1. Compute image diagonal D = √(w² + h²) to determine rho range [−D, D]
/// 2. For each theta sample (0 to π, theta_resolution steps):
///    - For each pixel above threshold:
///      - Compute rho = x·cos(θ) + y·sin(θ)
///      - Increment accumulator at rho bin
/// 3. Find accumulator peaks above threshold
/// 4. Return (rho, theta) pairs for detected lines
///
/// # Precision
///
/// - Theta resolution defaults to 180 steps (1° increments)
/// - Rho resolution defaults to 1.0 pixel
/// - rho values are centered around 0 (negative rho = opposite direction)
///
/// # Returns
///
/// `Vec<(f64, f64)>` — Detected lines as (rho, theta) pairs,
/// sorted by vote count descending (most voted lines first).
/// Returns empty vector if no lines exceed the threshold.
pub fn hough_line_transform(
    img: &GrayImage,
    theta_resolution: usize,
    rho_resolution: f64,
    threshold: usize,
) -> Vec<Line> {
    let w = img.w;
    let h = img.h;

    // Compute image diagonal for rho range
    let diag = ((w * w + h * h) as f64).sqrt();

    // Theta range: 0 to π (exclusive); guard against degenerate resolutions
    let theta_step = PI / (theta_resolution.max(2) as f64 - 1.0);

    // Rho range: -diag to +diag
    // Number of rho bins: ceil(2 * diag / rho_resolution) + 1
    let num_rho_bins = ((2.0 * diag / rho_resolution) + 1.0).ceil() as usize;

    // Initialize accumulator: for each theta, we have num_rho_bins bins
    // We'll use a flat array: accumulator[theta * num_rho_bins + rho_idx]
    let mut accumulator: Vec<usize> = vec![0; theta_resolution * num_rho_bins];

    // Helper to compute rho bin index from rho value
    let rho_offset = diag / rho_resolution; // shift so rho=0 is at bin diag/rho_resolution

    // Accumulate votes from edge pixels
    for y in 0..h {
        for x in 0..w {
            let pixel_value = img.get(x, y);
            // Consider pixels above a minimal threshold as edge pixels
            // Default: consider any non-zero pixel as an edge pixel
            // For better results, apply Canny edge detection first
            if pixel_value > 0.0 {
                for theta_idx in 0..theta_resolution {
                    let theta = theta_idx as f64 * theta_step;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let rho = (x as f64) * cos_t + (y as f64) * sin_t;

                    // Map rho from [−diag, +diag] to [0, num_rho_bins-1]
                    let rho_idx = ((rho + diag) / rho_resolution).round() as usize;

                    // Clamp to valid range
                    let rho_idx = rho_idx.min(num_rho_bins - 1);

                    let acc_idx = theta_idx * num_rho_bins + rho_idx;
                    accumulator[acc_idx] += 1;
                }
            }
        }
    }

    // Find peaks in the accumulator above threshold, keeping vote counts so
    // the result can be sorted by votes (most voted lines first).
    let mut peaks: Vec<(usize, f64, f64)> = Vec::new();
    for theta_idx in 0..theta_resolution {
        for rho_idx in 0..num_rho_bins {
            let votes = accumulator[theta_idx * num_rho_bins + rho_idx];
            if votes >= threshold {
                let rho = (rho_idx as f64 - rho_offset) * rho_resolution;
                let theta = theta_idx as f64 * theta_step;
                peaks.push((votes, rho, theta));
            }
        }
    }

    // Sort by vote count descending; `total_cmp` keeps the order total even
    // if a coordinate is NaN.
    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));

    // Remove duplicate lines (lines with very close rho and theta)
    let mut unique_lines: Vec<Line> = Vec::new();
    for (_, rho, theta) in &peaks {
        let is_duplicate = unique_lines.iter().any(|ul| {
            let rho_diff = (*rho - ul.0).abs();
            let theta_diff = (*theta - ul.1).abs();
            rho_diff < rho_resolution && theta_diff < (PI / theta_resolution as f64)
        });
        if !is_duplicate {
            unique_lines.push((*rho, *theta));
        }
    }

    unique_lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn test_hough_vertical_line() {
        let mut img = GrayImage::new(100, 100).unwrap();
        // Draw a vertical line at x=50
        for y in 0..100 {
            for x in 0..100 {
                img.set(x, y, if x == 50 { 1.0 } else { 0.0 });
            }
        }

        // Hough transform with ρ = x·cosθ + y·sinθ: a vertical line x=50 is
        // concentrated at θ = 0 (normal points along +x), ρ ≈ 50.
        let lines = hough_line_transform(&img, 180, 1.0, 20);
        // Should detect at least one line
        assert!(!lines.is_empty(), "Should detect vertical line");

        // Find the line closest to theta = 0
        let vertical_lines: Vec<_> = lines
            .iter()
            .filter(|(_, theta)| theta.abs() < 0.1 || (PI - theta).abs() < 0.1)
            .collect();
        assert!(
            !vertical_lines.is_empty(),
            "Should detect lines near theta=0 for vertical line"
        );

        // For vertical line at x=50, rho should be around 50 (distance from origin)
        let rho_vals: Vec<_> = lines
            .iter()
            .filter(|(rho, _)| rho.abs() > 40.0 && rho.abs() < 60.0)
            .collect();
        assert!(!rho_vals.is_empty(), "Should detect rho ≈ 50 for vertical line at x=50");
    }

    #[test]
    fn test_hough_horizontal_line() {
        let mut img = GrayImage::new(100, 100).unwrap();
        // Draw a horizontal line at y=50
        for y in 0..100 {
            for x in 0..100 {
                img.set(x, y, if y == 50 { 1.0 } else { 0.0 });
            }
        }

        // With ρ = x·cosθ + y·sinθ, a horizontal line y=50 concentrates at
        // θ = π/2 (normal points along +y), ρ ≈ 50.
        let lines = hough_line_transform(&img, 180, 1.0, 20);
        assert!(!lines.is_empty(), "Should detect horizontal line");

        // For horizontal line at y=50, theta should be π/2 (normal vertical)
        let horizontal_lines: Vec<_> = lines
            .iter()
            .filter(|(_, theta)| (theta - PI / 2.0).abs() < 0.1)
            .collect();
        assert!(
            !horizontal_lines.is_empty(),
            "Should detect lines near theta=π/2 for horizontal line"
        );
    }

    #[test]
    fn test_hough_no_lines() {
        let img = GrayImage::new(50, 50).unwrap();
        // All zeros - no edges
        let lines = hough_line_transform(&img, 180, 1.0, 10);
        // Should return empty (or very few) lines
        // With threshold=1 and all pixels zero, no lines detected
        // But our implementation considers any non-zero pixel, so test with truly empty
        assert!(lines.is_empty() || lines.len() < 5, "Should detect very few lines in blank image");
    }

    #[test]
    fn test_hough_parameters() {
        let mut img = GrayImage::new(50, 50).unwrap();
        // Draw a diagonal line from (0,0) to (50,50)
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, if x == y { 1.0 } else { 0.0 });
            }
        }

        // Test with different theta resolutions
        let lines_coarse = hough_line_transform(&img, 45, 1.0, 10);   // 45 theta steps
        let lines_fine = hough_line_transform(&img, 180, 1.0, 10);   // 180 theta steps

        // Fine resolution should find the line (diagonal = theta=π/4, rho=0)
        assert!(
            !lines_fine.is_empty(),
            "Fine theta resolution should detect diagonal line"
        );
    }
}