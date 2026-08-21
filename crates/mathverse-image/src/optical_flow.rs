//! Dense Optical Flow — pyramidal Lucas-Kanade
//!
//! Estimates pixel-level motion between two consecutive grayscale frames
//! using the multi-scale (pyramidal) Lucase-Kanade method of Bouguet.
//!
//! # Algorithm
//!
//! 1. Build Gaussian pyramids for both frames (`pyramid_levels` octaves,
//!    automatically capped so the coarsest level keeps ≥ 8 px per side).
//! 2. Start with a zero flow guess at the coarsest level.
//! 3. For each level, coarse → fine:
//!    - Propagate the coarser flow estimate: `g ← 2 · upsample(flow_coarse)`
//!      (bilinear interpolation, scaled by the level ratio).
//!    - Refine per pixel with Gauss–Newton iterations solving the 2×2
//!      Lucas-Kanade normal equations over a `(2r+1)²` window:
//!      `A·Δd = Σ ∇I·(I(q) − J(q + g + d))`,
//!      where `A = Σ ∇I∇Iᵀ` depends only on the previous frame's gradients
//!      and is therefore computed once per pixel per level.
//!    - Pixels whose window is degenerate (`det A` below an epsilon — flat
//!      regions and pure aperture problems) keep their propagated estimate
//!      instead of an unreliable solve.
//! 4. Return the level-0 flow field: `u[i]`, `v[i]` in row-major order,
//!    positive `u` = rightward motion, positive `v` = downward motion.
//!
//! # Typical Usage
//!
//! ```rust
//! use mathverse_image::optical_flow::dense_optical_flow;
//! use mathverse_image::GrayImage;
//!
//! let frame1 = GrayImage::new(256, 256).unwrap();
//! let frame2 = GrayImage::new(256, 256).unwrap();
//!
//! let flow = dense_optical_flow(&frame1, &frame2, 3);
//! assert_eq!(flow.u.len(), 256 * 256);
//! ```

use crate::GrayImage;

/// Optical flow result containing horizontal and vertical motion vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct OpticalFlow {
    /// Horizontal motion vectors (positive = rightward motion)
    pub u: Vec<f64>,
    /// Vertical motion vectors (positive = downward motion)
    pub v: Vec<f64>,
}

/// Half-width of the integration window (`7×7` with radius 3).
const WIN_RADIUS: i64 = 3;
/// Number of Gauss–Newton refinement iterations per level.
const REFINE_ITERS: usize = 3;
/// Smallest side length allowed at the coarsest pyramid level.
const MIN_COARSE_SIDE: i64 = 8;
/// Determinant threshold below which a window solve is considered degenerate.
const DET_EPSILON: f64 = 1e-6;

/// Bilinear sample of a scalar field stored row-major, with clamped borders.
fn sample_field(field: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let xf = x.clamp(0.0, (w - 1) as f64);
    let yf = y.clamp(0.0, (h - 1) as f64);
    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = xf - x0 as f64;
    let fy = yf - y0 as f64;
    let v00 = field[y0 * w + x0];
    let v10 = field[y0 * w + x1];
    let v01 = field[y1 * w + x0];
    let v11 = field[y1 * w + x1];
    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}

/// Halve an image: Gaussian anti-alias blur followed by 2× subsampling.
fn downsample(img: &GrayImage) -> GrayImage {
    let blurred = img.gaussian_blur(1, 1.0);
    let nw = ((blurred.w + 1) / 2).max(1);
    let nh = ((blurred.h + 1) / 2).max(1);
    let mut out = GrayImage::new(nw, nh).unwrap();
    for y in (0..blurred.h).step_by(2) {
        for x in (0..blurred.w).step_by(2) {
            out.set(x / 2, y / 2, blurred.get(x, y));
        }
    }
    out
}

/// Computes dense optical flow between two consecutive grayscale frames.
///
/// Uses the pyramidal Lucas-Kanade method: a Gaussian pyramid provides a
/// coarse-to-fine flow initialization so displacements larger than the
/// integration window can still be recovered.
///
/// # Arguments
///
/// * `prev` — The previous grayscale frame
/// * `curr` — The current grayscale frame (same dimensions as `prev`)
/// * `pyramid_levels` — Requested number of pyramid levels (≥ 1). The count
///   is capped so the coarsest level keeps at least 8 pixels per side; more
///   levels improve accuracy for large motions at proportional cost.
///
/// # Returns
///
/// [`OpticalFlow`] with dense motion vectors for every pixel (row-major):
/// `u[i]` = horizontal displacement (positive = right),
/// `v[i]` = vertical displacement (positive = down).
///
/// # Panics
///
/// Panics if the two frames differ in size or if `pyramid_levels == 0`.
///
/// # Precision
///
/// All calculations use `f64` arithmetic. Flow values may exceed one pixel
/// when multiple pyramid levels are used.
pub fn dense_optical_flow(prev: &GrayImage, curr: &GrayImage, pyramid_levels: usize) -> OpticalFlow {
    assert_eq!(prev.w, curr.w, "both frames must have the same width");
    assert_eq!(prev.h, curr.h, "both frames must have the same height");
    assert!(pyramid_levels >= 1, "pyramid_levels must be >= 1");

    // Cap the pyramid so the coarsest level never drops below MIN_COARSE_SIDE.
    let min_side = prev.w.min(prev.h) as i64;
    let mut levels = 1usize;
    while levels < pyramid_levels && (min_side >> levels) >= MIN_COARSE_SIDE {
        levels += 1;
    }

    // Build Gaussian pyramids (level 0 = original resolution).
    let mut prev_pyr = Vec::with_capacity(levels);
    let mut curr_pyr = Vec::with_capacity(levels);
    prev_pyr.push(prev.clone());
    curr_pyr.push(curr.clone());
    for lvl in 1..levels {
        prev_pyr.push(downsample(&prev_pyr[lvl - 1]));
        curr_pyr.push(downsample(&curr_pyr[lvl - 1]));
    }

    // Zero flow at the coarsest level.
    let coarse = levels - 1;
    let mut u = vec![0.0f64; prev_pyr[coarse].w * prev_pyr[coarse].h];
    let mut v = vec![0.0f64; prev_pyr[coarse].w * prev_pyr[coarse].h];

    for level in (0..levels).rev() {
        let iprev = &prev_pyr[level];
        let icurr = &curr_pyr[level];
        let (lw, lh) = (iprev.w, iprev.h);

        // Propagate the coarser flow guess: g = 2 · upsample(flow_coarser).
        if level < levels - 1 {
            let (gw, gh) = (prev_pyr[level + 1].w, prev_pyr[level + 1].h);
            let mut gu = vec![0.0f64; lw * lh];
            let mut gv = vec![0.0f64; lw * lh];
            for y in 0..lh {
                for x in 0..lw {
                    let cx = x as f64 / 2.0;
                    let cy = y as f64 / 2.0;
                    gu[y * lw + x] = 2.0 * sample_field(&u, gw, gh, cx, cy);
                    gv[y * lw + x] = 2.0 * sample_field(&v, gw, gh, cx, cy);
                }
            }
            u = gu;
            v = gv;
        }

        // Spatial gradients of the previous frame at this level
        // (central differences; borders get one-sided differences).
        let mut ix = vec![0.0f64; lw * lh];
        let mut iy = vec![0.0f64; lw * lh];
        for y in 0..lh {
            for x in 0..lw {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(lw - 1);
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(lh - 1);
                ix[y * lw + x] = (iprev.get(xp, y) - iprev.get(xm, y))
                    / (xp - xm) as f64;
                iy[y * lw + x] = (iprev.get(x, yp) - iprev.get(x, ym))
                    / (yp - ym) as f64;
            }
        }

        let mut next_u = u.clone();
        let mut next_v = v.clone();

        for y in 0..lh {
            for x in 0..lw {
                let g_u = u[y * lw + x];
                let g_v = v[y * lw + x];

                // Window sum of ∇I∇Iᵀ — independent of the warp, so computed once.
                let (mut a11, mut a12, mut a22) = (0.0f64, 0.0f64, 0.0f64);
                for dy in -WIN_RADIUS..=WIN_RADIUS {
                    for dx in -WIN_RADIUS..=WIN_RADIUS {
                        let sx = (x as i64 + dx).clamp(0, lw as i64 - 1) as usize;
                        let sy = (y as i64 + dy).clamp(0, lh as i64 - 1) as usize;
                        let gx = ix[sy * lw + sx];
                        let gy = iy[sy * lw + sx];
                        a11 += gx * gx;
                        a12 += gx * gy;
                        a22 += gy * gy;
                    }
                }
                let det = a11 * a22 - a12 * a12;
                if det <= DET_EPSILON {
                    // Textureless region or pure 1-D structure (aperture
                    // problem): keep the propagated estimate unsolved.
                    continue;
                }

                // Gauss–Newton refinement of d starting from the propagated guess.
                let mut d_u = 0.0f64;
                let mut d_v = 0.0f64;
                for _ in 0..REFINE_ITERS {
                    let (mut b1, mut b2) = (0.0f64, 0.0f64);
                    for dy in -WIN_RADIUS..=WIN_RADIUS {
                        for dx in -WIN_RADIUS..=WIN_RADIUS {
                            let sx = (x as i64 + dx).clamp(0, lw as i64 - 1) as usize;
                            let sy = (y as i64 + dy).clamp(0, lh as i64 - 1) as usize;
                            let j_val = sample_field(
                                &icurr.data,
                                lw,
                                lh,
                                sx as f64 + g_u + d_u,
                                sy as f64 + g_v + d_v,
                            );
                            // Residual of I(q) − J(q + g + d); its gradient
                            // step is Δd = A⁻¹·Σ ∇I·residual.
                            let residual = iprev.data[sy * lw + sx] - j_val;
                            b1 += ix[sy * lw + sx] * residual;
                            b2 += iy[sy * lw + sx] * residual;
                        }
                    }
                    d_u += (a22 * b1 - a12 * b2) / det;
                    d_v += (a11 * b2 - a12 * b1) / det;
                }

                next_u[y * lw + x] = g_u + d_u;
                next_v[y * lw + x] = g_v + d_v;
            }
        }

        u = next_u;
        v = next_v;
    }

    debug_assert_eq!(u.len(), prev.w * prev.h);
    debug_assert_eq!(v.len(), prev.w * prev.h);
    OpticalFlow { u, v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn test_optical_flow_basic() {
        let mut frame1 = GrayImage::new(32, 32).unwrap();
        let mut frame2 = GrayImage::new(32, 32).unwrap();

        // Bright square with corners (full 2-D texture) at (8, 8)..(15, 15),
        // shifted right by 2 pixels in frame2.
        for y in 8..16 {
            for x in 8..16 {
                frame1.set(x, y, 1.0);
                frame2.set((x + 2).min(31), y, 1.0);
            }
        }

        let flow = dense_optical_flow(&frame1, &frame2, 1);
        assert_eq!(flow.u.len(), 1024, "flow u should cover 32×32");
        assert_eq!(flow.v.len(), 1024, "flow v should cover 32×32");

        // Interior pixels of the square see the full pattern and should
        // recover the +2 px horizontal displacement.
        let interior_u: Vec<f64> = (9..15)
            .flat_map(|y| (9..15).map(move |x| (x, y)))
            .map(|(x, y)| flow.u[y * 32 + x])
            .collect();
        let mean_u: f64 = interior_u.iter().sum::<f64>() / interior_u.len() as f64;
        assert!(
            mean_u > 1.0,
            "interior should show rightward motion ≈ 2 px, mean_u={mean_u}"
        );
    }

    #[test]
    fn test_optical_flow_vertical_motion() {
        let mut frame1 = GrayImage::new(32, 32).unwrap();
        let mut frame2 = GrayImage::new(32, 32).unwrap();

        // Square shifted down by 2 pixels.
        for y in 8..16 {
            for x in 8..16 {
                frame1.set(x, y, 1.0);
                frame2.set(x, (y + 2).min(31), 1.0);
            }
        }

        let flow = dense_optical_flow(&frame1, &frame2, 1);
        let interior_v: Vec<f64> = (9..15)
            .flat_map(|y| (9..15).map(move |x| (x, y)))
            .map(|(x, y)| flow.v[y * 32 + x])
            .collect();
        let mean_v: f64 = interior_v.iter().sum::<f64>() / interior_v.len() as f64;
        assert!(
            mean_v > 1.0,
            "interior should show downward motion ≈ 2 px, mean_v={mean_v}"
        );
    }

    #[test]
    fn test_optical_flow_same_frame() {
        let mut frame1 = GrayImage::new(32, 32).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                frame1.set(x, y, (x + y) as f64 / 64.0);
            }
        }
        let frame2 = frame1.clone();

        let flow = dense_optical_flow(&frame1, &frame2, 1);
        // All flows should be near zero
        let u_sum: f64 = flow.u.iter().sum();
        let v_sum: f64 = flow.v.iter().sum();
        assert!(u_sum.abs() < 0.1, "identical frames should have zero flow u, sum={}", u_sum);
        assert!(v_sum.abs() < 0.1, "identical frames should have zero flow v, sum={}", v_sum);
    }

    #[test]
    fn test_optical_flow_pyramid_levels() {
        let mut frame1 = GrayImage::new(64, 64).unwrap();
        let mut frame2 = GrayImage::new(64, 64).unwrap();

        // Same textured block in both frames, shifted right by 3 px.
        for y in 20..30 {
            for x in 20..30 {
                let v = ((x / 3 + y / 2) % 2) as f64;
                frame1.set(x, y, v);
                frame2.set(x + 3, y, v);
            }
        }

        let flow1 = dense_optical_flow(&frame1, &frame2, 1);
        let flow2 = dense_optical_flow(&frame1, &frame2, 2);
        assert_eq!(flow1.u.len(), flow2.u.len());
        assert_eq!(flow1.v.len(), flow2.v.len());

        // Multi-scale should also recover the displacement direction.
        let mean_u2: f64 = (21..29)
            .flat_map(|y| (21..29).map(move |x| (x, y)))
            .map(|(x, y)| flow2.u[y * 64 + x])
            .sum::<f64>()
            / 64.0;
        assert!(mean_u2 > 1.0, "pyramid flow should track rightward motion, got {mean_u2}");
    }

    #[test]
    #[should_panic(expected = "same")]
    fn test_optical_flow_dimension_mismatch_panics() {
        let frame1 = GrayImage::new(32, 32).unwrap();
        let frame2 = GrayImage::new(16, 16).unwrap();
        let _ = dense_optical_flow(&frame1, &frame2, 1);
    }
}
