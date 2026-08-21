//! Template Matching
//!
//! Finds the best match of a template image within a larger image using
//! Normalized Cross-Correlation (NCC).
//!
//! # Algorithm
//!
//! For each valid template position:
//! 1. Center the image patch and template by subtracting their means
//! 2. Compute NCC = Σ(i−ī)(t−t̄) / (√Σ(i−ī)² · √Σ(t−t̄)²)
//! 3. Return the position with the highest NCC value (range: −1 to 1)
//!
//! # Notes
//!
//! - NCC of 1 = perfect match, 0 = no correlation, −1 = inverse match
//! - NCC requires contrast in **both** signals: a constant template (or a
//!   zero-variance image patch) makes the correlation undefined and scores 0
//! - Template must be no larger than the image in either dimension
//! - Boundary handling: the template must fit entirely within the image
//! - For performance, consider using a Gaussian pyramid for multi-scale matching
//!
//! # Example
//!
//! ```rust
//! use mathverse_image::template_matching::template_matching_with_template;
//! use mathverse_image::GrayImage;
//!
//! let mut img = GrayImage::new(64, 64).unwrap();
//! let mut template = GrayImage::new(8, 8).unwrap();
//! // Two-tone template (left half dark, right half bright)
//! for y in 0..8 {
//!     for x in 0..8 {
//!         let v = if x < 4 { 0.25 } else { 0.75 };
//!         template.set(x, y, v);
//!     }
//! }
//! // Uniform mid-gray background (avoids intensity-invariant alias matches)
//! for v_ in img.data.iter_mut() {
//!     *v_ = 0.5;
//! }
//! // Copy the template into the search image at (20, 16)
//! for y in 16..24 {
//!     for x in 20..28 {
//!         img.set(x, y, if x - 20 < 4 { 0.25 } else { 0.75 });
//!     }
//! }
//! let result = template_matching_with_template(&img, &template);
//! assert!(result.ncc > 0.9, "NCC should be close to 1.0 for exact match");
//! assert_eq!(result.x, 20.0);
//! assert_eq!(result.y, 16.0);
//! ```
//!
//! # Return Value
//!
//! [`TemplateResult`] contains:
//! - `x`, `y`: top-left corner of best match
//! - `ncc`: Normalized Cross-Correlation value (−1 to 1)

use crate::GrayImage;

/// Result of template matching operation.
///
/// Contains the position and NCC value of the best match found.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateResult {
    /// X coordinate of top-left corner of best match
    pub x: f64,
    /// Y coordinate of top-left corner of best match
    pub y: f64,
    /// Normalized Cross-Correlation value (-1 to 1, where 1 = perfect match)
    pub ncc: f64,
}

/// Finds the best match of `template` within `img` using Normalized Cross-Correlation.
///
/// The NCC at offset `(tx, ty)` is computed as
/// `Σ(i−ī)(t−t̄) / (√Σ(i−ī)² · √Σ(t−t̄)²)` over the `tw × th` window,
/// which makes the score invariant to affine intensity changes
/// (`i → a·i + b` with `a > 0`). Positions where either signal has zero
/// variance (flat patches, constant templates) score `0.0` because the
/// correlation direction is undefined there.
///
/// # Arguments
///
/// * `img` — The search image (grayscale)
/// * `template` — The template to find (must fit within `img`; should have contrast)
///
/// # Returns
///
/// [`TemplateResult`] with the best match position and NCC value.
/// Returns `TemplateResult { x: 0.0, y: 0.0, ncc: -1.0 }` if the template
/// is larger than the image or if no valid positions exist.
///
/// # Performance
///
/// Time complexity: O((W−tw+1) × (H−th+1) × tw × th). For production use,
/// consider integral-image precomputation or multi-scale pyramid search.
///
/// # Example
///
/// ```rust
/// use mathverse_image::template_matching::template_matching_with_template;
/// use mathverse_image::GrayImage;
///
/// let mut img = GrayImage::new(64, 64).unwrap();
/// let mut template = GrayImage::new(16, 16).unwrap();
/// // Two-tone template: dark left half, bright right half
/// for y in 0..16 {
///     for x in 0..16 {
///         template.set(x, y, if x < 8 { 0.25 } else { 0.75 });
///     }
/// }
/// // Uniform mid-gray background (avoids intensity-invariant alias matches)
/// for v_ in img.data.iter_mut() {
///     *v_ = 0.5;
/// }
/// // Copy the template into the image at (24, 16)
/// for y in 16..32 {
///     for x in 24..40 {
///         img.set(x, y, if x - 24 < 8 { 0.25 } else { 0.75 });
///     }
/// }
/// let result = template_matching_with_template(&img, &template);
/// assert!((result.x - 24.0).abs() < 1.0, "x should be near 24");
/// assert!((result.y - 16.0).abs() < 1.0, "y should be near 16");
/// assert!(result.ncc > 0.9, "NCC should be high for good match");
/// ```
pub fn template_matching_with_template(img: &GrayImage, template: &GrayImage) -> TemplateResult {
    // Validate sizes
    if template.w == 0 || template.h == 0 || template.w > img.w || template.h > img.h {
        return TemplateResult { x: 0.0, y: 0.0, ncc: -1.0 };
    }

    // Precompute template statistics once.
    let n_t = (template.w * template.h) as f64;
    let template_mean: f64 = template.data.iter().sum::<f64>() / n_t;
    let template_sum_sq: f64 =
        template.data.iter().map(|&v| (v - template_mean).powi(2)).sum();

    let mut best_ncc: f64 = f64::NEG_INFINITY;
    let mut best_x: f64 = 0.0;
    let mut best_y: f64 = 0.0;

    let search_limit_x = img.w - template.w;
    let search_limit_y = img.h - template.h;

    // For each valid position in the search image
    for ty in 0..=search_limit_y {
        for tx in 0..=search_limit_x {
            // Accumulate patch statistics and the cross term in one pass.
            let mut patch_sum = 0.0f64;
            let mut cross_sum = 0.0f64;

            for ty_off in 0..template.h {
                let row = (ty + ty_off) * img.w;
                let t_row = ty_off * template.w;
                for tx_off in 0..template.w {
                    let i_val = img.data[row + tx + tx_off];
                    let t_val = template.data[t_row + tx_off];
                    patch_sum += i_val;
                    cross_sum += i_val * t_val;
                }
            }

            // Patch mean/variance from raw sums (no second pass needed):
            //   Σ(i−ī)(t−t̄) = Σit − n·ī·t̄
            //   Σ(i−ī)²      = Σi² − n·ī²
            let n = n_t;
            let patch_mean = patch_sum / n;
            let t_mean = template_mean;
            let sum_i_sq = {
                let mut s = 0.0f64;
                for ty_off in 0..template.h {
                    let row = (ty + ty_off) * img.w;
                    for tx_off in 0..template.w {
                        let v = img.data[row + tx + tx_off];
                        s += v * v;
                    }
                }
                s
            };
            let numerator = cross_sum - n * patch_mean * t_mean;
            let denom_i = (sum_i_sq - n * patch_mean * patch_mean).sqrt();
            let denom_t = template_sum_sq.sqrt();

            let ncc = if denom_i > 1e-12 && denom_t > 1e-12 {
                numerator / (denom_i * denom_t)
            } else {
                0.0
            };

            if ncc > best_ncc {
                best_ncc = ncc;
                best_x = tx as f64;
                best_y = ty as f64;
            }
        }
    }

    TemplateResult {
        x: best_x,
        y: best_y,
        ncc: best_ncc.clamp(-1.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn test_template_matching_basic() {
        let mut img = GrayImage::new(64, 64).unwrap();
        // Uniform mid-gray background prevents intensity-invariant alias
        // matches against the image border (NCC is invariant to a·i + b).
        for v_ in img.data.iter_mut() {
            *v_ = 0.5;
        }
        // Two-tone 8×8 template copied to top-left corner (21, 17)
        let mut template = GrayImage::new(8, 8).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let v = if x < 4 { 0.25 } else { 0.75 };
                template.set(x, y, v);
                img.set(21 + x, 17 + y, v);
            }
        }

        let result = template_matching_with_template(&img, &template);
        // Should find match exactly at the square's top-left corner
        assert!((result.x - 21.0).abs() < 2.0, "x should be near 21, got {}", result.x);
        assert!((result.y - 17.0).abs() < 2.0, "y should be near 17, got {}", result.y);
        assert!(result.ncc > 0.9, "NCC should be high for good match, got {}", result.ncc);
    }

    #[test]
    fn test_template_matching_no_match() {
        let mut img = GrayImage::new(64, 64).unwrap();
        // Fill with uniform gray
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, 0.5);
            }
        }

        let mut template = GrayImage::new(8, 8).unwrap();
        // Template with contrast
        for y in 0..8 {
            for x in 0..8 {
                template.set(x, y, if x < 4 { 0.0 } else { 1.0 });
            }
        }

        let result = template_matching_with_template(&img, &template);
        // NCC is defined as 0 when the patch has zero variance
        assert_eq!(result.ncc, 0.0, "NCC should be 0 for zero-variance patches");
    }

    #[test]
    fn test_template_matching_template_larger_than_image() {
        let img = GrayImage::new(64, 64).unwrap();
        let template = GrayImage::new(128, 128).unwrap(); // Larger than image

        let result = template_matching_with_template(&img, &template);
        // Should return default result with ncc = -1
        assert_eq!(result.ncc, -1.0, "Should return -1 when template is larger");
        assert_eq!(result.x, 0.0);
        assert_eq!(result.y, 0.0);
    }

    #[test]
    fn test_template_matching_exact_match() {
        let mut img = GrayImage::new(32, 32).unwrap();
        let mut template = GrayImage::new(8, 8).unwrap();

        // Both have the same 4px checkerboard (note: an 8×8 window with an
        // 8px period would be constant — integer division — so use 4px cells)
        for y in 0..32 {
            for x in 0..32 {
                let v = if (x/4 + y/4) % 2 == 0 { 1.0 } else { 0.0 };
                img.set(x, y, v);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let v = if (x/4 + y/4) % 2 == 0 { 1.0 } else { 0.0 };
                template.set(x, y, v);
            }
        }

        let result = template_matching_with_template(&img, &template);
        // Should find match at (0, 0) with NCC = 1.0
        assert!((result.x - 0.0).abs() < 0.1, "x should be 0, got {}", result.x);
        assert!((result.y - 0.0).abs() < 0.1, "y should be 0, got {}", result.y);
        assert!((result.ncc - 1.0).abs() < 0.01, "NCC should be 1.0 for exact match, got {}", result.ncc);
    }

    #[test]
    fn test_template_matching_intensity_invariance() {
        // NCC is invariant under i → a·i + b (a > 0): halving every search
        // image intensity must not change the correlation score or position.
        let mut img = GrayImage::new(16, 16).unwrap();
        let mut template = GrayImage::new(8, 8).unwrap();

        for y in 0..16 {
            for x in 0..16 {
                let v = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.25 };
                img.set(x, y, v);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let v = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.25 };
                template.set(x, y, v);
            }
        }

        let result = template_matching_with_template(&img, &template);
        assert_eq!(result.x, 0.0, "pattern starts at origin");
        assert_eq!(result.y, 0.0, "pattern starts at origin");

        // Halved intensities: identical normalized correlation.
        let mut dim = GrayImage::new(16, 16).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                dim.set(x, y, img.get(x, y) * 0.5);
            }
        }
        let result2 = template_matching_with_template(&dim, &template);
        assert!(
            (result2.ncc - result.ncc).abs() < 1e-9,
            "NCC must be intensity-invariant: {} vs {}",
            result2.ncc,
            result.ncc
        );
    }
}
