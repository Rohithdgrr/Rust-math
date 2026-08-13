//! Template matching: slides a template over an image and produces a response
//! map, mirroring `cv2.matchTemplate` with all six comparison methods.
//!
//! The result has size `(W − w + 1) × (H − h + 1)` where the template is
//! `w × h` and the image `W × H`; use [`crate::utils::min_max`] on the result
//! to locate the best match.

use crate::Image;

/// Comparison methods for [`match_template`] (mirrors `cv2.TM_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Sum of squared differences. **Lower** is better.
    SqDiff,
    /// Normalized sum of squared differences. **Lower** is better.
    SqDiffNormed,
    /// Cross-correlation. **Higher** is better.
    CCorr,
    /// Normalized cross-correlation. **Higher** is better.
    CCorrNormed,
    /// Correlation coefficient (mean-subtracted). **Higher** is better.
    CCoeff,
    /// Normalized correlation coefficient. **Higher** is better.
    CCoeffNormed,
}

/// Slides `templ` over `img` and returns the response map.
///
/// The output dimensions are `(img.w − templ.w + 1) × (img.h − templ.h + 1)`.
/// For the `*Normed` methods the values lie in `[−1, 1]`.
///
/// # Panics
///
/// Panics if the template is larger than the image in either dimension.
pub fn match_template(img: &Image, templ: &Image, method: MatchMethod) -> Image {
    assert!(
        templ.w <= img.w && templ.h <= img.h,
        "match_template: template must not exceed image size"
    );
    let (ow, oh) = (img.w - templ.w + 1, img.h - templ.h + 1);
    let mut out = Image::new(ow, oh);

    let (tw, th) = (templ.w, templ.h);
    let n = (tw * th) as f64;

    for oy in 0..oh {
        for ox in 0..ow {
            let mut sq_diff = 0.0;
            let mut corr = 0.0;
            let mut sum_i = 0.0;
            let mut sum_i2 = 0.0;
            let mut sum_t = 0.0;
            let mut sum_t2 = 0.0;
            for ty in 0..th {
                for tx in 0..tw {
                    let iv = img.data[(oy + ty) * img.w + (ox + tx)];
                    let tv = templ.data[ty * tw + tx];
                    let d = iv - tv;
                    sq_diff += d * d;
                    corr += iv * tv;
                    sum_i += iv;
                    sum_i2 += iv * iv;
                    sum_t += tv;
                    sum_t2 += tv * tv;
                }
            }
            let cov = corr - sum_i * sum_t / n;
            out.data[oy * ow + ox] = match method {
                MatchMethod::SqDiff => sq_diff,
                MatchMethod::SqDiffNormed => {
                    let denom = (sum_i2 * sum_t2).sqrt();
                    if denom > 1e-12 { sq_diff / denom } else { f64::INFINITY }
                }
                MatchMethod::CCorr => corr,
                MatchMethod::CCorrNormed => {
                    let denom = (sum_i2 * sum_t2).sqrt();
                    if denom > 1e-12 { corr / denom } else { 0.0 }
                }
                MatchMethod::CCoeff => cov,
                MatchMethod::CCoeffNormed => {
                    let var_i = (sum_i2 - sum_i * sum_i / n).max(0.0);
                    let var_t = (sum_t2 - sum_t * sum_t / n).max(0.0);
                    let denom = (var_i * var_t).sqrt();
                    if denom > 1e-12 { cov / denom } else { 0.0 }
                }
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::min_max;

    #[test]
    fn exact_match_location() {
        // 10×10 image with a bright 4×4 square at (3, 2).
        let mut img = Image::new(10, 10);
        for y in 2..6 {
            for x in 3..7 {
                img.set(x, y, 1.0);
            }
        }
        // Template is the same square.
        let mut templ = Image::new(4, 4);
        for i in 0..16 {
            templ.data[i] = 1.0;
        }
        let res = match_template(&img, &templ, MatchMethod::SqDiff);
        assert_eq!((res.w, res.h), (7, 7));
        // For SQDIFF a smaller value is better, so use the min location.
        let (_, _, best, _) = min_max(&res);
        assert_eq!(best, (3, 2));
        // SqDiff at the best location must be ~0.
        assert!(res.get(3, 2) < 1e-12);
    }

    #[test]
    fn normalized_ccorr_finds_shifted_template() {
        let mut img = Image::new(8, 8);
        for y in 1..5 {
            for x in 2..6 {
                img.set(x, y, 0.8);
            }
        }
        let mut templ = Image::new(3, 3);
        for i in 0..9 {
            templ.data[i] = 0.8;
        }
        let res = match_template(&img, &templ, MatchMethod::CCorrNormed);
        let (_, max_v, _, best) = min_max(&res);
        assert!(max_v > 0.99, "max {max_v}");
        assert_eq!(best, (2, 1));
    }

    #[test]
    fn ccoeff_normed_finds_corner_template() {
        // Gradient background + bright square; template = 20×20 window whose
        // top-left corner region contains the square's top-left corner.
        let (w, h) = (96usize, 96usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, 0.15 + 0.4 * ((x + y) as f64 / (w + h) as f64));
            }
        }
        for y in 10..34 {
            for x in 10..34 {
                img.set(x, y, 0.95);
            }
        }
        let templ = crate::transform::crop(&img, 8, 8, 20, 20).unwrap();
        let res = match_template(&img, &templ, MatchMethod::CCoeffNormed);
        // `min_max` returns (min_val, max_val, min_loc, max_loc): for the best
        // match use the max value/location (4th tuple element).
        let (_, best_val, _, best_loc) = crate::utils::min_max(&res);
        // The exact location must be the strongest match (~1.0), and the worst
        // match must be strictly weaker.
        assert!((res.get(8, 8) - 1.0).abs() < 1e-9, "score at true loc {}", res.get(8, 8));
        assert_eq!(best_loc, (8, 8), "best {best_loc:?} val {best_val}");
    }

    #[test]
    fn methods_agree_on_identity() {
        // Template equal to the whole image: SqDiff=0, CCorrNormed=1, CCoeffNormed=1.
        let mut img = Image::new(4, 4);
        for i in 0..16 {
            img.data[i] = (i % 5) as f64 / 5.0;
        }
        let templ = img.clone();
        let sq = match_template(&img, &templ, MatchMethod::SqDiff);
        assert!(sq.get(0, 0) < 1e-12);
        let ccn = match_template(&img, &templ, MatchMethod::CCorrNormed);
        assert!((ccn.get(0, 0) - 1.0).abs() < 1e-9);
        let cc = match_template(&img, &templ, MatchMethod::CCoeffNormed);
        assert!((cc.get(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn template_larger_than_image_panics() {
        let img = Image::new(3, 3);
        let templ = Image::new(5, 5);
        let _ = match_template(&img, &templ, MatchMethod::SqDiff);
    }
}
