//! # Advanced Operations
//!
//! Histogram analysis, contour detection, and other computer vision utilities.

use crate::Image;

/// 256-bin histogram over the [0, 1] range.
///
/// Divides the [0, 1] range into 256 equal-width bins and counts pixel values
/// in each bin. Values are binned using `floor(v * 256)`, with the value `1.0`
/// mapping to bin 255.
///
/// # Returns
/// * `[usize; 256]` - Histogram bin counts
pub fn histogram(img: &Image) -> [usize; 256] {
    let mut bins = [0usize; 256];
    for &v in &img.data {
        let bin = (v * 256.0).floor() as usize;
        let bin = bin.min(255);
        bins[bin] += 1;
    }
    bins
}

/// Compute the gradient magnitude and direction using Sobel operators.
///
/// Returns both the magnitude image and the direction angles in radians.
///
/// # Returns
/// * `(Image, Vec<f64>)` - (magnitude image, direction angles flat vector)
pub fn sobel(img: &Image) -> (Image, Vec<f64>) {
    use crate::kernels::SOBEL_GX;
    use crate::kernels::SOBEL_GY;

    let mut magnitude = Image::new(img.w, img.h);
    let mut direction = vec![0.0; img.w * img.h];

    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            // Gx convolution
            let mut gx = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                gx += SOBEL_GX[ky * 3] * src[0] + SOBEL_GX[ky * 3 + 1] * src[1] + SOBEL_GX[ky * 3 + 2] * src[2];
            }

            // Gy convolution
            let mut gy = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                gy += SOBEL_GY[ky * 3] * src[0] + SOBEL_GY[ky * 3 + 1] * src[1] + SOBEL_GY[ky * 3 + 2] * src[2];
            }

            magnitude.set(x, y, (gx * gx + gy * gy).sqrt());
            direction[y * img.w + x] = gy.atan2(gx);
        }
    }

    (magnitude, direction)
}

/// Compute the Laplacian of an image.
///
/// Uses the 3×3 Laplacian kernel for second-derivative edge detection.
///
/// # Returns
/// * `Image` - Laplacian response image
pub fn laplacian(img: &Image) -> Image {
    let mut result = Image::new(img.w, img.h);
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let mut s = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                s += crate::kernels::LAPLACIAN[ky * 3] * src[0]
                    + crate::kernels::LAPLACIAN[ky * 3 + 1] * src[1]
                    + crate::kernels::LAPLACIAN[ky * 3 + 2] * src[2];
            }
            result.data[y * img.w + x] = s;
        }
    }
    result
}

/// Apply Canny edge detection approximation using Sobel + hysteresis.
///
/// This is a simplified Canny implementation that uses Sobel gradient magnitude
/// followed by double thresholding and hysteresis tracking.
///
/// # Arguments
/// * `low_threshold` - Low threshold for hysteresis
/// * `high_threshold` - High threshold for hysteresis
///
/// # Returns
/// * `Image` - Binary edge image (values 0.0 or 1.0)
pub fn canny(img: &Image, low_threshold: f64, high_threshold: f64) -> Image {
    // Step 1: Compute gradient magnitude and direction
    let (magnitude, _) = sobel(img);

    // Step 2: Non-maximum suppression
    let mut suppressed = Image::new(img.w, img.h);
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let mag = magnitude.get(x, y);
            if mag < low_threshold {
                suppressed.set(x, y, 0.0);
            } else if mag > high_threshold {
                suppressed.set(x, y, 1.0);
            } else {
                // Hysteresis: keep if connected to strong edges
                suppressed.set(x, y, 0.5); // sub-threshold
            }
        }
    }

    // Step 3: Hysteresis thresholding - track connected components
    // Pixels with value 1.0 are strong edges (keep)
    // Pixels with value 0.5 are potential edges (keep if connected to strong)
    // Pixels with value 0.0 are non-edges (discard)
    let mut result = Image::new(img.w, img.h);
    // First pass: mark strong edges
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            if suppressed.get(x, y) >= 1.0 {
                result.set(x, y, 1.0);
            }
        }
    }

    // Second pass: track connected components (4-connectivity).
    // Any un-promoted pixel (0.5 in `suppressed`, not yet in `result`) that is
    // 4-connected to a strong edge becomes 1.0. The `result` check is what
    // makes this converge: promoted pixels are never re-examined.
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..img.h - 1 {
            for x in 1..img.w - 1 {
                if suppressed.get(x, y) < 1.0
                    && suppressed.get(x, y) > 0.0
                    && result.get(x, y) < 1.0
                {
                    // Check 4-neighbors for strong edges
                    let has_strong_neighbor = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .any(|(dx, dy)| {
                            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                            nx >= 0 && ny >= 0 && nx < img.w as i64 && ny < img.h as i64
                                && result.get(nx as usize, ny as usize) >= 1.0
                        });
                    if has_strong_neighbor {
                        result.set(x, y, 1.0);
                        changed = true;
                    }
                }
            }
        }
    }

    result
}

/// Compute the central moments of an image.
///
/// Central moments are invariant to translation and describe the shape of the
/// image intensity distribution.
///
/// # Returns
/// * `Vec<f64>` - Central moments ordered as m_pq where p+x=order, q+y=order
///   The index is `p * (max_order + 1) + q` for max_order=2 gives 6 moments:
///   - index 0: m_00 (total mass/area)
///   - index 1: m_10 (centroid x)
///   - index 2: m_01 (centroid y)
///   - index 3: m_20 (variance-like, x²)
///   - index 4: m_11 (covariance, xy)
///   - index 5: m_02 (variance-like, y²)
pub fn central_moments(img: &Image, order: usize) -> Vec<f64> {
    let mut moments = vec![0.0; (order + 1) * (order + 1)];

    let total_mass = img.data.iter().sum::<f64>();
    if total_mass == 0.0 {
        return moments;
    }

    // Compute centroid
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for (i, &v) in img.data.iter().enumerate() {
        let x = (i % img.w) as f64;
        let y = (i / img.w) as f64;
        sum_x += v * x;
        sum_y += v * y;
    }
    let cx = sum_x / total_mass; // centroid x
    let cy = sum_y / total_mass; // centroid y

    // Compute central moments
    for (i, &v) in img.data.iter().enumerate() {
        let x = (i % img.w) as f64 - cx;
        let y = (i / img.w) as f64 - cy;
        let mut idx = 0;
        for p in 0..=order {
            for q in 0..=order - p {
                let _moment = (p as f64 * x + q as f64 * y).powf(p as f64 + q as f64);
                // Actually let us compute v * x^p * y^q
                let moment_val = v * x.powi(p as i32) * y.powi(q as i32);
                moments[idx] += moment_val;
                idx += 1;
            }
        }
    }

    moments
}

/// Compute bounding box of foreground pixels (values > 0.5).
///
/// # Returns
/// * `Option<(usize, usize, usize, usize)>` - (min_x, min_y, max_x, max_y) or None if no foreground
pub fn bounding_box(img: &Image) -> Option<(usize, usize, usize, usize)> {
    let mut min_x: Option<usize> = None;
    let mut min_y: Option<usize> = None;
    let mut max_x: Option<usize> = None;
    let mut max_y: Option<usize> = None;

    for (i, &v) in img.data.iter().enumerate() {
        if v > 0.5 {
            let x = i % img.w;
            let y = i / img.w;
            if min_x.is_none() || x < min_x.unwrap() { min_x = Some(x); }
            if min_y.is_none() || y < min_y.unwrap() { min_y = Some(y); }
            if max_x.is_none() || x > max_x.unwrap() { max_x = Some(x); }
            if max_y.is_none() || y > max_y.unwrap() { max_y = Some(y); }
        }
    }

    match (min_x, min_y, max_x, max_y) {
        (Some(_), Some(_), Some(_), Some(_)) => {
            Some((min_x.unwrap(), min_y.unwrap(), max_x.unwrap(), max_y.unwrap()))
        }
        _ => None,
    }
}

/// Compute the aspect ratio of the foreground bounding box.
///
/// # Returns
/// * `f64` - Aspect ratio (width / height), or 0.0 if no foreground pixels
pub fn aspect_ratio(img: &Image) -> f64 {
    if let Some((min_x, min_y, max_x, max_y)) = bounding_box(img) {
        let width = max_x - min_x;
        let height = max_y - min_y;
        if height > 0 {
            width as f64 / height as f64
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Histogram equalization: redistributes pixel intensities to flatten the
/// cumulative distribution, increasing contrast.
///
/// Equivalent to `cv2.equalizeHist(img)`. Uses a 256-bin cumulative
/// distribution over the `[0, 1]` range; output values stay in `[0, 1]`.
pub fn histogram_equalize(img: &Image) -> Image {
    let hist = histogram(img);
    let total = img.data.len() as f64;
    if total == 0.0 {
        return img.clone();
    }
    // Cumulative distribution, scaled so the smallest non-zero bin maps to 0.
    let mut cdf = [0usize; 256];
    let mut acc = 0usize;
    let mut min_nonzero = usize::MAX;
    for (i, &c) in hist.iter().enumerate() {
        acc += c;
        cdf[i] = acc;
        if c > 0 && min_nonzero == usize::MAX {
            min_nonzero = acc;
        }
    }
    let denom = (total - min_nonzero as f64).max(1.0);
    let mut lut = [0.0f64; 256];
    for (i, &c) in cdf.iter().enumerate() {
        lut[i] = ((c as f64 - min_nonzero as f64) / denom).clamp(0.0, 1.0);
    }
    let mut out = Image::new(img.w, img.h);
    for (i, &v) in img.data.iter().enumerate() {
        let bin = ((v * 256.0).floor() as usize).min(255);
        out.data[i] = lut[bin];
    }
    out
}

/// Min-max normalization: linearly rescales pixel values to the range
/// `[alpha, beta]`.
///
/// Equivalent to `cv2.normalize(img, None, alpha, beta, NORM_MINMAX)`. A
/// constant image is set to `alpha`.
pub fn normalize_minmax(img: &Image, alpha: f64, beta: f64) -> Image {
    let (min_v, max_v, _, _) = crate::utils::min_max(img);
    if (max_v - min_v).abs() < 1e-300 {
        return Image::from_data(img.w, img.h, vec![alpha; img.data.len()]);
    }
    let mut out = img.clone();
    for v in &mut out.data {
        *v = alpha + (*v - min_v) * (beta - alpha) / (max_v - min_v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Image;

    #[test]
    fn test_histogram() {
        let mut img = Image::new(4, 4);
        for i in 0..16 {
            img.data[i] = i as f64 / 15.0;
        }
        let h = histogram(&img);
        assert_eq!(h.len(), 256);
        // Bin 0 holds value 0.0 (pixel 0).
        assert_eq!(h[0], 1);
        // Value 1.0 (pixel 15) maps to bin 255.
        assert_eq!(h[255], 1);
        // Values are spread over distinct bins (no duplicates except edges).
        assert_eq!(h.iter().sum::<usize>(), 16);
    }

    #[test]
    fn test_sobel() {
        let img = Image::new(10, 10);
        let (mag, dir) = sobel(&img);
        assert_eq!(mag.w, 10);
        assert_eq!(mag.h, 10);
        assert_eq!(dir.len(), 100);
    }

    #[test]
    fn test_laplacian() {
        let img = Image::new(10, 10);
        let lap = laplacian(&img);
        assert_eq!(lap.w, 10);
        assert_eq!(lap.h, 10);
    }

    #[test]
    fn test_canny() {
        let mut img = Image::new(10, 10);
        // Create a vertical line
        for y in 0..10 {
            for x in 0..10 {
                img.set(x, y, if x >= 5 { 1.0 } else { 0.0 });
            }
        }
        let edges = canny(&img, 0.3, 0.7);
        // Should have some edge pixels
        let sum: f64 = edges.data.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_canny_converges_with_weak_edges() {
        // A gradient field creates weak (sub-threshold) edge pixels; the
        // hysteresis loop must converge quickly instead of spinning forever.
        let (w, h) = (96usize, 96usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, 0.15 + 0.4 * ((x + y) as f64 / (w + h) as f64));
            }
        }
        for y in 10..34 { for x in 10..34 { img.set(x, y, 0.95); } }
        for y in 60..78 { for x in 60..84 { img.set(x, y, 0.55); } }
        let t0 = std::time::Instant::now();
        let edges = canny(&img, 0.15, 0.45);
        assert!(t0.elapsed().as_secs() < 5, "canny did not converge quickly");
        assert!(edges.data.iter().sum::<f64>() > 0.0);
    }

    #[test]
    fn test_central_moments() {
        let mut img = Image::new(4, 4);
        // Set a simple shape
        img.set(1, 1, 1.0);
        img.set(1, 2, 1.0);
        img.set(2, 1, 1.0);
        img.set(2, 2, 1.0);
        let moms = central_moments(&img, 2);
        assert_eq!(moms.len(), 9); // (order+1)*(order+1) = 3*3 = 9
    }

    #[test]
    fn test_bounding_box() {
        let mut img = Image::new(10, 10);
        // Foreground in top-left 3×3
        for y in 0..3 {
            for x in 0..3 {
                img.set(x, y, 1.0);
            }
        }
        let box_ = bounding_box(&img);
        assert!(box_.is_some());
        let (min_x, min_y, max_x, max_y) = box_.unwrap();
        assert_eq!(min_x, 0);
        assert_eq!(min_y, 0);
        assert_eq!(max_x, 2);
        assert_eq!(max_y, 2);
    }

    #[test]
    fn test_bounding_box_no_foreground() {
        let img = Image::new(10, 10);
        let box_ = bounding_box(&img);
        assert!(box_.is_none());
    }

    #[test]
    fn test_aspect_ratio() {
        let mut img = Image::new(10, 10);
        // Square 5×5 in center
        for y in 3..8 {
            for x in 3..8 {
                img.set(x, y, 1.0);
            }
        }
        let ar = aspect_ratio(&img);
        assert!((ar - 1.0).abs() < 1e-10); // square
    }

    #[test]
    fn test_histogram_equalize_spreads() {
        // Dark image: everything in [0.1, 0.2]. Equalization should spread it.
        let mut img = Image::new(4, 4);
        for i in 0..16 {
            img.data[i] = 0.1 + (i % 4) as f64 * 0.025;
        }
        let eq = histogram_equalize(&img);
        assert!(eq.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // The brightest input pixel should map to ~1.0.
        let max_in = img.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_out = eq.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let idx = img.data.iter().position(|&v| v == max_in).unwrap();
        assert!(max_out > 0.9, "max out {max_out}");
        assert!((eq.data[idx] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_minmax() {
        let mut img = Image::new(2, 2);
        img.data = vec![0.0, 0.5, 1.0, 2.0];
        let n = normalize_minmax(&img, 0.0, 1.0);
        assert!((n.data[0] - 0.0).abs() < 1e-12);
        assert!((n.data[1] - 0.25).abs() < 1e-12);
        assert!((n.data[3] - 1.0).abs() < 1e-12);
        // Constant image maps to alpha.
        let flat = Image::from_data(2, 2, vec![0.7; 4]);
        let nf = normalize_minmax(&flat, 0.0, 1.0);
        assert!(nf.data.iter().all(|&v| v == 0.0));
    }
}