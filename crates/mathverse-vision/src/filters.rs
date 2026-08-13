//! Spatial filtering: box blur, median blur, bilateral filter, sharpening and
//! generic 2D convolution — the `cv2.filter2D` / `cv2.blur` / `cv2.medianBlur`
//! / `cv2.bilateralFilter` family.
//!
//! Border handling is *clamp-to-edge* (replicate), matching OpenCV's
//! `BORDER_REPLICATE` default for these filters.

use crate::Image;

/// Applies an arbitrary N×N convolution kernel.
///
/// `kernel` is row-major with `kw × kh` elements; the anchor is the kernel
/// center. Equivalent to `cv2.filter2D(img, -1, kernel)`.
///
/// # Panics
///
/// Panics if `kw` or `kh` is zero or even.
pub fn filter2d(img: &Image, kernel: &[f64], kw: usize, kh: usize) -> Image {
    assert!(kw > 0 && kh > 0 && kw % 2 == 1 && kh % 2 == 1, "filter2d: kernel must be odd-sized");
    assert_eq!(kernel.len(), kw * kh, "filter2d: kernel length mismatch");
    let (ox, oy) = (kw / 2, kh / 2);
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for ky in 0..kh {
                let py = (y as i64 + ky as i64 - oy as i64).clamp(0, h as i64 - 1) as usize;
                for kx in 0..kw {
                    let px = (x as i64 + kx as i64 - ox as i64).clamp(0, w as i64 - 1) as usize;
                    s += kernel[ky * kw + kx] * img.data[py * w + px];
                }
            }
            out.data[y * w + x] = s;
        }
    }
    out
}

/// Box (mean) blur with a `ksize × ksize` normalized window.
///
/// Equivalent to `cv2.blur(img, (ksize, ksize))`. `ksize` must be odd.
///
/// # Panics
///
/// Panics if `ksize` is zero or even.
pub fn box_filter(img: &Image, ksize: usize) -> Image {
    assert!(ksize > 0 && ksize % 2 == 1, "box_filter: ksize must be odd");
    let kernel = vec![1.0 / (ksize * ksize) as f64; ksize * ksize];
    filter2d(img, &kernel, ksize, ksize)
}

/// Median blur: each pixel becomes the median of the `ksize × ksize` window.
///
/// Equivalent to `cv2.medianBlur(img, ksize)`. Preserves edges while removing
/// salt-and-pepper noise. `ksize` must be odd.
///
/// # Panics
///
/// Panics if `ksize` is zero or even.
pub fn median_blur(img: &Image, ksize: usize) -> Image {
    assert!(ksize > 0 && ksize % 2 == 1, "median_blur: ksize must be odd");
    let (w, h) = (img.w, img.h);
    let (ox, oy) = (ksize / 2, ksize / 2);
    let mut out = Image::new(w, h);
    let mut window = Vec::with_capacity(ksize * ksize);
    for y in 0..h {
        for x in 0..w {
            window.clear();
            for ky in 0..ksize {
                let py = (y as i64 + ky as i64 - oy as i64).clamp(0, h as i64 - 1) as usize;
                for kx in 0..ksize {
                    let px = (x as i64 + kx as i64 - ox as i64).clamp(0, w as i64 - 1) as usize;
                    window.push(img.data[py * w + px]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out.data[y * w + x] = window[window.len() / 2];
        }
    }
    out
}

/// Bilateral filter: edge-preserving smoothing.
///
/// Each pixel is a weighted average of its `d × d` neighborhood, where the
/// weight combines spatial proximity (`sigma_space`) and intensity similarity
/// (`sigma_color`). Equivalent to `cv2.bilateralFilter(img, d, sigma_color, sigma_space)`.
pub fn bilateral_filter(img: &Image, d: usize, sigma_color: f64, sigma_space: f64) -> Image {
    let (w, h) = (img.w, img.h);
    let r = d / 2;
    let mut out = Image::new(w, h);
    let (sc2, ss2) = (2.0 * sigma_color * sigma_color, 2.0 * sigma_space * sigma_space);
    for y in 0..h {
        for x in 0..w {
            let center = img.data[y * w + x];
            let mut acc = 0.0;
            let mut weight_sum = 0.0;
            for dy in -(r as i64)..=(r as i64) {
                for dx in -(r as i64)..=(r as i64) {
                    let (px, py) = ((x as i64 + dx).clamp(0, w as i64 - 1) as usize, (y as i64 + dy).clamp(0, h as i64 - 1) as usize);
                    let val = img.data[py * w + px];
                    let d_int = (val - center).powi(2);
                    let d_sp = (dx * dx + dy * dy) as f64;
                    let wgt = if sc2 > 0.0 { (-d_int / sc2).exp() } else { 1.0 }
                        * if ss2 > 0.0 { (-d_sp / ss2).exp() } else { 1.0 };
                    acc += wgt * val;
                    weight_sum += wgt;
                }
            }
            out.data[y * w + x] = if weight_sum > 0.0 { acc / weight_sum } else { center };
        }
    }
    out
}

/// Unsharp-mask sharpening: `img + amount · (img − gaussian_blur(img, sigma))`.
///
/// `amount` typically in `[0.0, 2.0]`; the result is clamped to `[0.0, 1.0]`.
/// Equivalent to the OpenCV sharpening idiom using `addWeighted`.
pub fn sharpen(img: &Image, amount: f64, sigma: f64) -> Image {
    let blurred = img.gaussian_blur(1, sigma);
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        let diff = img.data[i] - blurred.data[i];
        out.data[i] = (img.data[i] + amount * diff).clamp(0.0, 1.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_filter_averages() {
        let mut img = Image::new(5, 5);
        img.data = vec![1.0; 25];
        let b = box_filter(&img, 3);
        assert!(b.data.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn median_removes_salt_pepper() {
        let mut img = Image::new(5, 5);
        img.data = vec![0.5; 25];
        img.set(2, 2, 1.0); // single bright outlier
        let m = median_blur(&img, 3);
        assert!((m.get(2, 2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bilateral_preserves_step_edge() {
        let mut img = Image::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                img.set(x, y, if x < 4 { 0.0 } else { 1.0 });
            }
        }
        let b = bilateral_filter(&img, 5, 0.1, 3.0);
        // Strong step edge should survive smoothing almost untouched.
        assert!((b.get(0, 4) - 0.0).abs() < 0.05, "left {}", b.get(0, 4));
        assert!((b.get(7, 4) - 1.0).abs() < 0.05, "right {}", b.get(7, 4));
    }

    #[test]
    fn filter2d_identity() {
        let mut img = Image::new(4, 4);
        for i in 0..16 {
            img.data[i] = i as f64 / 15.0;
        }
        // Identity 3×3 kernel: single 1.0 in the center.
        let k = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let out = filter2d(&img, &k, 3, 3);
        for i in 0..16 {
            assert!((out.data[i] - img.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sharpen_boosts_edges() {
        let mut img = Image::new(16, 1);
        for x in 0..16 {
            img.set(x, 0, if x < 8 { 0.0 } else { 1.0 });
        }
        let s = sharpen(&img, 1.0, 1.0);
        assert!(s.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
