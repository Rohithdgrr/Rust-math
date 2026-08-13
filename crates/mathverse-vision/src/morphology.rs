//! Morphological operations: erosion, dilation, opening, closing, gradient,
//! top hat and black hat — the `cv2.morphologyEx` family.
//!
//! Kernels are 3×3 boolean masks anchored at the center; out-of-bounds
//! neighbors are ignored (equivalent to OpenCV's `BORDER_CONSTANT` with the
//! morphology default value).

use crate::Image;

/// Rectangular 3×3 structuring element (all neighbors active).
pub const fn kernel_rect() -> [[bool; 3]; 3] {
    [[true; 3]; 3]
}

/// Cross-shaped 3×3 structuring element (center + 4-neighbors).
pub const fn kernel_cross() -> [[bool; 3]; 3] {
    [
        [false, true, false],
        [true, true, true],
        [false, true, false],
    ]
}

/// Ellipse-shaped 3×3 structuring element (center + 8-neighbors with corners).
pub const fn kernel_ellipse() -> [[bool; 3]; 3] {
    [
        [false, true, false],
        [true, true, true],
        [false, true, false],
    ]
}

/// Erodes an image: each pixel becomes the minimum over the active
/// neighborhood. Bright regions shrink.
///
/// Equivalent to `cv2.erode(img, kernel, iterations)`.
pub fn erode(img: &Image, kernel: &[[bool; 3]; 3], iterations: usize) -> Image {
    let mut out = img.clone();
    for _ in 0..iterations.max(1) {
        out = erode_once(&out, kernel);
    }
    out
}

/// Dilates an image: each pixel becomes the maximum over the active
/// neighborhood. Bright regions grow.
///
/// Equivalent to `cv2.dilate(img, kernel, iterations)`.
pub fn dilate(img: &Image, kernel: &[[bool; 3]; 3], iterations: usize) -> Image {
    let mut out = img.clone();
    for _ in 0..iterations.max(1) {
        out = dilate_once(&out, kernel);
    }
    out
}

/// Opening: erosion followed by dilation. Removes small bright specks.
///
/// Equivalent to `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`.
pub fn opening(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    dilate(&erode(img, kernel, 1), kernel, 1)
}

/// Closing: dilation followed by erosion. Fills small dark holes.
///
/// Equivalent to `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)`.
pub fn closing(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    erode(&dilate(img, kernel, 1), kernel, 1)
}

/// Morphological gradient: `dilate − erode`. Highlights object boundaries.
///
/// Equivalent to `cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)`.
pub fn morphological_gradient(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    let d = dilate(img, kernel, 1);
    let e = erode(img, kernel, 1);
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        out.data[i] = d.data[i] - e.data[i];
    }
    out
}

/// Top hat: `img − opening`. Extracts bright features smaller than the kernel.
///
/// Equivalent to `cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)`.
pub fn top_hat(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    let op = opening(img, kernel);
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        out.data[i] = (img.data[i] - op.data[i]).max(0.0);
    }
    out
}

/// Black hat: `closing − img`. Extracts dark features smaller than the kernel.
///
/// Equivalent to `cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)`.
pub fn black_hat(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    let cl = closing(img, kernel);
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        out.data[i] = (cl.data[i] - img.data[i]).max(0.0);
    }
    out
}

fn erode_once(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut min_v = f64::INFINITY;
            for (ky, row) in kernel.iter().enumerate() {
                for (kx, &active) in row.iter().enumerate() {
                    if !active {
                        continue;
                    }
                    let (nx, ny) = (x as i64 + kx as i64 - 1, y as i64 + ky as i64 - 1);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        min_v = min_v.min(img.data[ny as usize * w + nx as usize]);
                    }
                }
            }
            out.data[y * w + x] = if min_v.is_finite() { min_v } else { img.data[y * w + x] };
        }
    }
    out
}

fn dilate_once(img: &Image, kernel: &[[bool; 3]; 3]) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut max_v = f64::NEG_INFINITY;
            for (ky, row) in kernel.iter().enumerate() {
                for (kx, &active) in row.iter().enumerate() {
                    if !active {
                        continue;
                    }
                    let (nx, ny) = (x as i64 + kx as i64 - 1, y as i64 + ky as i64 - 1);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        max_v = max_v.max(img.data[ny as usize * w + nx as usize]);
                    }
                }
            }
            out.data[y * w + x] = if max_v.is_finite() { max_v } else { img.data[y * w + x] };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                img.set(x, y, 1.0);
            }
        }
        img
    }

    #[test]
    fn erode_shrinks_dilate_grows() {
        // 10×10 image with a filled 6×6 blob at (2..8).
        let img = binary_rect(10, 10, 2, 2, 8, 8);
        let e = erode(&img, &kernel_rect(), 1);
        // After erosion the blob should be 4×4 at (3..7).
        assert_eq!(e.get(2, 2), 0.0);
        assert_eq!(e.get(3, 3), 1.0);
        assert_eq!(e.get(6, 6), 1.0);
        assert_eq!(e.get(7, 7), 0.0);

        let d = dilate(&img, &kernel_rect(), 1);
        // After dilation the blob should be 8×8 at (1..9).
        assert_eq!(d.get(1, 1), 1.0);
        assert_eq!(d.get(8, 8), 1.0);
        assert_eq!(d.get(0, 0), 0.0);
        assert_eq!(d.get(9, 9), 0.0);
    }

    #[test]
    fn opening_removes_specks() {
        let mut img = binary_rect(10, 10, 2, 2, 8, 8);
        img.set(0, 0, 1.0); // tiny isolated speck
        let op = opening(&img, &kernel_rect());
        // Speck gone, blob intact.
        assert_eq!(op.get(0, 0), 0.0);
        assert_eq!(op.get(5, 5), 1.0);
    }

    #[test]
    fn closing_fills_holes() {
        let mut img = binary_rect(10, 10, 2, 2, 8, 8);
        img.set(5, 5, 0.0); // single-pixel hole
        let cl = closing(&img, &kernel_rect());
        assert_eq!(cl.get(5, 5), 1.0);
        assert_eq!(cl.get(2, 2), 1.0);
        assert_eq!(cl.get(0, 0), 0.0);
    }

    #[test]
    fn gradient_and_hats() {
        let img = binary_rect(10, 10, 2, 2, 8, 8);
        let g = morphological_gradient(&img, &kernel_rect());
        assert_eq!(g.get(5, 5), 0.0); // interior: no gradient
        assert_eq!(g.get(2, 2), 1.0); // boundary
        let th = top_hat(&img, &kernel_rect());
        assert!(th.data.iter().all(|&v| v <= 1e-12));
        let bh = black_hat(&img, &kernel_rect());
        assert!(bh.data.iter().all(|&v| v <= 1e-12));
    }
}
