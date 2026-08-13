#! Image transformations: resize, rotate, affine.

use crate::Image;

/// Resizes an image to the specified width and height using nearest-neighbor interpolation.
///
/// # Arguments
///
/// * `img` - Input image
/// * `new_w` - New width
/// * `new_h` - New height
///
/// # Returns
///
/// A new `Image` with the specified dimensions.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, transform::resize};
///
/// let mut img = Image::new(4, 4);
/// for y in 0..4 { for x in 0..4 { img.set(x, y, (x + y) as f64); } }
/// let small = resize(&img, 2, 2);
/// assert_eq!(small.w, 2);
/// assert_eq!(small.h, 2);
/// ```
pub fn resize(img: &Image, new_w: usize, new_h: usize) -> Image {
    let mut out = Image::new(new_w, new_h);
    
    if new_w == 0 || new_h == 0 {
        return out;
    }
    
    let scale_x = img.w as f64 / new_w as f64;
    let scale_y = img.h as f64 / new_h as f64;
    
    for y in 0..new_h {
        for x in 0..new_w {
            // Map output pixel to input coordinates (top-left mapping)
            let src_x = (x as f64 * scale_x).min((img.w - 1) as f64);
            let src_y = (y as f64 * scale_y).min((img.h - 1) as f64);
            
            let sx = src_x as usize;
            let sy = src_y as usize;
            
            out.data[y * new_w + x] = img.data[sy * img.w + sx];
        }
    }
    out
}

/// Rotates an image by the specified angle (in degrees) around its center.
///
/// Uses nearest-neighbor interpolation. The output image size is computed to
/// contain the entire rotated input image (bounding box).
///
/// # Arguments
///
/// * `img` - Input image
/// * `angle_deg` - Rotation angle in degrees (counter-clockwise)
///
/// # Returns
///
/// A new `Image` with the rotated content.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, transform::rotate};
///
/// let img = Image::new(4, 4);
/// let rotated = rotate(&img, 90.0);
/// assert_eq!(rotated.w, img.h);
/// assert_eq!(rotated.h, img.w);
/// ```
pub fn rotate(img: &Image, angle_deg: f64) -> Image {
    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    
    // Compute output bounds (bounding box of rotated rectangle)
    let w = img.w;
    let h = img.h;
    
    // Four corners of input image
    let corners = [
        (0.0, 0.0),
        (w as f64, 0.0),
        (0.0, h as f64),
        (w as f64, h as f64),
    ];
    
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    
    for (cx, cy) in &corners {
        let rx = cx * cos_a - cy * sin_a;
        let ry = cx * sin_a + cy * cos_a;
        min_x = min_x.min(rx);
        max_x = max_x.max(rx);
        min_y = min_y.min(ry);
        max_y = max_y.max(ry);
    }
    
    let out_w = ((max_x - min_x).ceil()) as usize;
    let out_h = ((max_y - min_y).ceil()) as usize;
    
    // Adjust origin so that min_x, min_y become 0
    let mut out = Image::new(out_w, out_h);
    
    // Center of original image
    let cx = (w - 1) as f64 / 2.0;
    let cy = (h - 1) as f64 / 2.0;
    
    for y in 0..out_h {
        for x in 0..out_w {
            // Map output pixel back to input space (inverse rotation)
            // Output (x, y) corresponds to input point:
            //   x_in = cx + (x - out_w/2) * cos - (y - out_h/2) * sin
            //   y_in = cy + (x - out_w/2) * sin + (y - out_h/2) * cos
            let dx = x as f64 - (out_w - 1) as f64 / 2.0;
            let dy = y as f64 - (out_h - 1) as f64 / 2.0;
            
            let src_x = cx + dx * cos_a - dy * sin_a;
            let src_y = cy + dx * sin_a + dy * cos_a;
            
            // Clip to input image bounds
            let sx = src_x.max(0.0).min((w - 1) as f64);
            let sy = src_y.max(0.0).min((h - 1) as f64);
            
            let ix = sx as usize;
            let iy = sy as usize;
            out.data[y * out_w + x] = img.data[iy * w + ix];
        }
    }
    out
}

/// Applies an affine transformation defined by a 2x3 matrix.
///
/// The transformation matrix is `[a, b, tx; c, d, ty]` where:
/// ```text
///   x' = a * x + b * y + tx
///   y' = c * x + d * y + ty
/// ```
///
/// Uses nearest-neighbor interpolation.
///
/// # Arguments
///
/// * `img` - Input image
/// * `a, b, tx` - Affine matrix coefficients for x transformation
/// * `c, d, ty` - Affine matrix coefficients for y transformation
///
/// # Returns
///
/// A new `Image` with the transformed content.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, transform::affine};
///
/// let img = Image::new(4, 4);
/// // Identity transformation (a=1, b=0, tx=0, c=0, d=1, ty=0)
/// let out = affine(&img, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
/// assert_eq!(out.w, img.w);
/// assert_eq!(out.h, img.h);
/// ```
pub fn affine(img: &Image, a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Image {
    let w = img.w;
    let h = img.h;
    let mut out = Image::new(w, h);
    
    for y in 0..h {
        for x in 0..w {
            // Inverse mapping: compute source position
            // [x']   [a b tx] [x]   [x]
            // [y'] = [c d ty] [y] = [y]
            // Actually for inverse, we solve for source given destination
            // Using the matrix directly on destination coordinates
            
            let src_x = (a * x as f64 + b * y as f64 + tx).max(0.0).min((w - 1) as f64);
            let src_y = (c * x as f64 + d * y as f64 + ty).max(0.0).min((h - 1) as f64);
            
            let sx = src_x as usize;
            let sy = src_y as usize;
            out.data[y * w + x] = img.data[sy * w + sx];
        }
    }
    out
}

/// Flips an image horizontally, vertically, or both.
///
/// `flip_code` follows OpenCV: `0` flips vertically (around the x-axis),
/// `1` flips horizontally (around the y-axis), and `-1` flips both axes.
///
/// # Panics
///
/// Panics if `flip_code` is not `-1`, `0`, or `1`.
pub fn flip(img: &Image, flip_code: i32) -> Image {
    assert!((-1..=1).contains(&flip_code), "flip: code must be -1, 0 or 1");
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = if flip_code == 0 { x } else { w - 1 - x };
            let sy = if flip_code == 1 { y } else { h - 1 - y };
            out.data[y * w + x] = img.data[sy * w + sx];
        }
    }
    out
}

/// Transposes an image: pixel `(x, y)` moves to `(y, x)`.
///
/// The output is `h × w`. Equivalent to `cv2.transpose`. Applying it twice
/// recovers the original image (without flipping).
pub fn transpose(img: &Image) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(h, w);
    for y in 0..h {
        for x in 0..w {
            out.data[x * h + y] = img.data[y * w + x];
        }
    }
    out
}

/// Crops a rectangular region of interest from an image.
///
/// Returns `None` if the region extends beyond the image bounds. The returned
/// image has size `w × h`. Equivalent to OpenCV's ROI slicing, e.g.
/// `img[y..y+h, x..x+w]`.
pub fn crop(img: &Image, x: usize, y: usize, w: usize, h: usize) -> Option<Image> {
    if x + w > img.w || y + h > img.h || w == 0 || h == 0 {
        return None;
    }
    let mut out = Image::new(w, h);
    for j in 0..h {
        let src_row = (y + j) * img.w + x;
        let dst_row = j * w;
        out.data[dst_row..dst_row + w].copy_from_slice(&img.data[src_row..src_row + w]);
    }
    Some(out)
}

/// Computes a 2×3 rotation (and scale) matrix about a center point.
///
/// Equivalent to `cv2.getRotationMatrix2D(center, angle, scale)`. The angle is
/// in degrees, counter-clockwise. The matrix is:
/// ```text
/// [ α  β  (1-α)·cx - β·cy ]
/// [-β  α   β·cx + (1-α)·cy ]
/// ```
/// with `α = scale·cos(θ)`, `β = scale·sin(θ)`.
pub fn get_rotation_matrix_2d(center: (f64, f64), angle_deg: f64, scale: f64) -> [[f64; 3]; 2] {
    let a = scale * angle_deg.to_radians().cos();
    let b = scale * angle_deg.to_radians().sin();
    let (cx, cy) = center;
    [
        [a, b, (1.0 - a) * cx - b * cy],
        [-b, a, b * cx + (1.0 - a) * cy],
    ]
}

/// Applies an affine transform (2×3 matrix) to an image using bilinear
/// interpolation, producing an output of the same size.
///
/// Equivalent to `cv2.warpAffine(img, M, (w, h), INTER_LINEAR)`. Pixels mapped
/// outside the source image become `0.0`.
pub fn warp_affine(img: &Image, m: &[[f64; 3]; 2]) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = m[0][0] * x as f64 + m[0][1] * y as f64 + m[0][2];
            let sy = m[1][0] * x as f64 + m[1][1] * y as f64 + m[1][2];
            out.data[y * w + x] = sample_bilinear(img, sx, sy);
        }
    }
    out
}

/// Applies a perspective transform (3×3 homography) to an image using bilinear
/// interpolation, producing an output of the same size.
///
/// Equivalent to `cv2.warpPerspective(img, H, (w, h), INTER_LINEAR)`. Pixels
/// mapped outside the source image become `0.0`. The matrix is row-major
/// `[h00, h01, h02, h10, h11, h12, h20, h21, h22]` (see [`crate::homography::Homography`]).
pub fn warp_perspective(img: &Image, h: &[f64; 9]) -> Image {
    let (w, hgt) = (img.w, img.h);
    let mut out = Image::new(w, hgt);
    for y in 0..hgt {
        for x in 0..w {
            let denom = h[6] * x as f64 + h[7] * y as f64 + h[8];
            if denom.abs() < 1e-12 {
                out.data[y * w + x] = 0.0;
                continue;
            }
            let sx = (h[0] * x as f64 + h[1] * y as f64 + h[2]) / denom;
            let sy = (h[3] * x as f64 + h[4] * y as f64 + h[5]) / denom;
            out.data[y * w + x] = sample_bilinear(img, sx, sy);
        }
    }
    out
}

/// Downsamples an image by a factor of two using a 5-tap Gaussian kernel
/// `[1, 4, 6, 4, 1] / 16` (same kernel as OpenCV's `pyrDown`).
///
/// The output is `⌈w/2⌉ × ⌈h/2⌉`.
pub fn pyr_down(img: &Image) -> Image {
    let blurred = gaussian_5x5(img);
    let (w, h) = (img.w, img.h);
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Image::new(nw, nh);
    for y in 0..nh {
        for x in 0..nw {
            out.data[y * nw + x] = blurred.data[(2 * y) * w + 2 * x];
        }
    }
    out
}

/// Upsamples an image by a factor of two: inserts zero rows/columns, then
/// applies the 5-tap Gaussian kernel scaled by 4 (OpenCV's `pyrUp`).
///
/// The output is `2·w × 2·h`.
pub fn pyr_up(img: &Image) -> Image {
    let (w, h) = (img.w, img.h);
    let (nw, nh) = (2 * w, 2 * h);
    // Insert zeros.
    let mut up = Image::new(nw, nh);
    for y in 0..h {
        for x in 0..w {
            up.data[(2 * y) * nw + 2 * x] = img.data[y * w + x];
        }
    }
    // Convolve with 4× the 5-tap kernel (separable: row then column).
    let k = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut tmp = Image::new(nw, nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut s = 0.0;
            for i in 0..5usize {
                let px = x as i64 + i as i64 - 2;
                if (0..nw as i64).contains(&px) {
                    s += k[i] * up.data[y * nw + px as usize];
                }
            }
            tmp.data[y * nw + x] = s / 4.0;
        }
    }
    let mut out = Image::new(nw, nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut s = 0.0;
            for i in 0..5usize {
                let py = y as i64 + i as i64 - 2;
                if (0..nh as i64).contains(&py) {
                    s += k[i] * tmp.data[py as usize * nw + x];
                }
            }
            out.data[y * nw + x] = s / 4.0;
        }
    }
    out
}

fn gaussian_5x5(img: &Image) -> Image {
    let k = [1.0, 4.0, 6.0, 4.0, 1.0];
    let (w, h) = (img.w, img.h);
    let mut tmp = Image::new(w, h);
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for i in 0..5usize {
                let px = x as i64 + i as i64 - 2;
                if (0..w as i64).contains(&px) {
                    s += k[i] * img.data[y * w + px as usize];
                }
            }
            tmp.data[y * w + x] = s / 16.0;
        }
    }
    // Vertical pass.
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for i in 0..5usize {
                let py = y as i64 + i as i64 - 2;
                if (0..h as i64).contains(&py) {
                    s += k[i] * tmp.data[py as usize * w + x];
                }
            }
            out.data[y * w + x] = s / 16.0;
        }
    }
    out
}

/// Bilinear sampling with clamping to the image border; samples far outside
/// the image return `0.0`.
fn sample_bilinear(img: &Image, x: f64, y: f64) -> f64 {
    if x < -1.0 || y < -1.0 || x > img.w as f64 || y > img.h as f64 {
        return 0.0;
    }
    let (w, h) = (img.w, img.h);
    let x0 = x.floor().clamp(0.0, (w - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (h - 1) as f64) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = (x - x0 as f64).clamp(0.0, 1.0);
    let dy = (y - y0 as f64).clamp(0.0, 1.0);
    let v00 = img.data[y0 * w + x0];
    let v01 = img.data[y1 * w + x0];
    let v10 = img.data[y0 * w + x1];
    let v11 = img.data[y1 * w + x1];
    v00 * (1.0 - dx) * (1.0 - dy) + v10 * dx * (1.0 - dy) + v01 * (1.0 - dx) * dy + v11 * dx * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_identity() {
        let mut img = Image::new(4, 4);
        for y in 0..4 { for x in 0..4 { img.set(x, y, (x + y) as f64); } }
        let small = resize(&img, 2, 2);
        // Nearest neighbor: each 2x2 block becomes 1 pixel
        assert_eq!(small.data[0], 0.0);   // (0,0) block -> pixel (0,0)
        assert_eq!(small.data[1], 2.0);   // (0,1) and (1,0) block -> pixel (0,0) had value 0... 
    }

    #[test]
    fn affine_identity() {
        let mut img = Image::new(4, 4);
        for y in 0..4 { for x in 0..4 { img.set(x, y, 1.0); } }
        let out = affine(&img, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        // Should be same size with same values
        assert_eq!(out.w, 4);
        assert_eq!(out.h, 4);
        assert_eq!(out.data.iter().all(|&v| v == 1.0), true);
    }

    #[test]
    fn flip_horizontal() {
        let mut img = Image::new(3, 1);
        img.data = vec![0.1, 0.2, 0.3];
        let f = flip(&img, 1);
        assert!((f.data[0] - 0.3).abs() < 1e-12);
        assert!((f.data[2] - 0.1).abs() < 1e-12);
        let ff = flip(&f, 1);
        for i in 0..3 {
            assert!((ff.data[i] - img.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn transpose_twice_is_identity() {
        let mut img = Image::new(4, 3);
        for i in 0..img.data.len() {
            img.data[i] = i as f64 / 11.0;
        }
        let t = transpose(&img);
        assert_eq!((t.w, t.h), (3, 4));
        let tt = transpose(&t);
        assert_eq!((tt.w, tt.h), (4, 3));
        for i in 0..img.data.len() {
            assert!((tt.data[i] - img.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn crop_region() {
        let mut img = Image::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                img.set(x, y, (x + 10 * y) as f64);
            }
        }
        let c = crop(&img, 2, 3, 4, 2).unwrap();
        assert_eq!((c.w, c.h), (4, 2));
        assert!((c.get(0, 0) - img.get(2, 3)).abs() < 1e-12);
        assert!((c.get(3, 1) - img.get(5, 4)).abs() < 1e-12);
        assert!(crop(&img, 6, 6, 4, 4).is_none());
    }

    #[test]
    fn warp_affine_identity() {
        let mut img = Image::new(6, 6);
        for i in 0..36 {
            img.data[i] = i as f64 / 35.0;
        }
        let m = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = warp_affine(&img, &m);
        for i in 0..36 {
            assert!((out.data[i] - img.data[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn warp_perspective_identity() {
        let mut img = Image::new(6, 6);
        for i in 0..36 {
            img.data[i] = i as f64 / 35.0;
        }
        let h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let out = warp_perspective(&img, &h);
        for i in 0..36 {
            assert!((out.data[i] - img.data[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn pyr_roundtrip() {
        let mut img = Image::new(8, 8);
        for i in 0..64 {
            img.data[i] = i as f64 / 63.0;
        }
        let small = pyr_down(&img);
        assert_eq!((small.w, small.h), (4, 4));
        let up = pyr_up(&small);
        assert_eq!((up.w, up.h), (8, 8));
    }

    #[test]
    fn rotation_matrix_90_deg() {
        // OpenCV's getRotationMatrix2D convention: with image coordinates
        // (y down), a positive angle rotates counter-clockwise visually, so
        // (1, 0) maps to (0, -1) under 90°.
        let m = get_rotation_matrix_2d((0.0, 0.0), 90.0, 1.0);
        let x = m[0][0] * 1.0 + m[0][1] * 0.0 + m[0][2];
        let y = m[1][0] * 1.0 + m[1][1] * 0.0 + m[1][2];
        assert!(x.abs() < 1e-9, "x {x}");
        assert!((y + 1.0).abs() < 1e-9, "y {y}");
        // Scale 2 doubles the displacement.
        let m2 = get_rotation_matrix_2d((0.0, 0.0), 90.0, 2.0);
        let x2 = m2[0][0] * 1.0 + m2[0][1] * 0.0 + m2[0][2];
        let y2 = m2[1][0] * 1.0 + m2[1][1] * 0.0 + m2[1][2];
        assert!(x2.abs() < 1e-9, "x2 {x2}");
        assert!((y2 + 2.0).abs() < 1e-9, "y2 {y2}");
    }
}