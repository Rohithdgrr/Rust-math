//! # Filter Kernel Bank
//!
//! Pre-computed filter kernels for common computer vision operations.
//! All kernels are 3×3 unless otherwise noted, and are normalized (sum = 1.0)
//! except derivative kernels (Sobel, Laplacian) which preserve sign information.

use crate::Image;

/// Gaussian blur kernel with the given sigma.
/// Kernel size is always 3×3 for the compact version.
/// For larger sigma, use `gaussian_kernel_5x5` or `gaussian_kernel_7x7`.
///
/// # Arguments
/// * `sigma` - Standard deviation of the Gaussian distribution
///
/// # Returns
/// * `[[f64; 3]; 3]` - 3×3 Gaussian kernel
pub fn gaussian_kernel(sigma: f64) -> [[f64; 3]; 3] {
    let mut kernel = [[0.0f64; 3]; 3];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for ky in 0..3 {
        for kx in 0..3 {
            let dx = kx as f64 - 1.0;
            let dy = ky as f64 - 1.0;
            let v = (-(dx * dx + dy * dy) / s2).exp();
            kernel[ky][kx] = v;
            sum += v;
        }
    }
    for row in &mut kernel {
        for v in row { *v /= sum; }
    }
    kernel
}

/// 5×5 Gaussian blur kernel with the given sigma.
pub fn gaussian_kernel_5x5(sigma: f64) -> [[f64; 5]; 5] {
    let mut kernel = [[0.0f64; 5]; 5];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for ky in 0..5 {
        for kx in 0..5 {
            let dx = kx as f64 - 2.0;
            let dy = ky as f64 - 2.0;
            let v = (-(dx * dx + dy * dy) / s2).exp();
            kernel[ky][kx] = v;
            sum += v;
        }
    }
    for row in &mut kernel {
        for v in row { *v /= sum; }
    }
    kernel
}

/// 7×7 Gaussian blur kernel with the given sigma.
pub fn gaussian_kernel_7x7(sigma: f64) -> [[f64; 7]; 7] {
    let mut kernel = [[0.0f64; 7]; 7];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for ky in 0..7 {
        for kx in 0..7 {
            let dx = kx as f64 - 3.0;
            let dy = ky as f64 - 3.0;
            let v = (-(dx * dx + dy * dy) / s2).exp();
            kernel[ky][kx] = v;
            sum += v;
        }
    }
    for row in &mut kernel {
        for v in row { *v /= sum; }
    }
    kernel
}

/// Sobel operator kernel for horizontal gradient (Gx).
pub const SOBEL_GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];

/// Sobel operator kernel for vertical gradient (Gy).
pub const SOBEL_GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

/// Laplacian kernel (second derivative).
pub const LAPLACIAN: [f64; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

/// Laplacian (combined) kernel - positive version.
pub const LAPLACIAN_POS: [f64; 9] = [1.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 1.0];

/// Prewitt operator kernel for horizontal gradient.
pub const PREWITT_GX: [f64; 9] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];

/// Prewitt operator kernel for vertical gradient.
pub const PREWITT_GY: [f64; 9] = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// Apply Gaussian blur to an Image using the kernel bank.
///
/// # Arguments
/// * `img` - Input Image (f64 [0.0, 1.0])
/// * `sigma` - Standard deviation for the Gaussian kernel
///
/// # Returns
/// * `Ok(Image)` - Blurred image
/// * `Err(String)` - On failure
pub fn apply_gaussian_blur(img: &Image, sigma: f64) -> Result<Image, String> {
    let kernel = gaussian_kernel(sigma);
    // Flatten 3×3 kernel into [f64; 9]
    let flat: [f64; 9] = [
        kernel[0][0], kernel[0][1], kernel[0][2],
        kernel[1][0], kernel[1][1], kernel[1][2],
        kernel[2][0], kernel[2][1], kernel[2][2],
    ];
    Ok(img.convolve3(&flat))
}

/// Apply Sobel edge detection to an Image.
/// Returns the gradient magnitude image.
///
/// # Returns
/// * `Ok(Image)` - Gradient magnitude image
/// * `Err(String)` - On failure
pub fn apply_sobel(img: &Image) -> Result<Image, String> {
    let mut mag = Image::new(img.w, img.h);
    let (gx_data, gy_data) = apply_sobel_components(img)?;
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let gx = gx_data.get(x, y);
            let gy = gy_data.get(x, y);
            let magnitude = (gx * gx + gy * gy).sqrt();
            mag.set(x, y, magnitude);
        }
    }
    Ok(mag)
}

/// Apply Sobel edge detection, returning both Gx and Gy components.
///
/// # Returns
/// * `Ok((Image, Image))` - (Gx gradient, Gy gradient)
/// * `Err(String)` - On failure
pub fn apply_sobel_components(img: &Image) -> Result<(Image, Image), String> {
    let mut gx = Image::new(img.w, img.h);
    let mut gy = Image::new(img.w, img.h);

    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            // Gx convolution
            let mut s = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                s += SOBEL_GX[ky * 3] * src[0] + SOBEL_GX[ky * 3 + 1] * src[1] + SOBEL_GX[ky * 3 + 2] * src[2];
            }
            gx.data[y * img.w + x] = s;

            // Gy convolution
            let mut s = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                s += SOBEL_GY[ky * 3] * src[0] + SOBEL_GY[ky * 3 + 1] * src[1] + SOBEL_GY[ky * 3 + 2] * src[2];
            }
            gy.data[y * img.w + x] = s;
        }
    }
    Ok((gx, gy))
}

/// Apply Laplacian edge detection to an Image.
/// Uses the standard 3×3 Laplacian kernel.
///
/// # Returns
/// * `Ok(Image)` - Laplacian response image
/// * `Err(String)` - On failure
pub fn apply_laplacian(img: &Image) -> Result<Image, String> {
    let mut laplacian = Image::new(img.w, img.h);
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let mut s = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                s += LAPLACIAN[ky * 3] * src[0] + LAPLACIAN[ky * 3 + 1] * src[1] + LAPLACIAN[ky * 3 + 2] * src[2];
            }
            laplacian.data[y * img.w + x] = s;
        }
    }
    Ok(laplacian)
}

/// Apply Scharr operator for more accurate gradient computation.
/// Scharr provides better rotation invariance than Sobel.
///
/// # Returns
/// * `Ok(Image)` - Gradient magnitude image
/// * `Err(String)` - On failure
pub fn apply_scharr(img: &Image) -> Result<Image, String> {
    let mut mag = Image::new(img.w, img.h);
    let (gx_data, gy_data) = apply_scharr_components(img)?;
    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let gx = gx_data.get(x, y);
            let gy = gy_data.get(x, y);
            let magnitude = (gx * gx + gy * gy).sqrt();
            mag.set(x, y, magnitude);
        }
    }
    Ok(mag)
}

/// Apply Scharr operator, returning both Gx and Gy components.
///
/// # Returns
/// * `Ok((Image, Image))` - (Gx gradient, Gy gradient)
/// * `Err(String)` - On failure
pub fn apply_scharr_components(img: &Image) -> Result<(Image, Image), String> {
    // Scharr kernels: [-3, 0, 3, -10, 0, 10, -3, 0, 3] and its transpose.
    const SX: [[f64; 3]; 3] = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
    const SY: [[f64; 3]; 3] = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];
    let mut gx = Image::new(img.w, img.h);
    let mut gy = Image::new(img.w, img.h);

    for y in 1..img.h - 1 {
        for x in 1..img.w - 1 {
            let mut sx_acc = 0.0;
            let mut sy_acc = 0.0;
            for ky in 0..3 {
                let base = (y + ky - 1) * img.w + x - 1;
                let src = &img.data[base..base + 3];
                sx_acc += SX[ky][0] * src[0] + SX[ky][1] * src[1] + SX[ky][2] * src[2];
                sy_acc += SY[ky][0] * src[0] + SY[ky][1] * src[1] + SY[ky][2] * src[2];
            }
            gx.data[y * img.w + x] = sx_acc;
            gy.data[y * img.w + x] = sy_acc;
        }
    }
    Ok((gx, gy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Image;

    #[test]
    fn test_gaussian_kernel() {
        let k = gaussian_kernel(1.0);
        // Sum should be 1.0
        let sum: f64 = k.iter().flatten().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Center should be the largest value
        assert!(k[1][1] >= k[0][0]);
    }

    #[test]
    fn test_gaussian_kernel_5x5() {
        let k = gaussian_kernel_5x5(1.0);
        let sum: f64 = k.iter().flatten().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel_7x7() {
        let k = gaussian_kernel_7x7(1.0);
        let sum: f64 = k.iter().flatten().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sobel_kernels() {
        // SOBEL_GX should detect horizontal changes
        let sum_gx: f64 = SOBEL_GX.iter().sum();
        assert_eq!(sum_gx, 0.0);

        // SOBEL_GY should detect vertical changes
        let sum_gy: f64 = SOBEL_GY.iter().sum();
        assert_eq!(sum_gy, 0.0);
    }

    #[test]
    fn test_laplacian_kernel() {
        let sum: f64 = LAPLACIAN.iter().sum();
        // The 4-neighbourhood Laplacian has zero DC gain.
        assert!((sum - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_gaussian_blur() {
        let img = Image::new(10, 10);
        let blurred = apply_gaussian_blur(&img, 1.0).unwrap();
        assert_eq!(blurred.w, img.w);
        assert_eq!(blurred.h, img.h);
    }

    #[test]
    fn test_apply_sobel() {
        let img = Image::new(10, 10);
        let mag = apply_sobel(&img).unwrap();
        assert_eq!(mag.w, img.w);
        assert_eq!(mag.h, img.h);
    }

    #[test]
    fn test_apply_laplacian() {
        let img = Image::new(10, 10);
        let lap = apply_laplacian(&img).unwrap();
        assert_eq!(lap.w, img.w);
        assert_eq!(lap.h, img.h);
    }

    #[test]
    fn test_apply_scharr() {
        let img = Image::new(10, 10);
        let mag = apply_scharr(&img).unwrap();
        assert_eq!(mag.w, img.w);
        assert_eq!(mag.h, img.h);
    }

    #[test]
    fn test_scharr_components() {
        let img = Image::new(10, 10);
        let (gx, gy) = apply_scharr_components(&img).unwrap();
        assert_eq!(gx.w, img.w);
        assert_eq!(gx.h, img.h);
        assert_eq!(gy.w, img.w);
        assert_eq!(gy.h, img.h);
    }
}