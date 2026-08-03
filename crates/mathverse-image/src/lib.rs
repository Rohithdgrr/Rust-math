//! # MathVerse Image Processing Library
//!
//! A production-grade Rust library for grayscale image processing, providing
//! efficient implementations of common computer vision algorithms.
//!
//! ## Features
//!
//! - **Core Operations**: Convolution, Gaussian blur, box blur, sharpening
//! - **Edge Detection**: Sobel gradients and Canny edge detection with hysteresis
//! - **Morphological Operations**: Erosion, dilation, opening, closing
//! - **Geometric Transforms**: Horizontal flip, 90° rotation, nearest-neighbor resize
//! - **Histogram Analysis**: 256-bin histogram computation
//! - **Image I/O**: Load and save images in common formats (PNG, JPEG, BMP)
//! - **Advanced Operations**: Thresholding, noise generation, arithmetic operations
//! - **Error Handling**: Comprehensive error types with `thiserror`
//!
//! ## Basic Usage
//!
//! ```rust
//! use mathverse_image::{GrayImage, box_blur, sharpen};
//!
//! // Create a new blank image
//! let mut img = GrayImage::new(64, 64);
//!
//! // Set some pixel values
//! img.set(10, 10, 0.5);
//!
//! // Apply blur
//! let blurred = box_blur(&img);
//!
//! // Apply sharpening
//! let sharpened = sharpen(&img);
//! ```
//!
//! ## Modules
//!
//! - [`canny`] - Canny edge detection algorithm
//! - [`morphology`] - Binary morphological operations
//! - [`io`] - Image loading and saving
//! - [`operations`] - Advanced image operations
//! - [`error`] - Error types and Result alias
//!
//! ## Image Representation
//!
//! Images are represented as [`GrayImage`] structs with row-major `f64` values
//! in the range `[0, 1]`. This provides high precision for mathematical operations
//! while maintaining compatibility with standard image formats through I/O operations.

pub mod canny;
pub mod error;
pub mod io;
pub mod morphology;
pub mod operations;

pub use crate::error::{ImageError, Result};

/// Grayscale image with row-major `f64` pixel values in `[0, 1]`.
///
/// The `GrayImage` struct represents a grayscale image where each pixel value
/// is stored as a `f64` in the range `[0, 1]`. This provides high precision for
/// mathematical operations while maintaining compatibility with standard image
/// formats through I/O operations.
///
/// # Fields
///
/// - `w` - Width of the image in pixels
/// - `h` - Height of the image in pixels
/// - `data` - Flat row-major vector of pixel values, length = w × h
///
/// # Examples
///
/// ```rust
/// use mathverse_image::GrayImage;
///
/// // Create a new blank image
/// let img = GrayImage::new(64, 64);
///
/// // Create from existing data
/// let data = vec![0.5; 64 * 64];
/// let mut img = GrayImage::from_data(64, 64, data).unwrap();
///
/// // Access and modify pixels
/// img.set(10, 10, 0.8);
/// let value = img.get(10, 10);
/// ```
#[derive(Debug, Clone)]
pub struct GrayImage {
    /// Image width in pixels
    pub w: usize,
    /// Image height in pixels
    pub h: usize,
    /// Flat row-major pixel data, values in [0, 1]
    pub data: Vec<f64>,
}

impl GrayImage {
    /// Creates a new blank image with the given dimensions.
    ///
    /// All pixels are initialized to 0.0 (black).
    ///
    /// # Arguments
    ///
    /// * `w` - Width of the image in pixels
    /// * `h` - Height of the image in pixels
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// assert_eq!(img.w, 64);
    /// assert_eq!(img.h, 64);
    /// assert!(img.data.iter().all(|&v| v == 0.0));
    /// ```
    pub fn new(w: usize, h: usize) -> GrayImage {
        GrayImage { w, h, data: vec![0.0; w * h] }
    }

    /// Creates an image from flat row-major data.
    ///
    /// # Arguments
    ///
    /// * `w` - Width of the image in pixels
    /// * `h` - Height of the image in pixels
    /// * `data` - Flat row-major pixel data, must have exactly w × h elements
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if width or height is zero.
    /// Returns [`ImageError::DataLengthMismatch`] if data length doesn't match dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let data = vec![0.5; 64 * 64];
    /// let img = GrayImage::from_data(64, 64, data).unwrap();
    /// ```
    pub fn from_data(w: usize, h: usize, data: Vec<f64>) -> Result<GrayImage> {
        if w == 0 || h == 0 {
            return Err(ImageError::InvalidDimensions { width: w, height: h });
        }
        if data.len() != w * h {
            return Err(ImageError::DataLengthMismatch {
                data_len: data.len(),
                expected_len: w * h,
                width: w,
                height: h,
            });
        }
        Ok(GrayImage { w, h, data })
    }

    /// Gets the pixel value at the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column), must be in range [0, w)
    /// * `y` - Y coordinate (row), must be in range [0, h)
    ///
    /// # Returns
    ///
    /// The pixel value in the range [0, 1].
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.w + x]
    }

    /// Sets the pixel value at the specified coordinates.
    ///
    /// The value is automatically clamped to the range [0, 1].
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column), must be in range [0, w)
    /// * `y` - Y coordinate (row), must be in range [0, h)
    /// * `v` - New pixel value, will be clamped to [0, 1]
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        self.data[y * self.w + x] = v.clamp(0.0, 1.0);
    }

    /// Applies a 3×3 kernel convolution with border clamping.
    ///
    /// Performs discrete convolution with the given 3×3 kernel. Pixels at the
    /// image borders use clamped boundary conditions (nearest neighbor).
    ///
    /// # Arguments
    ///
    /// * `kernel` - 3×3 kernel in row-major order (9 elements)
    ///
    /// # Returns
    ///
    /// A new image with the convolution applied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(32, 32);
    /// // Box blur kernel
    /// let kernel = [1.0/9.0; 9];
    /// let blurred = img.convolve3(&kernel);
    /// ```
    pub fn convolve3(&self, kernel: &[f64; 9]) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h);
        let (kw, kh) = (3, 3);
        let (ox, oy) = (kw / 2, kh / 2);
        for y in 0..self.h {
            for x in 0..self.w {
                let mut s = 0.0;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let px = (x as i64 + kx as i64 - ox as i64).clamp(0, self.w as i64 - 1) as usize;
                        let py = (y as i64 + ky as i64 - oy as i64).clamp(0, self.h as i64 - 1) as usize;
                        s += kernel[ky * kw + kx] * self.get(px, py);
                    }
                }
                out.set(x, y, s);
            }
        }
        out
    }

    /// Applies separable Gaussian blur with the given radius and sigma.
    ///
    /// Uses a separable implementation for efficiency: horizontal pass followed
    /// by vertical pass. The kernel size is `2r + 1` and uses clamped border conditions.
    ///
    /// # Arguments
    ///
    /// * `r` - Blur radius, kernel size will be 2r + 1
    /// * `sigma` - Standard deviation of the Gaussian distribution
    ///
    /// # Returns
    ///
    /// A new image with Gaussian blur applied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// let blurred = img.gaussian_blur(3, 1.5);
    /// ```
    pub fn gaussian_blur(&self, r: usize, sigma: f64) -> GrayImage {
        let n = 2 * r + 1;
        let mut kernel = Vec::with_capacity(n);
        for i in 0..n {
            let d = i as f64 - r as f64;
            kernel.push((-(d * d) / (2.0 * sigma * sigma)).exp());
        }
        let ksum: f64 = kernel.iter().sum();
        let kernel: Vec<f64> = kernel.iter().map(|v| v / ksum).collect();
        // horizontal pass
        let mut tmp = GrayImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let s: f64 = (0..n)
                    .map(|i| {
                        let px = (x as i64 + i as i64 - r as i64).clamp(0, self.w as i64 - 1) as usize;
                        kernel[i] * self.get(px, y)
                    })
                    .sum();
                tmp.set(x, y, s);
            }
        }
        // vertical pass
        let mut out = GrayImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let s: f64 = (0..n)
                    .map(|i| {
                        let py = (y as i64 + i as i64 - r as i64).clamp(0, self.h as i64 - 1) as usize;
                        kernel[i] * tmp.get(x, py)
                    })
                    .sum();
                out.set(x, y, s);
            }
        }
        out
    }

    /// Computes Sobel gradient magnitude and direction.
    ///
    /// Returns both the gradient magnitude image and the direction angles in radians.
    /// Uses the standard Sobel operators for horizontal and vertical gradients.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Gradient magnitude image
    /// - Direction angles in radians (flat vector, same order as pixel data)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// let (magnitude, direction) = img.sobel();
    /// ```
    pub fn sobel(&self) -> (GrayImage, Vec<f64>) {
        const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        let dx = self.convolve3(&GX);
        let dy = self.convolve3(&GY);
        let mut mag = GrayImage::new(self.w, self.h);
        let mut dir = vec![0.0; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let (gx, gy) = (dx.get(x, y), dy.get(x, y));
                mag.set(x, y, (gx * gx + gy * gy).sqrt());
                dir[y * self.w + x] = gy.atan2(gx);
            }
        }
        (mag, dir)
    }

    /// Computes a 256-bin histogram over the [0, 1] range.
    ///
    /// Divides the [0, 1] range into 256 equal bins and counts pixel values
    /// in each bin. Useful for image analysis and thresholding.
    ///
    /// # Returns
    ///
    /// An array of 256 bin counts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// let histogram = img.histogram();
    /// let total_pixels: usize = histogram.iter().sum();
    /// ```
    pub fn histogram(&self) -> [usize; 256] {
        let mut bins = [0usize; 256];
        for &v in &self.data {
            bins[(v.clamp(0.0, 0.999999) * 256.0) as usize] += 1;
        }
        bins
    }

    /// Flips the image horizontally (left-right mirror).
    ///
    /// # Returns
    ///
    /// A new horizontally flipped image.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// let flipped = img.flip_h();
    /// ```
    pub fn flip_h(&self) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(self.w - 1 - x, y, self.get(x, y));
            }
        }
        out
    }

    /// Rotates the image 90 degrees clockwise.
    ///
    /// The dimensions are swapped: a W×H image becomes H×W.
    ///
    /// # Returns
    ///
    /// A new rotated image with swapped dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 32);
    /// let rotated = img.rotate90();
    /// assert_eq!(rotated.w, 32);
    /// assert_eq!(rotated.h, 64);
    /// ```
    pub fn rotate90(&self) -> GrayImage {
        let mut out = GrayImage::new(self.h, self.w);
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(self.h - 1 - y, x, self.get(x, y));
            }
        }
        out
    }

    /// Resizes the image using nearest-neighbor interpolation.
    ///
    /// Simple but fast resizing method that selects the nearest pixel value.
    /// For higher quality, consider implementing bilinear or bicubic interpolation.
    ///
    /// # Arguments
    ///
    /// * `nw` - New width
    /// * `nh` - New height
    ///
    /// # Returns
    ///
    /// A new resized image.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_image::GrayImage;
    ///
    /// let img = GrayImage::new(64, 64);
    /// let resized = img.resize(128, 128);
    /// ```
    pub fn resize(&self, nw: usize, nh: usize) -> GrayImage {
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let sx = (x * self.w / nw).min(self.w - 1);
                let sy = (y * self.h / nh).min(self.h - 1);
                out.set(x, y, self.get(sx, sy));
            }
        }
        out
    }
}

/// Applies a 3×3 box blur (uniform averaging).
///
/// Uses a uniform kernel where all 9 elements are 1/9, effectively averaging
/// each pixel with its 8 neighbors.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// A new blurred image.
///
/// # Examples
///
/// ```rust
/// use mathverse_image::{GrayImage, box_blur};
///
/// let img = GrayImage::new(64, 64);
/// let blurred = box_blur(&img);
/// ```
pub fn box_blur(img: &GrayImage) -> GrayImage {
    img.convolve3(&[1.0 / 9.0; 9])
}

/// Applies sharpening using an unsharp mask kernel.
///
/// Uses the standard unsharp mask kernel:
/// ```text
///  0 -1  0
/// -1  5 -1
///  0 -1  0
/// ```
/// This enhances edges by subtracting a blurred version from the original.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// A new sharpened image.
///
/// # Examples
///
/// ```rust
/// use mathverse_image::{GrayImage, sharpen};
///
/// let img = GrayImage::new(64, 64);
/// let sharpened = sharpen(&img);
/// ```
pub fn sharpen(img: &GrayImage) -> GrayImage {
    img.convolve3(&[0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn black_white_step(w: usize, h: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, if x < w / 2 { 0.0 } else { 1.0 });
            }
        }
        img
    }

    #[test]
    fn blur_and_sobel() {
        let img = black_white_step(32, 16);
        let (mag, _dir) = img.sobel();
        // strong gradient near the step column
        assert!(mag.get(15, 8) > 0.5);
        assert!(mag.get(0, 8) < 0.1);
        // blur smooths the interior of a constant image exactly
        let flat = GrayImage::from_data(8, 8, vec![0.5; 64]).unwrap();
        assert!(flat.gaussian_blur(3, 1.5).data.iter().all(|v| (v - 0.5).abs() < 1e-12));
        // sobel of a constant image is zero
        let (mag, _) = flat.sobel();
        assert!(mag.data.iter().all(|v| *v < 1e-12));
    }

    #[test]
    fn histogram_and_transforms() {
        let img = black_white_step(16, 4);
        let h = img.histogram();
        assert_eq!(h[0], 8 * 4);
        assert_eq!(h[255], 8 * 4);
        // flip-then-flip is identity
        let twice = img.flip_h().flip_h();
        for (a, b) in img.data.iter().zip(&twice.data) {
            assert!((a - b).abs() < 1e-12);
        }
        // rotate 4 times is identity (square)
        let sq = GrayImage::from_data(6, 6, (0..36).map(|i| (i % 7) as f64 / 10.0).collect()).unwrap();
        let four = sq.rotate90().rotate90().rotate90().rotate90();
        for (a, b) in sq.data.iter().zip(&four.data) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(img.resize(8, 4).w, 8);
    }
}
