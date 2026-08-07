//! # mathverse-vision
//!
//! Computer vision primitives for the MathVerse ecosystem.
//!
//! Provides:
//! - **Camera**: pinhole camera model with projection and unprojection
//! - **Epipolar**: fundamental/essential matrix estimation, epipolar lines
//! - **Features**: corner detection (Harris), feature matching
//! - **Optical flow**: Lucas-Kanade sparse flow estimation
//! - **Homography**: homography estimation and RANSAC
//!
//! The [`Image`] type is a simple single-channel `f64` grayscale image with
//! basic convolution and Gaussian blur support.

pub mod camera;
pub mod epipolar;
pub mod features;
pub mod flow;
pub mod homography;

#[derive(Debug, Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    pub fn from_data(w: usize, h: usize, data: Vec<f64>) -> Self {
        assert_eq!(w * h, data.len());
        Self { w, h, data }
    }

    /// Read a pixel, returning `0.0` for out-of-bounds coordinates.
    ///
    /// The zero-extension semantics are deliberate: convolution kernels and
    /// finite differences (e.g. Sobel, Lucas–Kanade) call `get` near borders,
    /// where an implicit zero border is the standard padding convention.
    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.w && y < self.h { self.data[y * self.w + x] } else { 0.0 }
    }

    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        if x < self.w && y < self.h {
            self.data[y * self.w + x] = v;
        }
    }

    /// 3×3 convolution with a zero border. Images smaller than the kernel are
    /// returned unchanged (no filtering is applied).
    pub fn convolve3(&self, k: &[f64; 9]) -> Self {
        // Images smaller than 3×3 have no interior pixels to filter.
        if self.w < 3 || self.h < 3 {
            return self.clone();
        }
        let (w, h) = (self.w, self.h);
        let mut out = Self::new(w, h);
        // Interior pixels are always in bounds, so index the data directly to
        // skip the per-pixel bounds check of `get`.
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut s = 0.0;
                for ky in 0..3 {
                    let base = (y + ky - 1) * w + x - 1;
                    let src = &self.data[base..base + 3];
                    s += k[ky * 3] * src[0] + k[ky * 3 + 1] * src[1] + k[ky * 3 + 2] * src[2];
                }
                out.data[y * w + x] = s;
            }
        }
        out
    }

    /// Gaussian blur. Returns the image unchanged when `radius == 0`, when
    /// `sigma <= 0`, or when the kernel is larger than the image.
    pub fn gaussian_blur(&self, radius: usize, sigma: f64) -> Self {
        let size = 2 * radius + 1;
        // No-op for radius 0, non-positive sigma, or an oversized kernel.
        if radius == 0 || sigma <= 0.0 || self.w < size || self.h < size {
            return self.clone();
        }
        let mut kernel = vec![0.0f64; size * size];
        let mut sum = 0.0;
        let s2 = 2.0 * sigma * sigma;
        for ky in 0..size {
            for kx in 0..size {
                let dx = kx as f64 - radius as f64;
                let dy = ky as f64 - radius as f64;
                let v = (-(dx * dx + dy * dy) / s2).exp();
                kernel[ky * size + kx] = v;
                sum += v;
            }
        }
        for v in &mut kernel { *v /= sum; }
        let (w, h) = (self.w, self.h);
        let mut out = Self::new(w, h);
        // Interior pixels are always in bounds; index the data directly.
        for y in radius..h - radius {
            for x in radius..w - radius {
                let mut s = 0.0;
                for ky in 0..size {
                    let base = (y + ky - radius) * w + x - radius;
                    let src = &self.data[base..base + size];
                    for kx in 0..size {
                        s += kernel[ky * size + kx] * src[kx];
                    }
                }
                out.data[y * w + x] = s;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::camera::Camera;
    use crate::Image;

    const EPS: f64 = 1e-9;

    #[test]
    fn camera_projection() {
        let cam = Camera::new(800.0, 600.0, 320.0, 240.0);
        let (u, v) = cam.project(1.0, 2.0, 10.0);
        assert!((u - 400.0).abs() < EPS);
        assert!((v - 360.0).abs() < EPS);
    }

    #[test]
    fn image_basic() {
        let mut img = Image::new(4, 4);
        img.set(1, 2, 3.0);
        assert!((img.get(1, 2) - 3.0).abs() < EPS);
        assert!((img.get(0, 0)).abs() < EPS);
    }

    #[test]
    fn tiny_images_do_not_panic() {
        // Kernel filters on images smaller than the kernel must be no-ops.
        for (w, h) in [(0usize, 0usize), (1, 1), (2, 2), (3, 1), (1, 3)] {
            let img = Image::new(w, h);
            let c = img.convolve3(&[1.0; 9]);
            assert_eq!((c.w, c.h), (w, h));
            let b = img.gaussian_blur(2, 1.0);
            assert_eq!((b.w, b.h), (w, h));
        }
        // Radius larger than the image and non-positive sigma are no-ops.
        let img = Image::new(6, 6);
        assert_eq!(img.gaussian_blur(5, 1.0).data, img.data);
        assert_eq!(img.gaussian_blur(2, 0.0).data, img.data);
        assert_eq!(img.gaussian_blur(0, 1.0).data, img.data);
    }
}
