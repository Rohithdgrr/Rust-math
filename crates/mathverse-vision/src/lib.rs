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

    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.w && y < self.h { self.data[y * self.w + x] } else { 0.0 }
    }

    pub fn set(&mut self, x: usize, y: usize, v: f64) {
        if x < self.w && y < self.h {
            self.data[y * self.w + x] = v;
        }
    }

    pub fn convolve3(&self, k: &[f64; 9]) -> Self {
        let mut out = Self::new(self.w, self.h);
        for y in 1..self.h - 1 {
            for x in 1..self.w - 1 {
                let mut s = 0.0;
                for ky in 0..3 {
                    for kx in 0..3 {
                        s += k[ky * 3 + kx] * self.get(x + kx - 1, y + ky - 1);
                    }
                }
                out.set(x, y, s);
            }
        }
        out
    }

    pub fn gaussian_blur(&self, radius: usize, sigma: f64) -> Self {
        let size = 2 * radius + 1;
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
        let mut out = Self::new(self.w, self.h);
        for y in radius..self.h - radius {
            for x in radius..self.w - radius {
                let mut s = 0.0;
                for ky in 0..size {
                    for kx in 0..size {
                        s += kernel[ky * size + kx] * self.get(x + kx - radius, y + ky - radius);
                    }
                }
                out.set(x, y, s);
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
}
