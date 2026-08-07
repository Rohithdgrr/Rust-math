//! Integral (summed-area) image and fast O(1) box blur.

use crate::GrayImage;

/// Summed-area table for a grayscale image.
///
/// The internal buffer has dimensions `(w + 1) × (h + 1)` with the first row
/// and column zeroed, so that the sum of any axis-aligned rectangle
/// `[x0, x1) × [y0, y1)` can be computed with four table lookups.
#[derive(Debug, Clone)]
pub struct IntegralImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl IntegralImage {
    /// Build an integral image from `img`.
    pub fn from_image(img: &GrayImage) -> Self {
        let w = img.w + 1;
        let h = img.h + 1;
        let mut data = vec![0.0; w * h];
        for y in 1..h {
            let mut row_sum = 0.0;
            for x in 1..w {
                row_sum += img.get(x - 1, y - 1);
                data[y * w + x] = data[(y - 1) * w + x] + row_sum;
            }
        }
        Self { w, h, data }
    }

    /// Sum of the rectangle `[x0, x1) × [y0, y1)`.
    ///
    /// Panics if any index is out of range.
    pub fn sum_region(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        assert!(x0 <= x1 && y0 <= y1);
        assert!(x1 <= self.w && y1 <= self.h);
        self.data[y1 * self.w + x1]
            - self.data[y1 * self.w + x0]
            - self.data[y0 * self.w + x1]
            + self.data[y0 * self.w + x0]
    }
}

/// Fast box blur using an integral image.
///
/// Runs in O(w·h) regardless of `radius`, making it much faster than
/// separable 1-D convolution for large radii.
pub fn fast_box_blur(img: &GrayImage, radius: usize) -> GrayImage {
    let integral = IntegralImage::from_image(img);
    let mut out = GrayImage::new(img.w, img.h).unwrap();
    for y in 0..img.h {
        for x in 0..img.w {
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let x1 = (x + radius + 1).min(img.w);
            let y1 = (y + radius + 1).min(img.h);
            let actual_area = (x1 - x0) * (y1 - y0);
            let sum = integral.sum_region(x0, y0, x1, y1);
            // `actual_area` already accounts for border clamping; dividing by
            // `side²` as well would darken edge pixels by a factor of `side²`.
            out.set(x, y, sum / actual_area as f64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_integral_sum_region() {
        let img = GrayImage::from_data(4, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]).unwrap();
        let ii = IntegralImage::from_image(&img);
        // sum of rows [1,3) × cols [1,3) = 6+7+10+11 = 34
        assert_relative_eq!(ii.sum_region(1, 1, 3, 3), 34.0, epsilon = 1e-9);
        assert_relative_eq!(ii.sum_region(0, 0, 4, 4), 136.0, epsilon = 1e-9);
    }

    #[test]
    fn test_fast_box_blur_flat() {
        let img = GrayImage::from_data(8, 8, vec![0.5; 64]).unwrap();
        let blurred = fast_box_blur(&img, 3);
        assert!(blurred.data.iter().all(|v| (v - 0.5).abs() < 1e-12));
    }

    #[test]
    fn test_fast_box_blur_edge_brightness_preserved() {
        // A uniform image must stay uniform after blurring, including edges:
        // a wrong division (by area·side²) would darken border pixels.
        let img = GrayImage::from_data(6, 6, vec![0.7; 36]).unwrap();
        let blurred = fast_box_blur(&img, 2);
        for (i, v) in blurred.data.iter().enumerate() {
            assert!((v - 0.7).abs() < 1e-9, "pixel {i} darkens to {v}");
        }
    }
}
