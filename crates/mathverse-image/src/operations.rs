//! Additional image operations: thresholding, noise, arithmetic operations.

use crate::GrayImage;
use rand::Rng;

/// Draw a sample from `N(mean, std_dev)` using the Box–Muller transform.
///
/// Uses `Rng::random::<f64>` (the non-deprecated `rand` 0.9 API) for the two
/// uniform draws and guarantees a non-zero (hence finite) result.
fn normal_sample(rng: &mut impl Rng, mean: f64, std_dev: f64) -> f64 {
    let u1 = rng.random::<f64>().max(1e-12);
    let u2 = rng.random::<f64>();
    let mag = (-2.0 * u1.ln()).sqrt();
    let z = mag * (2.0 * core::f64::consts::PI * u2).cos();
    mean + std_dev * z
}

impl GrayImage {
    /// Apply simple thresholding: values >= threshold become 1.0, else 0.0.
    pub fn threshold(&self, threshold: f64) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for (i, &v) in self.data.iter().enumerate() {
            out.data[i] = if v >= threshold { 1.0 } else { 0.0 };
        }
        out
    }

    /// Apply adaptive thresholding using local mean.
    pub fn adaptive_threshold(&self, block_size: usize, c: f64) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        let half = block_size / 2;

        for y in 0..self.h {
            for x in 0..self.w {
                let mut sum = 0.0;
                let mut count = 0;

                // Calculate local mean
                for dy in -(half as i64)..=(half as i64) {
                    for dx in -(half as i64)..=(half as i64) {
                        let nx = (x as i64 + dx).clamp(0, self.w as i64 - 1) as usize;
                        let ny = (y as i64 + dy).clamp(0, self.h as i64 - 1) as usize;
                        sum += self.get(nx, ny);
                        count += 1;
                    }
                }

                let mean = sum / count as f64;
                let local_threshold = mean - c;
                out.set(x, y, if self.get(x, y) >= local_threshold { 1.0 } else { 0.0 });
            }
        }
        out
    }

    /// Add Gaussian noise to the image.
    pub fn add_gaussian_noise(&self, mean: f64, std_dev: f64) -> GrayImage {
        let mut out = self.clone();
        let mut rng = rand::thread_rng();

        for v in out.data.iter_mut() {
            let noise: f64 = normal_sample(&mut rng, mean, std_dev);
            *v = (*v + noise).clamp(0.0, 1.0);
        }
        out
    }

    /// Add Gaussian noise to the image with a reproducible seed.
    ///
    /// Uses a deterministic `StdRng` seeded with `seed`, so repeated calls
    /// with the same seed produce identical output.
    pub fn add_gaussian_noise_seeded(&self, mean: f64, std_dev: f64, seed: u64) -> GrayImage {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = self.clone();
        for v in out.data.iter_mut() {
            let noise: f64 = normal_sample(&mut rng, mean, std_dev);
            *v = (*v + noise).clamp(0.0, 1.0);
        }
        out
    }

    /// Add salt-and-pepper noise to the image.
    pub fn add_salt_pepper_noise(&self, density: f64) -> GrayImage {
        let mut out = self.clone();
        let mut rng = rand::thread_rng();

        for v in out.data.iter_mut() {
            if rng.random::<f64>() < density {
                *v = if rng.random::<bool>() { 1.0 } else { 0.0 };
            }
        }
        out
    }

    /// Apply median filter with the given radius (kernel size = 2*radius + 1).
    pub fn median_filter(&self, radius: usize) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        let size = 2 * radius + 1;
        let mut window: Vec<f64> = Vec::with_capacity(size * size);
        for y in 0..self.h {
            for x in 0..self.w {
                window.clear();
                for dy in -(radius as i64)..=(radius as i64) {
                    for dx in -(radius as i64)..=(radius as i64) {
                        let nx = (x as i64 + dx).clamp(0, self.w as i64 - 1) as usize;
                        let ny = (y as i64 + dy).clamp(0, self.h as i64 - 1) as usize;
                        window.push(self.get(nx, ny));
                    }
                }
                // total_cmp: NaN pixel values sort deterministically instead
                // of panicking via partial_cmp().unwrap().
                window.sort_by(|a, b| a.total_cmp(b));
                let median = window[window.len() / 2];
                out.set(x, y, median);
            }
        }
        out
    }

    /// Add two images element-wise with clamping.
    pub fn add(&self, other: &GrayImage) -> GrayImage {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);

        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] + other.data[i]).clamp(0.0, 1.0);
        }
        out
    }

    /// Subtract two images element-wise with clamping.
    pub fn subtract(&self, other: &GrayImage) -> GrayImage {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);

        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] - other.data[i]).clamp(0.0, 1.0);
        }
        out
    }

    /// Multiply two images element-wise.
    pub fn multiply(&self, other: &GrayImage) -> GrayImage {
        assert_eq!(self.w, other.w);
        assert_eq!(self.h, other.h);

        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] * other.data[i]).clamp(0.0, 1.0);
        }
        out
    }

    /// Multiply all pixel values by a scalar.
    pub fn scale(&self, factor: f64) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] * factor).clamp(0.0, 1.0);
        }
        out
    }

    /// Add a constant to all pixel values.
    pub fn offset(&self, value: f64) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] + value).clamp(0.0, 1.0);
        }
        out
    }

    /// Invert the image (1.0 - value).
    pub fn invert(&self) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = 1.0 - self.data[i];
        }
        out
    }

    /// Apply gamma correction: output = input^(1/gamma).
    pub fn gamma_correction(&self, gamma: f64) -> GrayImage {
        let inv_gamma = 1.0 / gamma;
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].powf(inv_gamma);
        }
        out
    }

    /// Compute the mean pixel value.
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.data.iter().sum();
        sum / self.data.len() as f64
    }

    /// Compute the standard deviation of pixel values.
    pub fn std_dev(&self) -> f64 {
        let mean = self.mean();
        let variance: f64 = self.data.iter()
            .map(|v| (v - mean).powi(2))
            .sum();
        (variance / self.data.len() as f64).sqrt()
    }

    /// Compute the minimum pixel value.
    pub fn min_value(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Compute the maximum pixel value.
    pub fn max_value(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Normalize the image to [0, 1] range based on min/max values.
    pub fn normalize(&self) -> GrayImage {
        let min = self.min_value();
        let max = self.max_value();
        let range = max - min;

        if range < 1e-10 {
            // Constant image
            return self.clone();
        }

        let mut out = GrayImage::new(self.w, self.h).unwrap();
        for i in 0..self.data.len() {
            out.data[i] = (self.data[i] - min) / range;
        }
        out
    }

    /// Apply contrast stretching: map [low, high] to [0, 1].
    pub fn contrast_stretch(&self, low: f64, high: f64) -> GrayImage {
        let mut out = GrayImage::new(self.w, self.h).unwrap();
        let range = high - low;

        for i in 0..self.data.len() {
            if self.data[i] <= low {
                out.data[i] = 0.0;
            } else if self.data[i] >= high {
                out.data[i] = 1.0;
            } else {
                out.data[i] = (self.data[i] - low) / range;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold() {
        let mut img = GrayImage::new(4, 4).unwrap();
        for i in 0..16 {
            img.data[i] = i as f64 / 16.0;
        }

        let thresholded = img.threshold(0.5);
        let count_ones: usize = thresholded.data.iter().filter(|v| **v > 0.5).count();
        assert_eq!(count_ones, 8);
    }

    #[test]
    fn test_invert() {
        let mut img = GrayImage::new(2, 2).unwrap();
        img.set(0, 0, 0.0);
        img.set(1, 0, 0.5);
        img.set(0, 1, 1.0);
        img.set(1, 1, 0.25);

        let inverted = img.invert();
        // invert(x) = 1 - x
        assert!((inverted.get(0, 0) - 1.0).abs() < 1e-10); // invert(0.0)
        assert!((inverted.get(1, 0) - 0.5).abs() < 1e-10); // invert(0.5)
        assert!((inverted.get(0, 1) - 0.0).abs() < 1e-10); // invert(1.0)
        assert!((inverted.get(1, 1) - 0.75).abs() < 1e-10); // invert(0.25)
    }

    #[test]
    fn test_arithmetic() {
        let a = GrayImage::from_data(2, 2, vec![0.2, 0.4, 0.6, 0.8]).unwrap();
        let b = GrayImage::from_data(2, 2, vec![0.1, 0.3, 0.5, 0.7]).unwrap();

        let sum = a.add(&b);
        assert!((sum.get(0, 0) - 0.3).abs() < 1e-10);
        assert!((sum.get(1, 1) - 1.0).abs() < 1e-10); // clamped

        let diff = a.subtract(&b);
        assert!((diff.get(0, 0) - 0.1).abs() < 1e-10);

        let prod = a.multiply(&b);
        assert!((prod.get(0, 0) - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_scale_offset() {
        let img = GrayImage::from_data(2, 2, vec![0.5; 4]).unwrap();

        let scaled = img.scale(2.0);
        assert!((scaled.get(0, 0) - 1.0).abs() < 1e-10); // clamped

        let offset = img.offset(0.25);
        assert!((offset.get(0, 0) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_statistics() {
        let img = GrayImage::from_data(2, 2, vec![0.0, 0.5, 1.0, 0.5]).unwrap();

        assert!((img.mean() - 0.5).abs() < 1e-10);
        assert!((img.min_value() - 0.0).abs() < 1e-10);
        assert!((img.max_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let img = GrayImage::from_data(2, 2, vec![0.25, 0.5, 0.75, 1.0]).unwrap();
        let normalized = img.normalize();

        assert!((normalized.min_value() - 0.0).abs() < 1e-10);
        assert!((normalized.max_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_correction() {
        let img = GrayImage::from_data(2, 2, vec![0.5; 4]).unwrap();
        let gamma = img.gamma_correction(2.0);

        // gamma = 2.0 means output = input^(0.5)
        assert!((gamma.get(0, 0) - 0.5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_seeded_noise_reproducible() {
        let img = GrayImage::from_data(2, 2, vec![0.5; 4]).unwrap();
        let a = img.add_gaussian_noise_seeded(0.0, 0.1, 42);
        let b = img.add_gaussian_noise_seeded(0.0, 0.1, 42);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_median_filter() {
        // Salt-and-pepper impulse: median filter should remove isolated spikes.
        let mut img = GrayImage::new(5, 5).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                img.set(x, y, 0.5);
            }
        }
        img.set(2, 2, 1.0);
        let filtered = img.median_filter(1);
        assert!((filtered.get(2, 2) - 0.5).abs() < 1e-10);
    }
}
