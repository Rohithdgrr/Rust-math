//! Radon transform for tomographic reconstruction.

/// Radon transform: projection sums of a 2D image over line integrals.
///
/// For each of `n_angles` equally spaced angles `θ ∈ [0, π)` and `n_offsets`
/// equally spaced offsets spanning `[-max, max]` (where `max = √(h² + w²)`),
/// sums every pixel whose projected coordinate `x·cos θ + y·sin θ` lies within
/// 1.0 of the offset. Returns a `[n_angles][n_offsets]` matrix (a sinogram).
pub fn radon_transform(image: &[Vec<f64>], n_angles: usize, n_offsets: usize) -> Vec<Vec<f64>> {
    let (h, w) = (image.len(), image[0].len());
    let max_offset = ((h * h + w * w) as f64).sqrt() as usize;
    let mut result = vec![vec![0.0; n_offsets]; n_angles];
    for theta_idx in 0..n_angles {
        let theta = theta_idx as f64 * core::f64::consts::PI / n_angles as f64;
        for offset_idx in 0..n_offsets {
            let offset = -(max_offset as f64) + 2.0 * max_offset as f64 * offset_idx as f64 / (n_offsets - 1) as f64;
            let mut sum = 0.0;
            for x in 0..w {
                for y in 0..h {
                    let proj = x as f64 * theta.cos() + y as f64 * theta.sin();
                    if (proj - offset).abs() < 1.0 { sum += image[y][x]; }
                }
            }
            result[theta_idx][offset_idx] = sum;
        }
    }
    result
}

/// Sinogram of an image: [`radon_transform`] with offsets spanning
/// `2·max_r = 2·√(h² + w²)`.
pub fn sinogram(image: &[Vec<f64>], n_angles: usize) -> Vec<Vec<f64>> {
    let (h, w) = (image.len(), image[0].len());
    let max_r = ((h * h + w * w) as f64).sqrt() as usize * 2;
    radon_transform(image, n_angles, max_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radon_test() {
        let image = vec![vec![1.0; 10]; 10];
        let result = radon_transform(&image, 18, 20);
        assert_eq!(result.len(), 18);
    }
}
