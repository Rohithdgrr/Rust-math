struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generates a synthetic classification dataset with n_classes.
#[must_use]
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        let mut row = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let shift = if f == 0 { class as f64 } else { 0.0 };
            row.push(rng.next_normal() + shift);
        }
        x.push(row);
        y.push(class as f64);
    }

    (x, y)
}

/// Generates a synthetic regression dataset with Gaussian noise.
#[must_use]
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let true_weights: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let row: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        let target: f64 = row
            .iter()
            .zip(true_weights.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + noise * rng.next_normal();
        x.push(row);
        y.push(target);
    }

    (x, y)
}

/// Generates isotropic Gaussian blobs for clustering.
#[must_use]
pub fn make_blobs(
    n_samples: usize,
    n_centers: usize,
    cluster_std: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let centers: Vec<Vec<f64>> = (0..n_centers)
        .map(|_| vec![rng.next_normal() * 5.0, rng.next_normal() * 5.0])
        .collect();

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let center = i % n_centers;
        let row = vec![
            centers[center][0] + cluster_std * rng.next_normal(),
            centers[center][1] + cluster_std * rng.next_normal(),
        ];
        x.push(row);
        y.push(center as f64);
    }

    (x, y)
}

/// Generates two interleaving half-circle moons for binary classification.
#[must_use]
pub fn make_moons(n_samples: usize, noise: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % 2;
        let t = rng.next_f64() * std::f64::consts::PI;
        // Generate interleaving crescents by offsetting the second moon
        let x0 = if class == 0 {
            t.cos() + noise * rng.next_normal()
        } else {
            (1.0 - t.cos()) + noise * rng.next_normal()
        };
        let x1 = if class == 0 {
            t.sin() + noise * rng.next_normal()
        } else {
            (0.5 - t.sin()) + noise * rng.next_normal()
        };
        x.push(vec![x0, x1]);
        y.push(class as f64);
    }

    (x, y)
}

/// Generates two concentric circles for binary classification.
#[must_use]
pub fn make_circles(
    n_samples: usize,
    noise: f64,
    factor: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % 2;
        let t = rng.next_f64() * 2.0 * std::f64::consts::PI;
        let r = if class == 0 { 1.0 } else { factor };
        let x0 = r * t.cos() + noise * rng.next_normal();
        let x1 = r * t.sin() + noise * rng.next_normal();
        x.push(vec![x0, x1]);
        y.push(class as f64);
    }

    (x, y)
}

/// Generates n_classes interleaving spirals for multi-class classification.
#[must_use]
pub fn make_spirals(
    n_samples: usize,
    noise: f64,
    n_classes: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        let t = rng.next_f64() * 2.0 * std::f64::consts::PI;
        let r = t / (2.0 * std::f64::consts::PI) + rng.next_normal() * noise;
        let offset = class as f64 * 2.0 * std::f64::consts::PI / n_classes as f64;
        let x0 = r * (t + offset).cos();
        let x1 = r * (t + offset).sin();
        x.push(vec![x0, x1]);
        y.push(class as f64);
    }

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_classification() {
        let (x, y) = make_classification(100, 4, 3, 42);
        assert_eq!(x.len(), 100);
        assert_eq!(x[0].len(), 4);
        assert_eq!(y.len(), 100);
        let classes: Vec<f64> = y.iter().copied().collect();
        assert!(classes.iter().any(|&c| c == 0.0));
        assert!(classes.iter().any(|&c| c == 1.0));
        assert!(classes.iter().any(|&c| c == 2.0));
    }

    #[test]
    fn test_make_regression() {
        let (x, y) = make_regression(50, 3, 0.1, 42);
        assert_eq!(x.len(), 50);
        assert_eq!(x[0].len(), 3);
        assert_eq!(y.len(), 50);
    }

    #[test]
    fn test_make_blobs() {
        let (x, y) = make_blobs(60, 3, 0.5, 42);
        assert_eq!(x.len(), 60);
        assert!(x[0].len() >= 2);
        assert_eq!(y.len(), 60);
    }

    #[test]
    fn test_make_moons() {
        let (x, y) = make_moons(100, 0.1, 42);
        assert_eq!(x.len(), 100);
        assert_eq!(x[0].len(), 2);
        assert_eq!(y.len(), 100);
        assert!(y.iter().any(|&c| c == 0.0));
        assert!(y.iter().any(|&c| c == 1.0));
    }

    #[test]
    fn test_make_circles() {
        let (x, y) = make_circles(100, 0.1, 0.5, 42);
        assert_eq!(x.len(), 100);
        assert_eq!(x[0].len(), 2);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_spirals() {
        let (x, y) = make_spirals(90, 0.1, 3, 42);
        assert_eq!(x.len(), 90);
        assert_eq!(x[0].len(), 2);
        assert_eq!(y.len(), 90);
    }
}
