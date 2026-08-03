//! K-means clustering.

use std::cmp::Ordering;

use crate::knn::euclidean;
use mathverse_core::error::MathResult;

/// K-means result.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Final centroid positions for each cluster.
    pub centroids: Vec<Vec<f64>>,
    /// Cluster assignment for each sample.
    pub labels: Vec<usize>,
    /// Sum of squared distances from each sample to its assigned centroid.
    pub inertia: f64,
    /// Number of iterations executed.
    pub n_iters: usize,
}

/// K-means++ initialization.
fn init_centroids(x: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut centroids = Vec::with_capacity(k);
    let mut state: u32 = 0xABCD_1234;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    centroids.push(x[(state as usize) % n].clone());
    for _ in 1..k {
        let dists: Vec<f64> = x
            .iter()
            .map(|xi| {
                centroids
                    .iter()
                    .map(|c| euclidean(xi, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = dists.iter().sum();
        let threshold = {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u32::MAX as f64) * total
        };
        let mut pushed = false;
        let mut cumsum = 0.0;
        for (i, d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                centroids.push(x[i].clone());
                pushed = true;
                break;
            }
        }
        if !pushed {
            centroids.push(x[0].clone());
        }
    }
    centroids
}

/// K-means clustering.
#[must_use]
pub fn kmeans(x: &[Vec<f64>], k: usize, max_iters: usize, tol: f64) -> MathResult<KMeansResult> {
    assert!(!x.is_empty() && k > 0 && k <= x.len());
    let n = x.len();
    let p = x[0].len();
    let mut centroids = init_centroids(x, k);
    let mut labels = vec![0usize; n];

    for iter in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let best = centroids
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    euclidean(&x[i], a.1)
                        .partial_cmp(&euclidean(&x[i], b.1))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if labels[i] != best {
                changed = true;
            }
            labels[i] = best;
        }
        // Update centroids
        let mut new_centroids = vec![vec![0.0; p]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            counts[labels[i]] += 1;
            for j in 0..p {
                new_centroids[labels[i]][j] += x[i][j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..p {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            } else {
                new_centroids[c] = centroids[c].clone();
            }
        }
        let shift: f64 = centroids
            .iter()
            .zip(&new_centroids)
            .map(|(old, new)| euclidean(old, new))
            .sum();
        centroids = new_centroids;
        if !changed || shift < tol {
            let inertia: f64 = (0..n)
                .map(|i| euclidean(&x[i], &centroids[labels[i]]).powi(2))
                .sum();
            return Ok(KMeansResult {
                centroids,
                labels,
                inertia,
                n_iters: iter + 1,
            });
        }
    }
    let inertia: f64 = (0..n)
        .map(|i| euclidean(&x[i], &centroids[labels[i]]).powi(2))
        .sum();
    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iters: max_iters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters() {
        let mut x: Vec<Vec<f64>> = Vec::new();
        for i in 0..10 {
            x.push(vec![i as f64, 0.0]);
        }
        for i in 0..10 {
            x.push(vec![i as f64 + 100.0, 0.0]);
        }
        let r = kmeans(&x, 2, 100, 1e-6).unwrap();
        assert_eq!(r.labels[0], r.labels[1]);
        assert_ne!(r.labels[0], r.labels[10]);
    }

    #[test]
    fn inertia_decreases() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let r1 = kmeans(&x, 3, 1, 1e-6).unwrap();
        let r2 = kmeans(&x, 3, 10, 1e-6).unwrap();
        assert!(r2.inertia <= r1.inertia + 1e-6);
    }
}
