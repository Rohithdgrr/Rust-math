//! Agglomerative hierarchical clustering.

use crate::knn::euclidean;

/// Linkage method.
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    /// Minimum distance between any two points in different clusters.
    Single,
    /// Maximum distance between any two points in different clusters.
    Complete,
    /// Average distance between all pairs of points in different clusters.
    Average,
}

/// Agglomerative clustering result.
#[derive(Debug, Clone)]
pub struct AgglomerativeResult {
    /// Cluster assignment for each sample.
    pub labels: Vec<usize>,
    /// Number of clusters in the final partition.
    pub n_clusters: usize,
    /// Merge history as (left, right, distance, merged_size) tuples.
    pub linkage_matrix: Vec<(usize, usize, f64, usize)>,
}

/// Agglomerative hierarchical clustering.
#[must_use]
pub fn agglomerative(x: &[Vec<f64>], n_clusters: usize, linkage: Linkage) -> AgglomerativeResult {
    let n = x.len();
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = euclidean(&x[i], &x[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut sizes = vec![1usize; n];
    let mut linkage_matrix = Vec::new();
    let mut active = vec![true; n];

    while clusters.iter().filter(|c| !c.is_empty()).count() > n_clusters {
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;
        let active_ids: Vec<usize> = (0..clusters.len())
            .filter(|&i| active[i] && !clusters[i].is_empty())
            .collect();
        for ai in 0..active_ids.len() {
            for aj in (ai + 1)..active_ids.len() {
                let ci = active_ids[ai];
                let cj = active_ids[aj];
                let d = compute_linkage(&clusters[ci], &clusters[cj], &dist, linkage);
                if d < best_dist {
                    best_dist = d;
                    best_i = ci;
                    best_j = cj;
                }
            }
        }
        let size = sizes[best_i] + sizes[best_j];
        linkage_matrix.push((best_i, best_j, best_dist, size));
        // Drain best_j into best_i
        let drained: Vec<usize> = clusters[best_j].drain(..).collect();
        clusters[best_i].extend(drained);
        sizes[best_i] = size;
        active[best_j] = false;
    }

    let mut labels = vec![0usize; n];
    let active_ids: Vec<usize> = (0..clusters.len())
        .filter(|&i| active[i] && !clusters[i].is_empty())
        .collect();
    for (label, &cid) in active_ids.iter().enumerate() {
        for &idx in &clusters[cid] {
            labels[idx] = label;
        }
    }
    AgglomerativeResult {
        labels,
        n_clusters: active_ids.len(),
        linkage_matrix,
    }
}

fn compute_linkage(a: &[usize], b: &[usize], dist: &[Vec<f64>], linkage: Linkage) -> f64 {
    match linkage {
        Linkage::Single => {
            let mut min_d = f64::INFINITY;
            for &ai in a {
                for &bi in b {
                    if dist[ai][bi] < min_d {
                        min_d = dist[ai][bi];
                    }
                }
            }
            min_d
        }
        Linkage::Complete => {
            let mut max_d = 0.0;
            for &ai in a {
                for &bi in b {
                    if dist[ai][bi] > max_d {
                        max_d = dist[ai][bi];
                    }
                }
            }
            max_d
        }
        Linkage::Average => {
            let mut sum = 0.0;
            let mut count = 0;
            for &ai in a {
                for &bi in b {
                    sum += dist[ai][bi];
                    count += 1;
                }
            }
            sum / count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters() {
        let mut x: Vec<Vec<f64>> = Vec::new();
        for i in 0..5 {
            x.push(vec![i as f64, 0.0]);
        }
        for i in 0..5 {
            x.push(vec![i as f64 + 100.0, 0.0]);
        }
        let r = agglomerative(&x, 2, Linkage::Single);
        assert_eq!(r.n_clusters, 2);
        assert_eq!(r.labels[0], r.labels[3]);
        assert_ne!(r.labels[0], r.labels[7]);
    }

    #[test]
    fn single_cluster() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let r = agglomerative(&x, 1, Linkage::Complete);
        assert_eq!(r.n_clusters, 1);
    }
}
