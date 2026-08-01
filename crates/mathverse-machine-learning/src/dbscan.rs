//! DBSCAN clustering.

/// DBSCAN result.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Cluster label for each sample (-1 indicates noise).
    pub labels: Vec<i32>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// DBSCAN clustering.
/// `eps`: neighborhood radius, `min_pts`: minimum points to form a core point.
#[must_use]
pub fn dbscan(x: &[Vec<f64>], eps: f64, min_pts: usize) -> DbscanResult {
    let n = x.len();
    let mut labels = vec![-1i32; n];
    let mut visited = vec![false; n];
    let mut cluster_id = 0i32;

    // Precompute distance matrix
    let dist = |a: usize, b: usize| -> f64 {
        x[a].iter()
            .zip(&x[b])
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let neighbors: Vec<usize> = (0..n).filter(|&j| dist(i, j) <= eps).collect();
        if neighbors.len() < min_pts {
            labels[i] = -1;
            continue;
        }
        labels[i] = cluster_id;
        let mut queue: Vec<usize> = neighbors;
        let mut qi = 0;
        while qi < queue.len() {
            let j = queue[qi];
            qi += 1;
            if !visited[j] {
                visited[j] = true;
                let j_neighbors: Vec<usize> = (0..n).filter(|&k| dist(j, k) <= eps).collect();
                if j_neighbors.len() >= min_pts {
                    for &k in &j_neighbors {
                        if !queue.contains(&k) {
                            queue.push(k);
                        }
                    }
                }
            }
            if labels[j] == -1 {
                labels[j] = cluster_id;
            }
        }
        cluster_id += 1;
    }
    DbscanResult {
        labels,
        n_clusters: cluster_id as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_with_noise() {
        let mut x: Vec<Vec<f64>> = Vec::new();
        // Cluster 0
        for i in 0..10 {
            x.push(vec![i as f64, 0.0]);
        }
        // Cluster 1
        for i in 0..10 {
            x.push(vec![i as f64 + 100.0, 0.0]);
        }
        // Noise point
        x.push(vec![500.0, 500.0]);
        let r = dbscan(&x, 1.0, 3);
        assert_eq!(r.n_clusters, 2);
        assert_eq!(r.labels[20], -1);
        assert_eq!(r.labels[0], r.labels[5]);
        assert_ne!(r.labels[0], r.labels[10]);
    }

    #[test]
    fn all_noise() {
        let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 100.0]).collect();
        let r = dbscan(&x, 1.0, 3);
        assert_eq!(r.n_clusters, 0);
        assert!(r.labels.iter().all(|&l| l == -1));
    }
}
