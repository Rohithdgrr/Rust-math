//! Feature selection: variance threshold, correlation filter, mutual information.

/// Remove features with variance below threshold.
#[must_use]
pub fn variance_threshold(x: &[Vec<f64>], threshold: f64) -> (Vec<usize>, Vec<Vec<f64>>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let p = x[0].len();
    let n = x.len() as f64;
    let selected: Vec<usize> = (0..p)
        .filter(|&j| {
            let mean: f64 = x.iter().map(|xi| xi[j]).sum::<f64>() / n;
            let var: f64 = x.iter().map(|xi| (xi[j] - mean).powi(2)).sum::<f64>() / n;
            var > threshold
        })
        .collect();
    let reduced: Vec<Vec<f64>> = x
        .iter()
        .map(|xi| selected.iter().map(|&j| xi[j]).collect())
        .collect();
    (selected, reduced)
}

/// Compute Pearson correlation between two vectors.
#[must_use]
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma: f64 = a.iter().sum::<f64>() / n;
    let mb: f64 = b.iter().sum::<f64>() / n;
    let cov: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / n;
    let sa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
    let sb: f64 = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>().sqrt();
    if sa < 1e-15 || sb < 1e-15 {
        0.0
    } else {
        cov / (sa * sb / n)
    }
}

/// Remove features with correlation above threshold (keep first).
#[must_use]
pub fn correlation_filter(x: &[Vec<f64>], threshold: f64) -> (Vec<usize>, Vec<Vec<f64>>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let p = x[0].len();
    let n = x.len();
    let mut selected = vec![true; p];

    for i in 0..p {
        if !selected[i] {
            continue;
        }
        for j in (i + 1)..p {
            if !selected[j] {
                continue;
            }
            let col_i: Vec<f64> = (0..n).map(|r| x[r][i]).collect();
            let col_j: Vec<f64> = (0..n).map(|r| x[r][j]).collect();
            if pearson_correlation(&col_i, &col_j).abs() > threshold {
                selected[j] = false;
            }
        }
    }

    let indices: Vec<usize> = selected
        .iter()
        .enumerate()
        .filter(|(_, &s)| s)
        .map(|(i, _)| i)
        .collect();
    let reduced: Vec<Vec<f64>> = x
        .iter()
        .map(|xi| indices.iter().map(|&j| xi[j]).collect())
        .collect();
    (indices, reduced)
}

/// Discretize a continuous vector into bins for mutual information.
fn discretize(values: &[f64], n_bins: usize) -> Vec<usize> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-15 {
        return vec![0; values.len()];
    }
    values
        .iter()
        .map(|&v| {
            let bin = ((v - min) / range * n_bins as f64) as usize;
            bin.min(n_bins - 1)
        })
        .collect()
}

/// Mutual information between two discrete variables.
#[must_use]
pub fn mutual_information(x: &[usize], y: &[usize], n_bins_x: usize, n_bins_y: usize) -> f64 {
    let n = x.len() as f64;
    let mut joint = vec![vec![0.0f64; n_bins_y]; n_bins_x];
    let mut px = vec![0.0f64; n_bins_x];
    let mut py = vec![0.0f64; n_bins_y];

    for (&xi, &yi) in x.iter().zip(y) {
        joint[xi][yi] += 1.0;
        px[xi] += 1.0;
        py[yi] += 1.0;
    }

    let mut mi = 0.0;
    for i in 0..n_bins_x {
        for j in 0..n_bins_y {
            if joint[i][j] > 0.0 && px[i] > 0.0 && py[j] > 0.0 {
                let pxy = joint[i][j] / n;
                let p_x = px[i] / n;
                let p_y = py[j] / n;
                mi += pxy * (pxy / (p_x * p_y)).ln();
            }
        }
    }
    mi
}

/// Select top-k features by mutual information with target.
#[must_use]
pub fn select_k_best(x: &[Vec<f64>], y: &[f64], k: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let p = x[0].len();
    let n_bins = 10;
    let y_disc = discretize(y, n_bins);

    let mut scores: Vec<(usize, f64)> = (0..p)
        .map(|j| {
            let col: Vec<f64> = x.iter().map(|xi| xi[j]).collect();
            let col_disc = discretize(&col, n_bins);
            let mi = mutual_information(&col_disc, &y_disc, n_bins, n_bins);
            (j, mi)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let selected: Vec<usize> = scores.iter().take(k).map(|(i, _)| *i).collect();
    let reduced: Vec<Vec<f64>> = x
        .iter()
        .map(|xi| selected.iter().map(|&j| xi[j]).collect())
        .collect();
    (selected, reduced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_threshold_test() {
        let x = vec![
            vec![1.0, 0.0, 5.0],
            vec![2.0, 0.0, 6.0],
            vec![3.0, 0.0, 7.0],
        ];
        let (idx, red) = variance_threshold(&x, 0.01);
        assert_eq!(idx, vec![0, 2]);
        assert_eq!(red[0].len(), 2);
    }

    #[test]
    fn pearson_test() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn correlation_filter_test() {
        let x = vec![
            vec![1.0, 2.0, 10.0],
            vec![2.0, 4.0, 20.0],
            vec![3.0, 6.0, 30.0],
        ];
        let (idx, _red) = correlation_filter(&x, 0.99);
        assert!(idx.len() < 3);
    }

    #[test]
    fn mutual_information_test() {
        let x = vec![0, 0, 1, 1, 2, 2];
        let y = vec![0, 0, 1, 1, 2, 2];
        let mi = mutual_information(&x, &y, 3, 3);
        assert!(mi > 1.0); // high MI for identical
    }

    #[test]
    fn select_k_best_test() {
        let x = vec![
            vec![1.0, 0.0, 5.0],
            vec![2.0, 0.0, 6.0],
            vec![3.0, 0.0, 7.0],
            vec![4.0, 0.0, 8.0],
        ];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let (idx, red) = select_k_best(&x, &y, 2);
        assert_eq!(idx.len(), 2);
        assert_eq!(red[0].len(), 2);
    }
}
