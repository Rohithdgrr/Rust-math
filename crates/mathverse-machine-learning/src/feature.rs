//! Feature preprocessing: standardization, normalization, one-hot encoding.

/// Standardize features to zero mean, unit variance.
#[must_use]
#[inline]
pub fn standardize(x: &mut [Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let p = x[0].len();
    let n = x.len() as f64;
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p {
        means[j] = x.iter().map(|row| row[j]).sum::<f64>() / n;
        let var: f64 = x.iter().map(|row| (row[j] - means[j]).powi(2)).sum::<f64>() / n;
        stds[j] = var.sqrt().max(1e-10);
    }
    for row in x.iter_mut() {
        for j in 0..p {
            row[j] = (row[j] - means[j]) / stds[j];
        }
    }
    (means, stds)
}

/// Apply standardization using pre-computed parameters.
pub fn standardize_apply(x: &mut [Vec<f64>], means: &[f64], stds: &[f64]) {
    for row in x.iter_mut() {
        for j in 0..means.len() {
            row[j] = (row[j] - means[j]) / stds[j];
        }
    }
}

/// Min-max normalize features to [0, 1].
#[must_use]
#[inline]
pub fn min_max(x: &mut [Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let p = x[0].len();
    let mut mins = vec![f64::INFINITY; p];
    let mut maxs = vec![f64::NEG_INFINITY; p];
    for row in x.iter() {
        for j in 0..p {
            if row[j] < mins[j] {
                mins[j] = row[j];
            }
            if row[j] > maxs[j] {
                maxs[j] = row[j];
            }
        }
    }
    for row in x.iter_mut() {
        for j in 0..p {
            let range = maxs[j] - mins[j];
            row[j] = if range.abs() < 1e-10 {
                0.0
            } else {
                (row[j] - mins[j]) / range
            };
        }
    }
    (mins, maxs)
}

#[allow(dead_code)]
/// One-hot encode integer labels.
pub(crate) fn one_hot_encode(labels: &[f64], num_classes: usize) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&l| {
            let mut row = vec![0.0; num_classes];
            let idx = l as usize;
            if idx < num_classes {
                row[idx] = 1.0;
            }
            row
        })
        .collect()
}

#[allow(dead_code)]
/// Decode one-hot vectors back to labels.
pub(crate) fn one_hot_decode(encoded: &[Vec<f64>]) -> Vec<f64> {
    encoded
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as f64)
                .unwrap_or(0.0)
        })
        .collect()
}

/// Add polynomial features (degree 2: x, x², x1*x2, ...).
#[must_use]
pub fn polynomial_features(x: &[Vec<f64>], degree: usize) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let p = row.len();
            let mut features = row.clone();
            if degree >= 2 {
                for i in 0..p {
                    for j in i..p {
                        features.push(row[i] * row[j]);
                    }
                }
            }
            if degree >= 3 {
                let _base_len = features.len();
                for i in 0..p {
                    for j in i..p {
                        for k in j..p {
                            features.push(row[i] * row[j] * row[k]);
                        }
                    }
                }
            }
            features
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardize_test() {
        let mut x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (means, _stds) = standardize(&mut x);
        assert!((means[0] - 3.0).abs() < 1e-10);
        assert!((means[1] - 4.0).abs() < 1e-10);
        let mean0: f64 = x.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        assert!(mean0.abs() < 1e-10);
    }

    #[test]
    fn min_max_test() {
        let mut x = vec![vec![0.0, 10.0], vec![5.0, 20.0], vec![10.0, 30.0]];
        let _ = min_max(&mut x);
        assert!((x[0][0] - 0.0).abs() < 1e-10);
        assert!((x[2][0] - 1.0).abs() < 1e-10);
        assert!((x[0][1] - 0.0).abs() < 1e-10);
        assert!((x[2][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn one_hot_test() {
        let labels = vec![0.0, 1.0, 2.0, 1.0];
        let encoded = one_hot_encode(&labels, 3);
        assert_eq!(
            encoded,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0]
            ]
        );
        let decoded = one_hot_decode(&encoded);
        assert_eq!(decoded, labels);
    }

    #[test]
    fn polynomial_features_test() {
        let x = vec![vec![2.0, 3.0]];
        let pf = polynomial_features(&x, 2);
        // [2, 3, 4, 6, 9] → x1, x2, x1², x1*x2, x2²
        assert_eq!(pf[0].len(), 5);
        assert!((pf[0][0] - 2.0).abs() < 1e-10);
        assert!((pf[0][2] - 4.0).abs() < 1e-10);
        assert!((pf[0][3] - 6.0).abs() < 1e-10);
        assert!((pf[0][4] - 9.0).abs() < 1e-10);
    }
}
