/// Encodes values as contiguous integer labels starting from 0.
#[must_use]
pub fn label_encode(values: &[f64]) -> Vec<f64> {
    let mut unique: Vec<f64> = values.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    values
        .iter()
        .map(|v| {
            unique
                .iter()
                .position(|u| (*u - v).abs() < 1e-10)
                .unwrap_or(0) as f64
        })
        .collect()
}

/// Encodes values by their position in the specified order, or -1 if not found.
#[must_use]
pub fn ordinal_encode(values: &[f64], order: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|v| {
            order
                .iter()
                .position(|o| (*o - *v).abs() < 1e-10)
                .map(|i| i as f64)
                .unwrap_or(-1.0)
        })
        .collect()
}

/// Replaces NaN values with column means.
pub fn impute_mean(x: &mut [Vec<f64>]) {
    if x.is_empty() {
        return;
    }
    let n_cols = x[0].len();
    for col in 0..n_cols {
        let mut sum = 0.0;
        let mut count = 0;
        for row in x.iter() {
            if !row[col].is_nan() {
                sum += row[row.len().min(col)];
                count += 1;
            }
        }
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        for row in x.iter_mut() {
            let c = col.min(row.len() - 1);
            if row[c].is_nan() {
                row[c] = mean;
            }
        }
    }
}

/// Replaces NaN values with column medians.
pub fn impute_median(x: &mut [Vec<f64>]) {
    if x.is_empty() {
        return;
    }
    let n_cols = x[0].len();
    for col in 0..n_cols {
        let mut vals: Vec<f64> = x
            .iter()
            .filter(|row| !row[col.min(row.len() - 1)].is_nan())
            .map(|row| row[col.min(row.len() - 1)])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if vals.is_empty() {
            0.0
        } else if vals.len().is_multiple_of(2) {
            (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
        } else {
            vals[vals.len() / 2]
        };
        for row in x.iter_mut() {
            let c = col.min(row.len() - 1);
            if row[c].is_nan() {
                row[c] = median;
            }
        }
    }
}

/// Replaces NaN values with a constant value.
pub fn impute_constant(x: &mut [Vec<f64>], val: f64) {
    for row in x.iter_mut() {
        for cell in row.iter_mut() {
            if cell.is_nan() {
                *cell = val;
            }
        }
    }
}

/// Applies power transformation using "yeo-johnson" or "box-cox" method.
#[must_use]
pub fn power_transform(x: &[Vec<f64>], method: &str) -> Vec<Vec<f64>> {
    match method {
        "yeo-johnson" => yeo_johnson(x),
        "box-cox" => box_cox(x),
        _ => x.to_vec(),
    }
}

fn yeo_johnson(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .map(|&v| {
                    if v >= 0.0 {
                        (v + 1.0).ln()
                    } else {
                        -((-v + 1.0).ln())
                    }
                })
                .collect()
        })
        .collect()
}

fn box_cox(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .map(|&v| if v > 0.0 { v.ln() } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Transforms features to uniform distribution using rank-based quantile mapping.
#[must_use]
pub fn quantile_transform(x: &[Vec<f64>], _n_quantiles: usize) -> Vec<Vec<f64>> {
    if x.is_empty() {
        return Vec::new();
    }
    let n_cols = x[0].len();
    let n = x.len();

    (0..n_cols)
        .map(|col| {
            let mut vals: Vec<(usize, f64)> =
                x.iter().enumerate().map(|(i, row)| (i, row[col])).collect();
            vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut result = vec![0.0; n];
            for (rank, &(idx, _)) in vals.iter().enumerate() {
                result[idx] = rank as f64 / (n - 1).max(1) as f64;
            }
            result
        })
        .collect::<Vec<Vec<f64>>>()
        .into_iter()
        .collect()
}

/// Quantile transform returning the same shape as input (fixed version).
#[must_use]
pub fn quantile_transform_fixed(x: &[Vec<f64>], _n_quantiles: usize) -> Vec<Vec<f64>> {
    if x.is_empty() {
        return Vec::new();
    }
    let n_cols = x[0].len();
    let n = x.len();

    let mut result = vec![vec![0.0; n_cols]; n];
    for col in 0..n_cols {
        let mut vals: Vec<(usize, f64)> =
            x.iter().enumerate().map(|(i, row)| (i, row[col])).collect();
        vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (rank, &(idx, _)) in vals.iter().enumerate() {
            result[idx][col] = rank as f64 / (n - 1).max(1) as f64;
        }
    }
    result
}

/// Scales features using median and IQR, returning (scaled, medians, iqrs).
#[must_use]
pub fn robust_scale(x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n_cols = x[0].len();
    let n = x.len();

    let mut medians = Vec::with_capacity(n_cols);
    let mut iqrs = Vec::with_capacity(n_cols);

    for col in 0..n_cols {
        let mut vals: Vec<f64> = x.iter().map(|row| row[col]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if n.is_multiple_of(2) {
            (vals[n / 2 - 1] + vals[n / 2]) / 2.0
        } else {
            vals[n / 2]
        };
        medians.push(median);

        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let iqr = vals[q3_idx] - vals[q1_idx];
        iqrs.push(iqr.max(1e-10));
    }

    let scaled: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(col, &v)| (v - medians[col]) / iqrs[col])
                .collect()
        })
        .collect();

    (scaled, medians, iqrs)
}

/// Normalizes each row to unit L1 norm (sum of absolute values = 1).
#[must_use]
pub fn normalize_l1(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let norm: f64 = row.iter().map(|v| v.abs()).sum();
            if norm < 1e-10 {
                row.clone()
            } else {
                row.iter().map(|v| v / norm).collect()
            }
        })
        .collect()
}

/// Normalizes each row to unit L2 norm (Euclidean length = 1).
#[must_use]
pub fn normalize_l2(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-10 {
                row.clone()
            } else {
                row.iter().map(|v| v / norm).collect()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{one_hot_decode, one_hot_encode};

    #[test]
    fn test_one_hot() {
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
    fn test_label_encode() {
        let values = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        let encoded = label_encode(&values);
        assert_eq!(encoded, vec![2.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_ordinal_encode() {
        let values = vec![1.0, 3.0, 2.0];
        let order = vec![1.0, 2.0, 3.0];
        let encoded = ordinal_encode(&values, &order);
        assert_eq!(encoded, vec![0.0, 2.0, 1.0]);
    }

    #[test]
    fn test_impute_mean() {
        let mut x = vec![vec![1.0, f64::NAN], vec![3.0, 4.0], vec![5.0, 6.0]];
        impute_mean(&mut x);
        assert!((x[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_median() {
        let mut x = vec![vec![1.0, f64::NAN], vec![3.0, 4.0], vec![5.0, 6.0]];
        impute_median(&mut x);
        assert!((x[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_constant() {
        let mut x = vec![vec![1.0, f64::NAN], vec![f64::NAN, 2.0]];
        impute_constant(&mut x, -1.0);
        assert_eq!(x[0][1], -1.0);
        assert_eq!(x[1][0], -1.0);
    }

    #[test]
    fn test_power_transform() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let yj = power_transform(&x, "yeo-johnson");
        assert!((yj[0][0] - 2.0_f64.ln()).abs() < 1e-10);
        let bc = power_transform(&x, "box-cox");
        assert!((bc[0][0] - 1.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_transform() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let qt = quantile_transform_fixed(&x, 4);
        assert_eq!(qt.len(), 4);
        assert!((qt[0][0]).abs() < 1e-10);
        assert!((qt[3][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_robust_scale() {
        let x = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];
        let (scaled, _medians, iqrs) = robust_scale(&x);
        assert_eq!(scaled.len(), 4);
        assert!(iqrs[0] > 0.0);
        assert!((scaled[0][0] + 1.0).abs() < 1.0);
    }

    #[test]
    fn test_normalize_l1() {
        let x = vec![vec![3.0, 4.0]];
        let normed = normalize_l1(&x);
        assert!((normed[0][0] - 3.0 / 7.0).abs() < 1e-10);
        assert!((normed[0][1] - 4.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_l2() {
        let x = vec![vec![3.0, 4.0]];
        let normed = normalize_l2(&x);
        let norm: f64 = normed[0].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
