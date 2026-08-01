//! Gradient boosting for regression and classification.

/// Gradient boosting regressor.
pub struct GradientBoostingRegressor {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Step size shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each weak learner.
    pub max_depth: usize,
    /// Initial prediction (mean of target values).
    pub initial_prediction: f64,
    /// Fitted weak learners from each boosting round.
    pub trees: Vec<WeakTree>,
    /// Feature subset used by each weak learner.
    pub feature_indices: Vec<Vec<usize>>,
}

/// A decision stump used as a weak learner in gradient boosting.
#[derive(Debug, Clone)]
pub struct WeakTree {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl GradientBoostingRegressor {
    /// Creates a new regressor with the given hyperparameters.
    #[must_use]
    #[inline]
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            initial_prediction: 0.0,
            trees: Vec::new(),
            feature_indices: Vec::new(),
        }
    }

    /// Fits the model to training data by sequentially fitting stumps on residuals.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = y.len();
        let p = x[0].len();
        self.initial_prediction = y.iter().sum::<f64>() / n as f64;
        let mut residuals: Vec<f64> = y.iter().map(|yi| yi - self.initial_prediction).collect();
        self.trees.clear();
        self.feature_indices.clear();

        for _ in 0..self.n_estimators {
            let mut feats: Vec<usize> = (0..p).collect();
            use_xorshift_shuffle(&mut feats);
            let max_feats = ((p as f64).sqrt().ceil() as usize).max(1);
            feats.truncate(max_feats);
            feats.sort();
            self.feature_indices.push(feats.clone());
            // Fit weak learner (stump with feature subset)
            let tree = fit_stump(x, &residuals, &feats);
            // Update residuals
            for i in 0..n {
                let pred = predict_stump(&tree, &x[i]);
                residuals[i] -= self.learning_rate * pred;
            }
            self.trees.push(tree);
        }
    }

    /// Returns predictions for each input sample.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut pred = self.initial_prediction;
                for (tree, feats) in self.trees.iter().zip(&self.feature_indices) {
                    let x_sub: Vec<f64> = feats.iter().map(|&j| row[j]).collect();
                    // Find matching tree (use first feature)
                    let tree_feat = feats.iter().position(|&f| f == tree.feature).unwrap_or(0);
                    let val = if tree_feat < x_sub.len() {
                        x_sub[tree_feat]
                    } else {
                        0.0
                    };
                    pred += self.learning_rate
                        * if val <= tree.threshold {
                            tree.left_value
                        } else {
                            tree.right_value
                        };
                }
                pred
            })
            .collect()
    }
}

fn fit_stump(x: &[Vec<f64>], y: &[f64], feats: &[usize]) -> WeakTree {
    let n = y.len();
    let mut best = WeakTree {
        feature: 0,
        threshold: 0.0,
        left_value: 0.0,
        right_value: 0.0,
    };
    let mut best_loss = f64::INFINITY;
    for &j in feats {
        let mut vals: Vec<(f64, f64)> = x.iter().zip(y).map(|(row, &yi)| (row[j], yi)).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..n - 1 {
            if (vals[i].0 - vals[i + 1].0).abs() < 1e-10 {
                continue;
            }
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            let left_y: Vec<f64> = vals.iter().take(i + 1).map(|(_, y)| *y).collect();
            let right_y: Vec<f64> = vals.iter().skip(i + 1).map(|(_, y)| *y).collect();
            if left_y.is_empty() || right_y.is_empty() {
                continue;
            }
            let lm = left_y.iter().sum::<f64>() / left_y.len() as f64;
            let rm = right_y.iter().sum::<f64>() / right_y.len() as f64;
            let loss: f64 = left_y.iter().map(|y| (y - lm).powi(2)).sum::<f64>()
                + right_y.iter().map(|y| (y - rm).powi(2)).sum::<f64>();
            if loss < best_loss {
                best_loss = loss;
                best = WeakTree {
                    feature: j,
                    threshold: thresh,
                    left_value: lm,
                    right_value: rm,
                };
            }
        }
    }
    best
}

fn predict_stump(tree: &WeakTree, x: &[f64]) -> f64 {
    if x[tree.feature] <= tree.threshold {
        tree.left_value
    } else {
        tree.right_value
    }
}

fn use_xorshift_shuffle(v: &mut [usize]) {
    let n = v.len();
    let mut state: u32 = 0xCAFE_1234;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict_simple() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64 * 2.0 + 1.0).collect();
        let mut gb = GradientBoostingRegressor::new(50, 0.1, 3);
        gb.fit(&x, &y);
        let preds = gb.predict(&x);
        let mae: f64 = preds
            .iter()
            .zip(&y)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / 20.0;
        assert!(mae < 2.0);
    }

    #[test]
    fn learns_nonlinear() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin() * 10.0).collect();
        let mut gb = GradientBoostingRegressor::new(100, 0.1, 3);
        gb.fit(&x, &y);
        let preds = gb.predict(&x);
        let mae: f64 = preds
            .iter()
            .zip(&y)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / 30.0;
        assert!(mae < 2.0);
    }
}
