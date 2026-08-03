#[derive(Debug, Clone)]
struct TreeNode {
    feature: usize,
    threshold: f64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: f64,
    is_leaf: bool,
}

impl TreeNode {
    fn leaf(value: f64) -> Self {
        Self {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            is_leaf: true,
        }
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        if self.is_leaf {
            return self.value;
        }
        if x[self.feature] <= self.threshold {
            // Safety: internal nodes always have both children
            self.left.as_ref().expect("internal node must have left child").predict_single(x)
        } else {
            // Safety: internal nodes always have both children
            self.right.as_ref().expect("internal node must have right child").predict_single(x)
        }
    }
}

#[derive(Debug, Clone)]
struct BoostTree {
    root: TreeNode,
}

impl BoostTree {
    fn fit(x: &[Vec<f64>], targets: &[f64], max_depth: usize, lambda: f64, gamma: f64) -> Self {
        let root = build_tree(x, targets, 0, max_depth, lambda, gamma);
        Self { root }
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.root.predict_single(xi)).collect()
    }
}

fn build_tree(
    x: &[Vec<f64>],
    targets: &[f64],
    depth: usize,
    max_depth: usize,
    lambda: f64,
    gamma: f64,
) -> TreeNode {
    if depth >= max_depth || x.len() <= 2 {
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        return TreeNode::leaf(mean);
    }

    let n_features = x[0].len();
    let mut best_mse = f64::INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut _best_left_val = 0.0;
    let mut _best_right_val = 0.0;
    let mut best_left_idx = Vec::new();
    let mut best_right_idx = Vec::new();

    for feature in 0..n_features {
        let mut vals: Vec<(f64, f64)> = x
            .iter()
            .zip(targets.iter())
            .map(|(xi, &ti)| (xi[feature], ti))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..vals.len() - 1 {
            if (vals[i].0 - vals[i + 1].0).abs() < 1e-12 {
                continue;
            }
            let threshold = (vals[i].0 + vals[i + 1].0) / 2.0;
            let left_vals: Vec<f64> = vals[..=i].iter().map(|(_, t)| *t).collect();
            let right_vals: Vec<f64> = vals[i + 1..].iter().map(|(_, t)| *t).collect();
            let lm = left_vals.iter().sum::<f64>() / left_vals.len() as f64;
            let rm = right_vals.iter().sum::<f64>() / right_vals.len() as f64;
            let mse: f64 = left_vals.iter().map(|t| (t - lm).powi(2)).sum::<f64>()
                + right_vals.iter().map(|t| (t - rm).powi(2)).sum::<f64>()
                + lambda * (lm * lm + rm * rm);

            if mse < best_mse - gamma {
                best_mse = mse;
                best_feature = feature;
                best_threshold = threshold;
                _best_left_val = lm;
                _best_right_val = rm;
                best_left_idx = (0..x.len())
                    .filter(|&i| x[i][feature] <= threshold)
                    .collect();
                best_right_idx = (0..x.len())
                    .filter(|&i| x[i][feature] > threshold)
                    .collect();
            }
        }
    }

    if best_mse == f64::INFINITY {
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        return TreeNode::leaf(mean);
    }

    let left_x: Vec<Vec<f64>> = best_left_idx.iter().map(|&i| x[i].clone()).collect();
    let left_t: Vec<f64> = best_left_idx.iter().map(|&i| targets[i]).collect();
    let right_x: Vec<Vec<f64>> = best_right_idx.iter().map(|&i| x[i].clone()).collect();
    let right_t: Vec<f64> = best_right_idx.iter().map(|&i| targets[i]).collect();

    let left = build_tree(&left_x, &left_t, depth + 1, max_depth, lambda, gamma);
    let right = build_tree(&right_x, &right_t, depth + 1, max_depth, lambda, gamma);

    TreeNode {
        feature: best_feature,
        threshold: best_threshold,
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
        value: 0.0,
        is_leaf: false,
    }
}

/// Gradient boosted regression tree ensemble for regression.
#[derive(Debug, Clone)]
pub struct XGBoostRegressor {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Step size shrinkage per round.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// L2 regularization term on weights.
    pub lambda: f64,
    /// Minimum loss reduction required for a split.
    pub gamma: f64,
    trees: Vec<BoostTree>,
    base_score: f64,
}

impl XGBoostRegressor {
    /// Create a new regressor with the given hyperparameters.
    #[must_use]
    #[inline]
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            lambda,
            gamma,
            trees: Vec::new(),
            base_score: 0.0,
        }
    }

    /// Fit the regressor to training data using gradient boosting.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.base_score = y.iter().sum::<f64>() / y.len() as f64;
        let mut predictions = vec![self.base_score; y.len()];
        self.trees.clear();

        for _ in 0..self.n_estimators {
            let residuals: Vec<f64> = y
                .iter()
                .zip(predictions.iter())
                .map(|(yi, pi)| yi - pi)
                .collect();

            let tree = BoostTree::fit(x, &residuals, self.max_depth, self.lambda, self.gamma);
            let tree_preds = tree.predict(x);

            for (pi, tp) in predictions.iter_mut().zip(tree_preds.iter()) {
                *pi += self.learning_rate * tp;
            }
            self.trees.push(tree);
        }
    }

    /// Predict target values for the given inputs.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut result = vec![self.base_score; x.len()];
        for tree in &self.trees {
            let preds = tree.predict(x);
            for (r, p) in result.iter_mut().zip(preds.iter()) {
                *r += self.learning_rate * p;
            }
        }
        result
    }
}

/// Gradient boosted tree ensemble for binary classification.
#[derive(Debug, Clone)]
pub struct XGBoostClassifier {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Step size shrinkage per round.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// L2 regularization term on weights.
    pub lambda: f64,
    /// Minimum loss reduction required for a split.
    pub gamma: f64,
    trees: Vec<BoostTree>,
    base_score: f64,
}

impl XGBoostClassifier {
    /// Create a new classifier with the given hyperparameters.
    #[must_use]
    #[inline]
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            lambda,
            gamma,
            trees: Vec::new(),
            base_score: 0.0,
        }
    }

    /// Fit the classifier to binary-labeled training data.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let pos_rate = y.iter().filter(|&&v| v > 0.5).count() as f64 / y.len() as f64;
        let pos_rate = pos_rate.clamp(0.01, 0.99);
        self.base_score = (pos_rate / (1.0 - pos_rate)).ln();
        let mut raw_preds = vec![self.base_score; y.len()];
        self.trees.clear();

        for _ in 0..self.n_estimators {
            let probs: Vec<f64> = raw_preds.iter().map(|&v| sigmoid(v)).collect();
            let gradients: Vec<f64> = probs.iter().zip(y.iter()).map(|(p, yi)| p - yi).collect();
            let hessians: Vec<f64> = probs.iter().map(|p| p * (1.0 - p)).collect();

            let tree = BoostTree::fit_with_hessians(
                x,
                &gradients,
                &hessians,
                self.max_depth,
                self.lambda,
                self.gamma,
            );
            let tree_preds = tree.predict(x);

            for (rp, tp) in raw_preds.iter_mut().zip(tree_preds.iter()) {
                *rp += self.learning_rate * tp;
            }
            self.trees.push(tree);
        }
    }

    /// Predict class probabilities for the given inputs.
    /// Predict class probabilities for the given inputs.
    #[must_use]
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let raw = self.predict_raw(x);
        raw.iter().map(|&v| sigmoid(v)).collect()
    }

    fn predict_raw(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut result = vec![self.base_score; x.len()];
        for tree in &self.trees {
            let preds = tree.predict(x);
            for (r, p) in result.iter_mut().zip(preds.iter()) {
                *r += self.learning_rate * p;
            }
        }
        result
    }

    /// Predict binary class labels (0.0 or 1.0) for the given inputs.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.predict_proba(x)
            .iter()
            .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
            .collect()
    }
}

impl BoostTree {
    fn fit_with_hessians(
        x: &[Vec<f64>],
        gradients: &[f64],
        hessians: &[f64],
        max_depth: usize,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        let root = build_tree_weighted(x, gradients, hessians, 0, max_depth, lambda, gamma);
        Self { root }
    }
}

fn build_tree_weighted(
    x: &[Vec<f64>],
    gradients: &[f64],
    hessians: &[f64],
    depth: usize,
    max_depth: usize,
    lambda: f64,
    gamma: f64,
) -> TreeNode {
    if depth >= max_depth || x.len() <= 2 {
        let sum_g: f64 = gradients.iter().sum();
        let sum_h: f64 = hessians.iter().sum();
        let weight = -sum_g / (sum_h + lambda);
        return TreeNode::leaf(weight);
    }

    let n_features = x[0].len();
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_left_idx = Vec::new();
    let mut best_right_idx = Vec::new();

    let sum_g: f64 = gradients.iter().sum();
    let sum_h: f64 = hessians.iter().sum();

    for feature in 0..n_features {
        let mut vals: Vec<(f64, f64, f64)> = x
            .iter()
            .zip(gradients.iter())
            .zip(hessians.iter())
            .map(|((xi, &gi), &hi)| (xi[feature], gi, hi))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_g = 0.0;
        let mut left_h = 0.0;
        let mut right_g = sum_g;
        let mut right_h = sum_h;

        for i in 0..vals.len() - 1 {
            left_g += vals[i].1;
            left_h += vals[i].2;
            right_g -= vals[i].1;
            right_h -= vals[i].2;

            if (vals[i].0 - vals[i + 1].0).abs() < 1e-12 {
                continue;
            }

            let gain = 0.5
                * (left_g * left_g / (left_h + lambda) + right_g * right_g / (right_h + lambda)
                    - sum_g * sum_g / (sum_h + lambda))
                - gamma;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_threshold = (vals[i].0 + vals[i + 1].0) / 2.0;
                best_left_idx = (0..x.len())
                    .filter(|&i| x[i][feature] <= best_threshold)
                    .collect();
                best_right_idx = (0..x.len())
                    .filter(|&i| x[i][feature] > best_threshold)
                    .collect();
            }
        }
    }

    if best_gain <= 0.0 {
        let sum_g: f64 = gradients.iter().sum();
        let sum_h: f64 = hessians.iter().sum();
        return TreeNode::leaf(-sum_g / (sum_h + lambda));
    }

    let left_x: Vec<Vec<f64>> = best_left_idx.iter().map(|&i| x[i].clone()).collect();
    let left_g: Vec<f64> = best_left_idx.iter().map(|&i| gradients[i]).collect();
    let left_h: Vec<f64> = best_left_idx.iter().map(|&i| hessians[i]).collect();
    let right_x: Vec<Vec<f64>> = best_right_idx.iter().map(|&i| x[i].clone()).collect();
    let right_g: Vec<f64> = best_right_idx.iter().map(|&i| gradients[i]).collect();
    let right_h: Vec<f64> = best_right_idx.iter().map(|&i| hessians[i]).collect();

    let left = build_tree_weighted(
        &left_x,
        &left_g,
        &left_h,
        depth + 1,
        max_depth,
        lambda,
        gamma,
    );
    let right = build_tree_weighted(
        &right_x,
        &right_g,
        &right_h,
        depth + 1,
        max_depth,
        lambda,
        gamma,
    );

    TreeNode {
        feature: best_feature,
        threshold: best_threshold,
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
        value: 0.0,
        is_leaf: false,
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xgb_regressor_fit_predict() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut model = XGBoostRegressor::new(100, 0.1, 3, 1.0, 0.0);
        model.fit(&x, &y);
        let preds = model.predict(&x);
        for (pred, target) in preds.iter().zip(y.iter()) {
            assert!((pred - target).abs() < 0.5, "pred={pred}, target={target}");
        }
    }

    #[test]
    fn xgb_classifier_fit_predict() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut model = XGBoostClassifier::new(100, 0.1, 3, 1.0, 0.0);
        model.fit(&x, &y);
        let preds = model.predict(&x);
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - *t).abs() < 0.5)
            .count();
        assert!(correct >= 4, "only {correct}/6 correct: {preds:?}");
    }
}
