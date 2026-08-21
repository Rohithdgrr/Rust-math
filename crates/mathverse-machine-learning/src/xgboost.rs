//! Gradient boosted trees (XGBoost-style objective).
//!
//! Both the regressor and classifier boost trees whose splits maximise the
//! second-order approximation of the regularised objective
//!
//! ```text
//!   gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
//!   leaf weight w* = −G/(H+λ)
//! ```
//!
//! where `G`, `H` are the sums of gradients and Hessians on each side. For
//! squared-error regression `(p−y)²` this gives `g = p − y`, `h = 1`; for
//! binary logistic loss it gives `g = σ(p) − y`, `h = σ(p)(1−σ(p))`.

use mathverse_core::error::{MathError, MathResult};

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
        let child = if x[self.feature] <= self.threshold {
            self.left.as_deref()
        } else {
            self.right.as_deref()
        };
        child.map_or(self.value, |c| c.predict_single(x))
    }
}

#[derive(Debug, Clone)]
struct BoostTree {
    root: TreeNode,
}

/// Minimum rows required to consider a split.
const MIN_SPLIT_ROWS: usize = 3;

impl BoostTree {
    /// Fit one tree on gradient/Hessian pairs using the exact greedy
    /// split search.
    ///
    /// Per node the search is `O(features · n log n)` (one sort per feature)
    /// with `O(1)` gain evaluation via running prefix sums — not the naive
    /// `O(n²)` rescan — and child partitions reuse row indices directly.
    fn fit(
        x: &[Vec<f64>],
        gradients: &[f64],
        hessians: &[f64],
        max_depth: usize,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        debug_assert_eq!(x.len(), gradients.len());
        debug_assert_eq!(x.len(), hessians.len());
        let idx: Vec<usize> = (0..x.len()).collect();
        let root = build_tree_weighted(x, gradients, hessians, &idx, 0, max_depth, lambda, gamma);
        Self { root }
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.root.predict_single(xi)).collect()
    }
}

/// Recursively grow a tree over the rows listed in `idx`.
fn build_tree_weighted(
    x: &[Vec<f64>],
    gradients: &[f64],
    hessians: &[f64],
    idx: &[usize],
    depth: usize,
    max_depth: usize,
    lambda: f64,
    gamma: f64,
) -> TreeNode {
    let sum_g: f64 = idx.iter().map(|&i| gradients[i]).sum();
    let sum_h: f64 = idx.iter().map(|&i| hessians[i]).sum();
    let leaf_weight = -sum_g / (sum_h + lambda);

    if depth >= max_depth || idx.len() < MIN_SPLIT_ROWS || x.is_empty() || x[0].is_empty() {
        return TreeNode::leaf(leaf_weight);
    }

    let n_features = x[0].len();
    // Scratch buffer reused across features to avoid reallocation.
    let mut order: Vec<usize> = Vec::with_capacity(idx.len());
    let mut best_gain = 0.0f64; // splits must beat gamma, so require gain > 0
    let mut best_feature = usize::MAX;
    let mut best_threshold = 0.0f64;
    let mut best_split_pos = 0usize;

    for feature in 0..n_features {
        // Sort this node's rows by feature value (NaN values sort last and
        // are skipped by the equal-neighbour check below).
        order.clear();
        order.extend_from_slice(idx);
        order.sort_by(|&a, &b| {
            x[a][feature]
                .partial_cmp(&x[b][feature])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut left_g = 0.0f64;
        let mut left_h = 0.0f64;

        for pos in 0..order.len() - 1 {
            let i = order[pos];
            left_g += gradients[i];
            left_h += hessians[i];

            let v_lo = x[order[pos]][feature];
            let v_hi = x[order[pos + 1]][feature];
            // Skip ties: a threshold between equal values separates nothing.
            if !(v_lo < v_hi) {
                continue;
            }

            let right_g = sum_g - left_g;
            let right_h = sum_h - left_h;
            let gain = 0.5
                * (left_g * left_g / (left_h + lambda)
                    + right_g * right_g / (right_h + lambda)
                    - sum_g * sum_g / (sum_h + lambda))
                - gamma;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_threshold = 0.5 * (v_lo + v_hi);
                best_split_pos = pos + 1;
            }
        }
    }

    if best_feature == usize::MAX {
        return TreeNode::leaf(leaf_weight);
    }

    // Partition this node's rows once, in the original index space.
    let mut left_idx = Vec::with_capacity(best_split_pos);
    let mut right_idx = Vec::with_capacity(idx.len() - best_split_pos);
    for &i in idx {
        if x[i][best_feature] <= best_threshold {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    let left = build_tree_weighted(
        x,
        gradients,
        hessians,
        &left_idx,
        depth + 1,
        max_depth,
        lambda,
        gamma,
    );
    let right = build_tree_weighted(
        x,
        gradients,
        hessians,
        &right_idx,
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
        value: leaf_weight,
        is_leaf: false,
    }
}

/// Validate shared training inputs; returns row count or an error.
fn validate_fit(x: &[Vec<f64>], y: &[f64]) -> MathResult<usize> {
    if x.len() != y.len() {
        return Err(MathError::InvalidArgument(
            "boosted trees: x and y must have the same number of rows",
        ));
    }
    if y.is_empty() {
        return Err(MathError::InvalidArgument(
            "boosted trees: training data must not be empty",
        ));
    }
    if x.iter().any(|row| row.len() != x[0].len()) {
        return Err(MathError::InvalidArgument(
            "boosted trees: all rows must have the same number of features",
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(MathError::InvalidArgument(
            "boosted trees: targets must be finite",
        ));
    }
    Ok(y.len())
}

/// Gradient boosted regression tree ensemble for regression.
///
/// Uses the true second-order XGBoost objective for squared-error loss:
/// each round fits gradients `g = p − y` with unit Hessians, so leaf
/// weights are the ridge-shrunk means `w* = −ΣG / (ΣH + λ)`.
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
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidArgument`] when `x`/`y` have mismatched
    /// shapes, contain non-finite targets, or are empty.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        let n = validate_fit(x, y)?;
        self.base_score = y.iter().sum::<f64>() / n as f64;
        self.trees.clear();

        let ones = vec![1.0; n];
        let mut predictions = vec![self.base_score; n];

        for _ in 0..self.n_estimators {
            // Squared-error loss: g = p − y, h = 1.
            let gradients: Vec<f64> = predictions
                .iter()
                .zip(y.iter())
                .map(|(pi, yi)| pi - yi)
                .collect();

            let tree = BoostTree::fit(
                x,
                &gradients,
                &ones,
                self.max_depth,
                self.lambda,
                self.gamma,
            );
            let tree_preds = tree.predict(x);

            for (pi, tp) in predictions.iter_mut().zip(tree_preds.iter()) {
                *pi += self.learning_rate * tp;
            }
            self.trees.push(tree);
        }
        Ok(())
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

    /// Fit the classifier to binary-labeled training data (`y ∈ {0, 1}`).
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidArgument`] when `x`/`y` have mismatched
    /// shapes, contain non-finite targets, or are empty.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        let n = validate_fit(x, y)?;
        let pos_rate = y.iter().filter(|&&v| v > 0.5).count() as f64 / n as f64;
        let pos_rate = pos_rate.clamp(0.01, 0.99);
        self.base_score = (pos_rate / (1.0 - pos_rate)).ln();
        self.trees.clear();

        let mut raw_preds = vec![self.base_score; n];

        for _ in 0..self.n_estimators {
            let probs: Vec<f64> = raw_preds.iter().map(|&v| sigmoid(v)).collect();
            // Logistic loss: g = σ(p) − y, h = σ(p)(1 − σ(p)).
            let gradients: Vec<f64> = probs.iter().zip(y.iter()).map(|(p, yi)| p - yi).collect();
            let hessians: Vec<f64> = probs.iter().map(|p| p * (1.0 - p)).collect();

            let tree = BoostTree::fit(
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
        Ok(())
    }

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

/// Numerically stable logistic function.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xgb_regressor_fit_predict() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut model = XGBoostRegressor::new(100, 0.1, 3, 1.0, 0.0);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x);
        for (pred, target) in preds.iter().zip(y.iter()) {
            assert!((pred - target).abs() < 0.5, "pred={pred}, target={target}");
        }
    }

    #[test]
    fn xgb_regressor_multivariate() {
        // y = 2a − b: exercises the multi-feature exact split search.
        let x: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64, (19 - i) as f64])
            .collect();
        let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 - (19 - i) as f64).collect();
        let mut model = XGBoostRegressor::new(80, 0.2, 4, 1.0, 0.0);
        model.fit(&x, &y).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let p = model.predict(std::slice::from_ref(xi))[0];
            assert!((p - yi).abs() < 1.5, "input {xi:?}: pred={p}, want≈{yi}");
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
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x);
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - *t).abs() < 0.5)
            .count();
        assert!(correct >= 4, "only {correct}/6 correct: {preds:?}");
    }

    #[test]
    fn fit_validates_inputs() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0];
        assert!(XGBoostRegressor::new(5, 0.1, 2, 1.0, 0.0).fit(&x, &y).is_err());

        let y_bad = vec![f64::NAN, 1.0];
        assert!(XGBoostRegressor::new(5, 0.1, 2, 1.0, 0.0).fit(&x, &y_bad).is_err());

        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        let yy = vec![1.0, 2.0];
        assert!(XGBoostRegressor::new(5, 0.1, 2, 1.0, 0.0).fit(&ragged, &yy).is_err());
    }

    #[test]
    fn sigmoid_extremes_are_finite() {
        assert!((sigmoid(1000.0) - 1.0).abs() < 1e-12);
        assert!(sigmoid(-1000.0).abs() < 1e-12);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }
}
