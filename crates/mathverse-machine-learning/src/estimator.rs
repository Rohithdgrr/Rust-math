//! Scikit-learn-style estimator and transformer traits with uniform wrappers.
//!
//! Every model in the crate exposes its own bespoke API. This module layers a
//! single [`Estimator`] / [`Transformer`] interface on top of the existing
//! implementations so pipelines, grid search, and cross-validation can treat
//! models uniformly.

use mathverse_core::error::{MathError, MathResult};

use crate::tree::DecisionTree;
use crate::validate;

/// A fitted supervised model with `fit` / `predict`.
pub trait Estimator {
    /// Fit the model on features `x` and targets `y`.
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()>;
    /// Predict targets for new samples.
    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>>;
}

/// A classifier: an estimator that also exposes class probabilities.
pub trait Classifier: Estimator {
    /// Predict class probabilities. Row `i` sums to ~1 across classes.
    fn predict_proba(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>>;
}

/// A feature transform with `fit` / `transform` (like sklearn `TransformerMixin`).
pub trait Transformer {
    /// Fit the transform on `x`.
    fn fit(&mut self, x: &[Vec<f64>]) -> MathResult<()>;
    /// Apply a fitted transform to `x`.
    fn transform(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>>;
    /// Fit then transform in one call.
    fn fit_transform(&mut self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        self.fit(x)?;
        self.transform(x)
    }
}

const SCORE_EPS: f64 = 1e-12;

fn accuracy(pred: &[f64], actual: &[f64]) -> f64 {
    if pred.is_empty() {
        return 0.0;
    }
    pred.iter()
        .zip(actual)
        .filter(|(p, a)| (*p - *a).abs() < SCORE_EPS)
        .count() as f64
        / pred.len() as f64
}

fn r2(pred: &[f64], actual: &[f64]) -> f64 {
    if pred.is_empty() {
        return 0.0;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_res: f64 = pred.iter().zip(actual).map(|(p, a)| (p - a).powi(2)).sum();
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    if ss_tot < SCORE_EPS {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Score a fitted model under k-fold cross-validation (higher is better).
pub fn cross_val_score_trait<M: Estimator + Clone>(
    model: &M,
    x: &[Vec<f64>],
    y: &[f64],
    k: usize,
    classification: bool,
) -> MathResult<Vec<f64>> {
    if k == 0 {
        return Err(MathError::InvalidArgument("k must be at least 1"));
    }
    let n = x.len();
    if n < k {
        return Err(MathError::InvalidArgument(
            "sample count must be >= number of folds",
        ));
    }
    let fold_size = n / k;
    if fold_size == 0 {
        return Err(MathError::InvalidArgument(
            "too many folds for the sample size",
        ));
    }
    let mut scores = Vec::with_capacity(k);
    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { (i + 1) * fold_size };
        let mut train_x = Vec::with_capacity(n - (end - start));
        let mut train_y = Vec::with_capacity(n - (end - start));
        let mut test_x = Vec::with_capacity(end - start);
        for (j, xi) in x.iter().enumerate() {
            if j >= start && j < end {
                test_x.push(xi.clone());
            } else {
                train_x.push(xi.clone());
                train_y.push(y[j]);
            }
        }
        let mut candidate = (*model).clone();
        candidate.fit(&train_x, &train_y)?;
        let preds = candidate.predict(&test_x)?;
        let score = if classification {
            accuracy(&preds, &y[start..end])
        } else {
            r2(&preds, &y[start..end])
        };
        scores.push(score);
    }
    Ok(scores)
}

/// Feature importance from permuting each column and measuring score drop.
///
/// Mirrors sklearn's `permutation_importance` (inspection): the model is used
/// as-is on `x`/`y`, each feature column is randomly shuffled `n_repeats` times,
/// and the score drop relative to the unpermuted baseline is recorded per
/// feature. Returns `(mean_importance, std_importance)` per column.
pub fn permutation_importance<M: Estimator>(
    model: &M,
    x: &[Vec<f64>],
    y: &[f64],
    classification: bool,
    n_repeats: usize,
    seed: u64,
) -> MathResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() || x.len() != y.len() {
        return Err(MathError::InvalidArgument(
            "feature matrix and targets must be non-empty with matching length",
        ));
    }
    if n_repeats == 0 {
        return Err(MathError::InvalidArgument("n_repeats must be at least 1"));
    }
    let p = x[0].len();
    let baseline_preds = model.predict(x)?;
    let baseline = if classification {
        accuracy(&baseline_preds, y)
    } else {
        r2(&baseline_preds, y)
    };

    let mut means = Vec::with_capacity(p);
    let mut stds = Vec::with_capacity(p);
    for j in 0..p {
        let mut drops = Vec::with_capacity(n_repeats);
        // Derive a fresh seed per feature and advance the state across repeats
        // so every permutation is distinct (xorshift is not a PRNG unless the
        // state is carried forward).
        let mut rng_state = seed
            .wrapping_mul(0x2545_F491)
            .wrapping_add(j as u64)
            .wrapping_mul(0x9E37_79B9) as u32;
        for _ in 0..n_repeats {
            let mut permuted: Vec<Vec<f64>> = x.to_vec();
            for t in (1..permuted.len()).rev() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let k = (rng_state as usize) % (t + 1);
                let tmp = permuted[t][j];
                permuted[t][j] = permuted[k][j];
                permuted[k][j] = tmp;
            }
            let preds = model.predict(&permuted)?;
            let scored = if classification {
                accuracy(&preds, y)
            } else {
                r2(&preds, y)
            };
            drops.push(baseline - scored);
        }
        let mean = drops.iter().sum::<f64>() / drops.len() as f64;
        let variance = drops.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / drops.len() as f64;
        means.push(mean.max(0.0));
        stds.push(variance.sqrt());
    }
    Ok((means, stds))
}

/// Cross-validation grid search over explicit candidate models.
///
/// Candidates are supplied pre-built with `(name, model)` pairs rather than a
/// param dictionary, so hyperparameter typing (float vs usize vs enum) stays
/// unchecked at compile time and searching adds no machinery.
pub struct GridSearchCV<M: Estimator + Clone> {
    candidates: Vec<(String, M)>,
    folds: usize,
    classification: bool,
    scores: Vec<(String, f64)>,
    best_name: Option<String>,
    best_model: Option<M>,
    best_score: f64,
}

impl<M: Estimator + Clone> GridSearchCV<M> {
    /// Create a search. Set `classification` to score with accuracy instead of R².
    #[must_use]
    pub fn new(candidates: Vec<(String, M)>, folds: usize, classification: bool) -> Self {
        Self {
            candidates,
            folds,
            classification,
            scores: Vec::new(),
            best_name: None,
            best_model: None,
            best_score: f64::NEG_INFINITY,
        }
    }

    /// Evaluate every candidate under cross-validation and keep the best.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        if self.candidates.is_empty() {
            return Err(MathError::InvalidArgument(
                "grid search needs at least one candidate",
            ));
        }
        self.scores.clear();
        self.best_model = None;
        self.best_name = None;
        self.best_score = f64::NEG_INFINITY;
        for (name, model) in &self.candidates {
            let scores = cross_val_score_trait(model, x, y, self.folds, self.classification)?;
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            self.scores.push((name.clone(), mean));
            if mean > self.best_score {
                self.best_score = mean;
                self.best_name = Some(name.clone());
                self.best_model = Some(model.clone());
            }
        }
        Ok(())
    }

    /// Mean cross-validation score of every candidate, in input order.
    #[must_use]
    pub fn results(&self) -> &[(String, f64)] {
        &self.scores
    }

    /// Mean score of the winning candidate.
    #[must_use]
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Name of the winning candidate.
    #[must_use]
    pub fn best_params(&self) -> Option<&str> {
        self.best_name.as_deref()
    }

    /// The winning fitted model, if `fit` has been called.
    #[must_use]
    pub fn best_model(&self) -> Option<&M> {
        self.best_model.as_ref()
    }
}

/// Randomized search over explicit candidate models.
///
/// Like [`GridSearchCV`] but evaluates a seeded random subset of `n_iter`
/// candidates instead of the full list, for cheap exploration of large
/// candidate sets.
pub struct RandomizedSearchCV<M: Estimator + Clone> {
    candidates: Vec<(String, M)>,
    folds: usize,
    classification: bool,
    n_iter: usize,
    seed: u64,
    scores: Vec<(String, f64)>,
    best_name: Option<String>,
    best_model: Option<M>,
    best_score: f64,
}

/// xorshift-ish shuffle used to pick the random candidate subset.
fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

impl<M: Estimator + Clone> RandomizedSearchCV<M> {
    /// Create a search over a random subset of `n_iter` candidates.
    #[must_use]
    pub fn new(
        candidates: Vec<(String, M)>,
        folds: usize,
        classification: bool,
        n_iter: usize,
        seed: u64,
    ) -> Self {
        Self {
            candidates,
            folds,
            classification,
            n_iter,
            seed,
            scores: Vec::new(),
            best_name: None,
            best_model: None,
            best_score: f64::NEG_INFINITY,
        }
    }

    /// Evaluate the random candidate subset under cross-validation.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        if self.candidates.is_empty() {
            return Err(MathError::InvalidArgument(
                "random search needs at least one candidate",
            ));
        }
        if self.n_iter == 0 {
            return Err(MathError::InvalidArgument(
                "n_iter must be at least 1",
            ));
        }
        self.scores.clear();
        self.best_model = None;
        self.best_name = None;
        self.best_score = f64::NEG_INFINITY;
        let order = shuffle_indices(self.candidates.len(), self.seed);
        for &ci in order.iter().take(self.n_iter) {
            let (name, model) = &self.candidates[ci];
            let scores = cross_val_score_trait(model, x, y, self.folds, self.classification)?;
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            self.scores.push((name.clone(), mean));
            if mean > self.best_score {
                self.best_score = mean;
                self.best_name = Some(name.clone());
                self.best_model = Some(model.clone());
            }
        }
        Ok(())
    }

    /// Mean cross-validation score of every evaluated candidate.
    #[must_use]
    pub fn results(&self) -> &[(String, f64)] {
        &self.scores
    }

    /// Mean score of the winning candidate.
    #[must_use]
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Name of the winning candidate.
    #[must_use]
    pub fn best_params(&self) -> Option<&str> {
        self.best_name.as_deref()
    }

    /// The winning fitted model, if `fit` has been called.
    #[must_use]
    pub fn best_model(&self) -> Option<&M> {
        self.best_model.as_ref()
    }
}

/// Ordinary least-squares linear regression.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Number of features seen at fit time.
    n_features: usize,
    /// Regression coefficients (one per feature).
    coefficients: Option<Vec<f64>>,
    /// Intercept term.
    intercept: Option<f64>,
}

impl LinearRegression {
    /// Create a new (un-fitted) linear regression.
    #[must_use]
    pub fn new() -> Self {
        Self { n_features: 0, coefficients: None, intercept: None }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        self.n_features = validate::validate_xy(x, y)?;
        let result = crate::linear::fit(x, y)?;
        self.coefficients = Some(result.coefficients);
        self.intercept = Some(result.intercept);
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let (coeffs, intercept) = match (&self.coefficients, self.intercept) {
            (Some(c), Some(i)) => (c, i),
            _ => return Err(MathError::InvalidArgument("model has not been fitted")),
        };
        validate::validate_x(x, self.n_features)?;
        crate::linear::predict(x, coeffs, intercept)
    }
}

/// Ridge linear regression with L2 penalty `alpha`.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    alpha: f64,
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
}

impl RidgeRegression {
    /// Create a ridge regression with the given L2 penalty.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self { alpha, coefficients: None, intercept: None }
    }
}

impl Estimator for RidgeRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        validate::validate_xy(x, y)?;
        let result = crate::linear::fit_ridge(x, y, self.alpha)?;
        self.coefficients = Some(result.coefficients);
        self.intercept = Some(result.intercept);
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let (coeffs, intercept) = match (&self.coefficients, self.intercept) {
            (Some(c), Some(i)) => (c, i),
            _ => return Err(MathError::InvalidArgument("model has not been fitted")),
        };
        validate::validate_x(x, coeffs.len())?;
        crate::linear::predict(x, coeffs, intercept)
    }
}

/// Binary logistic regression with L2 regularization.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    lr: f64,
    max_iters: usize,
    tol: f64,
    c: f64,
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
}

impl LogisticRegression {
    /// Create a logistic regression.
    ///
    /// `c` is inverse regularization strength; use `f64::INFINITY` for none.
    #[must_use]
    pub fn new(lr: f64, max_iters: usize, tol: f64, c: f64) -> Self {
        Self { lr, max_iters, tol, c, coefficients: None, intercept: None }
    }
}

impl Estimator for LogisticRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        let result = crate::logistic::fit(x, y, self.lr, self.max_iters, self.tol, self.c)?;
        self.coefficients = Some(result.coefficients);
        self.intercept = Some(result.intercept);
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let (coeffs, intercept) = match (&self.coefficients, self.intercept) {
            (Some(c), Some(i)) => (c, i),
            _ => return Err(MathError::InvalidArgument("model has not been fitted")),
        };
        validate::validate_x(x, coeffs.len())?;
        Ok(crate::logistic::predict(x, coeffs, intercept))
    }
}

impl Classifier for LogisticRegression {
    fn predict_proba(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        let (coeffs, intercept) = match (&self.coefficients, self.intercept) {
            (Some(c), Some(i)) => (c, i),
            _ => return Err(MathError::InvalidArgument("model has not been fitted")),
        };
        validate::validate_x(x, coeffs.len())?;
        let p1 = crate::logistic::predict_proba(x, coeffs, intercept);
        Ok(p1.into_iter().map(|p| vec![1.0 - p, p]).collect())
    }
}

/// k-Nearest-Neighbors classifier (lazy: stores the training set).
#[derive(Debug, Clone)]
pub struct KNNClassifier {
    k: usize,
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
    classes: Vec<f64>,
}

impl KNNClassifier {
    /// Create a KNN classifier with `k` neighbors.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, train_x: Vec::new(), train_y: Vec::new(), classes: Vec::new() }
    }
}

impl Estimator for KNNClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        if x.is_empty() || x.len() != y.len() {
            return Err(MathError::InvalidArgument(
                "training set must be non-empty with matching target length",
            ));
        }
        if self.k == 0 || self.k > x.len() {
            return Err(MathError::InvalidArgument(
                "k must be between 1 and the training size",
            ));
        }
        self.train_x = x.to_vec();
        self.train_y = y.to_vec();
        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        if self.train_x.is_empty() {
            return Err(MathError::InvalidArgument("model has not been fitted"));
        }
        let n_features = self.train_x[0].len();
        validate::validate_x(x, n_features)?;
        crate::knn::classify(&self.train_x, &self.train_y, x, self.k)
    }
}

impl Classifier for KNNClassifier {
    fn predict_proba(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        if self.train_x.is_empty() {
            return Err(MathError::InvalidArgument("model has not been fitted"));
        }
        let n_features = self.train_x[0].len();
        validate::validate_x(x, n_features)?;
        let mut probs = Vec::with_capacity(x.len());
        for query in x {
            let mut dists: Vec<(f64, f64)> = self
                .train_x
                .iter()
                .zip(&self.train_y)
                .map(|(tx, &ty)| (crate::knn::euclidean(query, tx), ty))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut counts = vec![0.0; self.classes.len()];
            for &(_, label) in dists.iter().take(self.k) {
                if let Some(pos) = self.classes.iter().position(|&c| c == label) {
                    counts[pos] += 1.0;
                }
            }
            let row: Vec<f64> = counts.iter().map(|&c| c / self.k as f64).collect();
            probs.push(row);
        }
        Ok(probs)
    }
}

/// k-Nearest-Neighbors regressor (lazy: stores the training set).
#[derive(Debug, Clone)]
pub struct KNNRegressor {
    k: usize,
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
}

impl KNNRegressor {
    /// Create a KNN regressor with `k` neighbors.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, train_x: Vec::new(), train_y: Vec::new() }
    }
}

impl Estimator for KNNRegressor {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        if x.is_empty() || x.len() != y.len() {
            return Err(MathError::InvalidArgument(
                "training set must be non-empty with matching target length",
            ));
        }
        if self.k == 0 || self.k > x.len() {
            return Err(MathError::InvalidArgument(
                "k must be between 1 and the training size",
            ));
        }
        self.train_x = x.to_vec();
        self.train_y = y.to_vec();
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        if self.train_x.is_empty() {
            return Err(MathError::InvalidArgument("model has not been fitted"));
        }
        let n_features = self.train_x[0].len();
        validate::validate_x(x, n_features)?;
        crate::knn::regress(&self.train_x, &self.train_y, x, self.k)
    }
}

/// Decision tree classifier.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<DecisionTree>,
    classes: Vec<f64>,
    n_features: usize,
}

impl DecisionTreeClassifier {
    /// Create a decision tree classifier.
    #[must_use]
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self { max_depth, min_samples_split, tree: None, classes: Vec::new(), n_features: 0 }
    }
}

impl Estimator for DecisionTreeClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        self.n_features = validate::validate_xy(x, y)?;
        let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split);
        tree.fit(x, y);
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.tree = Some(tree);
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        match &self.tree {
            Some(tree) => {
                validate::validate_x(x, self.n_features)?;
                Ok(tree.predict(x))
            }
            None => Err(MathError::InvalidArgument("model has not been fitted")),
        }
    }
}

impl Classifier for DecisionTreeClassifier {
    fn predict_proba(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        match &self.tree {
            Some(tree) => {
                validate::validate_x(x, self.n_features)?;
                if self.classes.is_empty() {
                    return Err(MathError::InvalidArgument("model has not been fitted"));
                }
                let proba = tree.predict_proba(x, &self.classes);
                if proba.len() != x.len() {
                    return Ok(vec![vec![1.0 / self.classes.len() as f64; self.classes.len()]; x.len()]);
                }
                Ok(proba)
            }
            None => Err(MathError::InvalidArgument("model has not been fitted")),
        }
    }
}

/// Random forest classifier.
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    n_trees: usize,
    max_depth: usize,
    max_features: usize,
    forest: Option<crate::forest::RandomForest>,
    classes: Vec<f64>,
    n_features: usize,
}

impl RandomForestClassifier {
    /// Create a random forest classifier.
    #[must_use]
    pub fn new(n_trees: usize, max_depth: usize, max_features: usize) -> Self {
        Self {
            n_trees,
            max_depth,
            max_features,
            forest: None,
            classes: Vec::new(),
            n_features: 0,
        }
    }
}

impl Estimator for RandomForestClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> MathResult<()> {
        self.n_features = validate::validate_xy(x, y)?;
        let mut forest = crate::forest::RandomForest::new(self.n_trees, self.max_depth, self.max_features);
        forest.fit(x, y);
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.forest = Some(forest);
        Ok(())
    }

    fn predict(&self, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        match &self.forest {
            Some(forest) => {
                validate::validate_x(x, self.n_features)?;
                Ok(forest.predict(x))
            }
            None => Err(MathError::InvalidArgument("model has not been fitted")),
        }
    }
}

impl Classifier for RandomForestClassifier {
    fn predict_proba(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        match &self.forest {
            Some(forest) => {
                validate::validate_x(x, self.n_features)?;
                if self.classes.is_empty() {
                    return Err(MathError::InvalidArgument("model has not been fitted"));
                }
                Ok(forest.predict_proba(x, &self.classes))
            }
            None => Err(MathError::InvalidArgument("model has not been fitted")),
        }
    }
}

/// Standardization transform wrapping [`crate::preprocessing_adv::StandardScaler`].
#[derive(Debug, Clone)]
pub struct StandardScaler {
    scaler: crate::preprocessing_adv::StandardScaler,
}

impl StandardScaler {
    /// Create a standard scaler.
    #[must_use]
    pub fn new() -> Self {
        Self { scaler: crate::preprocessing_adv::StandardScaler::new() }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, x: &[Vec<f64>]) -> MathResult<()> {
        if x.is_empty() || x[0].is_empty() {
            return Err(MathError::InvalidArgument(
                "feature matrix must be non-empty",
            ));
        }
        self.scaler.fit(x);
        Ok(())
    }

    fn transform(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        if !self.scaler.is_fitted() {
            return Err(MathError::InvalidArgument("scaler has not been fitted"));
        }
        Ok(self.scaler.transform(x))
    }
}

/// One-hot encoding transform for categorical columns.
///
/// `fit` records the distinct values seen per column; `transform` expands each
/// column into one binary indicator per category. Values unseen at fit time
/// map to an all-zero row (sklearn `handle_unknown="ignore"` semantics).
/// Category lookup is O(1) via a hash map over `f64` bit patterns.
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    /// Sorted distinct categories per input column.
    categories: Vec<Vec<f64>>,
    /// For each column, category value bits -> index within `categories[j]`.
    lookup: Vec<std::collections::HashMap<u64, usize>>,
    fitted: bool,
}

impl OneHotEncoder {
    /// Create a one-hot encoder.
    #[must_use]
    pub fn new() -> Self {
        Self { categories: Vec::new(), lookup: Vec::new(), fitted: false }
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for OneHotEncoder {
    fn fit(&mut self, x: &[Vec<f64>]) -> MathResult<()> {
        if x.is_empty() || x[0].is_empty() {
            return Err(MathError::InvalidArgument(
                "feature matrix must be non-empty",
            ));
        }
        let p = x[0].len();
        let mut cats = Vec::with_capacity(p);
        let mut lookup = Vec::with_capacity(p);
        for j in 0..p {
            let mut col: Vec<f64> = x.iter().map(|row| row[j]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            col.dedup();
            if col.is_empty() {
                return Err(MathError::InvalidArgument("column has no categories"));
            }
            let map: std::collections::HashMap<u64, usize> = col
                .iter()
                .enumerate()
                .map(|(i, &v)| (if v == 0.0 { 0.0 } else { v }.to_bits(), i))
                .collect();
            lookup.push(map);
            cats.push(col);
        }
        self.categories = cats;
        self.lookup = lookup;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(MathError::InvalidArgument("encoder has not been fitted"));
        }
        let n_categories: usize = self.categories.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            let mut encoded = vec![0.0; n_categories];
            let mut offset = 0usize;
            for (j, cats) in self.categories.iter().enumerate() {
                if let Some(value) = row.get(j).copied() {
                    let value = if value == 0.0 { 0.0 } else { value };
                    if let Some(&ci) = self.lookup[j].get(&value.to_bits()) {
                        encoded[offset + ci] = 1.0;
                    }
                }
                offset += cats.len();
            }
            out.push(encoded);
        }
        Ok(out)
    }
}

/// Min-max scaling transform wrapping [`crate::preprocessing_adv::MinMaxScaler`].
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    scaler: crate::preprocessing_adv::MinMaxScaler,
}

impl MinMaxScaler {
    /// Create a min-max scaler.
    #[must_use]
    pub fn new() -> Self {
        Self { scaler: crate::preprocessing_adv::MinMaxScaler::new() }
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, x: &[Vec<f64>]) -> MathResult<()> {
        if x.is_empty() || x[0].is_empty() {
            return Err(MathError::InvalidArgument(
                "feature matrix must be non-empty",
            ));
        }
        self.scaler.fit(x);
        Ok(())
    }

    fn transform(&self, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
        if !self.scaler.is_fitted() {
            return Err(MathError::InvalidArgument("scaler has not been fitted"));
        }
        Ok(self.scaler.transform(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_regression_roundtrip() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        let preds = m.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(&y) {
            assert!((p - t).abs() < 1e-6, "expected {t}, got {p}");
        }
    }

    #[test]
    fn logistic_binary_classification() {
        let x = vec![vec![0.0], vec![0.1], vec![1.0], vec![1.1]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut m = LogisticRegression::new(0.5, 1000, 1e-8, f64::INFINITY);
        m.fit(&x, &y).unwrap();
        let preds = m.predict(&x).unwrap();
        assert_eq!(preds, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn grid_search_picks_better_knn() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let candidates = vec![
            ("k=1".to_string(), KNNClassifier::new(1)),
            ("k=3".to_string(), KNNClassifier::new(3)),
        ];
        let mut gs = GridSearchCV::new(candidates, 3, true);
        gs.fit(&x, &y).unwrap();
        assert!(gs.best_params().is_some());
        assert!(gs.best_model().is_some());
        assert!(gs.best_score() > 0.0);
    }

    #[test]
    fn randomized_search_finds_best() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let candidates = vec![
            ("k=1".to_string(), KNNClassifier::new(1)),
            ("k=2".to_string(), KNNClassifier::new(2)),
            ("k=3".to_string(), KNNClassifier::new(3)),
        ];
        let mut rs = RandomizedSearchCV::new(candidates, 3, true, 2, 42);
        rs.fit(&x, &y).unwrap();
        assert!(rs.best_params().is_some());
        assert!(rs.best_score() > 0.0);
        assert_eq!(rs.results().len(), 2);
    }

    #[test]
    fn scaler_transform_matches_manual() {
        let x = vec![vec![1.0, 10.0], vec![3.0, 30.0]];
        let mut s = StandardScaler::new();
        let out = s.fit_transform(&x).unwrap();
        assert!((out[0][0] + out[1][0]).abs() < 1e-9);
        assert!((out[0][1] + out[1][1]).abs() < 1e-9);
    }

    #[test]
    fn one_hot_encoder_expands_columns() {
        let x = vec![vec![0.0, 10.0], vec![1.0, 20.0], vec![0.0, 30.0], vec![2.0, 40.0]];
        let mut enc = OneHotEncoder::new();
        let out = enc.fit_transform(&x).unwrap();
        // Col 0 has cats {0,1,2} -> 3 cols; col 1 has 4 cats -> 4 cols; total 7.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 7);
        // Row 0: cat 0 in col0, cat 10 in col1.
        assert_eq!(out[0], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        // Unknown value 2.0 in col0 maps to active cat index 2.
        assert_eq!(out[3], vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        // Unseen category maps to all-zero indicators.
        let unknown = enc.transform(&[vec![99.0, 99.0]]).unwrap();
        assert_eq!(unknown[0], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // -0.0 must hash like the fit category +0.0 in col0; col1 has no 0.0 cat.
        let neg_zero = enc.transform(&[vec![-0.0, -0.0]]).unwrap();
        assert_eq!(neg_zero[0], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn permutation_importance_ranks_relevant_feature() {
        let x: Vec<Vec<f64>> = (0..60)
            .map(|i| vec![i as f64, ((i * 7) as f64).sin()])
            .collect();
        let y: Vec<f64> = x.iter().map(|row| row[0]).collect(); // y = col0 exactly
        let mut m = LinearRegression::new();
        m.fit(&x, &y).unwrap();
        let (means, _stds) = permutation_importance(&m, &x, &y, false, 5, 1).unwrap();
        assert_eq!(means.len(), 2);
        assert!(means[0] > means[1] * 1.5, "relevant feature should dominate: {means:?}");
    }
}