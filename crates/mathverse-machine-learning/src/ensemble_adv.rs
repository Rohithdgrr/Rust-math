use mathverse_core::error::{MathError, MathResult};
use std::f64;

#[derive(Debug, Clone)]
pub enum BaggingBase {
    DecisionTree,
    KNN { k: usize },
}

#[derive(Debug, Clone)]
struct SimpleDecisionTree {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl SimpleDecisionTree {
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n_features = x[0].len();
        let mut best_mse = f64::INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left_val = 0.0;
        let mut best_right_val = 0.0;

        for feature in 0..n_features {
            let mut vals: Vec<(f64, f64)> = x
                .iter()
                .zip(y.iter())
                .map(|(xi, &yi)| (xi[feature], yi))
                .collect();
            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for i in 0..vals.len() - 1 {
                if vals[i].0 == vals[i + 1].0 {
                    continue;
                }
                let threshold = (vals[i].0 + vals[i + 1].0) / 2.0;
                let left: Vec<f64> = vals[..=i].iter().map(|(_, y)| *y).collect();
                let right: Vec<f64> = vals[i + 1..].iter().map(|(_, y)| *y).collect();
                let left_mean: f64 = left.iter().sum::<f64>() / left.len() as f64;
                let right_mean: f64 = right.iter().sum::<f64>() / right.len() as f64;
                let mse: f64 = left
                    .iter()
                    .map(|y| (y - left_mean).powi(2))
                    .chain(right.iter().map(|y| (y - right_mean).powi(2)))
                    .sum();
                if mse < best_mse {
                    best_mse = mse;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_val = left_mean;
                    best_right_val = right_mean;
                }
            }
        }

        Self {
            feature: best_feature,
            threshold: best_threshold,
            left_value: best_left_val,
            right_value: best_right_val,
        }
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        if x[self.feature] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_single(xi)).collect()
    }
}

fn knn_predict(train_x: &[Vec<f64>], train_y: &[f64], x: &[Vec<f64>], k: usize) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            let mut dists: Vec<(f64, f64)> = train_x
                .iter()
                .zip(train_y.iter())
                .map(|(tx, &ty)| {
                    let d: f64 = xi
                        .iter()
                        .zip(tx.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (d, ty)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k = k.min(dists.len());
            let sum: f64 = dists[..k].iter().map(|(_, y)| y).sum();
            sum / k as f64
        })
        .collect()
}

fn majority_vote(labels: &[f64]) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for &l in labels {
        *counts.entry((l * 1000.0).round() as i64).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(k, _)| k as f64 / 1000.0)
        .unwrap_or(0.0)
}

fn bootstrap_sample_indices(n: usize, sample_size: usize, seed: u64) -> Vec<usize> {
    let mut rng_state = seed;
    (0..sample_size)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as usize % n
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct BaggingClassifier {
    pub n_estimators: usize,
    pub max_samples: f64,
    pub base_estimator: BaggingBase,
    trees: Vec<SimpleDecisionTree>,
    knn_data: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl BaggingClassifier {
    pub fn new(n_estimators: usize, max_samples: f64, base_estimator: BaggingBase) -> Self {
        Self {
            n_estimators,
            max_samples,
            base_estimator: base_estimator.clone(),
            trees: Vec::new(),
            knn_data: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.trees.clear();
        self.knn_data.clear();
        let n = x.len();
        let sample_size = (n as f64 * self.max_samples) as usize;

        for i in 0..self.n_estimators {
            let indices = bootstrap_sample_indices(n, sample_size, (i as u64) * 31 + 42);
            let boot_x: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

            match &self.base_estimator {
                BaggingBase::DecisionTree => {
                    self.trees.push(SimpleDecisionTree::fit(&boot_x, &boot_y));
                }
                BaggingBase::KNN { k: _ } => {
                    self.knn_data.push((boot_x, boot_y));
                }
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut all_preds: Vec<Vec<f64>> = Vec::new();

        for (i, tree) in self.trees.iter().enumerate() {
            let _ = i;
            all_preds.push(tree.predict(x));
        }

        for (train_x, train_y) in &self.knn_data {
            let k = match &self.base_estimator {
                BaggingBase::KNN { k } => *k,
                _ => 3,
            };
            all_preds.push(knn_predict(train_x, train_y, x, k));
        }

        (0..n)
            .map(|i| {
                let votes: Vec<f64> = all_preds.iter().map(|p| p[i]).collect();
                majority_vote(&votes)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct AdaBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    trees: Vec<SimpleDecisionTree>,
    alphas: Vec<f64>,
}

impl AdaBoostClassifier {
    pub fn new(n_estimators: usize, learning_rate: f64) -> Self {
        Self {
            n_estimators,
            learning_rate,
            trees: Vec::new(),
            alphas: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        let mut weights = vec![1.0 / n as f64; n];
        let y_sign: Vec<f64> = y.iter().map(|&yi| if yi > 0.5 { 1.0 } else { -1.0 }).collect();

        self.trees.clear();
        self.alphas.clear();

        for iter_i in 0..self.n_estimators {
            // Resample according to weights
            let mut rng_state = (iter_i as u64) * 31 + 42;
            let indices: Vec<usize> = (0..n)
                .map(|_| {
                    let r = {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (rng_state >> 33) as f64 / (1u64 << 31) as f64
                    };
                    let mut cumsum = 0.0;
                    let mut chosen = 0;
                    for (j, &w) in weights.iter().enumerate() {
                        cumsum += w;
                        if r <= cumsum {
                            chosen = j;
                            break;
                        }
                    }
                    chosen
                })
                .collect();

            let boot_x: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let tree = SimpleDecisionTree::fit(&boot_x, &boot_y);
            let preds = tree.predict(x);

            let err_raw: f64 = (0..n)
                .map(|i| {
                    let pred_sign = if preds[i] > 0.5 { 1.0 } else { -1.0 };
                    if (pred_sign - y_sign[i]).abs() > 0.5 { weights[i] } else { 0.0 }
                })
                .sum();
            let err = err_raw.clamp(1e-10, 1.0 - 1e-10);

            let alpha = self.learning_rate * ((1.0 - err) / err).sqrt();

            for i in 0..n {
                let pred_sign = if preds[i] > 0.5 { 1.0 } else { -1.0 };
                if (pred_sign - y_sign[i]).abs() > 0.5 {
                    weights[i] *= alpha.exp();
                } else {
                    weights[i] *= (-alpha).exp();
                }
            }

            let w_sum: f64 = weights.iter().sum();
            if w_sum > 1e-12 {
                for w in weights.iter_mut() {
                    *w /= w_sum;
                }
            }

            self.trees.push(tree);
            self.alphas.push(alpha);

            if err < 1e-6 {
                break;
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut scores = vec![0.0; n];

        for (tree, &alpha) in self.trees.iter().zip(self.alphas.iter()) {
            let preds = tree.predict(x);
            for (s, &p) in scores.iter_mut().zip(preds.iter()) {
                let sign = if p > 0.5 { 1.0 } else { -1.0 };
                *s += alpha * sign;
            }
        }

        scores
            .iter()
            .map(|&s| if s > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }
}

fn weighted_targets(y: &[f64], weights: &[f64]) -> Vec<f64> {
    // For stump fitting, return weighted version
    y.iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| yi * wi)
        .collect()
}

#[derive(Debug, Clone)]
pub enum StackingBase {
    Logistic,
    KNN,
    DecisionTree,
}

#[derive(Debug, Clone)]
pub enum StackingMeta {
    Logistic,
    Linear,
}

#[derive(Debug, Clone)]
struct SimpleLogistic {
    weights: Vec<f64>,
    bias: f64,
}

impl SimpleLogistic {
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n_features = x[0].len();
        let mut w = vec![0.0; n_features];
        let mut b = 0.0;
        let lr = 0.1;

        for _ in 0..200 {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let logit: f64 = w.iter().zip(xi.iter()).map(|(wi, xij)| wi * xij).sum::<f64>() + b;
                let pred = 1.0 / (1.0 + (-logit).exp());
                let error = pred - yi;
                for (wi, &xij) in w.iter_mut().zip(xi.iter()) {
                    *wi -= lr * error * xij / x.len() as f64;
                }
                b -= lr * error / x.len() as f64;
            }
        }
        Self { weights: w, bias: b }
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|xi| {
                let logit: f64 = self.weights.iter().zip(xi.iter()).map(|(w, x)| w * x).sum::<f64>() + self.bias;
                1.0 / (1.0 + (-logit).exp())
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct SimpleLinear {
    weights: Vec<f64>,
    bias: f64,
}

impl SimpleLinear {
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n_features = x[0].len();
        let mut w = vec![0.0; n_features];
        let mut b = 0.0;
        let lr = 0.01;

        for _ in 0..200 {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let pred: f64 = w.iter().zip(xi.iter()).map(|(wi, xij)| wi * xij).sum::<f64>() + b;
                let error = pred - yi;
                for (wi, &xij) in w.iter_mut().zip(xi.iter()) {
                    *wi -= lr * error * xij / x.len() as f64;
                }
                b -= lr * error / x.len() as f64;
            }
        }
        Self { weights: w, bias: b }
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|xi| {
                self.weights.iter().zip(xi.iter()).map(|(w, x)| w * x).sum::<f64>() + self.bias
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct StackingClassifier {
    pub base_models: Vec<StackingBase>,
    pub meta_model: StackingMeta,
    base_trained: Vec<BaseTrained>,
    meta_trained: Option<MetaTrained>,
}

#[derive(Debug, Clone)]
enum BaseTrained {
    Logistic(SimpleLogistic),
    KNN { train_x: Vec<Vec<f64>>, train_y: Vec<f64>, k: usize },
    DecisionTree(SimpleDecisionTree),
}

#[derive(Debug, Clone)]
enum MetaTrained {
    Logistic(SimpleLogistic),
    Linear(SimpleLinear),
}

impl StackingClassifier {
    pub fn new(base_models: Vec<StackingBase>, meta_model: StackingMeta) -> Self {
        Self {
            base_models,
            meta_model,
            base_trained: Vec::new(),
            meta_trained: None,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.base_trained.clear();

        // Train base models and get meta-features
        let mut meta_features: Vec<Vec<f64>> = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            meta_features.push(Vec::with_capacity(self.base_models.len()));
        }

        for base in &self.base_models {
            let trained = match base {
                StackingBase::Logistic => {
                    let model = SimpleLogistic::fit(x, y);
                    let probas = model.predict_proba(x);
                    for (mf, p) in meta_features.iter_mut().zip(probas.iter()) {
                        mf.push(*p);
                    }
                    BaseTrained::Logistic(model)
                }
                StackingBase::KNN => {
                    let k = 3.min(x.len());
                    BaseTrained::KNN {
                        train_x: x.to_vec(),
                        train_y: y.to_vec(),
                        k,
                    }
                }
                StackingBase::DecisionTree => {
                    let model = SimpleDecisionTree::fit(x, y);
                    let preds = model.predict(x);
                    for (mf, p) in meta_features.iter_mut().zip(preds.iter()) {
                        mf.push(*p);
                    }
                    BaseTrained::DecisionTree(model)
                }
            };
            self.base_trained.push(trained);
        }

        // Fill in KNN predictions
        for trained in &self.base_trained {
            if let BaseTrained::KNN { train_x, train_y, k } = trained {
                let preds = knn_predict(train_x, train_y, x, *k);
                for (mf, p) in meta_features.iter_mut().zip(preds.iter()) {
                    mf.push(*p);
                }
            }
        }

        // Train meta model
        self.meta_trained = Some(match &self.meta_model {
            StackingMeta::Logistic => MetaTrained::Logistic(SimpleLogistic::fit(&meta_features, y)),
            StackingMeta::Linear => MetaTrained::Linear(SimpleLinear::fit(&meta_features, y)),
        });
    }

    fn get_meta_features(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut meta_features: Vec<Vec<f64>> = Vec::with_capacity(x.len());
        for _ in 0..x.len() {
            meta_features.push(Vec::with_capacity(self.base_models.len()));
        }

        for trained in &self.base_trained {
            let preds = match trained {
                BaseTrained::Logistic(model) => model.predict_proba(x),
                BaseTrained::KNN { train_x, train_y, k } => knn_predict(train_x, train_y, x, *k),
                BaseTrained::DecisionTree(model) => model.predict(x),
            };
            for (mf, p) in meta_features.iter_mut().zip(preds.iter()) {
                mf.push(*p);
            }
        }
        meta_features
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let meta_features = self.get_meta_features(x);
        match self.meta_trained.as_ref().unwrap() {
            MetaTrained::Logistic(model) => model
                .predict_proba(&meta_features)
                .iter()
                .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
                .collect(),
            MetaTrained::Linear(model) => model
                .predict(&meta_features)
                .iter()
                .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn bagging_fit_predict() {
        let (x, y) = simple_data();
        let mut model = BaggingClassifier::new(10, 0.8, BaggingBase::DecisionTree);
        model.fit(&x, &y);
        let preds = model.predict(&x);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| (**p - *t).abs() < 0.5).count();
        assert!(correct >= 4, "only {correct}/6 correct");
    }

    #[test]
    fn bagging_knn() {
        let (x, y) = simple_data();
        let mut model = BaggingClassifier::new(5, 0.8, BaggingBase::KNN { k: 3 });
        model.fit(&x, &y);
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn adaboost_fit_predict() {
        let x = vec![
            vec![0.0], vec![1.0], vec![2.0], vec![3.0],
            vec![2.5], vec![3.5], vec![4.0], vec![5.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut model = AdaBoostClassifier::new(3, 0.5);
        model.fit(&x, &y);
        assert!(model.trees.len() >= 1, "should have at least 1 tree, got {}", model.trees.len());
        assert!(!model.alphas.is_empty());
    }

    #[test]
    fn stacking_fit_predict() {
        let (x, y) = simple_data();
        let mut model = StackingClassifier::new(
            vec![StackingBase::Logistic, StackingBase::DecisionTree],
            StackingMeta::Logistic,
        );
        model.fit(&x, &y);
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 6);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| (**p - *t).abs() < 0.5).count();
        assert!(correct >= 4, "only {correct}/6 correct");
    }
}
