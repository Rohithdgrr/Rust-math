//! Ensemble methods: voting classifier, stacking, blending.

type Predictor = Box<dyn Fn(&[Vec<f64>]) -> Vec<f64>>;
type TrainablePredictor = Box<dyn Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>>;
type MetaModel = dyn Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>;

/// Voting classifier: majority vote from multiple models.
pub struct VotingClassifier {
    /// Base classifiers.
    pub models: Vec<Predictor>,
    /// Weight for each classifier.
    pub weights: Vec<f64>,
}

impl Default for VotingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl VotingClassifier {
    /// Create an empty voting classifier.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a base classifier with its weight.
    pub fn add_model<F: Fn(&[Vec<f64>]) -> Vec<f64> + 'static>(&mut self, model: F, weight: f64) {
        self.models.push(Box::new(model));
        self.weights.push(weight);
    }

    /// Predict by weighted majority vote.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let all_preds: Vec<Vec<f64>> = self.models.iter().map(|m| m(x)).collect();

        (0..n)
            .map(|i| {
                let mut votes = std::collections::HashMap::new();
                for (preds, &w) in all_preds.iter().zip(&self.weights) {
                    let p = preds[i];
                    // Normalize -0.0 to 0.0 to avoid hash key collision
                    let normalized = if p == 0.0 { 0.0 } else { p };
                    *votes.entry(normalized.to_bits()).or_insert(0.0) += w;
                }
                let best = votes
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                f64::from_bits(*best.0)
            })
            .collect()
    }
}

/// Voting regressor: weighted average.
pub struct VotingRegressor {
    /// Base regressors.
    pub models: Vec<Predictor>,
    /// Weight for each regressor.
    pub weights: Vec<f64>,
}

impl Default for VotingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl VotingRegressor {
    /// Create an empty voting regressor.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a base regressor with its weight.
    pub fn add_model<F: Fn(&[Vec<f64>]) -> Vec<f64> + 'static>(&mut self, model: F, weight: f64) {
        self.models.push(Box::new(model));
        self.weights.push(weight);
    }

    /// Predict by weighted average of base regressors.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let all_preds: Vec<Vec<f64>> = self.models.iter().map(|m| m(x)).collect();
        let total_w: f64 = self.weights.iter().sum();

        (0..n)
            .map(|i| {
                let sum: f64 = all_preds
                    .iter()
                    .zip(&self.weights)
                    .map(|(preds, &w)| preds[i] * w)
                    .sum();
                sum / total_w
            })
            .collect()
    }
}

/// Blending: splits data into train/base-metamodel layers.
#[must_use]
pub fn blending(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_val: &[Vec<f64>],
    y_val: &[f64],
    base_models: &[TrainablePredictor],
    meta_model: &MetaModel,
) -> Vec<f64> {
    // Get base model predictions on validation set as meta-features
    let mut meta_features: Vec<Vec<f64>> = (0..x_val.len()).map(|_| Vec::new()).collect();
    for model in base_models {
        let preds = model(x_train, y_train, x_val);
        for (i, &p) in preds.iter().enumerate() {
            meta_features[i].push(p);
        }
    }

    // Train meta-model on base predictions
    meta_model(x_val, y_val, &meta_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voting_classifier_test() {
        let mut vc = VotingClassifier::new();
        vc.add_model(
            |x| {
                x.iter()
                    .map(|xi| if xi[0] > 0.5 { 1.0 } else { 0.0 })
                    .collect()
            },
            1.0,
        );
        vc.add_model(
            |x| {
                x.iter()
                    .map(|xi| if xi[0] > 0.5 { 1.0 } else { 0.0 })
                    .collect()
            },
            2.0,
        );
        let x = vec![vec![0.2], vec![0.8]];
        let pred = vc.predict(&x);
        assert_eq!(pred, vec![0.0, 1.0]);
    }

    #[test]
    fn voting_regressor_test() {
        let mut vr = VotingRegressor::new();
        vr.add_model(|x| x.iter().map(|xi| xi[0] * 2.0).collect(), 1.0);
        vr.add_model(|x| x.iter().map(|xi| xi[0] * 4.0).collect(), 1.0);
        let x = vec![vec![1.0]];
        let pred = vr.predict(&x);
        assert!((pred[0] - 3.0).abs() < 1e-9);
    }
}
