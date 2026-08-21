#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::approx_constant)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::needless_range_loop)]

//! Classical machine learning: supervised, unsupervised, ensemble, model evaluation.

pub mod boosting;
/// Synthetic dataset generators for testing and benchmarking.
pub mod datasets;
pub mod dbscan;
pub mod elastic_net;
pub mod ensemble;
/// Advanced ensemble methods (bagging, AdaBoost, stacking).
pub mod ensemble_adv;
/// sklearn-style estimator/transformer traits and uniform wrappers.
pub mod estimator;
/// Cross-validation, learning curves, and bootstrap evaluation.
pub mod evaluation;
pub mod feature;
pub mod feature_selection;
pub mod forest;
/// Gaussian process regression with configurable kernels.
pub mod gaussian_process;
pub mod gmm;
pub mod hierarchical;
pub mod isolation_forest;
pub mod kmeans;
pub mod knn;
pub mod linear;
pub mod logistic;
/// Advanced metrics (MCC, kappa, log-loss, Brier score, NDCG).
pub mod metrics_adv;
pub mod model_selection;
pub mod naive_bayes;
/// Neural network layers and basic feed-forward architecture.
pub mod neural_net;
pub mod pca;
/// End-to-end ML pipeline with serialization.
pub mod pipeline;
/// Advanced preprocessing (imputation, power transforms, robust scaling).
pub mod preprocessing_adv;
pub mod svm;
pub mod tree;
mod validate;
/// Run a complete ML workflow: generate a synthetic dataset, fit a pipeline,
/// and evaluate via cross-validation.
///
/// Returns the negated mean cross-validated score (higher is better).
///
/// # Example
///
/// ```
/// use mathverse_ml::learn;
/// use mathverse_ml::pipeline::{Pipeline, PipelineStep, ModelType};
/// use mathverse_ml::datasets::make_classification;
///
/// let (x, y) = make_classification(200, 4, 2, 42);
/// let pipeline = Pipeline::new(vec![
///     PipelineStep::Standardize,
///     PipelineStep::Model(ModelType::Logistic),
/// ]);
/// let score = learn(&x, &y, &pipeline, 5);
/// println!("Negated mean cross-validated loss: {:.4}", score);
/// ```
#[must_use]
pub fn learn(
    x: &[Vec<f64>],
    y: &[f64],
    pipeline: &pipeline::Pipeline,
    k: usize,
) -> f64 {
    let scores = evaluation::cross_val_score(x, y, k, |train_x, train_y, test_x| {
        pipeline.predict(test_x)
    });
    // `cross_val_score` already yields negative MSE per fold, so the mean is
    // directly the "higher is better" aggregate.
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_classification, make_regression};
    use crate::pipeline::{Pipeline, PipelineStep, ModelType};

    #[test]
    fn test_learn_logistic() {
        let (x, y) = make_classification(100, 4, 2, 42);
        let pipeline = Pipeline::new(vec![
            PipelineStep::Standardize,
            PipelineStep::Model(ModelType::Logistic),
        ]);
        let mean_mse = learn(&x, &y, &pipeline, 5);
        assert!(mean_mse < 0.0, "Mean MSE should be negative");
    }

    #[test]
    fn test_learn_linear() {
        let (x, y) = make_regression(100, 3, 0.1, 42);
        let pipeline = Pipeline::new(vec![
            PipelineStep::Standardize,
            PipelineStep::Model(ModelType::Linear),
        ]);
        let score = learn(&x, &y, &pipeline, 5);
        assert!(score.is_finite(), "cross-validated score must be finite");
    }
}

/// XGBoost gradient boosting implementation.
pub mod xgboost;
