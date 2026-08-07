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
/// XGBoost gradient boosting implementation.
pub mod xgboost;
