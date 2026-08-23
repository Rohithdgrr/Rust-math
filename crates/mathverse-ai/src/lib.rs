//! AI/ML mathematical primitives: tensors, activations, losses, optimizers, attention, layers, models.
//!
//! # Features
//!
//! - N-dimensional tensor with broadcasting, reshaping, and element-wise math
//! - Activation functions: `ReLU`, GELU, Swish, Mish, Softmax, Sigmoid, Tanh, and more
//! - Loss functions: MSE, MAE, Huber, Cross-Entropy, Binary Cross-Entropy, KL Divergence, Hinge
//! - Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC, R², Explained Variance
//! - Stateful optimizers: SGD (with momentum), Adam, `AdamW`
//! - Learning rate schedulers: Constant, Step Decay, Cosine Annealing, Linear Warmup
//! - Attention math: QKV projection, scaled dot-product attention, multi-head attention
//! - Positional encodings: sinusoidal (additive), rotary (`RoPE`)
//! - Layer/Batch/RMS normalization

#![warn(missing_docs)]
#![forbid(unsafe_code)]
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

extern crate alloc;

mod internal;

pub mod activations;
pub mod attention;
pub mod attention_adv;
pub mod autograd;
pub mod conv_transpose;
pub mod data;
pub mod generative;
pub mod layers;
pub mod losses;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod recurrent;
pub mod registry;
pub mod tensor;
pub mod vision;

pub use activations::{gelu, relu, sigmoid, softmax, swish, tanh as tanh_act};
pub use attention::{
    apply_rope, multi_head_attention, scaled_dot_product_attention, sinusoidal_encoding,
};
pub use autograd::{ComputationGraph, GradTensor};
pub use data::{Batch, DataLoader, train_test_split};
pub use layers::{BatchNorm, Dropout, LayerNorm, Linear};
pub use losses::{binary_cross_entropy_with_logits, cross_entropy, mse};
pub use metrics::{accuracy, confusion_matrix, f1};
pub use models::{Activation, MLP, Sequential, TransformerBlock};
pub use optimizers::{Adam, AdamW, LrScheduler, Schedule, Sgd};
pub use tensor::Tensor;
