//! AI/ML mathematical primitives: tensors, activations, losses, optimizers, attention, layers, models.
//!
//! # Features
//!
//! - N-dimensional tensor with broadcasting, reshaping, and element-wise math
//! - Activation functions: ReLU, GELU, Swish, Mish, Softmax, Sigmoid, Tanh, and more
//! - Loss functions: MSE, MAE, Huber, Cross-Entropy, Binary Cross-Entropy, KL Divergence, Hinge
//! - Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC AUC, R², Explained Variance
//! - Stateful optimizers: SGD (with momentum), Adam, AdamW
//! - Learning rate schedulers: Constant, Step Decay, Cosine Annealing, Linear Warmup
//! - Attention math: QKV projection, scaled dot-product attention, multi-head attention
//! - Positional encodings: sinusoidal (additive), rotary (RoPE)
//! - Layer/Batch/RMS normalization

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
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

pub mod tensor;
pub mod activations;
pub mod losses;
pub mod metrics;
pub mod optimizers;
pub mod attention;
pub mod autograd;
pub mod layers;
pub mod models;
pub mod data;
pub mod conv_transpose;
pub mod recurrent;
pub mod attention_adv;
pub mod vision;
pub mod generative;
pub mod registry;

pub use tensor::Tensor;
pub use activations::{relu, sigmoid, softmax, gelu, swish, tanh as tanh_act};
pub use losses::{mse, cross_entropy, binary_cross_entropy_with_logits};
pub use metrics::{accuracy, f1, confusion_matrix};
pub use optimizers::{Sgd, Adam, AdamW, LrScheduler, Schedule};
pub use attention::{scaled_dot_product_attention, multi_head_attention, sinusoidal_encoding, apply_rope};
pub use layers::{Linear, LayerNorm, BatchNorm, Dropout};
pub use models::{Sequential, MLP, TransformerBlock, Activation};
pub use data::{DataLoader, Batch, train_test_split};
pub use autograd::{GradTensor, add, sub, mul, div, neg, sigmoid as autograd_sigmoid, tanh as autograd_tanh, matmul as autograd_matmul, relu_op, sum as autograd_sum, mse_loss, backward, clear_graph, get_grad, node_id};
