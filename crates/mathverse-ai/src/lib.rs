//! AI/ML mathematical primitives: tensors, activations, losses, optimizers, attention.
//!
//! This crate provides the numerical building blocks for neural networks:
//! - N-dimensional tensor with broadcasting
//! - Common activation functions and their derivatives
//! - Loss functions (regression + classification)
//! - Evaluation metrics
//! - Stateful optimizers (SGD, Adam, AdamW) with LR schedulers
//! - Attention math (QKV, scaled dot-product, RoPE, multi-head)

pub mod tensor;
pub mod activations;
pub mod losses;
pub mod metrics;
pub mod optimizers;
pub mod attention;

pub use tensor::Tensor;
pub use activations::{relu, sigmoid, softmax, gelu, swish, tanh as tanh_act};
pub use losses::{mse, cross_entropy, binary_cross_entropy_with_logits};
pub use metrics::{accuracy, f1, confusion_matrix};
pub use optimizers::{Sgd, Adam, AdamW, LrScheduler, Schedule};
pub use attention::{scaled_dot_product_attention, multi_head_attention, sinusoidal_encoding, apply_rope};
