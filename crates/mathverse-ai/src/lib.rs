//! AI/ML mathematical primitives: tensors, activations, losses, optimizers, attention, layers, models.

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
