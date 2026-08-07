//! # mathverse-gpu
//!
//! GPU-accelerated matrix and vector operations using `wgpu`.
//!
//! Provides GPU kernels for:
//! - Matrix multiplication
//! - Element-wise operations (add, sub, mul, scale)
//! - Reductions (sum, dot product)
//! - Activation functions (sigmoid, relu, tanh)
//!
//! # Examples
//!
//! ```no_run
//! use mathverse_gpu::{GpuContext, gpu_mat_mul};
//! use mathverse_matrix::Matrix;
//!
//! # async fn example() {
//! let ctx = GpuContext::new().await.unwrap();
//! let a = Matrix::ones(256, 256);
//! let b = Matrix::ones(256, 256);
//! let result = gpu_mat_mul(&ctx, &a, &b).unwrap();
//! # }
//! ```

extern crate alloc;

pub mod context;
pub mod ops;

/// Inlined WGSL shader sources used by the compute kernels.
pub mod shaders;

pub use context::GpuContext;
pub use ops::*;
