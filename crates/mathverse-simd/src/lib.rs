//! # mathverse-simd
//!
//! Portable SIMD acceleration kernels for f64 operations.
//!
//! Provides optimized implementations of common mathematical operations
//! using manual loop unrolling and cache-friendly access patterns. These
//! kernels serve as the fallback when platform-specific SIMD (SSE/AVX/NEON)
//! is not available, and as the reference implementation for correctness.
//!
//! # Operations
//!
//! - **Arithmetic**: add, sub, mul, div, scale, negate
//! - **Reductions**: sum, dot product, max, min
//! - **Math**: exp, log, sqrt, abs, sign
//! - **Activation**: sigmoid, tanh, relu, gelu, softmax
//! - **Linear algebra**: axpy, gemv (matrix-vector), dot blocked

extern crate alloc;

pub mod arithmetic;
pub mod math;
pub mod activation;
pub mod linalg;

pub use arithmetic::*;
pub use math::*;
pub use activation::*;
pub use linalg::*;
