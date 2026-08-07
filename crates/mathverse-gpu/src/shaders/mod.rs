//! Inlined WGSL compute shader sources.

/// WGSL source for element-wise kernels (add, sub, scale, activations).
pub const ELEMENTWISE_WGSL: &str = include_str!("elementwise.wgsl");

/// WGSL source for the matrix-multiplication kernel.
pub const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
