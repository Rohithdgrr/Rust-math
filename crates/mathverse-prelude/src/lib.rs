//! MathVerse Prelude
//!
//! A single import surface for the MathVerse ecosystem.
//!
//! Flat glob re-exports of every crate caused name collisions (multiple
//! crates define `Error`, `Result`, `Vector`-like types, etc.), so this crate
//! now exposes each dependency under its own namespace module:
//!
//! ```rust
//! use mathverse_prelude::ai::Tensor;
//! use mathverse_prelude::matrix::Matrix;
//! ```
//!
//! The curated [`prelude`] module re-exports only the most common,
//! collision-free items for `use mathverse_prelude::prelude::*;`.
//!
//! The crate itself defines no math; it exists to aggregate. For a trimmed
//! dependency graph, depend on the individual crates directly instead.

/// Everything from `mathverse-algebra`, under its own namespace.
pub mod algebra {
    pub use mathverse_algebra::*;
}

/// Everything from `mathverse-ai`, under its own namespace.
pub mod ai {
    pub use mathverse_ai::*;
}

/// Everything from `mathverse-calculus`, under its own namespace.
pub mod calculus {
    pub use mathverse_calculus::*;
}

/// Everything from `mathverse-combinatorics`, under its own namespace.
pub mod combinatorics {
    pub use mathverse_combinatorics::*;
}

/// Everything from `mathverse-complex`, under its own namespace.
pub mod complex {
    pub use mathverse_complex::*;
}

/// Core error taxonomy and shared traits, under its own namespace.
pub mod core {
    pub use mathverse_core::*;
}

/// Everything from `mathverse-equations`, under its own namespace.
pub mod equations {
    pub use mathverse_equations::*;
}

/// Everything from `mathverse-graphics`, under its own namespace.
pub mod graphics {
    pub use mathverse_graphics::*;
}

/// Everything from `mathverse-graph`, under its own namespace.
pub mod graph {
    pub use mathverse_graph::*;
}

/// Everything from `mathverse-linear-algebra`, under its own namespace.
pub mod linear_algebra {
    pub use mathverse_linear_algebra::*;
}

/// Everything from `mathverse-matrix`, under its own namespace.
pub mod matrix {
    pub use mathverse_matrix::*;
}

/// Machine learning, under its own namespace.
pub mod ml {
    pub use mathverse_machine_learning::*;
}

/// ndarray interop, under its own namespace.
pub mod ndarray_interop {
    pub use mathverse_ndarray_interop::*;
}

/// Everything from `mathverse-number-theory`, under its own namespace.
pub mod number_theory {
    pub use mathverse_number_theory::*;
}

/// Numerical methods, under its own namespace.
pub mod numerical {
    pub use mathverse_numerical::*;
}

/// Parallel helpers, under its own namespace.
pub mod parallel {
    pub use mathverse_parallel::*;
}

/// Plotting, gated behind the `plot` feature flag.
#[cfg(feature = "plot")]
pub mod plot {
    pub use mathverse_plot::*;
}

/// Probability & statistics, under its own namespace.
pub mod probability {
    pub use mathverse_probability::*;
}

/// Signal processing, under its own namespace.
pub mod signal {
    pub use mathverse_signal::*;
}

/// Special functions, under its own namespace.
pub mod special {
    pub use mathverse_special::*;
}

/// Descriptive statistics, under its own namespace.
pub mod statistics {
    pub use mathverse_statistics::*;
}

/// Transforms (FFT, DCT, ...), under its own namespace.
pub mod transforms {
    pub use mathverse_transforms::*;
}

/// Trigonometry, under its own namespace.
pub mod trigonometry {
    pub use mathverse_trigonometry::*;
}

/// Vectors, under its own namespace.
pub mod vector {
    pub use mathverse_vector::*;
}

/// Zero-copy views, under its own namespace.
pub mod views {
    pub use mathverse_views::*;
}

/// Computer vision, under its own namespace.
pub mod vision {
    pub use mathverse_vision::*;
}

/// Curated set of the most commonly used, collision-free MathVerse items.
pub mod prelude {
    // Core error taxonomy
    pub use mathverse_core::error::{MathError, MathResult};

    // Fundamental numeric types
    pub use mathverse_complex::{Complex, C32, C64};
    pub use mathverse_matrix::Matrix;
    pub use mathverse_vector::Vector;

    // Tensors & learning
    pub use mathverse_ai::Tensor;
    pub use mathverse_machine_learning::pipeline::{ModelType, Pipeline, PipelineStep};

    // Algebra
    pub use mathverse_algebra::Polynomial;
}
