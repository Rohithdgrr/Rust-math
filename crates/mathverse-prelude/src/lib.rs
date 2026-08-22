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

pub mod algebra {
    pub use mathverse_algebra::*;
}

pub mod ai {
    pub use mathverse_ai::*;
}

pub mod calculus {
    pub use mathverse_calculus::*;
}

pub mod combinatorics {
    pub use mathverse_combinatorics::*;
}

pub mod complex {
    pub use mathverse_complex::*;
}

pub mod core {
    pub use mathverse_core::*;
}

pub mod equations {
    pub use mathverse_equations::*;
}

pub mod graphics {
    pub use mathverse_graphics::*;
}

pub mod graph {
    pub use mathverse_graph::*;
}

pub mod linear_algebra {
    pub use mathverse_linear_algebra::*;
}

pub mod matrix {
    pub use mathverse_matrix::*;
}

pub mod ml {
    pub use mathverse_machine_learning::*;
}

pub mod ndarray_interop {
    pub use mathverse_ndarray_interop::*;
}

pub mod number_theory {
    pub use mathverse_number_theory::*;
}

pub mod numerical {
    pub use mathverse_numerical::*;
}

pub mod parallel {
    pub use mathverse_parallel::*;
}

/// Plotting, gated behind the `plot` feature flag.
#[cfg(feature = "plot")]
pub mod plot {
    pub use mathverse_plot::*;
}

pub mod probability {
    pub use mathverse_probability::*;
}

pub mod signal {
    pub use mathverse_signal::*;
}

pub mod special {
    pub use mathverse_special::*;
}

pub mod statistics {
    pub use mathverse_statistics::*;
}

pub mod transforms {
    pub use mathverse_transforms::*;
}

pub mod trigonometry {
    pub use mathverse_trigonometry::*;
}

pub mod vector {
    pub use mathverse_vector::*;
}

pub mod views {
    pub use mathverse_views::*;
}

pub mod vision {
    pub use mathverse_vision::*;
}

/// Curated set of the most commonly used, collision-free MathVerse items.
pub mod prelude {
    // Core error taxonomy
    pub use mathverse_core::error::{MathError, MathResult};

    // Fundamental numeric types
    pub use mathverse_complex::{C32, C64, Complex};
    pub use mathverse_matrix::Matrix;
    pub use mathverse_vector::Vector;

    // Tensors & learning
    pub use mathverse_ai::Tensor;
    pub use mathverse_machine_learning::pipeline::{ModelType, Pipeline, PipelineStep};

    // Algebra
    pub use mathverse_algebra::Polynomial;
}
