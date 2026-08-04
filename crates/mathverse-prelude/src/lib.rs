//! MathVerse Prelude
//!
//! A single re-export surface for the whole MathVerse ecosystem:
//! `use mathverse_prelude::*;` brings in the public API of every workspace
//! crate (core, algebra, calculus, transforms, ML, vision, graphics, ...) at
//! once.
//!
//! The crate itself defines no math; it exists to aggregate. Re-exports are
//! unconditionally enabled except for `mathverse-plot`, which is gated behind
//! the `plot` feature flag — for a trimmed dependency graph, depend on the
//! individual crates directly instead.

pub use mathverse_core::*;
pub use mathverse_algebra::*;
pub use mathverse_calculus::*;
pub use mathverse_trigonometry::*;
pub use mathverse_statistics::*;
pub use mathverse_probability::*;
pub use mathverse_linear_algebra::*;
pub use mathverse_matrix::*;
pub use mathverse_vector::*;
pub use mathverse_complex::*;
pub use mathverse_special::*;
pub use mathverse_number_theory::*;
pub use mathverse_combinatorics::*;
pub use mathverse_graph::*;
pub use mathverse_transforms::*;
pub use mathverse_signal::*;
pub use mathverse_optimization::*;
pub use mathverse_numerical::*;
pub use mathverse_equations::*;
pub use mathverse_ai::*;
pub use mathverse_machine_learning::*;
pub use mathverse_vision::*;
pub use mathverse_graphics::*;

// New ecosystem crates (Feature 1-9)
pub use mathverse_lazy::*;
pub use mathverse_ndarray_interop::*;
pub use mathverse_serde::*;
pub use mathverse_simd::*;
pub use mathverse_parallel::*;
pub use mathverse_views::*;
pub use mathverse_wasm::*;

#[cfg(feature = "plot")]
pub use mathverse_plot::*;

/// Namespace that re-exports the entire prelude, for
/// `use mathverse_prelude::prelude::*;`.
pub mod prelude {
    pub use crate::*;
}
