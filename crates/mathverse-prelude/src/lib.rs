//! MathVerse Prelude
//!
//! Re-exports commonly used items from across the MathVerse workspace.

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

pub mod prelude {
    pub use crate::*;
}
