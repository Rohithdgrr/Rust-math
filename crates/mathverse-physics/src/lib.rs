//! Physics domain applications for `MathVerse`
//!
//! This crate provides physics-specific calculations and constants including:
//! - Classical mechanics (kinematics, dynamics)
//! - Thermodynamics
//! - Electromagnetism
//! - Waves and optics
//! - Physical constants with uncertainty
//! - Special relativity
//! - Quantum mechanics
//! - Fluid dynamics
//! - Astrophysics / orbital mechanics
//! - Nuclear physics

pub mod mechanics;
pub mod thermodynamics;
pub mod electromagnetism;
pub mod waves;
pub mod constants;
pub mod relativity;
pub mod quantum;
pub mod fluid_dynamics;
pub mod astrophysics;
pub mod nuclear;

// Glob re-exports intentionally expose every public item at the crate root;
// items only referenced through the glob trigger the unused-imports lint, so
// it is allowed here.
#[allow(unused_imports)]
pub use mechanics::*;
#[allow(unused_imports)]
pub use thermodynamics::*;
#[allow(unused_imports)]
pub use electromagnetism::*;
#[allow(unused_imports)]
pub use waves::*;
#[allow(unused_imports)]
pub use constants::*;
#[allow(unused_imports)]
pub use relativity::*;
#[allow(unused_imports)]
pub use quantum::*;
#[allow(unused_imports)]
pub use fluid_dynamics::*;
#[allow(unused_imports)]
pub use astrophysics::*;
#[allow(unused_imports)]
pub use nuclear::*;
