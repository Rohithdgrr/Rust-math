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

pub use mechanics::*;
pub use thermodynamics::*;
pub use electromagnetism::*;
pub use waves::*;
pub use constants::*;
pub use relativity::*;
pub use quantum::*;
pub use fluid_dynamics::*;
pub use astrophysics::*;
pub use nuclear::*;
