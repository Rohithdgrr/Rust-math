//! Physics domain applications for `MathVerse`
//!
//! This crate provides physics-specific calculations and constants including:
//! - Classical mechanics (kinematics, dynamics) with compile-time unit checking
//! - Thermodynamics
//! - Electromagnetism
//! - Waves and optics
//! - Physical constants with uncertainty
//! - Special relativity
//! - Quantum mechanics
//! - Fluid dynamics
//! - Astrophysics / orbital mechanics
//! - Nuclear physics

pub mod astrophysics;
pub mod constants;
pub mod dynamics;
pub mod electromagnetism;
pub mod fluent;
pub mod fluid_dynamics;
pub mod kinematics;
pub mod mechanics;
pub mod nuclear;
pub mod quantum;
pub mod relativity;
pub mod rotational;
pub mod thermodynamics;
pub mod waves;

pub use fluent::FluentPhysicsUnits;

// Explicit re-exports for names defined by two modules: an explicit `use`
// shadows the glob re-exports below and keeps the crate root unambiguous.
pub use astrophysics::schwarzschild_radius;
pub use kinematics::VelocityQuant;

// Glob re-exports intentionally expose every public item at the crate root;
#[allow(unused_imports)]
pub use astrophysics::*;
#[allow(unused_imports)]
pub use constants::*;
#[allow(unused_imports)]
pub use dynamics::*;
#[allow(unused_imports)]
pub use electromagnetism::*;
#[allow(unused_imports)]
pub use fluid_dynamics::*;
#[allow(unused_imports)]
pub use kinematics::*;
#[allow(unused_imports)]
pub use kinematics::*;
#[allow(unused_imports)]
pub use mechanics::*;
#[allow(unused_imports)]
pub use mechanics::*;
#[allow(unused_imports)]
pub use nuclear::*;
#[allow(unused_imports)]
pub use quantum::*;
#[allow(unused_imports)]
pub use relativity::*;
#[allow(unused_imports)]
pub use rotational::*;
#[allow(unused_imports)]
pub use rotational::*;
#[allow(unused_imports)]
pub use rotational::*;
#[allow(unused_imports)]
pub use thermodynamics::*;
#[allow(unused_imports)]
pub use waves::*;
