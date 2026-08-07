//! Calculus: numerical differentiation, integration, vector calculus, ODEs, root finding.
//!
//! `mathverse-calculus` provides numerical methods for calculus operations,
//! from elementary derivatives and integrals to vector calculus and ODE solvers.
//!
//! # Quick Start
//!
//! ```rust
//! use mathverse_calculus::prelude::*;
//!
//! // Derivative of sin at 0 ≈ 1
//! let d = derivative(&f64::sin, 0.0);
//! assert!((d - 1.0).abs() < 1e-8);
//!
//! // Integral of sin from 0 to π ≈ 2
//! let i = integrate(&f64::sin, 0.0, core::f64::consts::PI, 1e-10);
//! assert!((i - 2.0).abs() < 1e-8);
//!
//! // Solve dy/dt = y, y(0) = 1 → y = e^t
//! let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0).solve().unwrap();
//! let y_final = sol.last().unwrap().1;
//! assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`derivative`] | Numerical derivatives (central differences, partial, nth, discrete gradient) |
//! | [`integrate`] | Numerical integration (trapezoid, Simpson, adaptive, Gaussian, Romberg, 2D) |
//! | [`ode`] | ODE solvers (Euler, midpoint, RK4) with builder API |
//! | [`root_finding`] | Root finding with auto-differentiation, plus re-exports from `mathverse-numerical` |
//! | [`vector_calculus`] | Gradient, divergence, curl, Laplacian, Jacobian, Hessian |
//!
//! # Python parity
//!
//! | Rust function | `SciPy` equivalent |
//! |---|---|
//! | `nth_derivative` | `scipy.misc.derivative` |
//! | `gaussian_quadrature` | `scipy.integrate.fixed_quad` |
//! | `romberg` | `scipy.integrate.romberg` |
//! | `integrate_2d` | `scipy.integrate.dblquad` |
//! | `OdeProblem` | `scipy.integrate.solve_ivp` |
//! | `discrete_gradient` | `numpy.gradient` |
//!
//! # Error Handling
//!
//! Fallible functions return [`MathResult<T>`] (an alias for `Result<T, MathError>`).
//! See [`mathverse_core::error`] for the full error taxonomy.

pub mod derivative;
pub mod integrate;
pub mod ode;
pub mod root_finding;
pub mod vector_calculus;

/// Common imports for convenient use.
pub mod prelude {
    pub use mathverse_core::error::{MathError, MathResult};
    pub use crate::derivative::{derivative, second_derivative, partial_derivative, nth_derivative, discrete_gradient};
    pub use crate::integrate::{trapezoid, simpson, integrate, gaussian_quadrature, romberg, integrate_2d};
    pub use crate::ode::{euler, midpoint, runge_kutta_4, runge_kutta_4_system, OdeProblem, OdeMethod};
    pub use crate::root_finding::{newton_raphson_auto, find_critical_point};
    pub use crate::vector_calculus::{gradient, divergence, curl, laplacian, jacobian, hessian, directional_derivative};
}

// Re-export commonly used items at crate root
pub use derivative::{derivative, second_derivative, partial_derivative, nth_derivative, discrete_gradient};
pub use integrate::{trapezoid, simpson, integrate, gaussian_quadrature, romberg, integrate_2d};
pub use ode::{euler, midpoint, runge_kutta_4, runge_kutta_4_system, OdeProblem, OdeMethod};
pub use root_finding::{newton_raphson_auto, find_critical_point};
pub use vector_calculus::{gradient, divergence, curl, laplacian, jacobian, hessian, directional_derivative};
