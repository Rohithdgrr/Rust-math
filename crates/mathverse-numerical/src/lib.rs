//! Numerical methods: root finding, ODE integration, interpolation,
//! least-squares approximation, optimization, linear solvers, eigenvalue methods.

use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

pub mod root;
pub mod ode;
pub mod interpolation;
pub mod integration;
pub mod linear_solvers;
pub mod eigenvalue;

pub use root::{secant, false_position, muller, brent, illinois, steffensen, halley, householder, fixed_point, aitken_delta_squared};
pub use ode::{ODEState, RKF45, DormandPrince, AdamsBashforth, BackwardEuler, CrankNicolson};
pub use interpolation::{CubicSpline, HermiteInterpolation, BarycentricInterpolation, RBFInterpolation, MultilinearInterpolation, ChebyshevInterpolation, NearestNeighbor};
pub use integration::{GaussianQuadrature, RombergIntegration, AdaptiveSimpson, MonteCarloIntegration, SimpsonRule, MidpointRule, BooleRule, ClenshawCurtis, DoubleExponential};
pub use linear_solvers::{Jacobi, GaussSeidel, SOR, ConjugateGradient, PreconditionedCG, GMRES, BiCGSTAB, ILUPreconditioner};
pub use eigenvalue::{PowerMethod, InversePowerMethod, RayleighQuotientIteration, QRAlgorithm, Lanczos, SubspaceIteration, JacobiEigenvalue};

/// Bisection on a bracket with a sign change; error otherwise.
pub fn bisection(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> MathResult<f64> {
    let (mut lo, mut hi) = (a, b);
    let flo = f(lo);
    let fhi = f(hi);
    if flo * fhi > 0.0 {
        return Err(MathError::InvalidArgument("no sign change on bracket"));
    }
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if f(m) == 0.0 || (hi - lo).abs() <= tol {
            return Ok(m);
        }
        if f(m) * flo <= 0.0 {
            hi = m;
        } else {
            lo = m;
        }
    }
    Err(MathError::NotConverged("bisection"))
}

/// Newton–Raphson with configurable tolerance and iteration cap.
///
/// ```
/// use mathverse_numerical::newton_raphson;
/// let root = newton_raphson(&|x| x * x - 2.0, &|x| 2.0 * x, 1.5, 1e-12, 100).unwrap();
/// assert!((root - 2.0f64.sqrt()).abs() < 1e-10);
/// ```
pub fn newton_raphson(
    f: &dyn Fn(f64) -> f64,
    fp: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x = x0;
    for _ in 0..max_iters {
        let fx = f(x);
        if fx.abs() < tol {
            return Ok(x);
        }
        let d = fp(x);
        if d == 0.0 {
            return Err(MathError::Domain);
        }
        let nx = x - fx / d;
        if (nx - x).abs() < tol {
            return Ok(nx);
        }
        x = nx;
    }
    Err(MathError::NotConverged("newton-raphson"))
}

/// One RK4 step for `dy/dt = f(t, y)`.
pub fn rk4_step(f: &dyn Fn(f64, f64) -> f64, t: f64, y: f64, h: f64) -> f64 {
    let k1 = f(t, y);
    let k2 = f(t + h / 2.0, y + h / 2.0 * k1);
    let k3 = f(t + h / 2.0, y + h / 2.0 * k2);
    let k4 = f(t + h, y + h * k3);
    y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

/// RK4 over `[t0, t1]` in `n` steps; returns `(t, y)` samples.
///
/// ```
/// use mathverse_numerical::rk4;
/// let sol = rk4(&|_, y| y, 1.0, 0.0, 1.0, 100);
/// assert!((sol.last().unwrap().1 - core::f64::consts::E).abs() < 1e-6);
/// ```
pub fn rk4(f: &dyn Fn(f64, f64) -> f64, y0: f64, t0: f64, t1: f64, n: usize) -> Vec<(f64, f64)> {
    let h = (t1 - t0) / n as f64;
    let mut out = Vec::with_capacity(n + 1);
    let (mut t, mut y) = (t0, y0);
    out.push((t, y));
    for _ in 0..n {
        y = rk4_step(f, t, y, h);
        t += h;
        out.push((t, y));
    }
    out
}

/// Piecewise-linear interpolation; clamps outside the data range.
/// `xs` must be sorted ascending.
pub fn linear_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[xs.len() - 1];
    }
    let i = xs.partition_point(|&v| v <= x) - 1;
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

/// Lagrange interpolation through `(xs, ys)`; exact on polynomials of
/// degree `< len(xs)`.
pub fn lagrange_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    xs.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let l: f64 = xs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &xj)| (x - xj) / (xi - xj))
                .product();
            l * ys[i]
        })
        .sum()
}

/// Polynomial least-squares fit (normal equations, Vandermonde).
/// Returns coefficients lowest-degree first.
///
/// ```
/// use mathverse_numerical::least_squares_poly;
/// let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
/// let c = least_squares_poly(&xs, &ys, 2).unwrap();
/// assert!(c.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-8));
/// ```
pub fn least_squares_poly(xs: &[f64], ys: &[f64], degree: usize) -> MathResult<Vec<f64>> {
    let m = degree + 1;
    let mut ata = Matrix::zeros(m, m);
    let mut aty = Vector::zeros(m);
    for (x, y) in xs.iter().zip(ys) {
        let mut p = vec![1.0; m];
        for k in 1..m {
            p[k] = p[k - 1] * x;
        }
        for k in 0..m {
            aty.set(k, aty.get(k) + p[k] * y);
            for l in 0..m {
                ata.set(k, l, ata.get(k, l) + p[k] * p[l]);
            }
        }
    }
    Ok(ata.solve(&aty)?.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_finders() {
        let r = bisection(&|x| x * x - 2.0, 0.0, 2.0, 1e-12).unwrap();
        assert!((r - 2.0f64.sqrt()).abs() < 1e-10);
        assert!(bisection(&|x| x * x + 1.0, 0.0, 2.0, 1e-12).is_err());
        // cos(x) = x
        let r = newton_raphson(&|x| x.cos() - x, &|x| -x.sin() - 1.0, 0.5, 1e-12, 50).unwrap();
        assert!((r - 0.7390851332151607).abs() < 1e-10);
        assert!(newton_raphson(&|_| 1.0, &|_| 0.0, 0.0, 1e-12, 10).is_err());
    }

    #[test]
    fn rk4_accuracy() {
        let sol = rk4(&|_, y| y, 1.0, 0.0, 1.0, 100);
        assert!((sol.last().unwrap().1 - core::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn interpolation() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 1.0, 4.0, 9.0]; // y = x²
        assert!((lagrange_interp(&xs, &ys, 1.5) - 2.25).abs() < 1e-12);
        assert!((linear_interp(&xs, &ys, 0.5) - 0.5).abs() < 1e-12);
        assert_eq!(linear_interp(&xs, &ys, -1.0), 0.0); // clamped
        assert_eq!(linear_interp(&xs, &ys, 5.0), 9.0);
    }

    #[test]
    fn least_squares() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
        let c = least_squares_poly(&xs, &ys, 2).unwrap();
        assert!(c.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-8));
        // Noisy linear data: slope ≈ 2
        let noisy: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0 + 1e-3 * (x * 100.0).sin()).collect();
        let c1 = least_squares_poly(&xs, &noisy, 1).unwrap();
        assert!((c1[1] - 2.0).abs() < 1e-3);
    }
}
