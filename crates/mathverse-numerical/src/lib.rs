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
///
/// Evaluates `f` exactly once per iteration.
pub fn bisection(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> MathResult<f64> {
    let (mut lo, mut hi) = (a, b);
    let mut flo = f(lo);
    let fhi = f(hi);
    if flo * fhi > 0.0 {
        return Err(MathError::InvalidArgument("no sign change on bracket"));
    }
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        let fm = f(m);
        if fm == 0.0 || (hi - lo).abs() <= tol {
            return Ok(m);
        }
        if fm * flo <= 0.0 {
            hi = m;
        } else {
            lo = m;
            flo = fm;
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
///
/// For strict in-range evaluation use [`linear_interp_checked`].
#[must_use]
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

/// Checked piecewise-linear interpolation: errors on inputs outside the
/// data range instead of silently clamping.
///
/// # Errors
///
/// - [`MathError::DimensionMismatch`] if `xs` and `ys` differ in length or
///   contain fewer than 2 points.
/// - [`MathError::OutOfRange`] if `x` lies outside `[xs[0], xs[last]]`.
pub fn linear_interp_checked(xs: &[f64], ys: &[f64], x: f64) -> MathResult<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return Err(MathError::InvalidArgument(
            "linear_interp_checked requires equal-length inputs with at least 2 points",
        ));
    }
    if !(xs[0]..=xs[xs.len() - 1]).contains(&x) {
        return Err(MathError::OutOfRange);
    }
    Ok(linear_interp(xs, ys, x))
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

/// Polynomial least-squares fit via Householder QR on the Vandermonde
/// matrix. Unlike normal equations, this does not square the condition
/// number, so fits remain stable at higher degrees.
/// Returns coefficients lowest-degree first.
///
/// # Errors
///
/// - [`MathError::InvalidArgument`] if the inputs are empty, differ in
///   length, or have fewer points than `degree + 1`.
/// - [`MathError::Singular`] if the fit is rank-deficient (e.g. too few
///   distinct `x` values).
///
/// ```
/// use mathverse_numerical::least_squares_poly;
/// let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
/// let c = least_squares_poly(&xs, &ys, 2).unwrap();
/// assert!(c.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-8));
/// ```
pub fn least_squares_poly(xs: &[f64], ys: &[f64], degree: usize) -> MathResult<Vec<f64>> {
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let m = degree + 1;
    if xs.len() < m {
        return Err(MathError::InvalidArgument(
            "least_squares_poly needs at least degree + 1 points",
        ));
    }

    // Vandermonde matrix A (n × m), row i = [1, x_i, x_i², …].
    let n = xs.len();
    let mut a = Matrix::zeros(n, m);
    for (i, &x) in xs.iter().enumerate() {
        let mut p = 1.0;
        for k in 0..m {
            a.set(i, k, p);
            p *= x;
        }
    }

    // A = Q R with Q (n × n) orthogonal and R upper triangular; solve
    // R x = Qᵀ y by back substitution using the leading m × m block of R.
    let qr = a.qr()?;
    let mut qty = Vector::zeros(m);
    for k in 0..m {
        let mut s = 0.0;
        for i in 0..n {
            s += qr.q.get(i, k) * ys[i];
        }
        qty.set(k, s);
    }

    let mut coeffs = vec![0.0; m];
    for k in (0..m).rev() {
        let mut s = qty.get(k);
        for l in (k + 1)..m {
            s -= qr.r.get(k, l) * coeffs[l];
        }
        let rkk = qr.r.get(k, k);
        if rkk.abs() < 1e-12 {
            return Err(MathError::Singular);
        }
        coeffs[k] = s / rkk;
    }
    Ok(coeffs)
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

    #[test]
    fn bisection_evaluates_once_per_iteration() {
        use core::cell::Cell;
        let count = Cell::new(0u32);
        let f = |x: f64| {
            count.set(count.get() + 1);
            x * x - 2.0
        };
        let r = bisection(&f, 0.0, 2.0, 1e-12).unwrap();
        assert!((r - 2.0f64.sqrt()).abs() < 1e-10);
        // 2 bracket evaluations + one per iteration; the old code needed 3
        // evaluations per iteration.
        assert!(count.get() < 60, "too many f() calls: {}", count.get());
    }

    #[test]
    fn linear_interp_checked_rejects_extrapolation() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 2.0, 4.0];
        assert!((linear_interp_checked(&xs, &ys, 1.5).unwrap() - 3.0).abs() < 1e-12);
        assert_eq!(linear_interp_checked(&xs, &ys, -0.5), Err(MathError::OutOfRange));
        assert_eq!(linear_interp_checked(&xs, &ys, 2.5), Err(MathError::OutOfRange));
        assert!(linear_interp_checked(&[0.0], &[1.0], 0.0).is_err());
        assert!(linear_interp_checked(&[0.0, 1.0], &[1.0], 0.5).is_err());
    }

    #[test]
    fn least_squares_stable_at_high_degree() {
        // Degree 12 over [-1, 1]: normal equations (cond²) lose accuracy here;
        // QR must recover the exact coefficients.
        let xs: Vec<f64> = (-20..=20).map(|i| f64::from(i) / 20.0).collect();
        let coef_ref = [0.5, -1.0, 2.0, 0.25, -0.75, 0.1, 0.05, -0.02, 0.01, 0.004, -0.002, 0.001, 0.0005];
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                coef_ref
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| c * x.powi(k as i32))
                    .sum()
            })
            .collect();
        let c = least_squares_poly(&xs, &ys, 12).unwrap();
        for (got, want) in c.iter().zip(coef_ref) {
            assert!((got - want).abs() < 1e-6, "coef {got} vs {want}");
        }
    }

    #[test]
    fn least_squares_input_validation() {
        assert_eq!(
            least_squares_poly(&[0.0, 1.0], &[1.0], 0),
            Err(MathError::DimensionMismatch)
        );
        assert!(least_squares_poly(&[], &[], 2).is_err());
        // Fewer points than degree + 1 is underdetermined.
        assert!(least_squares_poly(&[0.0, 1.0], &[0.0, 1.0], 3).is_err());
        // Duplicate x values make the Vandermonde rank-deficient.
        assert!(least_squares_poly(&[1.0, 1.0, 1.0, 1.0], &[1.0, 2.0, 3.0, 4.0], 1).is_err());
    }
}
