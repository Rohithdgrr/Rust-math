//! Ordinary differential equation solvers.
//!
//! Provides numerical methods for solving first-order ODEs of the form dy/dt = f(t, y).

use mathverse_core::error::MathResult;

/// Forward Euler method for solving dy/dt = f(t, y).
///
/// Returns a vector of (t, y) pairs.
///
    /// ```
    /// use mathverse_calculus::ode::euler;
    /// // dy/dt = y, y(0) = 1, solution: y = e^t
    /// let result = euler(&|t, y| y, 0.0, 1.0, 1.0, 100).unwrap();
    /// let y_final = result.last().unwrap().1;
    /// assert!((y_final - 1.0_f64.exp()).abs() < 0.02);
    /// ```
    pub fn euler(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> MathResult<Vec<(f64, f64)>> {
    if steps == 0 {
        return Err(mathverse_core::error::MathError::InvalidArgument("euler: steps must be > 0"));
    }
    let dt = (t_end - t0) / steps as f64;
    let mut result = Vec::with_capacity(steps + 1);
    let mut t = t0;
    let mut y = y0;
    result.push((t, y));
    for _ in 0..steps {
        y += dt * f(t, y);
        t += dt;
        result.push((t, y));
    }
    Ok(result)
}

/// Fourth-order Runge-Kutta method (RK4) for solving dy/dt = f(t, y).
///
/// Returns a vector of (t, y) pairs. Much more accurate than Euler.
///
    /// ```
    /// use mathverse_calculus::ode::runge_kutta_4;
    /// // dy/dt = y, y(0) = 1, solution: y = e^t
    /// let result = runge_kutta_4(&|t, y| y, 0.0, 1.0, 1.0, 10).unwrap();
    /// let y_final = result.last().unwrap().1;
    /// assert!((y_final - 1.0_f64.exp()).abs() < 1e-5);
    /// ```
    pub fn runge_kutta_4(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> MathResult<Vec<(f64, f64)>> {
    if steps == 0 {
        return Err(mathverse_core::error::MathError::InvalidArgument("runge_kutta_4: steps must be > 0"));
    }
    let dt = (t_end - t0) / steps as f64;
    let mut result = Vec::with_capacity(steps + 1);
    let mut t = t0;
    let mut y = y0;
    result.push((t, y));
    for _ in 0..steps {
        let k1 = dt * f(t, y);
        let k2 = dt * f(t + dt / 2.0, y + k1 / 2.0);
        let k3 = dt * f(t + dt / 2.0, y + k2 / 2.0);
        let k4 = dt * f(t + dt, y + k3);
        y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        t += dt;
        result.push((t, y));
    }
    Ok(result)
}

/// Midpoint method (second-order Runge-Kutta) for solving dy/dt = f(t, y).
///
/// Returns a `Result` of (t, y) pairs.
///
/// ```
/// use mathverse_calculus::ode::midpoint;
/// // dy/dt = y, y(0) = 1, solution: y = e^t
/// let result = midpoint(&|t, y| y, 0.0, 1.0, 1.0, 100).unwrap();
/// let y_final = result.last().unwrap().1;
/// assert!((y_final - 1.0_f64.exp()).abs() < 0.001);
/// ```
pub fn midpoint(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> MathResult<Vec<(f64, f64)>> {
    if steps == 0 {
        return Err(mathverse_core::error::MathError::InvalidArgument("midpoint: steps must be > 0"));
    }
    let dt = (t_end - t0) / steps as f64;
    let mut result = Vec::with_capacity(steps + 1);
    let mut t = t0;
    let mut y = y0;
    result.push((t, y));
    for _ in 0..steps {
        let k = dt * f(t + dt / 2.0, y + dt * f(t, y) / 2.0);
        y += k;
        t += dt;
        result.push((t, y));
    }
    Ok(result)
}

/// System of ODEs solver using RK4.
///
/// Solves dy/dt = f(t, y) where y is a vector.
///
    /// ```
    /// use mathverse_calculus::ode::runge_kutta_4_system;
    /// // Harmonic oscillator: d²x/dt² = -x
    /// // Convert to system: dx/dt = v, dv/dt = -x
    /// let f = |t: f64, y: &[f64]| -> Vec<f64> {
    ///     vec![y[1], -y[0]]
    /// };
    /// let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * core::f64::consts::PI, 100).unwrap();
    /// let y_final = &result.last().unwrap().1;
    /// assert!((y_final[0] - 1.0).abs() < 1e-6); // Should return to initial position
    /// ```
pub fn runge_kutta_4_system(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    steps: usize,
) -> MathResult<Vec<(f64, Vec<f64>)>> {
    if steps == 0 {
        return Err(mathverse_core::error::MathError::InvalidArgument("runge_kutta_4_system: steps must be > 0"));
    }
    let dt = (t_end - t0) / steps as f64;
    let n = y0.len();
    let mut result = Vec::with_capacity(steps + 1);
    let mut t = t0;
    let mut y = y0.to_vec();
    result.push((t, y.clone()));
    for _ in 0..steps {
        let k1 = f(t, &y).iter().map(|&v| dt * v).collect::<Vec<_>>();
        let y1: Vec<f64> = y.iter().zip(k1.iter()).map(|(&yi, &k1i)| yi + k1i / 2.0).collect();
        let k2 = f(t + dt / 2.0, &y1).iter().map(|&v| dt * v).collect::<Vec<_>>();
        let y2: Vec<f64> = y.iter().zip(k2.iter()).map(|(&yi, &k2i)| yi + k2i / 2.0).collect();
        let k3 = f(t + dt / 2.0, &y2).iter().map(|&v| dt * v).collect::<Vec<_>>();
        let y3: Vec<f64> = y.iter().zip(k3.iter()).map(|(&yi, &k3i)| yi + k3i).collect();
        let k4 = f(t + dt, &y3).iter().map(|&v| dt * v).collect::<Vec<_>>();
        for i in 0..n {
            y[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        t += dt;
        result.push((t, y.clone()));
    }
    Ok(result)
}

/// Adaptive Runge–Kutta–Fehlberg (RKF45) with automatic step-size control.
///
/// Uses the embedded 4th/5th-order pair: every step computes both a 4th and
/// 5th order estimate; their difference bounds the local error, and the step
/// is rejected/retried or grown accordingly. Steps shrink near sharp features
/// (singularities, stiff transients) and grow on smooth stretches, unlike the
/// fixed-step [`euler`]/[`runge_kutta_4`] solvers.
///
/// Returns `(t, y)` samples at each accepted step. `tol` is the per-step
/// error target; `max_steps` caps total accepted steps.
///
/// # Errors
///
/// - [`MathError::InvalidArgument`] if `tol <= 0` or `t0 == t_end`.
/// - [`MathError::NotConverged`] if `max_steps` is exhausted before reaching
///   `t_end` (error cannot be met — likely a singularity).
///
/// ```
/// use mathverse_calculus::ode::rkf45;
/// // dy/dt = y, y(0) = 1 → y(1) = e
/// let sol = rkf45(&|_, y| y, 0.0, 1.0, 1.0, 1e-8, 10_000).unwrap();
/// let y_final = sol.last().unwrap().1;
/// assert!((y_final - core::f64::consts::E).abs() < 1e-6);
/// ```
pub fn rkf45(
    f: &dyn Fn(f64, f64) -> f64,
    t0: f64,
    y0: f64,
    t_end: f64,
    tol: f64,
    max_steps: usize,
) -> MathResult<Vec<(f64, f64)>> {
    if tol <= 0.0 || !tol.is_finite() {
        return Err(mathverse_core::error::MathError::InvalidArgument(
            "rkf45: tolerance must be positive and finite",
        ));
    }
    if t0 == t_end {
        return Err(mathverse_core::error::MathError::InvalidArgument(
            "rkf45: t0 must differ from t_end",
        ));
    }

    // Fehlberg tableau (6 stages, embedded orders 4 and 5).
    const A: [[f64; 5]; 5] = [
        [0.25, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ];
    const C: [f64; 5] = [0.25, 0.375, 12.0 / 13.0, 1.0, 0.5];
    // 5th-order weights (the accepted solution).
    const B5: [f64; 6] = [
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    ];
    // 4th-order weights (used only for the error estimate).
    const B4: [f64; 6] = [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0];

    let dir = (t_end - t0).signum();
    let span = (t_end - t0).abs();
    let mut h = dir * span * 1e-3; // conservative initial step
    let min_h = span * 1e-12;

    let mut result = Vec::new();
    let (mut t, mut y) = (t0, y0);
    result.push((t, y));

    for _ in 0..max_steps {
        if (t_end - t).abs() < h.abs() {
            h = t_end - t; // land exactly on t_end
            if h == 0.0 {
                return Ok(result);
            }
        }

        let mut k = [f(t, y), 0.0, 0.0, 0.0, 0.0, 0.0];
        let (y5, err);
        let mut trial = 0u32;
        loop {
            for (stage, coeffs) in A.iter().enumerate() {
                let ti = t + C[stage] * h;
                let yi = y + h * (coeffs[0] * k[0]
                    + coeffs[1] * k[1]
                    + coeffs[2] * k[2]
                    + coeffs[3] * k[3]
                    + coeffs[4] * k[4]);
                k[stage + 1] = f(ti, yi);
            }
            let y5_trial = y + h * (B5[0] * k[0]
                + B5[1] * k[1]
                + B5[2] * k[2]
                + B5[3] * k[3]
                + B5[4] * k[4]
                + B5[5] * k[5]);
            // B4/B5 agree on stage 1 (both weights are 0), so it cancels.
            let err_trial = (h * (k[0] * (B5[0] - B4[0])
                + k[2] * (B5[2] - B4[2])
                + k[3] * (B5[3] - B4[3])
                + k[4] * (B5[4] - B4[4])
                + k[5] * (B5[5] - B4[5])))
            .abs();

            if err_trial <= tol || h.abs() <= min_h {
                y5 = y5_trial;
                err = err_trial;
                break;
            }
            if trial >= 60 {
                return Err(mathverse_core::error::MathError::NotConverged(
                    "rkf45: step size collapsed below minimum",
                ));
            }
            // Reject: retry with a smaller step.
            h *= (0.9 * tol / err_trial).powf(0.2).max(0.2);
            trial += 1;
        }

        y = y5;
        t += h;
        result.push((t, y));

        // Grow the next step based on the error just achieved (bounded 5x),
        // capped at 10% of the span so we never blow past t_end.
        h *= (0.9 * tol / err.max(f64::MIN_POSITIVE)).powf(0.2).clamp(0.2, 5.0);
        if h.abs() > span * 0.1 {
            h = dir * span * 0.1;
        }
    }

    if (t - t_end).abs() > span * 1e-9 {
        return Err(mathverse_core::error::MathError::NotConverged(
            "rkf45: exceeded max_steps before reaching t_end",
        ));
    }
    Ok(result)
}

/// ODE solver methods for the [`OdeProblem`] builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Forward Euler (1st order, fixed step)
    Euler,
    /// Midpoint / RK2 (2nd order, fixed step)
    Midpoint,
    /// Classical RK4 (4th order, fixed step)
    Rk4,
}

/// Builder for ODE problems — scipy-like API.
///
/// ```
/// use mathverse_calculus::ode::{OdeProblem, OdeMethod};
/// // dy/dt = y, y(0) = 1, solve to t=1
/// let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0)
///     .method(OdeMethod::Rk4)
///     .steps(100)
///     .solve()
///     .unwrap();
/// let y_final = sol.last().unwrap().1;
/// assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);
/// ```
#[derive(Clone)]
pub struct OdeProblem<'a> {
    f: &'a dyn Fn(f64, f64) -> f64,
    t_span: (f64, f64),
    y0: f64,
    method: OdeMethod,
    steps: usize,
}

impl<'a> OdeProblem<'a> {
    /// Create a new ODE problem.
    pub fn new(f: &'a dyn Fn(f64, f64) -> f64, t_span: (f64, f64), y0: f64) -> Self {
        Self {
            f,
            t_span,
            y0,
            method: OdeMethod::Rk4,
            steps: 1000,
        }
    }

    /// Set the solver method.
    #[must_use]
    pub fn method(mut self, m: OdeMethod) -> Self {
        self.method = m;
        self
    }

    /// Set the number of steps.
    #[must_use]
    pub fn steps(mut self, n: usize) -> Self {
        self.steps = n;
        self
    }

    /// Solve the ODE.
    pub fn solve(self) -> MathResult<Vec<(f64, f64)>> {
        match self.method {
            OdeMethod::Euler => euler(self.f, self.t_span.0, self.y0, self.t_span.1, self.steps),
            OdeMethod::Midpoint => midpoint(self.f, self.t_span.0, self.y0, self.t_span.1, self.steps),
            OdeMethod::Rk4 => runge_kutta_4(self.f, self.t_span.0, self.y0, self.t_span.1, self.steps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn euler_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = euler(&|t, y| y, 0.0, 1.0, 1.0, 100).unwrap();
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 0.02);
    }

    #[test]
    fn runge_kutta_4_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = runge_kutta_4(&|t, y| y, 0.0, 1.0, 1.0, 10).unwrap();
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 1e-5);
    }

    #[test]
    fn midpoint_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = midpoint(&|t, y| y, 0.0, 1.0, 1.0, 100).unwrap();
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 0.001);
    }

    #[test]
    fn runge_kutta_4_system_test() {
        // Harmonic oscillator: d²x/dt² = -x
        let f = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };
        let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * PI, 100).unwrap();
        let y_final = &result.last().unwrap().1;
        assert!((y_final[0] - 1.0).abs() < 1e-6);
        assert!((y_final[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn harmonic_oscillator_energy() {
        // Energy should be conserved: E = 0.5 * (x² + v²)
        let f = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };
        let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 10.0 * PI, 10000).unwrap();
        let y_final = &result.last().unwrap().1;
        let energy = 0.5 * (y_final[0] * y_final[0] + y_final[1] * y_final[1]);
        assert!((energy - 0.5).abs() < 1e-4, "energy drifted: {energy}");
    }

    #[test]
    fn steps_zero_returns_err() {
        assert!(euler(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(midpoint(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(runge_kutta_4(&|_, y| y, 0.0, 1.0, 1.0, 0).is_err());
        assert!(runge_kutta_4_system(&|_, y: &[f64]| y.to_vec(), 0.0, &[1.0], 1.0, 0).is_err());
    }

    #[test]
    fn ode_problem_builder() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0)
            .method(OdeMethod::Rk4)
            .steps(100)
            .solve()
            .unwrap();
        let y_final = sol.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);

        // Test Euler method
        let sol_euler = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0)
            .method(OdeMethod::Euler)
            .steps(1000)
            .solve()
            .unwrap();
        let y_euler = sol_euler.last().unwrap().1;
        assert!((y_euler - 1.0_f64.exp()).abs() < 0.02);
    }

    #[test]
    fn decay_test() {
        // dy/dt = -y, y(0) = 1, solution: y = e^(-t)
        let result = runge_kutta_4(&|_t, y| -y, 0.0, 1.0, 1.0, 10).unwrap();
        let y_final = result.last().unwrap().1;
        assert!((y_final - (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn rkf45_accuracy() {
        use core::f64::consts::E;
        // Smooth exponential: adaptive solver hits tight tolerance.
        let sol = rkf45(&|_, y| y, 0.0, 1.0, 1.0, 1e-9, 10_000).unwrap();
        let y_final = sol.last().unwrap().1;
        assert!((y_final - E).abs() < 1e-7, "got {y_final}");
        // Lands exactly on t_end.
        assert_eq!(sol.last().unwrap().0, 1.0);
    }

    #[test]
    fn rkf45_grows_steps_on_smooth_problems() {
        // dy/dt = t + 1 over [0, 100] → y(100) = 100²/2 + 100 = 5100.
        // Perfectly smooth: an adaptive solver should take far fewer steps
        // than a fixed RK4 needing h < 0.01 for comparable accuracy.
        let sol = rkf45(&|t, _| t + 1.0, 0.0, 0.0, 100.0, 1e-6, 10_000).unwrap();
        assert!((sol.last().unwrap().1 - 5100.0).abs() < 1e-3);
        assert!(
            sol.len() < 200,
            "expected step growth on linear problem, took {} steps",
            sol.len()
        );
    }

    #[test]
    fn rkf45_shrinks_steps_near_sharp_feature() {
        // A boundary layer at t ≈ 0 (y' = -1000(y - 1)) demands tiny initial
        // steps; fixed-step RK4 with h=0.01 would blow up, adaptive survives.
        let lambda = 1000.0;
        let sol = rkf45(&move |_, y| -lambda * (y - 1.0), 0.0, 0.0, 1.0, 1e-6, 100_000).unwrap();
        let y_final = sol.last().unwrap().1;
        assert!((y_final - 1.0).abs() < 1e-4, "got {y_final}");
        // Early steps must be small relative to the span.
        let first_dt = sol[1].0 - sol[0].0;
        assert!(first_dt < 1e-2, "first step too large for stiff start");
    }

    #[test]
    fn rkf45_backwards_integration() {
        // Integrate dy/dt = y backwards: y(0) = e → y(-1) = 1.
        let sol = rkf45(&|_, y| y, 0.0, core::f64::consts::E, -1.0, 1e-8, 10_000).unwrap();
        let y_final = sol.last().unwrap().1;
        assert!((y_final - 1.0).abs() < 1e-5, "got {y_final}");
        // Lands exactly on t_end.
        assert_eq!(sol.last().unwrap().0, -1.0);
    }

    #[test]
    fn rkf45_input_validation() {
        assert!(rkf45(&|_, y| y, 0.0, 1.0, 0.0, 1e-8, 100).is_err());
        assert!(rkf45(&|_, y| y, 0.0, 1.0, 1.0, 0.0, 100).is_err());
        assert!(rkf45(&|_, y| y, 0.0, 1.0, 1.0, -1e-8, 100).is_err());
        // Impossible tolerance → NotConverged, not a panic or hang.
        assert!(rkf45(&|_, y| y, 0.0, 1.0, 1.0, 1e-300, 100).is_err());
    }
}
