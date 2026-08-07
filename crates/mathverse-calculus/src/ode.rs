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
}
