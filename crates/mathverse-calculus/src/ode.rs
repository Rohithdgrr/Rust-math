//! Ordinary differential equation solvers.
//!
//! Provides numerical methods for solving first-order ODEs of the form dy/dt = f(t, y).

/// Forward Euler method for solving dy/dt = f(t, y).
///
/// Returns a vector of (t, y) pairs.
///
/// ```
/// use mathverse_calculus::ode::euler;
/// // dy/dt = y, y(0) = 1, solution: y = e^t
/// let result = euler(&|t, y| y, 0.0, 1.0, 1.0, 100);
/// let y_final = result.last().unwrap().1;
/// assert!((y_final - 1.0_f64.exp()).abs() < 0.02);
/// ```
pub fn euler(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> Vec<(f64, f64)> {
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
    result
}

/// Fourth-order Runge-Kutta method (RK4) for solving dy/dt = f(t, y).
///
/// Returns a vector of (t, y) pairs. Much more accurate than Euler.
///
/// ```
/// use mathverse_calculus::ode::runge_kutta_4;
/// // dy/dt = y, y(0) = 1, solution: y = e^t
/// let result = runge_kutta_4(&|t, y| y, 0.0, 1.0, 1.0, 10);
/// let y_final = result.last().unwrap().1;
/// assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);
/// ```
pub fn runge_kutta_4(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> Vec<(f64, f64)> {
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
    result
}

/// Midpoint method (second-order Runge-Kutta) for solving dy/dt = f(t, y).
///
/// Returns a vector of (t, y) pairs.
///
/// ```
/// use mathverse_calculus::ode::midpoint;
/// // dy/dt = y, y(0) = 1, solution: y = e^t
/// let result = midpoint(&|t, y| y, 0.0, 1.0, 1.0, 100);
/// let y_final = result.last().unwrap().1;
/// assert!((y_final - 1.0_f64.exp()).abs() < 0.001);
/// ```
pub fn midpoint(f: &dyn Fn(f64, f64) -> f64, t0: f64, y0: f64, t_end: f64, steps: usize) -> Vec<(f64, f64)> {
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
    result
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
/// let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * core::f64::consts::PI, 100);
/// let y_final = &result.last().unwrap().1;
/// assert!((y_final[0] - 1.0).abs() < 1e-6); // Should return to initial position
/// ```
pub fn runge_kutta_4_system(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    steps: usize,
) -> Vec<(f64, Vec<f64>)> {
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
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn euler_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = euler(&|t, y| y, 0.0, 1.0, 1.0, 100);
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 0.02);
    }

    #[test]
    fn runge_kutta_4_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = runge_kutta_4(&|t, y| y, 0.0, 1.0, 1.0, 10);
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);
    }

    #[test]
    fn midpoint_test() {
        // dy/dt = y, y(0) = 1, solution: y = e^t
        let result = midpoint(&|t, y| y, 0.0, 1.0, 1.0, 100);
        let y_final = result.last().unwrap().1;
        assert!((y_final - 1.0_f64.exp()).abs() < 0.001);
    }

    #[test]
    fn runge_kutta_4_system_test() {
        // Harmonic oscillator: d²x/dt² = -x
        let f = |t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };
        let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * PI, 100);
        let y_final = &result.last().unwrap().1;
        assert!((y_final[0] - 1.0).abs() < 1e-6);
        assert!((y_final[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn decay_test() {
        // dy/dt = -y, y(0) = 1, solution: y = e^(-t)
        let result = runge_kutta_4(&|t, y| -y, 0.0, 1.0, 1.0, 10);
        let y_final = result.last().unwrap().1;
        assert!((y_final - (-1.0_f64).exp()).abs() < 1e-6);
    }
}
