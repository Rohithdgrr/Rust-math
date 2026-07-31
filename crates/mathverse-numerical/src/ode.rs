//! Advanced ODE integrators: adaptive RK, multistep methods, stiff solvers.

use mathverse_core::error::{MathError, MathResult};

/// ODE state: (t, y) where y can be scalar or vector.
#[derive(Debug, Clone)]
pub struct ODEState {
    pub t: f64,
    pub y: Vec<f64>,
}

impl ODEState {
    pub fn new(t: f64, y: Vec<f64>) -> Self {
        ODEState { t, y }
    }
}

/// Adaptive Runge-Kutta-Fehlberg (RKF45) with error control.
pub struct RKF45 {
    pub min_step: f64,
    pub max_step: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
}

impl RKF45 {
    pub fn new(min_step: f64, max_step: f64, abs_tol: f64, rel_tol: f64) -> Self {
        RKF45 {
            min_step,
            max_step,
            abs_tol,
            rel_tol,
        }
    }

    /// Integrate ODE dy/dt = f(t, y) from t0 to t1.
    pub fn integrate(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        t0: f64,
        y0: &[f64],
        t1: f64,
    ) -> MathResult<Vec<ODEState>> {
        let mut result = Vec::new();
        let mut state = ODEState::new(t0, y0.to_vec());
        result.push(state.clone());
        
        let mut h = self.max_step.min((t1 - t0).abs());
        
        while (state.t - t1).abs() > 1e-12 {
            if state.t + h > t1 {
                h = t1 - state.t;
            }
            if h < self.min_step {
                h = self.min_step;
            }
            
            let (new_state, error, new_h) = self.step(f, &state, h)?;
            
            if error <= 1.0 {
                state = new_state;
                result.push(state.clone());
            }
            
            h = new_h.clamp(self.min_step, self.max_step);
            
            if h < self.min_step && (state.t - t1).abs() > 1e-6 {
                return Err(MathError::NotConverged("RKF45 step size too small"));
            }
        }
        
        Ok(result)
    }

    /// Single RKF45 step with error estimation.
    fn step(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        state: &ODEState,
        h: f64,
    ) -> MathResult<(ODEState, f64, f64)> {
        let t = state.t;
        let y = &state.y;
        
        // RKF45 coefficients
        let k1 = f(t, y);
        
        let y2: Vec<f64> = y.iter()
            .zip(&k1)
            .map(|(&yi, &k1i)| yi + h * k1i / 4.0)
            .collect();
        let k2 = f(t + h / 4.0, &y2);
        
        let y3: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k2)
            .map(|((&yi, &k1i), &k2i)| yi + h * (3.0 * k1i + 9.0 * k2i) / 32.0)
            .collect();
        let k3 = f(t + 3.0 * h / 8.0, &y3);
        
        let y4: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k2)
            .zip(&k3)
            .map(|(((&yi, &k1i), &k2i), &k3i)| yi + h * (1932.0 * k1i - 7200.0 * k2i + 7296.0 * k3i) / 2197.0)
            .collect();
        let k4 = f(t + 12.0 * h / 13.0, &y4);
        
        let y5: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .map(|(((&yi, &k1i), &k3i), &k4i)| yi + h * (439.0 * k1i / 216.0 - 8.0 * k3i + 3680.0 * k4i / 513.0))
            .collect();
        let k5 = f(t + h, &y5);
        
        let y6: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .map(|((((&yi, &k1i), &k3i), &k4i), &k5i)| yi + h * (-8.0 * k1i / 27.0 + 2.0 * k3i - 3544.0 * k4i / 2565.0 + 1859.0 * k5i / 4104.0))
            .collect();
        let k6 = f(t + h / 2.0, &y6);
        
        // 4th order solution
        let y4th: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .map(|((((&yi, &k1i), &k3i), &k4i), &k5i)| yi + h * (25.0 * k1i / 216.0 + 1408.0 * k3i / 2565.0 + 2197.0 * k4i / 4104.0 - k5i / 5.0))
            .collect();
        
        // 5th order solution
        let y5th: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .zip(&k6)
            .map(|(((((&yi, &k1i), &k3i), &k4i), &k5i), &k6i)| yi + h * (16.0 * k1i / 135.0 + 6656.0 * k3i / 12825.0 + 28561.0 * k4i / 56430.0 - 9.0 * k5i / 50.0 + 2.0 * k6i / 55.0))
            .collect();
        
        // Error estimation
        let error: f64 = y4th.iter()
            .zip(&y5th)
            .map(|(&y4i, &y5i)| {
                let scale = self.abs_tol + self.rel_tol * y5i.abs();
                (y4i - y5i).abs() / scale
            })
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        
        // Step size adjustment
        let safety = 0.9;
        let new_h = if error > 0.0 {
            h * safety * (1.0 / error).powf(0.2)
        } else {
            h * 2.0
        };
        
        Ok((ODEState::new(t + h, y5th), error, new_h))
    }
}

/// Dormand-Prince 4(5) adaptive RK method.
pub struct DormandPrince {
    pub min_step: f64,
    pub max_step: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
}

impl DormandPrince {
    pub fn new(min_step: f64, max_step: f64, abs_tol: f64, rel_tol: f64) -> Self {
        DormandPrince {
            min_step,
            max_step,
            abs_tol,
            rel_tol,
        }
    }

    /// Integrate ODE using Dormand-Prince method.
    pub fn integrate(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        t0: f64,
        y0: &[f64],
        t1: f64,
    ) -> MathResult<Vec<ODEState>> {
        let mut result = Vec::new();
        let mut state = ODEState::new(t0, y0.to_vec());
        result.push(state.clone());
        
        let mut h = self.max_step.min((t1 - t0).abs());
        
        while (state.t - t1).abs() > 1e-12 {
            if state.t + h > t1 {
                h = t1 - state.t;
            }
            if h < self.min_step {
                h = self.min_step;
            }
            
            let (new_state, error, new_h) = self.step(f, &state, h)?;
            
            if error <= 1.0 {
                state = new_state;
                result.push(state.clone());
            }
            
            h = new_h.clamp(self.min_step, self.max_step);
        }
        
        Ok(result)
    }

    fn step(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        state: &ODEState,
        h: f64,
    ) -> MathResult<(ODEState, f64, f64)> {
        // Dormand-Prince coefficients (simplified)
        let t = state.t;
        let y = &state.y;
        
        let k1 = f(t, y);
        
        let y2: Vec<f64> = y.iter()
            .zip(&k1)
            .map(|(&yi, &k1i)| yi + h * k1i / 5.0)
            .collect();
        let k2 = f(t + h / 5.0, &y2);
        
        let y3: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k2)
            .map(|((&yi, &k1i), &k2i)| yi + h * (3.0 * k1i + 9.0 * k2i) / 40.0)
            .collect();
        let k3 = f(t + 3.0 * h / 10.0, &y3);
        
        let y4: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .map(|((&yi, &k1i), &k3i)| yi + h * (44.0 * k1i / 45.0 - 56.0 * k3i / 15.0))
            .collect();
        let k4 = f(t + 4.0 * h / 5.0, &y4);
        
        let y5: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .map(|(((&yi, &k1i), &k3i), &k4i)| yi + h * (19372.0 * k1i / 6561.0 - 25360.0 * k3i / 2187.0 + 64448.0 * k4i / 6561.0))
            .collect();
        let k5 = f(t + 8.0 * h / 9.0, &y5);
        
        let y6: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .map(|((((&yi, &k1i), &k3i), &k4i), &k5i)| yi + h * (9017.0 * k1i / 3168.0 - 355.0 * k3i / 33.0 + 46732.0 * k4i / 5247.0 + 49.0 * k5i / 176.0))
            .collect();
        let k6 = f(t + h, &y6);
        
        let y7: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .zip(&k6)
            .map(|(((((&yi, &k1i), &k3i), &k4i), &k5i), &k6i)| yi + h * (35.0 * k1i / 384.0 + 500.0 * k3i / 1113.0 + 125.0 * k4i / 192.0 - 2187.0 * k5i / 6784.0 + 11.0 * k6i / 84.0))
            .collect();
        let k7 = f(t + h, &y7);
        
        // 5th order solution
        let y5th: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .zip(&k6)
            .zip(&k7)
            .map(|((((((&yi, &k1i), &k3i), &k4i), &k5i), &k6i), &k7i)| yi + h * (5179.0 * k1i / 57600.0 + 7571.0 * k3i / 16695.0 + 393.0 * k4i / 640.0 - 92097.0 * k5i / 339200.0 + 187.0 * k6i / 2100.0 + k7i / 40.0))
            .collect();
        
        // 4th order solution (for error estimation)
        let y4th: Vec<f64> = y.iter()
            .zip(&k1)
            .zip(&k3)
            .zip(&k4)
            .zip(&k5)
            .zip(&k6)
            .map(|(((((&yi, &k1i), &k3i), &k4i), &k5i), &k6i)| yi + h * (35.0 * k1i / 384.0 + 500.0 * k3i / 1113.0 + 125.0 * k4i / 192.0 - 2187.0 * k5i / 6784.0 + 11.0 * k6i / 84.0))
            .collect();
        
        let error: f64 = y4th.iter()
            .zip(&y5th)
            .map(|(&y4i, &y5i)| {
                let scale = self.abs_tol + self.rel_tol * y5i.abs();
                (y4i - y5i).abs() / scale
            })
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();
        
        let safety = 0.9;
        let new_h = if error > 0.0 {
            h * safety * (1.0 / error).powf(0.2)
        } else {
            h * 2.0
        };
        
        Ok((ODEState::new(t + h, y5th), error, new_h))
    }
}

/// Adams-Bashforth multistep method (explicit).
pub struct AdamsBashforth {
    pub order: usize,
}

impl AdamsBashforth {
    pub fn new(order: usize) -> Self {
        AdamsBashforth { order }
    }

    /// Integrate using Adams-Bashforth method (requires startup with RK4).
    pub fn integrate(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        t0: f64,
        y0: &[f64],
        t1: f64,
        steps: usize,
    ) -> MathResult<Vec<ODEState>> {
        let n = y0.len();
        let h = (t1 - t0) / steps as f64;
        
        // Startup with RK4
        let mut result = Vec::new();
        let mut states = Vec::new();
        
        let mut t = t0;
        let mut y = y0.to_vec();
        states.push(ODEState::new(t, y.clone()));
        result.push(states[0].clone());
        
        // Generate initial steps with RK4
        for _ in 0..self.order - 1 {
            let k1 = f(t, &y);
            let y2: Vec<f64> = y.iter()
                .zip(&k1)
                .map(|(&yi, &k1i)| yi + h * k1i / 2.0)
                .collect();
            let k2 = f(t + h / 2.0, &y2);
            
            let y3: Vec<f64> = y.iter()
                .zip(&k2)
                .map(|(&yi, &k2i)| yi + h * k2i / 2.0)
                .collect();
            let k3 = f(t + h / 2.0, &y3);
            
            let y4: Vec<f64> = y.iter()
                .zip(&k3)
                .map(|(&yi, &k3i)| yi + h * k3i)
                .collect();
            let k4 = f(t + h, &y4);
            
            y = y.iter()
                .zip(&k1)
                .zip(&k2)
                .zip(&k3)
                .zip(&k4)
                .map(|((((&yi, &k1i), &k2i), &k3i), &k4i)| yi + h * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0)
                .collect();
            
            t += h;
            states.push(ODEState::new(t, y.clone()));
            result.push(states.last().unwrap().clone());
        }
        
        // Adams-Bashforth steps
        let coeffs = self.adams_bashforth_coefficients();
        
        for _ in self.order - 1..steps {
            let mut dy = vec![0.0; n];
            
            for (j, &coeff) in coeffs.iter().enumerate() {
                let state = &states[states.len() - 1 - j];
                let fj = f(state.t, &state.y);
                for k in 0..n {
                    dy[k] += coeff * fj[k];
                }
            }
            
            y = y.iter().zip(&dy).map(|(&yi, &dyi)| yi + h * dyi).collect();
            t += h;
            
            states.push(ODEState::new(t, y.clone()));
            result.push(states.last().unwrap().clone());
            
            // Keep only needed history
            if states.len() > self.order {
                states.remove(0);
            }
        }
        
        Ok(result)
    }

    fn adams_bashforth_coefficients(&self) -> Vec<f64> {
        match self.order {
            2 => vec![3.0 / 2.0, -1.0 / 2.0],
            3 => vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
            4 => vec![55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
            5 => vec![1901.0 / 720.0, -2774.0 / 720.0, 2616.0 / 720.0, -1274.0 / 720.0, 251.0 / 720.0],
            _ => vec![1.0], // Fallback to Euler
        }
    }
}

/// Backward Euler method (implicit, for stiff equations).
pub struct BackwardEuler {
    pub max_iters: usize,
    pub tol: f64,
}

impl BackwardEuler {
    pub fn new(max_iters: usize, tol: f64) -> Self {
        BackwardEuler { max_iters, tol }
    }

    /// Integrate using Backward Euler with Newton iteration.
    pub fn integrate(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        jacobian: &dyn Fn(f64, &[f64]) -> Vec<Vec<f64>>,
        t0: f64,
        y0: &[f64],
        t1: f64,
        steps: usize,
    ) -> MathResult<Vec<ODEState>> {
        let n = y0.len();
        let h = (t1 - t0) / steps as f64;
        
        let mut result = Vec::new();
        let mut t = t0;
        let mut y = y0.to_vec();
        
        result.push(ODEState::new(t, y.clone()));
        
        for _ in 0..steps {
            // Newton iteration for implicit step
            let mut y_new = y.clone();
            
            for _ in 0..self.max_iters {
                let fy = f(t + h, &y_new);
                let j = jacobian(t + h, &y_new);
                
                // Solve (I - h*J) * delta = h*fy - (y_new - y)
                let mut rhs: Vec<f64> = fy.iter()
                    .zip(&y_new)
                    .zip(&y)
                    .map(|((&fyi, &yni), &yi)| h * fyi - (yni - yi))
                    .collect();
                
                // Simplified: use diagonal approximation
                let delta: Vec<f64> = rhs.iter()
                    .zip(&j)
                    .map(|(&rhsi, ji)| {
                        let diag = 1.0 - h * ji[0];
                        rhsi / diag
                    })
                    .collect();
                
                y_new = y_new.iter().zip(&delta).map(|(&yn, &d)| yn - d).collect();
                
                let delta_norm: f64 = delta.iter().map(|&d| d * d).sum::<f64>().sqrt();
                if delta_norm < self.tol {
                    break;
                }
            }
            
            y = y_new;
            t += h;
            result.push(ODEState::new(t, y.clone()));
        }
        
        Ok(result)
    }
}

/// Crank-Nicolson method (implicit, second-order accurate).
pub struct CrankNicolson {
    pub max_iters: usize,
    pub tol: f64,
}

impl CrankNicolson {
    pub fn new(max_iters: usize, tol: f64) -> Self {
        CrankNicolson { max_iters, tol }
    }

    /// Integrate using Crank-Nicolson method.
    pub fn integrate(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        jacobian: &dyn Fn(f64, &[f64]) -> Vec<Vec<f64>>,
        t0: f64,
        y0: &[f64],
        t1: f64,
        steps: usize,
    ) -> MathResult<Vec<ODEState>> {
        let n = y0.len();
        let h = (t1 - t0) / steps as f64;
        
        let mut result = Vec::new();
        let mut t = t0;
        let mut y = y0.to_vec();
        
        result.push(ODEState::new(t, y.clone()));
        
        for _ in 0..steps {
            let fy_current = f(t, &y);
            
            // Newton iteration for implicit step
            let mut y_new = y.clone();
            
            for _ in 0..self.max_iters {
                let fy_new = f(t + h, &y_new);
                let j = jacobian(t + h, &y_new);
                
                // Residual: y_new - y - h/2 * (f(t,y) + f(t+h,y_new))
                let mut residual: Vec<f64> = y_new.iter()
                    .zip(&y)
                    .zip(&fy_current)
                    .zip(&fy_new)
                    .map(|(((&yn, &yi), &fci), &fni)| yn - yi - h / 2.0 * (fci + fni))
                    .collect();
                
                // Jacobian of residual: I - h/2 * J
                let mut delta: Vec<f64> = residual.iter()
                    .zip(&j)
                    .map(|(&ri, ji)| {
                        let diag = 1.0 - h / 2.0 * ji[0];
                        ri / diag
                    })
                    .collect();
                
                y_new = y_new.iter().zip(&delta).map(|(&yn, &d)| yn - d).collect();
                
                let delta_norm: f64 = delta.iter().map(|&d| d * d).sum::<f64>().sqrt();
                if delta_norm < self.tol {
                    break;
                }
            }
            
            y = y_new;
            t += h;
            result.push(ODEState::new(t, y.clone()));
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rkf45() {
        let rkf45 = RKF45::new(1e-6, 1.0, 1e-8, 1e-8);
        let f = |_: f64, y: &[f64]| vec![y[0]]; // dy/dt = y
        
        let result = rkf45.integrate(&f, 0.0, &[1.0], 1.0).unwrap();
        let final_y = result.last().unwrap().y[0];
        
        assert!((final_y - core::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_dormand_prince() {
        let dp = DormandPrince::new(1e-6, 1.0, 1e-8, 1e-8);
        let f = |_: f64, y: &[f64]| vec![y[0]];
        
        let result = dp.integrate(&f, 0.0, &[1.0], 1.0).unwrap();
        let final_y = result.last().unwrap().y[0];
        
        assert!((final_y - core::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_adams_bashforth() {
        let ab = AdamsBashforth::new(3);
        let f = |_: f64, y: &[f64]| vec![y[0]];
        
        let result = ab.integrate(&f, 0.0, &[1.0], 1.0, 100).unwrap();
        let final_y = result.last().unwrap().y[0];
        
        assert!((final_y - core::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_backward_euler() {
        let be = BackwardEuler::new(100, 1e-10);
        let f = |_: f64, y: &[f64]| vec![-10.0 * y[0]]; // dy/dt = -10y (stiff)
        let jacobian = |_: f64, _: &[f64]| vec![vec![-10.0]];
        
        let result = be.integrate(&f, &jacobian, 0.0, &[1.0], 1.0, 100).unwrap();
        let final_y = result.last().unwrap().y[0];
        
        // y(1) = e^(-10) ≈ 4.54e-5
        assert!((final_y - (-10.0_f64).exp()).abs() < 0.01);
    }

    #[test]
    fn test_crank_nicolson() {
        let cn = CrankNicolson::new(100, 1e-10);
        let f = |_: f64, y: &[f64]| vec![-10.0 * y[0]];
        let jacobian = |_: f64, _: &[f64]| vec![vec![-10.0]];
        
        let result = cn.integrate(&f, &jacobian, 0.0, &[1.0], 1.0, 100).unwrap();
        let final_y = result.last().unwrap().y[0];
        
        assert!((final_y - (-10.0_f64).exp()).abs() < 0.01);
    }
}
