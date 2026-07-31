//! Stateful optimizers: SGD (with momentum + weight decay), Adam, AdamW,
//! and learning rate schedulers.

/// SGD with momentum and optional weight decay.
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocity: Vec<f64>,
    initialized: bool,
}

impl Sgd {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Self { lr, momentum, weight_decay, velocity: Vec::new(), initialized: false }
    }

    /// Perform one optimization step. `params` and `grads` must have same length.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.velocity = vec![0.0; params.len()];
            self.initialized = true;
        }
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * *p;
            }
            self.velocity[i] = self.momentum * self.velocity[i] + grad;
            *p -= self.lr * self.velocity[i];
        }
    }
}

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    initialized: bool,
}

impl Adam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay, m: Vec::new(), v: Vec::new(), t: 0, initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
            self.initialized = true;
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 {
                grad += self.weight_decay * *p;
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn zero_grad(&mut self) { self.t = 0; self.m.clear(); self.v.clear(); self.initialized = false; }
}

/// AdamW optimizer (decoupled weight decay).
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    initialized: bool,
}

impl AdamW {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay, m: Vec::new(), v: Vec::new(), t: 0, initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
            self.initialized = true;
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            // Decoupled weight decay: decay applied directly to params, not via grad
            if self.weight_decay > 0.0 {
                *p *= 1.0 - self.lr * self.weight_decay;
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * *g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * *g * *g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn zero_grad(&mut self) { self.t = 0; self.m.clear(); self.v.clear(); self.initialized = false; }
}

// ---------------------------------------------------------------------------
// Learning rate schedulers
// ---------------------------------------------------------------------------

/// Learning rate schedule type.
pub enum Schedule {
    Constant,
    StepDecay { step_size: usize, gamma: f64 },
    CosineAnnealing { t_max: usize, eta_min: f64 },
    LinearWarmup { warmup_steps: usize, base_lr: f64, target_lr: f64 },
}

pub struct LrScheduler {
    schedule: Schedule,
    step_count: usize,
}

impl LrScheduler {
    pub fn new(schedule: Schedule) -> Self {
        Self { schedule, step_count: 0 }
    }

    /// Get current learning rate.
    pub fn get_lr(&self) -> f64 {
        match &self.schedule {
            Schedule::Constant => 1.0,
            Schedule::StepDecay { step_size, gamma } => {
                gamma.powf((self.step_count / step_size) as f64)
            }
            Schedule::CosineAnnealing { t_max, eta_min } => {
                let t = (self.step_count % t_max) as f64;
                eta_min + 0.5 * (1.0 - eta_min) * (1.0 + (std::f64::consts::PI * t / *t_max as f64).cos())
            }
            Schedule::LinearWarmup { warmup_steps, base_lr, target_lr } => {
                if self.step_count < *warmup_steps {
                    base_lr + (target_lr - base_lr) * self.step_count as f64 / *warmup_steps as f64
                } else {
                    *target_lr
                }
            }
        }
    }

    /// Advance the schedule by one step.
    pub fn step(&mut self) { self.step_count += 1; }
}

// ---------------------------------------------------------------------------
// Additional optimizers
// ---------------------------------------------------------------------------

/// RMSprop optimizer.
pub struct RMSprop {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    avg_sq: Vec<f64>,
    velocity: Vec<f64>,
    initialized: bool,
}

impl RMSprop {
    pub fn new(lr: f64, alpha: f64, eps: f64, weight_decay: f64, momentum: f64) -> Self {
        Self { lr, alpha, eps, weight_decay, momentum, avg_sq: Vec::new(), velocity: Vec::new(), initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.avg_sq = vec![0.0; params.len()];
            self.velocity = vec![0.0; params.len()];
            self.initialized = true;
        }
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 { grad += self.weight_decay * *p; }
            self.avg_sq[i] = self.alpha * self.avg_sq[i] + (1.0 - self.alpha) * grad * grad;
            self.velocity[i] = self.momentum * self.velocity[i] + grad / (self.avg_sq[i].sqrt() + self.eps);
            *p -= self.lr * self.velocity[i];
        }
    }

    pub fn zero_grad(&mut self) { self.avg_sq.clear(); self.velocity.clear(); self.initialized = false; }
}

/// Lion optimizer (EvoLved Sign Momentum).
pub struct Lion {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    m: Vec<f64>,
    initialized: bool,
}

impl Lion {
    pub fn new(lr: f64, beta1: f64, beta2: f64, weight_decay: f64) -> Self {
        Self { lr, beta1, beta2, weight_decay, m: Vec::new(), initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = vec![0.0; params.len()];
            self.initialized = true;
        }
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let update = self.m[i] * self.beta1 + *g * (1.0 - self.beta1);
            let sign = update.signum();
            *p -= self.lr * (*p * self.weight_decay + sign);
            self.m[i] = self.m[i] * self.beta2 + *g * (1.0 - self.beta2);
        }
    }

    pub fn zero_grad(&mut self) { self.m.clear(); self.initialized = false; }
}

/// AdaGrad optimizer.
pub struct AdaGrad {
    pub lr: f64,
    pub eps: f64,
    pub weight_decay: f64,
    sum_sq: Vec<f64>,
    initialized: bool,
}

impl AdaGrad {
    pub fn new(lr: f64, eps: f64, weight_decay: f64) -> Self {
        Self { lr, eps, weight_decay, sum_sq: Vec::new(), initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.sum_sq = vec![0.0; params.len()];
            self.initialized = true;
        }
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 { grad += self.weight_decay * *p; }
            self.sum_sq[i] += grad * grad;
            *p -= self.lr * grad / (self.sum_sq[i].sqrt() + self.eps);
        }
    }

    pub fn zero_grad(&mut self) { self.sum_sq.clear(); self.initialized = false; }
}

/// AdaDelta optimizer.
pub struct AdaDelta {
    pub lr: f64,
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
    avg_sq: Vec<f64>,
    avg_dx: Vec<f64>,
    initialized: bool,
}

impl AdaDelta {
    pub fn new(lr: f64, rho: f64, eps: f64, weight_decay: f64) -> Self {
        Self { lr, rho, eps, weight_decay, avg_sq: Vec::new(), avg_dx: Vec::new(), initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.avg_sq = vec![0.0; params.len()];
            self.avg_dx = vec![0.0; params.len()];
            self.initialized = true;
        }
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 { grad += self.weight_decay * *p; }
            self.avg_sq[i] = self.rho * self.avg_sq[i] + (1.0 - self.rho) * grad * grad;
            let dx = (self.avg_dx[i] + self.eps).sqrt() / (self.avg_sq[i] + self.eps).sqrt() * grad;
            self.avg_dx[i] = self.rho * self.avg_dx[i] + (1.0 - self.rho) * dx * dx;
            *p -= self.lr * dx;
        }
    }

    pub fn zero_grad(&mut self) { self.avg_sq.clear(); self.avg_dx.clear(); self.initialized = false; }
}

/// Nadam optimizer (Nesterov-accelerated Adam).
pub struct Nadam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    initialized: bool,
}

impl Nadam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay, m: Vec::new(), v: Vec::new(), t: 0, initialized: false }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
            self.initialized = true;
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let bc1_next = 1.0 - self.beta1.powi((self.t + 1) as i32);
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            let mut grad = *g;
            if self.weight_decay > 0.0 { grad += self.weight_decay * *p; }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            let m_nesterov = self.beta1 * m_hat + (1.0 - self.beta1) * grad / bc1_next;
            *p -= self.lr * m_nesterov / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn zero_grad(&mut self) { self.t = 0; self.m.clear(); self.v.clear(); self.initialized = false; }
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-6;

    #[test]
    fn sgd_converges() {
        // Minimize f(x) = x² → grad = 2x, optimum at x=0
        let mut opt = Sgd::new(0.1, 0.9, 0.0);
        let mut x = [10.0];
        for _ in 0..200 {
            let g = [2.0 * x[0]];
            opt.step(&mut x, &g);
        }
        assert!(x[0].abs() < 0.1);
    }

    #[test]
    fn adam_converges() {
        let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
        let mut x = [10.0];
        for _ in 0..300 {
            let g = [2.0 * x[0]];
            opt.step(&mut x, &g);
        }
        assert!(x[0].abs() < 0.01);
    }

    #[test]
    fn adamw_converges() {
        let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);
        let mut x = [10.0];
        for _ in 0..500 {
            let g = [2.0 * x[0]];
            opt.step(&mut x, &g);
        }
        assert!(x[0].abs() < 0.1);
    }

    #[test]
    fn adam_vs_adamw_weight_decay() {
        // With weight decay, AdamW converges faster on simple problems
        let mut adam = Adam::new(0.01, 0.9, 0.999, 1e-8, 0.1);
        let mut adamw = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.1);
        let mut x_adam = [10.0];
        let mut x_adamw = [10.0];
        for _ in 0..200 {
            let g1 = [2.0 * x_adam[0]];
            adam.step(&mut x_adam, &g1);
            let g2 = [2.0 * x_adamw[0]];
            adamw.step(&mut x_adamw, &g2);
        }
        // Both should converge, AdamW should be at least as good
        assert!(x_adamw[0].abs() < x_adam[0].abs() + 0.5);
    }

    #[test]
    fn step_decay_test() {
        let mut s = LrScheduler::new(Schedule::StepDecay { step_size: 10, gamma: 0.5 });
        assert!((s.get_lr() - 1.0).abs() < E);
        for _ in 0..10 { s.step(); }
        assert!((s.get_lr() - 0.5).abs() < E);
        for _ in 0..10 { s.step(); }
        assert!((s.get_lr() - 0.25).abs() < E);
    }

    #[test]
    fn cosine_annealing_test() {
        let mut s = LrScheduler::new(Schedule::CosineAnnealing { t_max: 100, eta_min: 0.0 });
        for _ in 0..50 { s.step(); }
        let lr_mid = s.get_lr();
        assert!((lr_mid - 0.5).abs() < 0.01);
        for _ in 0..49 { s.step(); }
        let lr_end = s.get_lr();
        assert!(lr_end < 0.01); // near eta_min at end of cycle
    }

    #[test]
    fn linear_warmup_test() {
        let mut s = LrScheduler::new(Schedule::LinearWarmup {
            warmup_steps: 100, base_lr: 0.0, target_lr: 1.0,
        });
        for _ in 0..50 { s.step(); }
        assert!((s.get_lr() - 0.5).abs() < E);
        for _ in 0..50 { s.step(); }
        assert!((s.get_lr() - 1.0).abs() < E);
    }
}
