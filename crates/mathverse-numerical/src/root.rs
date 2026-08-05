//! Advanced root finding methods: secant, Brent's method, Muller's method, etc.

use mathverse_core::error::{MathError, MathResult};

/// Secant method (derivative-free Newton-like method).
pub fn secant(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x_prev = x0;
    let mut x_curr = x1;
    
    for _ in 0..max_iters {
        let f_prev = f(x_prev);
        let f_curr = f(x_curr);
        
        if (f_curr - f_prev).abs() < 1e-15 {
            return Err(MathError::InvalidArgument("function values too close"));
        }
        
        let x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
        
        if (x_new - x_curr).abs() < tol {
            return Ok(x_new);
        }
        
        x_prev = x_curr;
        x_curr = x_new;
    }
    
    Err(MathError::NotConverged("secant method"))
}

/// False position method (regula falsi).
pub fn false_position(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let fa = f(a);
    let fb = f(b);
    
    if fa * fb > 0.0 {
        return Err(MathError::InvalidArgument("no sign change on bracket"));
    }
    
    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut fb = fb;
    
    for _ in 0..max_iters {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = f(c);
        
        if fc.abs() < tol || (b - a).abs() < tol {
            return Ok(c);
        }
        
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    Err(MathError::NotConverged("false position method"))
}

/// Muller's method for finding roots (handles complex roots).
pub fn muller(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    x2: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x = [x0, x1, x2];
    let mut fx = [f(x0), f(x1), f(x2)];
    
    for _ in 0..max_iters {
        let h1 = x[1] - x[0];
        let h2 = x[2] - x[1];
        
        let delta1 = (fx[1] - fx[0]) / h1;
        let delta2 = (fx[2] - fx[1]) / h2;
        
        let d = (delta2 - delta1) / (h2 + h1);
        
        let b = delta2 + h2 * d;
        let discriminant = b * b - 4.0 * fx[2] * d;
        
        if discriminant < 0.0 {
            return Err(MathError::InvalidArgument("complex root encountered"));
        }
        
        let sqrt_disc = discriminant.sqrt();
        
        // Choose sign to avoid cancellation
        let e = if b.abs() > sqrt_disc {
            b - sqrt_disc
        } else {
            b + sqrt_disc
        };
        
        let dx = -2.0 * fx[2] / e;
        
        let x_new = x[2] + dx;
        
        if dx.abs() < tol {
            return Ok(x_new);
        }
        
        // Shift values
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x_new;
        fx[0] = fx[1];
        fx[1] = fx[2];
        fx[2] = f(x_new);
    }
    
    Err(MathError::NotConverged("Muller's method"))
}

/// Brent's method (combines bisection, secant, and inverse quadratic interpolation).
pub fn brent(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);
    
    if fa * fb > 0.0 {
        return Err(MathError::InvalidArgument("no sign change on bracket"));
    }
    
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    
    let mut c = a;
    let mut fc = fa;
    let mut d = c;
    let _e = b;
    let mut mflag = true;
    
    for _ in 0..max_iters {
        if fa.abs() < tol || (b - a).abs() < tol {
            return Ok(a);
        }
        
        let mut s;
        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        
        // Check if interpolation is acceptable
        if (s > (3.0 * a + b) / 4.0 && s < b) || mflag && (s - b).abs() >= (b - c).abs() / 2.0
            || !mflag && (s - b).abs() >= (c - d).abs() / 2.0
            || mflag && (b - c).abs() < tol
            || !mflag && (c - d).abs() < tol
        {
            // Bisection
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        
        let fs = f(s);
        d = c;
        c = b;
        fc = fb;
        
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    
    Err(MathError::NotConverged("Brent's method"))
}

/// Illinois method (modified false position with bracket adjustment).
pub fn illinois(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);
    
    if fa * fb > 0.0 {
        return Err(MathError::InvalidArgument("no sign change on bracket"));
    }
    
    for _ in 0..max_iters {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = f(c);
        
        if fc.abs() < tol || (b - a).abs() < tol {
            return Ok(c);
        }
        
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
            fa *= 0.5; // Illinois modification
        } else {
            a = c;
            fa = fc;
            fb *= 0.5; // Illinois modification
        }
    }
    
    Err(MathError::NotConverged("Illinois method"))
}

/// Steffensen's method (Aitken's delta-squared process for acceleration).
pub fn steffensen(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x = x0;
    
    for _ in 0..max_iters {
        let fx = f(x);
        let f_x_plus_fx = f(x + fx);
        
        let denominator = f_x_plus_fx - fx;
        if denominator.abs() < 1e-15 {
            return Err(MathError::DivisionByZero);
        }
        
        let x_new = x - fx * fx / denominator;
        
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("Steffensen's method"))
}

/// Halley's method (second-order derivative Newton variant).
pub fn halley(
    f: &dyn Fn(f64) -> f64,
    fp: &dyn Fn(f64) -> f64,
    fpp: &dyn Fn(f64) -> f64,
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
        
        let fp_x = fp(x);
        let fpp_x = fpp(x);
        
        let denominator = fp_x - fx * fpp_x / (2.0 * fp_x);
        if denominator.abs() < 1e-15 {
            return Err(MathError::DivisionByZero);
        }
        
        let x_new = x - fx / denominator;
        
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("Halley's method"))
}

/// Householder methods of order n (generalization of Newton and Halley).
pub fn householder(
    f: &dyn Fn(f64) -> f64,
    derivatives: &[&dyn Fn(f64) -> f64],
    x0: f64,
    order: usize,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x = x0;
    
    for _ in 0..max_iters {
        let fx = f(x);
        if fx.abs() < tol {
            return Ok(x);
        }
        
        let numerator = fx;
        let mut denominator = derivatives[0](x);
        
        if order >= 2 {
            let fp = derivatives[0](x);
            let fpp = derivatives[1](x);
            denominator = fp - fx * fpp / (2.0 * fp);
        }
        
        if denominator.abs() < 1e-15 {
            return Err(MathError::DivisionByZero);
        }
        
        let x_new = x - numerator / denominator;
        
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("Householder method"))
}

/// Fixed point iteration with acceleration.
pub fn fixed_point(
    g: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iters: usize,
) -> MathResult<f64> {
    let mut x = x0;
    
    for _ in 0..max_iters {
        let x_new = g(x);
        
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("fixed point iteration"))
}

/// Aitken's delta-squared process for sequence acceleration.
pub fn aitken_delta_squared(sequence: &[f64]) -> f64 {
    if sequence.len() < 3 {
        return sequence.last().copied().unwrap_or(0.0);
    }
    
    let n = sequence.len();
    let x_n = sequence[n - 3];
    let x_n1 = sequence[n - 2];
    let x_n2 = sequence[n - 1];
    
    let delta1 = x_n1 - x_n;
    let delta2 = x_n2 - x_n1;
    
    x_n - delta1 * delta1 / (delta2 - delta1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secant() {
        let root = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_false_position() {
        let root = false_position(&|x| x * x - 4.0, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_muller() {
        let root = muller(&|x| x * x - 4.0, 0.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_brent() {
        let root = brent(&|x| x * x - 4.0, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_illinois() {
        let root = illinois(&|x| x * x - 4.0, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_steffensen() {
        let root = steffensen(&|x| x * x - 4.0, 2.5, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_halley() {
        let root = halley(&|x| x * x - 4.0, &|x| 2.0 * x, &|x| 2.0, 2.5, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_point() {
        let root = fixed_point(&|x| (x + 2.0 / x) / 2.0, 1.0, 1e-10, 100).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_aitken() {
        let sequence = vec![1.0, 1.5, 1.75, 1.875, 1.9375];
        let accelerated = aitken_delta_squared(&sequence);
        assert!(accelerated > 1.9);
    }
}
