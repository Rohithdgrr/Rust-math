pub fn newton(f: impl Fn(f64) -> f64, df: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol { return Some(x); }
        let dfx = df(x);
        if dfx.abs() < 1e-30 { return None; }
        x -= fx / dfx;
    }
    Some(x)
}

pub fn secant(f: impl Fn(f64) -> f64, x0: f64, x1: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let (mut a, mut b) = (x0, x1);
    for _ in 0..max_iter {
        let fb = f(b);
        if fb.abs() < tol { return Some(b); }
        let fa = f(a);
        let diff = fb - fa;
        if diff.abs() < 1e-30 { return None; }
        let c = b - fb * (b - a) / diff;
        a = b;
        b = c;
    }
    Some(b)
}

pub fn bisection(f: impl Fn(f64) -> f64, a0: f64, b0: f64, tol: f64) -> Option<f64> {
    let (mut a, mut b) = (a0, b0);
    if f(a) * f(b) > 0.0 { return None; }
    for _ in 0..1000 {
        let mid = (a + b) / 2.0;
        if (b - a).abs() < tol { return Some(mid); }
        if f(mid) * f(a) <= 0.0 { b = mid; } else { a = mid; }
    }
    Some((a + b) / 2.0)
}

pub fn newton_system(f: &[impl Fn(&[f64]) -> f64], j: &impl Fn(&[f64]) -> Vec<Vec<f64>>, x0: &[f64], tol: f64, max_iter: usize) -> Option<Vec<f64>> {
    let n = x0.len();
    let mut x = x0.to_vec();
    for _ in 0..max_iter {
        let fx: Vec<f64> = f.iter().map(|fi| fi(&x)).collect();
        if fx.iter().map(|v| v * v).sum::<f64>().sqrt() < tol { return Some(x); }
        let jac = j(&x);
        let dx = super::matrix_eq::solve_gauss(&jac, &fx)?;
        for i in 0..n { x[i] -= dx[i]; }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_sqrt() {
        let x = newton(|x| x*x - 2.0, |x| 2.0*x, 1.0, 1e-15, 100).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn secant_method() {
        let x = secant(|x| x*x - 2.0, 1.0, 2.0, 1e-15, 100).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn bisect() {
        let x = bisection(|x| x*x - 2.0, 0.0, 2.0, 1e-12).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
