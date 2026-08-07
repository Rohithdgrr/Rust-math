pub fn fibonacci_search(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    // Fibonacci search needs at least 3 evaluations (k ≥ 2) to bracket.
    if n < 3 {
        return (a + b) / 2.0;
    }
    let fibs: Vec<f64> = {
        let mut f = vec![1.0f64, 1.0];
        for _ in 2..n { let l = f.len(); f.push(f[l-1] + f[l-2]); }
        f
    };
    let k = n.min(fibs.len() - 1);
    let (mut l, mut r) = (a, b);
    let mut x1 = l + (r - l) * fibs[k - 2] / fibs[k];
    let mut x2 = l + (r - l) * fibs[k - 1] / fibs[k];
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for i in (2..k).rev() {
        if f1 < f2 {
            r = x2;
            x2 = x1;
            f2 = f1;
            x1 = l + (r - l) * fibs[i - 2] / fibs[i];
            f1 = f(x1);
        } else {
            l = x1;
            x1 = x2;
            f1 = f2;
            x2 = l + (r - l) * fibs[i - 1] / fibs[i];
            f2 = f(x2);
        }
    }
    (l + r) / 2.0
}

// Scalar minimizers: Fibonacci search, golden section, ternary search, Brent's method.

/// Golden-section search for a scalar minimum in `[a, b]`.
pub fn golden_section(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let (mut l, mut r) = (a, b);
    while (r - l).abs() > tol {
        let x1 = r - (r - l) / phi;
        let x2 = l + (r - l) / phi;
        if f(x1) < f(x2) { r = x2; } else { l = x1; }
    }
    (l + r) / 2.0
}

pub fn ternary_search(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    let (mut l, mut r) = (a, b);
    while (r - l).abs() > tol {
        let m1 = l + (r - l) / 3.0;
        let m2 = r - (r - l) / 3.0;
        if f(m1) < f(m2) { r = m2; } else { l = m1; }
    }
    (l + r) / 2.0
}

pub fn convex_search(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    golden_section(f, a, b, tol)
}

pub fn brent_min(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    let mut x = (a + b) / 2.0;
    let mut fx = f(x);
    let (mut w, mut fw) = (x, fx);
    let (mut v, mut fv) = (x, fx);
    let (mut a0, mut b0) = (a, b);
    for _ in 0..1000 {
        let xm = (a0 + b0) / 2.0;
        let tol1 = (b0 - a0).abs() * 1e-15 + tol / 3.0;
        if (x - xm).abs() <= 2.0 * tol1 - (b0 - a0) / 2.0 { break; }
        if (fx - fw).abs() > 1e-30 && (fx - fv).abs() > 1e-30 {
            let u = x - ((x - w).powi(2) * (fx - fv) - (x - v).powi(2) * (fx - fw))
                / (2.0 * ((x - w) * (fx - fv) - (x - v) * (fx - fw)));
            if u > a0 + tol1 && u < b0 - tol1 && (u - xm).abs() < tol1 - (b0 - a0) / 2.0 {
                let fu = f(u);
                if fu < fx {
                    if u < xm { b0 = u; } else { a0 = u; }
                    if fu <= fw || (w - x).abs() < 1e-30 { (w, fw) = (u, fu); }
                    else if fu <= fv || (v - x).abs() < 1e-30 || (v - w).abs() < 1e-30 { (v, fv) = (u, fu); }
                    (x, fx) = (u, fu);
                    continue;
                }
            }
        }
        let u = if x < xm { x + 0.3819660112501051 * (b0 - x) } else { x - 0.3819660112501051 * (x - a0) };
        let fu = f(u);
        if fu <= fx {
            if u < xm { b0 = u; } else { a0 = u; }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else if fu <= fw || (v - x).abs() < 1e-30 || (v - u).abs() < 1e-30 {
            (v, fv) = (u, fu);
        } else {
            (v, fv) = (w, fw);
            (w, fw) = (u, fu);
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden() {
        let x = golden_section(|x| (x - 2.0).powi(2), 0.0, 5.0, 1e-10);
        assert!((x - 2.0).abs() < 1e-8);
    }

    #[test]
    fn ternary() {
        let x = ternary_search(|x| (x - 3.0).powi(2), 0.0, 6.0, 1e-10);
        assert!((x - 3.0).abs() < 1e-8);
    }

    #[test]
    fn brent() {
        let x = brent_min(|x| (x - 1.5).powi(2), 0.0, 3.0, 1e-10);
        assert!((x - 1.5).abs() < 1e-8);
    }

    #[test]
    fn fibonacci_degnerate_iterations() {
        // n < 3 returns the interval midpoint without panicking.
        let f = |x: f64| (x - 2.0).powi(2);
        for n in [0usize, 1, 2] {
            let x = fibonacci_search(f, 0.0, 4.0, n);
            assert!((x - 2.0).abs() < 2.0, "n={n} gave x={x}");
        }
        // A real run still finds the minimum.
        let x = fibonacci_search(f, 0.0, 4.0, 40);
        assert!((x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn brent_no_zero_in_interval() {
        let x = brent_min(|x| (x - 15.0).powi(2), 10.0, 20.0, 1e-10);
        assert!((x - 15.0).abs() < 1e-8);
    }
}
