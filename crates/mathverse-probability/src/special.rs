/// Lanczos approximation coefficients shared by `gamma_fn` and `ln_gamma`.
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

#[must_use]
pub fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        let sin_px = (core::f64::consts::PI * x).sin();
        if sin_px == 0.0 {
            return f64::INFINITY;
        }
        core::f64::consts::PI / (sin_px * gamma_fn(1.0 - x))
    } else {
        let z = x - 1.0;
        let g = 7.0;
        let coeffs = LANCZOS_COEFFS;

        let mut a = coeffs[0];
        for (i, &c) in coeffs.iter().enumerate().skip(1) {
            a += c / (z + i as f64);
        }

        let t = z + g + 0.5;
        (2.0 * core::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
    }
}

#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

#[must_use]
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

#[must_use]
pub fn lower_gamma(s: f64, x: f64) -> f64 {
    reg_lower_gamma(s, x) * gamma_fn(s)
}

/// Regularized lower incomplete gamma `P(s, x) = gamma(s, x) / Gamma(s)`.
///
/// Computed directly in probability space so the result stays finite even
/// when `Gamma(s)` overflows (s >= ~172).
#[must_use]
pub(crate) fn reg_lower_gamma(s: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-14;
    const MAX_ITERS: usize = 10_000;
    const FPMIN: f64 = 1e-300;

    if x < 0.0 || s <= 0.0 || !s.is_finite() || !x.is_finite() {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    let gln = ln_gamma(s);
    if x < s + 1.0 {
        // Series for P(s, x).
        let mut ap = s;
        let mut sum = 1.0 / s;
        let mut delta = sum;
        for _ in 0..MAX_ITERS {
            ap += 1.0;
            delta *= x / ap;
            sum += delta;
            if delta.abs() <= sum.abs() * EPS {
                break;
            }
        }
        sum * (s * x.ln() - x - gln).exp()
    } else {
        // Continued fraction for Q(s, x), then P = 1 - Q.
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b.max(FPMIN);
        let mut h = d;

        for i in 1..=MAX_ITERS {
            let i_f = i as f64;
            let an = -i_f * (i_f - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < EPS {
                break;
            }
        }

        let q = (s * x.ln() - x - gln).exp() * h;
        (1.0 - q).clamp(0.0, 1.0)
    }
}

#[must_use]
pub fn upper_gamma(s: f64, x: f64) -> f64 {
    gamma_fn(s) - lower_gamma(s, x)
}

#[must_use]
pub fn beta(a: f64, b: f64) -> f64 {
    gamma_fn(a) * gamma_fn(b) / gamma_fn(a + b)
}

#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.round() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        let sin_px = (core::f64::consts::PI * x).sin();
        if sin_px == 0.0 {
            return f64::INFINITY;
        }
        return core::f64::consts::PI.ln() - sin_px.abs().ln() - ln_gamma(1.0 - x);
    }

    let z = x - 1.0;
    let g = 7.0;
    let coeffs = LANCZOS_COEFFS;

    let mut a = coeffs[0];
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }

    let t = z + g + 0.5;
    0.5 * (2.0 * core::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Adaptive Simpson quadrature of `f` over `[a, b]` with absolute tolerance
/// `tol` (plus a relative term), capped at `max_eval` function evaluations.
/// Returns the estimated integral.
#[must_use]
pub(crate) fn adaptive_simpson(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_eval: usize,
) -> f64 {
    fn simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let m = 0.5 * (a + b);
        (b - a) / 6.0 * (f(a) + 4.0 * f(m) + f(b))
    }

    fn step(
        f: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        fa: f64,
        fm: f64,
        fb: f64,
        whole: f64,
        tol: f64,
        depth: usize,
        max_eval: usize,
        evals: &mut usize,
    ) -> f64 {
        let m = 0.5 * (a + b);
        let lm = 0.5 * (a + m);
        let rm = 0.5 * (m + b);
        let flm = f(lm);
        let frm = f(rm);
        *evals += 2;
        let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
        let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
        let delta = left + right - whole;
        if depth >= 24 || *evals > max_eval || (delta.abs() <= 15.0 * tol) {
            return left + right + delta / 15.0;
        }
        let tol_half = 0.5 * tol;
        step(f, a, m, fa, flm, fm, left, tol_half, depth + 1, max_eval, evals)
            + step(f, m, b, fm, frm, fb, right, tol_half, depth + 1, max_eval, evals)
    }

    if b <= a {
        return 0.0;
    }
    let fa = f(a);
    let fm = f(0.5 * (a + b));
    let fb = f(b);
    let mut evals = 3usize;
    let whole = simpson(f, a, b);
    step(f, a, b, fa, fm, fb, whole, tol, 0, max_eval, &mut evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_values_are_reasonable() {
        assert!((gamma_fn(5.0) - 24.0).abs() < 1e-10);
        assert!((gamma_fn(0.5) - core::f64::consts::PI.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn ln_gamma_large_argument_stays_finite() {
        let lg = ln_gamma(200.0);
        assert!(lg.is_finite());
        assert!((lg - 857.933_669_825_857_5).abs() < 1e-6);
    }

    #[test]
    fn lower_gamma_matches_gamma_for_large_x() {
        let s = 5.0;
        let lg = lower_gamma(s, 100.0);
        let g = gamma_fn(s);
        assert!((lg - g).abs() / g < 1e-10);
    }

    #[test]
    fn adaptive_simpson_integrates_polynomials_exactly() {
        let f = |x: f64| x * x * x - 2.0 * x + 1.0;
        let i = adaptive_simpson(&f, 0.0, 3.0, 1e-12, 100_000);
        assert!((i - (81.0 / 4.0 - 9.0 + 3.0)).abs() < 1e-9);
    }

    #[test]
    fn adaptive_simpson_integrates_gaussian_tail() {
        let f = |x: f64| (-0.5 * x * x).exp() / (2.0 * core::f64::consts::PI).sqrt();
        let i = adaptive_simpson(&f, -5.0, 5.0, 1e-10, 100_000);
        assert!((i - 1.0).abs() < 1e-7);
    }
}
