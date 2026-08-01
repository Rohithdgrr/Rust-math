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
        let coeffs = [
            0.999_999_999_999_809_9,
            676.5203681218851,
            -1259.1392167224028,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507343278686905,
            -0.13857109526572012,
            9.984_369_578_019_572e-6,
            1.5056327351493116e-7,
        ];

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
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
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
    if x < 0.0 || s <= 0.0 || !s.is_finite() || !x.is_finite() {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    const EPS: f64 = 1e-14;
    const MAX_ITERS: usize = 10_000;
    const FPMIN: f64 = 1e-300;

    let gln = ln_gamma(s);
    if x < s + 1.0 {
        // Series for P(s, x), then scale by Γ(s).
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
        let p = sum * (s * x.ln() - x - gln).exp();
        p * gln.exp()
    } else {
        // Continued fraction for Q(s, x), then use γ(s,x) = Γ(s) * (1 - Q(s,x)).
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
        let p = (1.0 - q).clamp(0.0, 1.0);
        p * gln.exp()
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
    let coeffs = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let mut a = coeffs[0];
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }

    let t = z + g + 0.5;
    0.5 * (2.0 * core::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
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
}
