#[must_use]
pub fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        core::f64::consts::PI / ((core::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let x = x - 1.0;
        let g: f64 = 7.0;
        let c = [
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
        let mut t = c[0];
        for i in 1..g as usize + 2 {
            t += c[i] / (x + i as f64);
        }
        let tt = (x + g + 0.5).ln();
        ((x + 0.5) * tt).exp()
            * tt.exp()
            * (-g - 0.5).exp()
            * (2.0 * core::f64::consts::PI).sqrt()
            * t
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
    if x < 0.0 {
        return 0.0;
    }
    let n = 200;
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;
    for k in 1..=n {
        term *= x / (s + k as f64);
        sum += term;
    }
    sum * x.powf(s) * (-x).exp()
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
    gamma_fn(x).ln()
}
