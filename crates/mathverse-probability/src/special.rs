pub fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        core::f64::consts::PI / ((core::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let x = x - 1.0;
        let g = 7.0;
        let c = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        ];
        let mut t = c[0];
        for i in 1..g as usize + 2 { t += c[i] / (x + i as f64); }
        let tt = (x + g as f64 + 0.5).ln();
        ((x + 0.5) * tt).exp() * tt.exp() * (-g - 0.5).exp() * (2.0 * core::f64::consts::PI).sqrt() * t
    }
}

pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn erfc(x: f64) -> f64 { 1.0 - erf(x) }

pub fn lower_gamma(s: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    let n = 200;
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;
    for k in 1..=n {
        term *= x / (s + k as f64);
        sum += term;
    }
    sum * x.powf(s) * (-x).exp()
}

pub fn upper_gamma(s: f64, x: f64) -> f64 {
    gamma_fn(s) - lower_gamma(s, x)
}

pub fn beta(a: f64, b: f64) -> f64 {
    gamma_fn(a) * gamma_fn(b) / gamma_fn(a + b)
}

pub fn ln_gamma(x: f64) -> f64 { gamma_fn(x).ln() }
