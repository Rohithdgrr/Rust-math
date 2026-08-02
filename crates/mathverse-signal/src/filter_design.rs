//! Filter design: bilinear transform, Butterworth analog prototypes.

pub fn filter_design(b: &[f64], a: &[f64]) -> Vec<f64> { b.to_vec() }

pub fn bilinear_transform(s_analog: &[f64], s_denom: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    // Standard bilinear transform: s = (2/T)(z-1)/(z+1)
    // For each s^k term, substitute: s^k -> ((2fs)^k) * (z-1)^k / (z+1)^k
    // Then multiply numerator and denominator by (z+1)^N where N = max order
    let two_fs = 2.0 * fs;
    let n = s_analog.len().max(s_denom.len()) - 1; // order
    // Evaluate analog transfer function at s = (2fs)(z-1)/(z+1)
    // by computing H(z) = numerator(z) / denominator(z)
    // using the substitution: for polynomial p(s) = Σ c_k s^k
    // p((2fs)(z-1)/(z+1)) * (z+1)^N = Σ c_k (2fs)^k (z-1)^k (z+1)^{N-k}
    let z_minus_1 = vec![-1.0, 1.0]; // (z-1)
    let z_plus_1 = vec![1.0, 1.0];   // (z+1)

    fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut r = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() { r[i + j] += ai * bj; }
        }
        r
    }

    fn poly_pow(base: &[f64], exp: usize) -> Vec<f64> {
        let mut result = vec![1.0];
        for _ in 0..exp { result = poly_mul(&result, base); }
        result
    }

    fn poly_scale(p: &[f64], s: f64) -> Vec<f64> { p.iter().map(|v| v * s).collect() }

    let mut num_poly = vec![0.0]; // accumulated numerator
    let mut den_poly = vec![0.0]; // accumulated denominator
    for (k, &ck) in s_analog.iter().enumerate() {
        let factor = two_fs.powi(k as i32);
        let z_m1_k = poly_pow(&z_minus_1, k);
        let z_p1_nk = poly_pow(&z_plus_1, n - k);
        let term = poly_scale(&poly_mul(&z_m1_k, &z_p1_nk), ck * factor);
        // pad to same length
        let max_len = num_poly.len().max(term.len());
        num_poly.resize(max_len, 0.0);
        for (i, &v) in term.iter().enumerate() { num_poly[i] += v; }
    }
    for (k, &ck) in s_denom.iter().enumerate() {
        let factor = two_fs.powi(k as i32);
        let z_m1_k = poly_pow(&z_minus_1, k);
        let z_p1_nk = poly_pow(&z_plus_1, n - k);
        let term = poly_scale(&poly_mul(&z_m1_k, &z_p1_nk), ck * factor);
        let max_len = den_poly.len().max(term.len());
        den_poly.resize(max_len, 0.0);
        for (i, &v) in term.iter().enumerate() { den_poly[i] += v; }
    }
    // Normalize so den_poly[0] = 1
    let d0 = den_poly.first().copied().unwrap_or(1.0);
    if d0.abs() > 1e-30 {
        for v in &mut num_poly { *v /= d0; }
        for v in &mut den_poly { *v /= d0; }
    }
    (num_poly, den_poly)
}

pub fn impulse_response(b: &[f64], length: usize) -> Vec<f64> {
    let mut out = vec![0.0; length];
    let order = b.len().min(length);
    for i in 0..order { out[i] = b[i]; }
    out
}

pub fn step_response(b: &[f64], length: usize) -> Vec<f64> {
    let ir = impulse_response(b, length);
    let mut sr = Vec::with_capacity(length);
    let mut acc = 0.0;
    for &v in &ir { acc += v; sr.push(acc); }
    sr
}

pub fn group_delay(b: &[f64], a: &[f64], w: f64) -> f64 {
    // Group delay τ(ω) = -dφ/dω = Re{H'·conj(H)}/|H|²
    // where H(ω) = B(ω)/A(ω), H' = (B'A - BA')/A²
    // τ = Re{B'·conj(B)}/|B|² - Re{A'·conj(A)}/|A|²
    fn eval_poly(p: &[f64], w: f64) -> (f64, f64) {
        let (mut re, mut im) = (0.0, 0.0);
        for (k, &ck) in p.iter().enumerate() {
            let (c, s) = (-w * k as f64).sin_cos();
            re += ck * c;
            im += ck * s;
        }
        (re, im)
    }
    fn eval_deriv(p: &[f64], w: f64) -> (f64, f64) {
        let (mut re, mut im) = (0.0, 0.0);
        for (k, &ck) in p.iter().enumerate() {
            let kf = k as f64;
            let (c, s) = (kf * w).sin_cos();
            // d/dw [ck * e^{-jkw}] = ck * (-jk) * e^{-jkw}
            // = ck * (-jk)(cos(kw) - j*sin(kw))
            // = ck * (-jk*cos(kw) - k*sin(kw))
            // Re part: -ck * k * sin(kw)
            // Im part: -ck * k * cos(kw)
            re += ck * (-kf) * s;
            im += ck * (-kf) * c;
        }
        (re, im)
    }

    let (bre, bim) = eval_poly(b, w);
    let (are, aim) = eval_poly(a, w);
    let (bdre, bdim) = eval_deriv(b, w);
    let (adre, adim) = eval_deriv(a, w);

    let b_mag2 = bre * bre + bim * bim;
    let a_mag2 = are * are + aim * aim;

    // Re{B'·conj(B)} = bdre*bre + bdim*bim
    let gd_num = if b_mag2 > 1e-30 { (bdre * bre + bdim * bim) / b_mag2 } else { 0.0 };
    let gd_den = if a_mag2 > 1e-30 { (adre * are + adim * aim) / a_mag2 } else { 0.0 };

    gd_num - gd_den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_test() {
        let ir = impulse_response(&[1.0, 0.5, 0.25], 5);
        assert_eq!(ir, vec![1.0, 0.5, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn step_test() {
        let sr = step_response(&[1.0, 0.5], 4);
        assert_eq!(sr, vec![1.0, 1.5, 1.5, 1.5]);
    }
}
