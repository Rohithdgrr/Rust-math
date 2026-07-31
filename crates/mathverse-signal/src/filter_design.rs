pub fn filter_design(b: &[f64], a: &[f64]) -> Vec<f64> { b.to_vec() }

pub fn bilinear_transform(s_analog: &[f64], s_denom: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let t = 1.0 / fs;
    let n = s_analog.len().max(s_denom.len());
    let mut b = vec![0.0; n];
    let mut a = vec![0.0; n];
    let mut sum_b = 0.0;
    let mut sum_a = 0.0;
    for (i, &bn) in s_analog.iter().enumerate() {
        let mut term = bn;
        for j in 0..i { term *= -t; }
        sum_b += term;
        b[i] = term;
    }
    for (i, &an) in s_denom.iter().enumerate() {
        let mut term = an;
        for j in 0..i { term *= -t; }
        sum_a += term;
        a[i] = term;
    }
    if sum_a.abs() > 1e-15 { for v in &mut b { *v /= sum_a; } for v in &mut a { *v /= sum_a; } }
    (b, a)
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
    let n = b.len().max(a.len());
    let mut num_re = 0.0; let mut num_im = 0.0;
    let mut den_re = 0.0; let mut den_im = 0.0;
    for i in 0..n {
        let bi = b.get(i).copied().unwrap_or(0.0);
        let ai = a.get(i).copied().unwrap_or(0.0);
        let (cos_wi, sin_wi) = (-w * i as f64).sin_cos();
        num_re += bi * cos_wi;
        num_im += bi * sin_wi;
        den_re += ai * cos_wi;
        den_im += ai * sin_wi;
    }
    let num = num_re * num_im;
    let den = den_re * den_im;
    if den.abs() < 1e-30 { 0.0 } else { -(num - den) / (num_re * num_re + num_im * num_im + den_re * den_re + den_im * den_im) }
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
