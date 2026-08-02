//! Convex analysis: convexity check, convex hull, simplex projection, box constraints.

/// Checks if a 1D function is convex on `[a, b]` by sampling.
pub fn is_convex_1d(f: &dyn Fn(f64) -> f64, a: f64, b: f64, steps: usize) -> bool {
    let dx = (b - a) / steps as f64;
    let (mut f_prev, mut f_curr) = (f(a), f(a + dx));
    for i in 2..=steps {
        let x = a + i as f64 * dx;
        let f_next = f(x);
        if f_curr > (f_prev + f_next) / 2.0 + 1e-10 { return false; }
        f_prev = f_curr;
        f_curr = f_next;
    }
    true
}

pub fn convex_hull_1d(points: &[f64]) -> (f64, f64) {
    (points.iter().cloned().fold(f64::INFINITY, f64::min),
     points.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

pub fn convex_combination(points: &[[f64; 2]], weights: &[f64]) -> [f64; 2] {
    assert_eq!(points.len(), weights.len());
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    let mut result = [0.0; 2];
    for (p, w) in points.iter().zip(weights) {
        result[0] += w * p[0];
        result[1] += w * p[1];
    }
    result
}

pub fn projection_simplex(v: &[f64], lambda: f64) -> Vec<f64> {
    let n = v.len();
    let mut sorted: Vec<(f64, usize)> = v.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut result = vec![0.0; n];
    let mut rho = 0;
    let mut theta = 0.0;
    for k in 0..n {
        let val = sorted[k].0;
        theta += val;
        if val - (theta - lambda) / (k as f64 + 1.0) > 1e-15 { rho = k + 1; } else { break; }
    }
    let tau = (theta - lambda) / rho as f64;
    for i in 0..n { result[i] = (v[i] - tau).max(0.0); }
    result
}

pub fn box_constraint(x: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    x.iter().zip(lo).zip(hi).map(|((xi, &l), &h)| xi.clamp(l, h)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_1d() {
        assert!(is_convex_1d(&|x| x * x, -2.0, 2.0, 100));
        assert!(!is_convex_1d(&|x| x * x * x, -2.0, 2.0, 100));
    }

    #[test]
    fn simplex_proj() {
        let v = vec![1.0, 2.0, 3.0];
        let p = projection_simplex(&v, 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
