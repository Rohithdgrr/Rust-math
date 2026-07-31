pub fn solve_linear(a: f64, b: f64) -> Vec<f64> {
    if a.abs() < 1e-15 {
        if b.abs() < 1e-15 { vec![0.0] } else { vec![] }
    } else {
        vec![-b / a]
    }
}

pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-15 { return solve_linear(b, c); }
    let disc = b * b - 4.0 * a * c;
    if disc < -1e-15 { vec![] }
    else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
    else {
        let sqrt_disc = disc.sqrt();
        vec![(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    }
}

pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < 1e-15 { return solve_quadratic(b, c, d); }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc > 1e-15 {
        let sq = disc.sqrt();
        let u = (-q / 2.0 + sq).cbrt();
        let v = (-q / 2.0 - sq).cbrt();
        vec![u + v - b / 3.0]
    } else if disc.abs() < 1e-15 {
        let u = if q.abs() < 1e-30 { 0.0 } else { (-q / 2.0).cbrt() };
        vec![2.0 * u - b / 3.0, -u - b / 3.0]
    } else {
        let r = (-p * p * p / 27.0).sqrt();
        let theta = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        let m = 2.0 * r.cbrt();
        vec![
            m * (theta / 3.0).cos() - b / 3.0,
            m * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0,
            m * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() - b / 3.0,
        ]
    }
}

pub fn solve_quartic(a: f64, b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    if a.abs() < 1e-15 { return solve_cubic(b, c, d, e); }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;
    let p = c - 3.0 * b * b / 8.0;
    let q = b * b * b / 8.0 - b * c / 2.0 + d;
    let r = -3.0 * b * b * b * b / 256.0 + c * b * b / 16.0 - b * d / 4.0 + e;
    if q.abs() < 1e-15 {
        let disc = p * p - 4.0 * r;
        if disc < -1e-15 { return vec![]; }
        let sqrt_disc = disc.sqrt();
        let mut roots = Vec::new();
        for sign in [-1.0, 1.0] {
            let inner = (-p + sign * sqrt_disc) / 2.0;
            if inner >= 0.0 {
                let sq = inner.sqrt();
                roots.push(sq - b / 4.0);
                roots.push(-sq - b / 4.0);
            }
        }
        roots
    } else {
        let cubic_roots = solve_cubic(1.0, -p, -4.0 * r, 4.0 * p * r - q * q);
        let y = cubic_roots[0];
        let sq1 = (2.0 * y - p).max(0.0).sqrt();
        let sq2 = ((y * y - r).max(0.0)).sqrt();
        let mut roots = Vec::new();
        for sign in [-1.0, 1.0] {
            let inner = -y + sign * 2.0 * q / (2.0 * sq1 + 1e-30);
            if inner >= -1e-15 {
                roots.push(sq1 / 2.0 + inner.max(0.0).sqrt() / 2.0 - b / 4.0);
                roots.push(sq1 / 2.0 - inner.max(0.0).sqrt() / 2.0 - b / 4.0);
            }
        }
        roots
    }
}

pub fn polynomial_eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear() {
        assert_eq!(solve_linear(2.0, -4.0), vec![2.0]);
        assert_eq!(solve_linear(0.0, 1.0), Vec::<f64>::new());
    }

    #[test]
    fn quadratic() {
        let r = solve_quadratic(1.0, -3.0, 2.0);
        assert_eq!(r.len(), 2);
        assert!(r.contains(&1.0) && r.contains(&2.0));
    }

    #[test]
    fn cubic() {
        let r = solve_cubic(1.0, -6.0, 11.0, -6.0);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn eval() {
        assert!((polynomial_eval(&[-6.0, 11.0, -6.0, 1.0], 2.0)).abs() < 1e-10);
    }
}
