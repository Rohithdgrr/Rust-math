pub fn l1(v: &[f64]) -> f64 { v.iter().map(|x| x.abs()).sum() }
pub fn l2(v: &[f64]) -> f64 { v.iter().map(|x| x*x).sum::<f64>().sqrt() }
pub fn lp(v: &[f64], p: f64) -> f64 { v.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0/p) }
pub fn linf(v: &[f64]) -> f64 { v.iter().map(|x| x.abs()).fold(0.0, f64::max) }
pub fn l0(v: &[f64]) -> usize { v.iter().filter(|&&x| x != 0.0).count() }
pub fn l_neg_inf(v: &[f64]) -> f64 { v.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn l2_test() { assert!((l2(&[3.0,4.0]) - 5.0).abs() < 1e-10); }
    #[test] fn linf_test() { assert!((linf(&[-5.0,3.0,7.0]) - 7.0).abs() < 1e-10); }
}
