/// Arithmetic mean of a slice.
pub fn mean(v: &[f64]) -> f64 { crate::operations::sum_fast(v) / v.len() as f64 }

/// Population variance of a slice.
pub fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}
/// Population standard deviation.
pub fn std_dev(v: &[f64]) -> f64 { variance(v).sqrt() }

/// Covariance of two equal-length slices.
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let ma = mean(a); let mb = mean(b);
    a.iter().zip(b).map(|(x,y)| (x-ma)*(y-mb)).sum::<f64>() / a.len() as f64
}
/// Pearson correlation coefficient of two slices. Returns 0.0 if either has zero variance.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let c = covariance(a, b);
    let sa = std_dev(a); let sb = std_dev(b);
    if sa == 0.0 || sb == 0.0 { 0.0 } else { c / (sa * sb) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn mean_test() { assert!((mean(&[1.0,2.0,3.0]) - 2.0).abs() < 1e-10); }
    #[test] fn corr_test() {
        let a = vec![1.0,2.0,3.0,4.0];
        let b = vec![2.0,4.0,6.0,8.0];
        assert!((correlation(&a, &b) - 1.0).abs() < 1e-10);
    }
}
