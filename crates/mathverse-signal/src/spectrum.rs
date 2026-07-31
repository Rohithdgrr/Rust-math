pub fn periodogram(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for k in 0..n {
        for i in 0..n {
            let angle = -2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re[k] += (x[i] - mean) * angle.cos();
            im[k] += (x[i] - mean) * angle.sin();
        }
    }
    (0..n).map(|k| (re[k] * re[k] + im[k] * im[k]) / n as f64).collect()
}

pub fn welch_psd(x: &[f64], segment_size: usize, overlap: usize) -> Vec<f64> {
    if x.is_empty() || segment_size == 0 || overlap >= segment_size || segment_size > x.len() {
        return periodogram(x);
    }
    let hop = segment_size - overlap;
    let n_segments = (x.len() - segment_size) / hop + 1;
    if n_segments == 0 { return periodogram(x); }
    let mut psd = vec![0.0; segment_size / 2 + 1];
    for seg in 0..n_segments {
        let start = seg * hop;
        let segment = &x[start..start + segment_size];
        let windowed: Vec<f64> = segment.iter().enumerate().map(|(i, &v)| {
            let w = 0.54 - 0.46 * (2.0 * core::f64::consts::PI * i as f64 / (segment_size - 1) as f64).cos();
            v * w
        }).collect();
        let pg = periodogram(&windowed);
        for i in 0..=segment_size / 2 { psd[i] += pg[i]; }
    }
    psd.iter().map(|v| v / n_segments as f64).collect()
}

pub fn autocorrelation(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var == 0.0 { return vec![1.0; n]; }
    (0..n).map(|lag| {
        let sum: f64 = (0..n - lag).map(|i| (x[i] - mean) * (x[i + lag] - mean)).sum();
        sum / (n as f64 * var)
    }).collect()
}

pub fn energy(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum() }

pub fn parseval(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let time_energy: f64 = x.iter().map(|v| v * v).sum();
    let mean: f64 = x.iter().sum::<f64>() / n;
    let freq_energy: f64 = periodogram(x).iter().sum();
    (time_energy - freq_energy).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodogram_test() {
        let x: Vec<f64> = (0..64).map(|i| (2.0 * core::f64::consts::PI * i as f64 / 16.0).sin()).collect();
        let psd = periodogram(&x);
        assert_eq!(psd.len(), 64);
    }
}
