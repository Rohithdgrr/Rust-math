//! Signal detection: envelope follower, threshold crossing, peak detection.

pub fn envelope(x: &[f64]) -> Vec<f64> {
    let mut env = Vec::with_capacity(x.len());
    let mut state = 0.0;
    let attack = 0.1;
    let release = 0.01;
    for &xi in x {
        let target = xi.abs();
        let coeff = if target > state { attack } else { release };
        state = state + coeff * (target - state);
        env.push(state);
    }
    env
}

pub fn onset_detection(x: &[f64], hop: usize) -> Vec<usize> {
    let mut onsets = Vec::new();
    for i in hop..x.len() - hop {
        let left: f64 = (i - hop..i).map(|j| x[j].abs()).sum();
        let right: f64 = (i..i + hop).map(|j| x[j].abs()).sum();
        if right > 1.2 * left && right > 0.1 { onsets.push(i); }
    }
    onsets
}

pub fn zero_crossing_rate(x: &[f64]) -> f64 {
    if x.len() < 2 { return 0.0; }
    let crosses = x.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    crosses as f64 / (x.len() - 1) as f64
}

pub fn spectral_centroid(magnitudes: &[f64], sample_rate: f64) -> f64 {
    let n = magnitudes.len();
    let mut weighted_sum = 0.0;
    let mut total = 0.0;
    for (i, &m) in magnitudes.iter().enumerate() {
        let freq = i as f64 * sample_rate / (2.0 * n as f64);
        weighted_sum += freq * m;
        total += m;
    }
    if total == 0.0 { 0.0 } else { weighted_sum / total }
}

pub fn spectral_rolloff(magnitudes: &[f64], threshold: f64) -> usize {
    let total: f64 = magnitudes.iter().sum();
    let mut cumsum = 0.0;
    for (i, &m) in magnitudes.iter().enumerate() {
        cumsum += m;
        if cumsum >= threshold * total { return i; }
    }
    magnitudes.len() - 1
}

pub fn dynamic_range(x: &[f64]) -> f64 {
    if x.is_empty() { return 0.0; }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = x.iter().cloned().filter(|v| *v > 0.0).fold(f64::INFINITY, f64::min);
    if !min.is_finite() || min <= 0.0 { f64::INFINITY } else { 20.0 * (max / min).log10() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zcr_test() {
        let x = [1.0, -1.0, 1.0, -1.0];
        assert!((zero_crossing_rate(&x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn envelope_test() {
        let x = [0.0, 1.0, 0.0, 1.0];
        let env = envelope(&x);
        assert_eq!(env.len(), 4);
    }
}
