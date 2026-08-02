//! Amplitude modulation (AM), frequency modulation (FM), phase modulation (PM).

pub fn amplitude_modulate(carrier: &[f64], modulating: &[f64], depth: f64) -> Vec<f64> {
    let n = carrier.len().min(modulating.len());
    (0..n).map(|i| carrier[i] * (1.0 + depth * modulating[i])).collect()
}

pub fn frequency_modulate(carrier: &[f64], modulating: &[f64], mod_index: f64, sample_rate: f64) -> Vec<f64> {
    let n = carrier.len().min(modulating.len());
    let mut phase = 0.0;
    (0..n).map(|i| {
        let inst_freq = 1.0 + mod_index * modulating[i];
        phase += 2.0 * core::f64::consts::PI * inst_freq / sample_rate;
        phase.sin()
    }).collect()
}

pub fn am_demodulate(signal: &[f64], carrier_freq: f64, sample_rate: f64) -> Vec<f64> {
    let n = signal.len();
    let mut envelope = Vec::with_capacity(n);
    let mut state = 0.0;
    for (i, &s) in signal.iter().enumerate() {
        let carrier = (2.0 * core::f64::consts::PI * carrier_freq * i as f64 / sample_rate).sin();
        let demod = s * carrier;
        let target = demod.abs();
        let coeff = if target > state { 0.1 } else { 0.01 };
        state += coeff * (target - state);
        envelope.push(state);
    }
    envelope
}

pub fn fsk_modulate(bits: &[bool], freq0: f64, freq1: f64, samples_per_bit: usize, sample_rate: f64) -> Vec<f64> {
    let mut signal = Vec::new();
    for &bit in bits {
        let freq = if bit { freq1 } else { freq0 };
        for i in 0..samples_per_bit {
            signal.push((2.0 * core::f64::consts::PI * freq * i as f64 / sample_rate).sin());
        }
    }
    signal
}

pub fn bpsk_modulate(bits: &[bool], carrier_freq: f64, samples_per_bit: usize, sample_rate: f64) -> Vec<f64> {
    let mut signal = Vec::new();
    for &bit in bits {
        let phase = if bit { 0.0 } else { core::f64::consts::PI };
        for i in 0..samples_per_bit {
            signal.push((2.0 * core::f64::consts::PI * carrier_freq * i as f64 / sample_rate + phase).sin());
        }
    }
    signal
}

pub fn db_to_linear(db: f64) -> f64 { 10.0_f64.powf(db / 20.0) }

pub fn linear_to_db(linear: f64) -> f64 { 20.0 * linear.abs().max(1e-15).log10() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn am_test() {
        let carrier = vec![1.0; 100];
        let modulating: Vec<f64> = (0..100).map(|i| (2.0 * core::f64::consts::PI * i as f64 / 50.0).sin()).collect();
        let am = amplitude_modulate(&carrier, &modulating, 0.5);
        assert_eq!(am.len(), 100);
    }

    #[test]
    fn db_test() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
    }
}
