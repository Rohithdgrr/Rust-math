//! IIR filter implementation using second-order sections (biquad) for stability.

/// Second-order IIR filter section (biquad) with transfer function
/// H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2).
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    /// Feedforward coefficient for the current input sample.
    pub b0: f64,
    /// Feedforward coefficient for the previous input sample.
    pub b1: f64,
    /// Feedforward coefficient for the input two samples back.
    pub b2: f64,
    /// Feedback coefficient for the previous output sample.
    pub a1: f64,
    /// Feedback coefficient for the output two samples back.
    pub a2: f64,
}

impl Biquad {
    /// Design a low-pass biquad (RBJ cookbook, Butterworth Q).
    #[must_use]
    pub fn lowpass(fs: f64, fc: f64) -> Self {
        let w = 2.0 * core::f64::consts::PI * fc / fs;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / 2.0f64.sqrt();
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cs) / 2.0 / a0;
        Self {
            b0,
            b1: 2.0 * b0,
            b2: b0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
    /// Design a high-pass biquad (RBJ cookbook, Butterworth Q).
    #[must_use]
    pub fn highpass(fs: f64, fc: f64) -> Self {
        let w = 2.0 * core::f64::consts::PI * fc / fs;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / 2.0f64.sqrt();
        let a0 = 1.0 + alpha;
        let b0 = f64::midpoint(1.0, cs) / a0;
        Self {
            b0,
            b1: -2.0 * b0,
            b2: b0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
    /// Design a band-pass biquad with peak gain unity and quality factor `q`.
    #[must_use]
    pub fn bandpass(fs: f64, fc: f64, q: f64) -> Self {
        let w = 2.0 * core::f64::consts::PI * fc / fs;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
    /// Design a band-stop (notch) biquad with quality factor `q`.
    #[must_use]
    pub fn bandstop(fs: f64, fc: f64, q: f64) -> Self {
        let w = 2.0 * core::f64::consts::PI * fc / fs;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = 1.0 / a0;
        Self {
            b0,
            b1: -2.0 * cs / a0,
            b2: b0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
    /// Filter an input signal, returning the filtered output samples.
    #[must_use]
    pub fn process(&self, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len());
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for &xn in x {
            let yn = self.b0 * xn + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            out.push(yn);
            x2 = x1;
            x1 = xn;
            y2 = y1;
            y1 = yn;
        }
        out
    }
}

/// Magnitude response of a biquad at normalized angular frequency `w`.
#[must_use]
pub fn biquad_magnitude(b: &Biquad, w: f64) -> f64 {
    let num = (b.b0 + b.b1 * (-w).cos() + b.b2 * (-2.0 * w).cos()).powi(2)
        + (b.b1 * (-w).sin() + b.b2 * (-2.0 * w).sin()).powi(2);
    let den = (1.0 + b.a1 * (-w).cos() + b.a2 * (-2.0 * w).cos()).powi(2)
        + (b.a1 * (-w).sin() + b.a2 * (-2.0 * w).sin()).powi(2);
    (num / den).sqrt()
}

/// Apply multiple biquads in series (cascade form).
#[must_use]
pub fn cascade(biquads: &[Biquad], x: &[f64]) -> Vec<f64> {
    biquads.iter().fold(x.to_vec(), |acc, bq| bq.process(&acc))
}

/// Sum the outputs of multiple biquads running in parallel (parallel form).
#[must_use]
pub fn parallel(biquads: &[Biquad], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for bq in biquads {
        let y = bq.process(x);
        for i in 0..n {
            out[i] += y[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biquad_test() {
        let bq = Biquad::lowpass(48000.0, 2000.0);
        assert!((biquad_magnitude(&bq, 0.0) - 1.0).abs() < 1e-9);
        assert!(biquad_magnitude(&bq, core::f64::consts::PI) < 1e-2);
    }
}
