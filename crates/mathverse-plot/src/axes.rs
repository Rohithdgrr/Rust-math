//! Mathematical axes: scales, "nice" tick selection, and data ranges.

/// A 1-D numeric range `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    /// Lower bound.
    pub min: f64,
    /// Upper bound.
    pub max: f64,
}

impl Range {
    /// The range spanning all values, or `None` for empty input.
    /// Non-finite values are skipped.
    pub fn compute(values: impl IntoIterator<Item = f64>) -> Option<Self> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut seen = false;
        for v in values {
            if !v.is_finite() {
                continue;
            }
            seen = true;
            min = min.min(v);
            max = max.max(v);
        }
        seen.then_some(Self { min, max })
    }

    /// Uniform outward padding by `frac` of the span. A zero-width span
    /// (constant or single value) expands by a relative epsilon instead of
    /// producing NaN.
    #[must_use]
    pub fn pad(self, frac: f64) -> Self {
        let span = self.span();
        if span > 0.0 {
            let d = span * frac;
            Self {
                min: self.min - d,
                max: self.max + d,
            }
        } else {
            let c = self.min;
            let h = if c == 0.0 { 0.5 } else { c.abs() * frac };
            Self {
                min: c - h,
                max: c + h,
            }
        }
    }

    /// Width of the range (`max - min`).
    #[inline]
    #[must_use]
    pub fn span(self) -> f64 {
        self.max - self.min
    }
}

impl Default for Range {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

/// Mapping between data space and a kernel space in which ticks are uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Scale {
    /// Identity: ordinary linear axis.
    #[default]
    Linear,
    /// Natural logarithm; data must be strictly positive.
    Log,
    /// Signed logarithm `sign(v) * ln(1 + |v|)`, continuous through zero.
    SymLog,
    /// Square root; data must be non-negative.
    Sqrt,
}

impl Scale {
    /// Map a data value into kernel space.
    #[must_use]
    pub fn transform(self, value: f64) -> f64 {
        match self {
            Self::Linear => value,
            Self::Log => value.ln(),
            Self::SymLog => value.signum() * (1.0 + value.abs()).ln(),
            Self::Sqrt => value.sqrt(),
        }
    }

    /// Inverse of [`Scale::transform`].
    #[must_use]
    pub fn inverse(self, kernel: f64) -> f64 {
        match self {
            Self::Linear => kernel,
            Self::Log => kernel.exp(),
            Self::SymLog => kernel.signum() * (kernel.abs().exp() - 1.0),
            Self::Sqrt => kernel * kernel,
        }
    }

    /// "Nice" tick values between `lo` and `hi` (inclusive), targeting about
    /// `count` ticks in data space. Falls back to the endpoints when the
    /// kernel span is empty or non-finite.
    #[must_use]
    pub fn ticks(self, lo: f64, hi: f64, count: usize) -> Vec<f64> {
        let count = count.max(1) as f64;
        let (klo, khi) = (self.transform(lo), self.transform(hi));
        if !(klo.is_finite() && khi.is_finite()) || khi - klo <= 0.0 {
            return if klo.is_finite() {
                vec![lo]
            } else {
                Vec::new()
            };
        }
        let raw = (khi - klo) / count;
        let step = nice_step(raw);
        let start = (klo / step).ceil() * step;

        let mut out = Vec::with_capacity((count as usize) * 2 + 2);
        let mut k = start;
        while k <= khi + step * 1e-9 && out.len() < count as usize * 8 + 1 {
            out.push(self.inverse(k));
            k += step;
        }
        if out.is_empty() {
            out.push(lo);
        }
        out
    }
}

/// Round `raw` up to a "nice" step (1, 2, or 5 times a power of ten).
#[must_use]
fn nice_step(raw: f64) -> f64 {
    let mag = 10f64.powf(raw.log10().floor());
    let n = raw / mag;
    let nice = if n < 1.5 {
        1.0
    } else if n < 3.0 {
        2.0
    } else if n < 7.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

/// Resolved axis mapping: forward/inverse transforms and padded kernel
/// bounds.
type KernelAxis = (Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>, Range);

/// Resolved axis mapping: forward/inverse transforms and padded kernel
/// bounds. Padding happens in kernel space, where tick spacing is uniform.
/// Falls back to a linear identity mapping when the scale transform is
/// degenerate on `data` (e.g. `Log` on non-positive values), so rendering
/// never emits NaN.
#[must_use]
pub fn axis_kernel(scale: Scale, data: Range) -> KernelAxis {
    let klo = scale.transform(data.min);
    let khi = scale.transform(data.max);
    if klo.is_finite() && khi.is_finite() && klo < khi {
        let span = khi - klo;
        let k = Range {
            min: klo - span * 0.05,
            max: khi + span * 0.05,
        };
        (
            Box::new(move |v| scale.transform(v)),
            Box::new(move |v| scale.inverse(v)),
            k,
        )
    } else {
        let k = data.pad(0.05);
        (Box::new(|v| v), Box::new(|v| v), k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_compute_and_pad() {
        assert_eq!(Range::compute(std::iter::empty::<f64>()), None);
        assert_eq!(
            Range::compute([3.0, 1.0, 2.0]),
            Some(Range { min: 1.0, max: 3.0 })
        );
        assert_eq!(
            Range::compute([f64::NAN, 2.0]),
            Some(Range { min: 2.0, max: 2.0 })
        );
    }

    #[test]
    fn zero_width_pad_no_nan() {
        let r = Range { min: 5.0, max: 5.0 }.pad(0.05);
        assert!(r.min.is_finite() && r.max.is_finite());
        assert!(r.min < 5.0 && r.max > 5.0);
        let r0 = Range { min: 0.0, max: 0.0 }.pad(0.05);
        assert_eq!(
            r0,
            Range {
                min: -0.5,
                max: 0.5
            }
        );
    }

    #[test]
    fn linear_ticks_are_nice() {
        let ticks = Scale::Linear.ticks(0.0, 10.0, 5);
        assert_eq!(ticks, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert!(ticks.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn scale_inverse_roundtrip() {
        let s = Scale::SymLog;
        for v in [-1e3, -1.0, 0.0, 1.0, 1e3] {
            assert!((s.inverse(s.transform(v)) - v).abs() < 1e-9);
        }
    }
}
