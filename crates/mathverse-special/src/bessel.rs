//! Bessel functions of the first, second, and modified kinds for real
//! arguments: `J₀/J₁/Jₙ`, `Y₀/Y₁`, `I₀/I₁`, `K₀/K₁`.
//!
//! Power series per DLMF §10.2 (J), §10.8 (Y), §10.31 (I/K). Accuracy is
//! ~1e-10 for `|x| ≲ 20` and degrades gradually for larger arguments.

use core::f64::consts::PI;

use crate::gamma::EULER_GAMMA;

/// Σₖ (−1)^{alt·k} (x/2)^{2k} / (k! · (k+a)!). Core Bessel block: `a=0`
/// yields the J₀/I₀ series, `a=1` the J₁/I₁ inner series.
fn series_block(x: f64, a: u64, alternating: bool) -> f64 {
    let quarter = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        term *= quarter / (k * (k + a as f64));
        sum += if alternating && k % 2.0 == 1.0 { -term } else { term };
        if term.abs() < sum.abs() * 1e-16 || k > 500.0 {
            break;
        }
        k += 1.0;
    }
    sum
}

/// Harmonic numbers H_m, H_0 = 0.
fn harmonic(m: u64) -> f64 {
    let mut h = 0.0;
    for i in 1..=m {
        h += 1.0 / i as f64;
    }
    h
}

/// Bessel J₀(x), first kind, order zero. Even: J₀(−x) = J₀(x).
///
/// ```
/// use mathverse_special::bessel_j0;
/// assert!((bessel_j0(0.0) - 1.0).abs() < 1e-12);
/// assert!((bessel_j0(1.0) - 0.765_197_686_557_967).abs() < 1e-10);
/// ```
pub fn bessel_j0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    series_block(x.abs(), 0, true)
}

/// Bessel J₁(x), first kind, order one. Odd: J₁(−x) = −J₁(x).
///
/// ```
/// use mathverse_special::bessel_j1;
/// assert!((bessel_j1(0.0) - 0.0).abs() < 1e-12);
/// assert!((bessel_j1(1.0) - 0.440_050_585_744_933_55).abs() < 1e-10);
/// assert!((bessel_j1(-1.0) + 0.440_050_585_744_933_55).abs() < 1e-10);
/// ```
pub fn bessel_j1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x / 2.0 * series_block(x.abs(), 1, true)
}

/// Bessel J_n(x), first kind, integer order n ≥ 0, by forward recurrence
/// J_{ν+1} = (2ν/x) J_ν − J_{ν−1}. Stable for x ≳ n; for x < n the upward
/// recurrence loses accuracy (use Miller's algorithm for small-x large-n).
///
/// ```
/// use mathverse_special::{bessel_j0, bessel_jn};
/// // J₀(2) = 0.2238907791, J₁(2) = 0.5767248078
/// assert!((bessel_jn(0, 2.0) - 0.223_890_779_141_236).abs() < 1e-10);
/// assert!((bessel_jn(2, 2.0) - 0.352_834_028_615_606).abs() < 1e-8);
/// ```
pub fn bessel_jn(n: u64, x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    match n {
        0 => bessel_j0(x),
        1 => bessel_j1(x),
        _ => {
            let mut j_prev = bessel_j0(x);
            let mut j_cur = bessel_j1(x);
            for nu in 1..n {
                let j_next = (2.0 * nu as f64 / x) * j_cur - j_prev;
                j_prev = j_cur;
                j_cur = j_next;
            }
            j_cur
        }
    }
}

/// Bessel Y₀(x), second kind, order zero, DLMF 10.8.2. Defined for x > 0.
///
/// ```
/// use mathverse_special::bessel_y0;
/// assert!((bessel_y0(1.0) - 0.088_256_964_215_676_96).abs() < 1e-8);
/// ```
pub fn bessel_y0(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let two_over_pi = 2.0 / PI;
    let j0 = bessel_j0(x);
    let mut log_series = 0.0; // Σ_{m≥1} (−1)^{m+1} H_m (x/2)^{2m}/(m!)²
    let half2 = x * x / 4.0;
    let mut term = half2; // m = 1 term: H_1 (x/2)² / (1!)²
    let mut m = 1u64;
    loop {
        let t = harmonic(m) * term;
        log_series += if m % 2 == 0 { -t } else { t };
        m += 1;
        if t.abs() < log_series.abs() * 1e-16 || m > 500 {
            break;
        }
        // term_{m+1} = term_m · (x/2)² / (m+1)²
        term *= half2 / ((m * m) as f64);
    }
    two_over_pi * ((x / 2.0).ln() + EULER_GAMMA) * j0 + two_over_pi * log_series
}

/// Bessel Y₁(x), second kind, order one. Derived from the identity
/// `Y₁ = −Y₀′` applied term-by-term to the Y₀ series (DLMF 10.8.2), which is
/// exact and avoids the ψ-function series of DLMF 10.8.1. Defined for x > 0.
///
/// ```
/// use mathverse_special::bessel_y1;
/// assert!((bessel_y1(1.0) + 0.781_212_821_300_288).abs() < 1e-7);
/// ```
pub fn bessel_y1(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let half2 = x * x / 4.0;
    // Σ_{m≥1} (−1)^{m+1} H_m (2m/x) (x/2)^{2m}/(m!)², the x-derivative of the
    // Y₀ harmonic series.
    let mut term = half2; // m = 1 term
    let mut d_series = harmonic(1) * (2.0 * 1.0 / x) * term;
    let mut m = 2u64;
    loop {
        term *= half2 / ((m * m) as f64);
        let t = harmonic(m) * (2.0 * m as f64 / x) * term;
        d_series += if m % 2 == 0 { -t } else { t };
        if t.abs() < d_series.abs() * 1e-16 || m > 500 {
            break;
        }
        m += 1;
    }
    let two_over_pi = 2.0 / PI;
    two_over_pi * (bessel_j1(x) * ((x / 2.0).ln() + EULER_GAMMA) - bessel_j0(x) / x + d_series)
}

/// Modified Bessel I₀(x). Even, I₀(0) = 1.
///
/// ```
/// use mathverse_special::bessel_i0;
/// assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
/// assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-10);
/// ```
pub fn bessel_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    series_block(x.abs(), 0, false)
}

/// Modified Bessel I₁(x). Odd: I₁(−x) = −I₁(x).
///
/// ```
/// use mathverse_special::bessel_i1;
/// assert!((bessel_i1(0.0) - 0.0).abs() < 1e-12);
/// assert!((bessel_i1(1.0) - 0.565_159_103_992_485).abs() < 1e-10);
/// ```
pub fn bessel_i1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x / 2.0 * series_block(x.abs(), 1, false)
}

/// Modified Bessel K₀(x), x > 0.
/// K₀(x) = −I₀(x)(ln(x/2) + γ) + Σ_{m≥1} H_m (x/2)^{2m}/(m!)² (DLMF 10.31.2).
///
/// ```
/// use mathverse_special::bessel_k0;
/// assert!((bessel_k0(1.0) - 0.421_024_438_240_708).abs() < 1e-10);
/// ```
pub fn bessel_k0(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let half2 = x * x / 4.0;
    let mut term = half2; // m = 1
    let mut sum = harmonic(1) * term;
    let mut m = 2u64;
    loop {
        term *= half2 / ((m * m) as f64);
        let t = harmonic(m) * term;
        sum += t;
        if t.abs() < sum.abs() * 1e-16 || m > 500 {
            break;
        }
        m += 1;
    }
    -bessel_i0(x) * ((x / 2.0).ln() + EULER_GAMMA) + sum
}

/// Modified Bessel K₁(x), x > 0. Derived as −K₀′(x):
/// K₁(x) = I₁(x)(ln(x/2) + γ) + I₀(x)/x − Σ_{m≥1} H_m·(2m/x)·(x/2)^{2m}/(m!)².
///
/// ```
/// use mathverse_special::bessel_k1;
/// assert!((bessel_k1(1.0) - 0.601_907_230_552_613).abs() < 1e-9);
/// ```
pub fn bessel_k1(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let half2 = x * x / 4.0;
    let mut term = half2; // m = 1
    let mut sum = harmonic(1) * (2.0 * 1.0 / x) * term;
    let mut m = 2u64;
    loop {
        term *= half2 / ((m * m) as f64);
        let t = harmonic(m) * (2.0 * m as f64 / x) * term;
        sum += t;
        if t.abs() < sum.abs() * 1e-16 || m > 500 {
            break;
        }
        m += 1;
    }
    bessel_i1(x) * ((x / 2.0).ln() + EULER_GAMMA) + bessel_i0(x) / x - sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j0_j1_reference() {
        // mpmath values
        assert!((bessel_j0(2.0) - 0.223_890_779_141_236).abs() < 1e-10);
        assert!((bessel_j1(2.0) - 0.576_724_807_756_873).abs() < 1e-10);
        assert!((bessel_j0(0.5) - 0.938_469_807_240_813).abs() < 1e-10);
        assert!((bessel_j1(0.5) - 0.242_268_457_674_873).abs() < 1e-10);
        assert!((bessel_jn(5, 1.0) - 0.000_249_757_730_211).abs() < 1e-9);
    }

    #[test]
    fn y0_y1_reference() {
        assert!((bessel_y0(0.5) + 0.444_518_733_509_355).abs() < 1e-7);
        assert!((bessel_y0(2.0) - 0.510_375_672_649_065).abs() < 1e-7);
        assert!((bessel_y1(2.0) - (-0.107_032_431_540_936)).abs() < 1e-5);
    }

    #[test]
    fn i0_i1_reference() {
        assert!((bessel_i0(2.0) - 2.279_585_302_336_068).abs() < 1e-8);
        assert!((bessel_i1(2.0) - 1.590_636_854_637_329).abs() < 1e-8);
    }

    #[test]
    fn k0_k1_reference() {
        assert!((bessel_k0(0.5) - 0.924_419_071_227_666).abs() < 1e-8);
        assert!((bessel_k0(2.0) - 0.113_893_872_749_533).abs() < 1e-8);
        assert!((bessel_k1(0.5) - 1.656_441_120_003_531).abs() < 1e-8);
        assert!((bessel_k1(2.0) - 0.139_865_881_816_973).abs() < 1e-8);
    }

    #[test]
    fn symmetry_and_domains() {
        assert!((bessel_j0(-1.0) - bessel_j0(1.0)).abs() < 1e-12);
        assert!((bessel_j1(-1.0) + bessel_j1(1.0)).abs() < 1e-12);
        assert!((bessel_i0(-1.0) - bessel_i0(1.0)).abs() < 1e-12);
        assert!((bessel_i1(-1.0) + bessel_i1(1.0)).abs() < 1e-12);
        assert!(bessel_y0(0.0).is_nan());
        assert!(bessel_y1(-1.0).is_nan());
        assert!(bessel_k0(-1.0).is_nan());
        assert!(bessel_k1(0.0).is_nan());
    }
}
