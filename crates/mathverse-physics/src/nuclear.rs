//! Nuclear physics: radioactive decay and the semi-empirical mass formula.



/// Radioactive decay: N(t) = N₀ · exp(−λ · t).
///
/// `half_life` in seconds. Returns the fraction of nuclei remaining after
/// `time` seconds.
pub fn decay_remaining(half_life: f64, time: f64) -> f64 {
    let lambda = decay_constant(half_life);
    (-lambda * time).exp()
}

/// Decay constant λ = ln(2) / t½ from a half-life.
pub fn decay_constant(half_life: f64) -> f64 {
    std::f64::consts::LN_2 / half_life
}

/// Half-life from decay constant: t½ = ln(2) / λ.
///
/// Returns `None` when `lambda == 0`.
pub fn half_life(lambda: f64) -> Option<f64> {
    if lambda == 0.0 {
        return None;
    }
    Some(std::f64::consts::LN_2 / lambda)
}

/// Binding energy per nucleon via the semi-empirical (Bethe–Weizsäcker) mass
/// formula, simplified:
///
/// ```text
/// BE/A ≈ 15.75
///        − 17.8·A⁻¹/³          (surface)
///        − 0.711·Z²/A⁴/³        (Coulomb)
///        − 23.7·(A−2Z)²/A²      (asymmetry)
///        ± 12.0·δ/A             (pairing; δ = +1 even-even, −1 odd-odd, 0 mixed)
/// ```
///
/// Result is in MeV. Divide by 1e6 to get joules (roughly).
///
/// # Panics
///
/// Never panics; returns `0.0` for `a == 0`.
pub fn binding_energy_per_nucleon(a: u32, z: u32) -> f64 {
    if a == 0 {
        return 0.0;
    }
    let a = a as f64;
    let z = z as f64;
    let volume = 15.75;
    let surface = -17.8 * a.powf(-1.0 / 3.0);
    let coulomb = -0.711 * z * z / a.powf(4.0 / 3.0);
    // All terms here are per-nucleon; the asymmetry term of the total binding
    // energy is −a_A·(A−2Z)²/A, so dividing by A again yields the per-nucleon
    // form −a_A·(A−2Z)²/A².
    let asymmetry = -23.7 * (a - 2.0 * z).powi(2) / a.powi(2);
    let pairing = {
        let a_usize = a as usize;
        if a_usize % 2 == 0 {
            if z as usize % 2 == 0 {
                12.0 / a // even-even → +δ
            } else {
                -12.0 / a // odd-odd → −δ
            }
        } else {
            0.0 // mixed → δ = 0
        }
    };
    volume + surface + coulomb + asymmetry + pairing
}

/// Number of un-decayed nuclei after `time` seconds given a known half-life
/// and an initial quantity `n0`.
pub fn remaining_nuclei(n0: f64, half_life: f64, time: f64) -> f64 {
    n0 * decay_remaining(half_life, time)
}

/// Activity (decays per second) from a half-life and the number of undecayed
/// nuclei `n`.
pub fn activity(n: f64, half_life: f64) -> f64 {
    n * decay_constant(half_life)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_decay_remaining() {
        // After exactly one half-life (t = t½ = 1 s), half should remain
        assert_relative_eq!(decay_remaining(1.0, 1.0), 0.5, epsilon = 1e-15);
        // At t = 0, all remain
        assert_relative_eq!(decay_remaining(10.0, 0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_decay_constant() {
        assert_relative_eq!(decay_constant(std::f64::consts::LN_2), 1.0, epsilon = 1e-15);
        assert_relative_eq!(half_life(1.0).unwrap(), std::f64::consts::LN_2, epsilon = 1e-15);
        assert!(half_life(0.0).is_none());
    }

    #[test]
    fn test_binding_energy_per_nucleon() {
        let be_iron = binding_energy_per_nucleon(56, 26);
        // Iron-56 has ~8.8 MeV/nucleon
        assert!(be_iron > 8.0 && be_iron < 9.0);
        // Light nuclei have lower binding energy per nucleon
        let be_hydrogen = binding_energy_per_nucleon(1, 1);
        assert!(be_hydrogen < be_iron);
        assert!(binding_energy_per_nucleon(0, 0) == 0.0);
    }

    #[test]
    fn test_activity() {
        let act = activity(1000.0, std::f64::consts::LN_2);
        assert_relative_eq!(act, 1000.0, epsilon = 1e-12);
    }
}
