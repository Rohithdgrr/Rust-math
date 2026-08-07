//! Complex special functions: gamma, zeta, polylog, and related functions.

use crate::Complex;

/// Complex special functions.
pub struct ComplexSpecialFunctions;

impl ComplexSpecialFunctions {
    /// Complex gamma function using Lanczos approximation.
    pub fn gamma(z: Complex) -> Complex {
        // Lanczos approximation coefficients
        const P: [f64; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        // Reflection formula for negative real part
        if z.re < 0.5 {
            let pi = Complex::new(core::f64::consts::PI, 0.0);
            let sin_pi_z = (pi * z).sin();
            let gamma_reflected = Self::gamma(Complex::one() - z);
            return pi / (sin_pi_z * gamma_reflected);
        }

        // Lanczos approximation
        let z_minus_1 = z - Complex::one();
        let mut x = Complex::real(P[0]);

        for i in 1..P.len() {
            let coeff = Complex::real(P[i]);
            let denominator = z_minus_1 + Complex::real(i as f64);
            x = x + coeff / denominator;
        }

        let sqrt_2pi = (2.0 * core::f64::consts::PI).sqrt();
        let t = z_minus_1 + Complex::real((P.len() - 2) as f64) + Complex::real(0.5);

        let sqrt_2pi_t = Complex::real(sqrt_2pi) * t.pow(z - Complex::real(0.5));
        let exp_t = (-t).exp();

        sqrt_2pi_t * exp_t * x
    }

    /// Complex digamma function (derivative of log gamma).
    pub fn digamma(z: Complex) -> Complex {
        // Use series expansion for large |z|
        if z.norm() >= 10.0 {
            let inv_z = Complex::one() / z;
            let inv_z2 = inv_z * inv_z;
            let inv_z4 = inv_z2 * inv_z2;

            // Asymptotic expansion: ψ(z) ~ ln(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - 1/(252z⁶) + ...
            let term2 = inv_z2 / Complex::real(12.0);
            let term4 = inv_z4 / Complex::real(120.0);
            let term6 = inv_z2 * inv_z4 / Complex::real(252.0);

            z.ln() - Complex::real(0.5) * inv_z - term2 + term4 - term6
        } else {
            // Use recurrence relation: ψ(z+1) = ψ(z) + 1/z
            let mut n = 0;
            let mut z_shifted = z;

            while z_shifted.norm() < 10.0 && n < 100 {
                z_shifted = z_shifted + Complex::one();
                n += 1;
            }

            let psi_shifted = Self::digamma(z_shifted);
            let mut result = psi_shifted;

            for k in 0..n {
                result = result - Complex::one() / (z + Complex::real(k as f64));
            }

            result
        }
    }

    /// Riemann zeta function for complex arguments.
    pub fn zeta(z: Complex, iterations: usize) -> Complex {
        // For Re(z) < 0, use the functional equation:
        // ζ(s) = 2^s · π^(s-1) · sin(πs/2) · Γ(1-s) · ζ(1-s)
        // (terminates because Re(1-s) > 1)
        if z.re < 0.0 {
            let s = z;
            let pi = Complex::new(std::f64::consts::PI, 0.0);
            let factor = Complex::real(2.0).pow(s)
                * pi.pow(s - Complex::one())
                * (pi * s / Complex::real(2.0)).sin()
                * Self::gamma(Complex::one() - s);
            return factor * Self::zeta(Complex::one() - s, iterations);
        }

        // For Re(z) >= 0, use the Euler-Maclaurin formula (accurate for all s != 1):
        // ζ(s) = Σ_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2
        //        + Σ_{k=1..M} B_{2k}/(2k)! · s(s+1)···(s+2k-2) · N^(-s-2k+1)
        Self::zeta_euler_maclaurin(z, iterations)
    }

    fn zeta_euler_maclaurin(s: Complex, n_terms: usize) -> Complex {
        // Bernoulli numbers B2..B12
        const BERNOULLI: [f64; 6] = [
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2730.0,
        ];

        let n = n_terms.max(10) as f64;
        let n_c = Complex::real(n);

        let mut sum = Complex::zero();
        for k in 1..n as usize {
            sum = sum + Complex::real(k as f64).pow(-s);
        }
        sum = sum
            + n_c.pow(Complex::one() - s) / (s - Complex::one())
            + n_c.pow(-s) / Complex::real(2.0);

        for (i, &b) in BERNOULLI.iter().enumerate() {
            let k2 = 2 * (i + 1);
            let mut rising = s;
            for j in 1..k2 - 1 {
                rising = rising * (s + Complex::real(j as f64));
            }
            let mut factorial = 1.0;
            for j in 2..=k2 {
                factorial *= j as f64;
            }
            let term = Complex::real(b) * rising / Complex::real(factorial)
                * n_c.pow(-s - Complex::real((k2 - 1) as f64));
            sum = sum + term;
        }

        sum
    }

    /// Polylogarithm Li_s(z).
    ///
    /// Computed from the defining series, which converges absolutely for
    /// `|z| < 1` (and conditionally on `|z| = 1` when `Re(s) > 1`).
    ///
    /// For `|z| > 1` the analytic continuation is **not** implemented (the
    /// naive inversion series is numerically unreliable for general complex
    /// `s`); `NaN` is returned rather than silently wrong values.
    pub fn polylog(s: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1.0 {
            // Series expansion: Li_s(z) = Σ_{n=1}^∞ z^n / n^s
            let mut sum = Complex::zero();
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let term = z.pow(n_complex) / n_complex.pow(s);
                sum = sum + term;
            }
            sum
        } else if z.norm() > 1.0 {
            Complex::new(f64::NAN, f64::NAN)
        } else {
            // |z| == 1: series converges for Re(s) > 1; partial sum otherwise
            let mut sum = Complex::zero();
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let term = z.pow(n_complex) / n_complex.pow(s);
                sum = sum + term;
            }
            sum
        }
    }

    /// Complex error function.
    ///
    /// Taylor series for `|z| < 3`; asymptotic expansion with optimal
    /// truncation for `|z| ≥ 3` (where the Taylor series needs hundreds of
    /// terms to converge).
    pub fn erf(z: Complex, iterations: usize) -> Complex {
        // erf is odd: erf(−z) = −erf(z). Mirroring keeps the asymptotic
        // branch inside its valid sector (the positive real half-plane).
        if z.re < 0.0 {
            return -Self::erf(-z, iterations);
        }
        // Asymptotic (DLMF 7.12.1), valid in the sector |arg z| ≤ π/4:
        //   erf(z) ~ 1 − exp(−z²)/(z·√π) · Σₙ (−1)ⁿ (2n−1)!!/(2z²)ⁿ
        // The series is divergent; truncate just past the smallest term.
        if z.norm() >= 3.0 && z.re >= z.im.abs() {
            let z2 = z * z;
            let mut sum = Complex::one();
            let mut term = Complex::one();
            let mut prev_mag = 1.0;
            for n in 1..=iterations {
                term = term * Complex::real((2 * n - 1) as f64) / (Complex::real(2.0) * z2);
                let mag = term.norm();
                if mag > prev_mag {
                    break;
                }
                prev_mag = mag;
                if n % 2 == 0 {
                    sum = sum + term;
                } else {
                    sum = sum - term;
                }
            }
            return Complex::one()
                - (-z2).exp() * sum / (z * Complex::real(core::f64::consts::PI.sqrt()));
        }

        // Taylor series (converges for all z; fast for small |z|)
        let mut sum = Complex::zero();
        for n in 0..=iterations {
            let sign = if n % 2 == 0 {
                Complex::one()
            } else {
                -Complex::one()
            };
            let factorial: f64 = (1..=n).map(|x| x as f64).product();
            let denominator = Complex::real((2 * n + 1) as f64 * factorial);
            let term = sign * z.pow(Complex::real((2 * n + 1) as f64)) / denominator;
            sum = sum + term;
        }
        sum * Complex::new(2.0 / core::f64::consts::PI.sqrt(), 0.0)
    }

    /// Complex complementary error function.
    ///
    /// Uses the direct asymptotic form for large `|z|` so that `erfc` does
    /// not suffer catastrophic cancellation inside `1 − erf(z)`.
    pub fn erfc(z: Complex, iterations: usize) -> Complex {
        // erfc(z) = 2 − erfc(−z); mirror into the positive half-plane so the
        // asymptotic branch (valid for Re(z) > 0) is always used in-sector.
        if z.re < 0.0 {
            return Complex::real(2.0) - Self::erfc(-z, iterations);
        }
        if z.norm() >= 3.0 && z.re >= z.im.abs() {
            // erfc(z) ≈ exp(−z²)/(z·√π) · Σₙ (−1)ⁿ (2n−1)!!/(2z²)ⁿ
            let z2 = z * z;
            let mut sum = Complex::one();
            let mut term = Complex::one();
            let mut prev_mag = 1.0;
            for n in 1..=iterations {
                term = term * Complex::real((2 * n - 1) as f64) / (Complex::real(2.0) * z2);
                let mag = term.norm();
                if mag > prev_mag {
                    break;
                }
                prev_mag = mag;
                if n % 2 == 0 {
                    sum = sum + term;
                } else {
                    sum = sum - term;
                }
            }
            return (-z2).exp() * sum / (z * Complex::real(core::f64::consts::PI.sqrt()));
        }
        Complex::one() - Self::erf(z, iterations)
    }

    /// Complex exponential integral Ei(z).
    pub fn exponential_integral(z: Complex, iterations: usize) -> Complex {
        if z.norm() < 20.0 {
            // Series expansion (globally convergent, slow for large |z|):
            // Ei(z) = γ + ln z + Σ_{n>=1} z^n/(n·n!)
            let mut sum = Complex::real(0.57721566490153286060651209008240243104215933593992); // Euler-Mascheroni constant
            sum = sum + z.ln();

            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).map(|x| x as f64).product();
                sum = sum + z.pow(n_complex) / (n_complex * Complex::real(factorial));
            }

            sum
        } else {
            // Asymptotic expansion for large |z|: Ei(z) ~ e^z · Σ n!/z^(n+1).
            // The series is divergent; truncate at the minimal term.
            let inv_z = Complex::one() / z;
            let mut sum = inv_z;
            let mut prev_mag = sum.norm();

            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).map(|x| x as f64).product();
                let term = Complex::real(factorial) * inv_z.pow(n_complex + Complex::one());
                let mag = term.norm();
                if mag > prev_mag {
                    break;
                }
                sum = sum + term;
                prev_mag = mag;
            }

            sum * z.exp()
        }
    }

    /// Complex Fresnel integrals C(z) and S(z).
    ///
    /// Convention without π/2 factors: C(z) = Σ (-1)^n z^(4n+1)/((2n)! (4n+1)),
    /// S(z) = Σ (-1)^n z^(4n+3)/((2n+1)! (4n+3)).
    pub fn fresnel(z: Complex, iterations: usize) -> (Complex, Complex) {
        let mut c = Complex::zero();
        let mut s = Complex::zero();

        for n in 0..=iterations {
            let _n_complex = Complex::real(n as f64);
            let sign = if n % 2 == 0 {
                Complex::one()
            } else {
                -Complex::one()
            };

            let factorial_2n: f64 = (1..=2 * n).map(|x| x as f64).product();
            let factorial_2n1: f64 = (1..=2 * n + 1).map(|x| x as f64).product();

            let term_c = sign * z.pow(Complex::real(4.0 * n as f64 + 1.0))
                / Complex::real((4 * n + 1) as f64 * factorial_2n);
            let term_s = sign * z.pow(Complex::real(4.0 * n as f64 + 3.0))
                / Complex::real((4 * n + 3) as f64 * factorial_2n1);

            c = c + term_c;
            s = s + term_s;
        }

        (c, s)
    }

    /// Complex Airy functions Ai(z) and Bi(z).
    ///
    /// Ai(z) = c1·f(z) - c2·g(z), Bi(z) = √3·(c1·f(z) + c2·g(z)) with
    /// f(z) = Σ z^(3k)/(9^k k! Γ(k+2/3)), g(z) = Σ z^(3k+1)/(9^k k! Γ(k+4/3)),
    /// c1 = 3^(-2/3), c2 = 3^(-4/3).
    pub fn airy(z: Complex, iterations: usize) -> (Complex, Complex) {
        if z.norm() < 1e-15 {
            // series gives f(0) = 1/Γ(2/3), g(0) = 0
            let c1 =
                Complex::real(3.0_f64.powf(-2.0 / 3.0)) / Self::gamma(Complex::real(2.0 / 3.0));
            let sqrt3 = Complex::real(3.0_f64.sqrt());
            return (c1, sqrt3 * c1);
        }
        // Use series expansion for small |z|
        if z.norm() < 2.0 {
            let mut f = Complex::zero();
            let mut g = Complex::zero();

            for n in 0..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).map(|x| x as f64).product();
                let nine_pow_n = 9.0_f64.powi(n as i32);

                let gamma_f = Self::gamma(n_complex + Complex::real(2.0 / 3.0));
                let gamma_g = Self::gamma(n_complex + Complex::real(4.0 / 3.0));

                f = f + z.pow(Complex::real(3.0 * n as f64))
                    / (Complex::real(nine_pow_n * factorial) * gamma_f);
                g = g + z.pow(Complex::real(3.0 * n as f64 + 1.0))
                    / (Complex::real(nine_pow_n * factorial) * gamma_g);
            }

            let c1 = Complex::real(3.0_f64.powf(-2.0 / 3.0));
            let c2 = Complex::real(3.0_f64.powf(-4.0 / 3.0));
            let sqrt3 = Complex::real(3.0_f64.sqrt());

            let ai = c1 * f - c2 * g;
            let bi = sqrt3 * (c1 * f + c2 * g);
            (ai, bi)
        } else {
            // Asymptotic expansion for large |z|:
            // Ai(z) ~ exp(-ζ) / (2√π z^(1/4)), Bi(z) ~ exp(ζ) / (√π z^(1/4)),
            // ζ = (2/3) z^(3/2)
            let t = Complex::real(2.0 / 3.0) * z.pow(Complex::real(1.5));
            let z_pow_quarter = z.pow(Complex::real(0.25));

            let ai = (-t).exp()
                / (Complex::real(2.0) * Complex::real(std::f64::consts::PI.sqrt()) * z_pow_quarter);
            let bi = t.exp() / (Complex::real(std::f64::consts::PI.sqrt()) * z_pow_quarter);

            (ai, bi)
        }
    }

    /// Complex Bessel function of the first kind J_v(z).
    pub fn bessel_j(v: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1e-15 {
            return if v == Complex::zero() {
                Complex::one()
            } else {
                Complex::zero()
            };
        }
        if z.norm() < 10.0 {
            // Series expansion (DLMF 10.2.2), converges for all finite z
            return Self::bessel_j_series(v, z, iterations);
        }
        // Large |z|: leading-order asymptotic (DLMF 10.17.3),
        // J_v(z) ~ sqrt(2/(πz)) · cos(z − vπ/2 − π/4),
        // valid for |arg z| < π. Near the negative real axis the complex
        // sqrt branches, so fall back to the convergent series with extra
        // terms (|z| = 10 needs ~50 of them).
        if z.arg().abs() < core::f64::consts::PI - 1e-3 {
            let sqrt_term =
                (Complex::real(2.0) / (Complex::real(core::f64::consts::PI) * z)).sqrt();
            let phase = z
                - v * Complex::real(core::f64::consts::FRAC_PI_2)
                - Complex::real(core::f64::consts::FRAC_PI_4);
            sqrt_term * phase.cos()
        } else {
            Self::bessel_j_series(v, z, iterations.saturating_mul(2))
        }
    }

    /// Series expansion of J_v(z) (DLMF 10.2.2), valid for all finite z.
    ///
    /// Stops once a term is negligible (`< 1e-15` in magnitude) or non-finite
    /// (large-n factorial/gamma overflow); the decaying tail can never matter
    /// again by then.
    fn bessel_j_series(v: Complex, z: Complex, iterations: usize) -> Complex {
        let mut sum = Complex::zero();
        for n in 0..=iterations {
            let n_complex = Complex::real(n as f64);
            let factorial_n: f64 = if n == 0 {
                1.0
            } else {
                (1..=n).map(|x| x as f64).product()
            };
            let gamma_v_n = Self::gamma(v + n_complex + Complex::one());
            let term = (Complex::real((-1.0_f64).powi(n as i32))
                * (z / Complex::real(2.0)).pow(v + Complex::real(2.0) * n_complex))
                / (Complex::real(factorial_n) * gamma_v_n);
            if term.is_nan() || (n > 10 && term.norm() < 1e-15) {
                break;
            }
            sum = sum + term;
        }
        sum
    }

    /// Complex Bessel function of the second kind Y_v(z).
    pub fn bessel_y(v: Complex, z: Complex, iterations: usize) -> Complex {
        // Integer order: (J_v cos(vπ) - J_-v)/sin(vπ) is 0/0, use the
        // limit series (DLMF 10.8.1) instead.
        if v.im.abs() < 1e-15 && (v.re - v.re.round()).abs() < 1e-15 {
            let n = v.re.round() as i64;
            let n_abs = n.unsigned_abs() as usize;
            let y = Self::bessel_y_integer(n_abs, z, iterations);
            if n < 0 && n % 2 != 0 {
                -y
            } else {
                y
            }
        } else {
            let j_v = Self::bessel_j(v, z, iterations);
            let j_minus_v = Self::bessel_j(-v, z, iterations);

            let cos_v_pi = (v * Complex::new(std::f64::consts::PI, 0.0)).cos();

            (j_v * cos_v_pi - j_minus_v) / (v * Complex::new(std::f64::consts::PI, 0.0)).sin()
        }
    }

    /// Y_n(z) for integer order n >= 0:
    /// (2/π) ln(z/2) J_n(z) - (1/π) Σ_{k<n} (n-k-1)!/k! (z/2)^(2k-n)
    ///   - (1/π) Σ_k (-1)^k (ψ(k+1)+ψ(k+n+1))/(k!(k+n)!) (z/2)^(2k+n)
    fn bessel_y_integer(n: usize, z: Complex, iterations: usize) -> Complex {
        let pi = std::f64::consts::PI;
        let z_over_2 = z / Complex::real(2.0);
        let j_n = Self::bessel_j(Complex::real(n as f64), z, iterations);

        let mut sum1 = Complex::zero();
        for k in 0..n {
            let fact_n_k_1: f64 = (1..=n - k - 1).map(|x| x as f64).product();
            let fact_k: f64 = (1..=k).map(|x| x as f64).product();
            sum1 = sum1
                + Complex::real(fact_n_k_1 / fact_k)
                    * z_over_2.pow(Complex::real((2 * k) as f64 - n as f64));
        }

        let mut sum2 = Complex::zero();
        for k in 0..=iterations {
            let sign = if k % 2 == 0 {
                Complex::one()
            } else {
                -Complex::one()
            };
            let fact_k: f64 = (1..=k).map(|x| x as f64).product();
            let fact_k_n: f64 = (1..=k + n).map(|x| x as f64).product();
            let psi = Self::digamma(Complex::real((k + 1) as f64))
                + Self::digamma(Complex::real((k + n + 1) as f64));
            sum2 = sum2
                + sign * psi / Complex::real(fact_k * fact_k_n)
                    * z_over_2.pow(Complex::real((2 * k + n) as f64));
        }

        Complex::real(2.0 / pi) * z_over_2.ln() * j_n - Complex::real(1.0 / pi) * (sum1 + sum2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        // Γ(1) = 1
        let result = ComplexSpecialFunctions::gamma(Complex::one());
        assert!((result.re - 1.0).abs() < 0.1);

        // Γ(2) = 1
        let result = ComplexSpecialFunctions::gamma(Complex::real(2.0));
        assert!((result.re - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_digamma() {
        // ψ(1) should be -γ (Euler-Mascheroni constant)
        let result = ComplexSpecialFunctions::digamma(Complex::one());
        let gamma = 0.57721566490153286060651209008240243104215933593992;
        assert!((result.re + gamma).abs() < 0.1);
    }

    #[test]
    fn test_zeta() {
        // ζ(2) = π²/6
        let result = ComplexSpecialFunctions::zeta(Complex::real(2.0), 1000);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((result.re - expected).abs() < 1e-6);
    }

    #[test]
    fn test_polylog() {
        // Li_1(z) = -ln(1-z)
        let z = Complex::real(0.5);
        let result = ComplexSpecialFunctions::polylog(Complex::one(), z, 100);
        let expected = -(Complex::one() - z).ln();

        assert!((result.re - expected.re).abs() < 0.1);
        assert!((result.im - expected.im).abs() < 0.1);
    }

    #[test]
    fn test_erf() {
        // erf(0) = 0
        let result = ComplexSpecialFunctions::erf(Complex::zero(), 50);
        assert!(result.norm() < 0.1);
    }

    #[test]
    fn test_bessel_j() {
        // J_0(0) = 1
        let result = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::zero(), 50);
        assert!((result.re - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ
        let result = ComplexSpecialFunctions::digamma(Complex::one());
        let gamma = 0.57721566490153286060651209008240243104215933593992;
        assert!((result.re + gamma).abs() < 1e-6);

        // ψ(5) = -γ + H_4 = -γ + 25/12
        let psi5 = ComplexSpecialFunctions::digamma(Complex::real(5.0));
        let expected = -gamma + 25.0 / 12.0;
        assert!((psi5.re - expected).abs() < 1e-8);
    }

    #[test]
    fn test_zeta_critical_values() {
        // ζ(0) = -1/2, ζ(-1) = -1/12, ζ(0.5) ≈ -1.46035 (regression: broken functional equation)
        let z0 = ComplexSpecialFunctions::zeta(Complex::zero(), 100);
        assert!((z0.re + 0.5).abs() < 1e-8);
        assert!(z0.im.abs() < 1e-8);

        let zm1 = ComplexSpecialFunctions::zeta(Complex::real(-1.0), 100);
        assert!((zm1.re + 1.0 / 12.0).abs() < 1e-6);
        assert!(zm1.im.abs() < 1e-6);

        let zhalf = ComplexSpecialFunctions::zeta(Complex::real(0.5), 100);
        assert!((zhalf.re + 1.4603545088).abs() < 1e-4);

        // ζ(2) = π²/6 (existing)
        let z2 = ComplexSpecialFunctions::zeta(Complex::real(2.0), 1000);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((z2.re - expected).abs() < 1e-6);

        // ζ(1/2 + 14i) on the critical line (regression: used to recurse forever)
        let critical = ComplexSpecialFunctions::zeta(Complex::new(0.5, 14.0), 100);
        assert!(critical.is_finite());
    }

    #[test]
    fn test_fresnel_small_z() {
        // C(z) ≈ z, S(z) ≈ z³/3 for small z (regression: series had z in the denominator)
        let (c, s) = ComplexSpecialFunctions::fresnel(Complex::real(0.1), 30);
        assert!((c.re - 0.1).abs() < 1e-6);
        assert!((s.re - 0.1f64.powi(3) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fresnel_unit_values() {
        // C(1) ≈ 0.904524, S(1) ≈ 0.310268 (no-π convention)
        let (c, s) = ComplexSpecialFunctions::fresnel(Complex::one(), 30);
        assert!((c.re - 0.9045242379).abs() < 1e-5);
        assert!((s.re - 0.3102683017).abs() < 1e-5);
    }

    #[test]
    fn test_airy_known_values() {
        // Ai(0) ≈ 0.355028, Bi(0) ≈ 0.614927, Ai(1) ≈ 0.135292, Bi(1) ≈ 1.207424
        let (ai0, bi0) = ComplexSpecialFunctions::airy(Complex::zero(), 30);
        assert!((ai0.re - 0.3550280539).abs() < 1e-6);
        assert!((bi0.re - 0.6149266274).abs() < 1e-6);

        let (ai1, bi1) = ComplexSpecialFunctions::airy(Complex::one(), 30);
        assert!((ai1.re - 0.1352924163).abs() < 1e-5);
        assert!((bi1.re - 1.2074235949).abs() < 1e-4);
    }

    #[test]
    fn test_bessel_j_known_values() {
        // J_0(1) ≈ 0.765198, J_1(1) ≈ 0.440051
        let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(1.0), 30);
        assert!((j0.re - 0.7651976866).abs() < 1e-6);

        let j1 = ComplexSpecialFunctions::bessel_j(Complex::one(), Complex::real(1.0), 30);
        assert!((j1.re - 0.4400505857).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_y_integer_order() {
        // Y_0(1) ≈ 0.088257, Y_1(1) ≈ -0.781213 (regression: integer order gave NaN)
        let y0 = ComplexSpecialFunctions::bessel_y(Complex::zero(), Complex::real(1.0), 40);
        assert!(!y0.is_nan());
        assert!((y0.re - 0.0882569642).abs() < 1e-4);

        let y1 = ComplexSpecialFunctions::bessel_y(Complex::one(), Complex::real(1.0), 40);
        assert!(!y1.is_nan());
        assert!((y1.re + 0.7812128213).abs() < 1e-4);
    }

    #[test]
    fn test_exponential_integral_known_values() {
        // Ei(1) ≈ 1.89512 (series branch), Ei(5) ≈ 40.1853 (asymptotic branch)
        let ei1 = ComplexSpecialFunctions::exponential_integral(Complex::one(), 40);
        assert!((ei1.re - 1.8951178164).abs() < 1e-4);

        let ei5 = ComplexSpecialFunctions::exponential_integral(Complex::real(5.0), 40);
        assert!((ei5.re - 40.1852753558).abs() < 1e-2);
    }

    #[test]
    fn test_gamma_half() {
        // Γ(1/2) = √π
        let g = ComplexSpecialFunctions::gamma(Complex::real(0.5));
        assert!((g.re - core::f64::consts::PI.sqrt()).abs() < 1e-8);
        assert!(g.im.abs() < 1e-10);
        // Γ(3/2) = √π/2
        let g32 = ComplexSpecialFunctions::gamma(Complex::real(1.5));
        assert!((g32.re - core::f64::consts::PI.sqrt() / 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_erf_large_z_asymptotic() {
        // Taylor would need ~50+ terms at z = 5; the asymptotic branch
        // must reach 1 − erf(5) ≈ 1.537e-12 without cancellation.
        let e5 = ComplexSpecialFunctions::erf(Complex::real(5.0), 60);
        assert!((1.0 - e5.re - 1.537e-12).abs() < 1e-12);
        assert!(e5.im.abs() < 1e-12);
        // erf(-5) ≈ -1 + 1.537e-12
        let em5 = ComplexSpecialFunctions::erf(Complex::real(-5.0), 60);
        assert!((em5.re + 1.0 - 1.537e-12).abs() < 1e-12);
        // erf(1) stays on the series branch
        let e1 = ComplexSpecialFunctions::erf(Complex::one(), 50);
        assert!((e1.re - 0.8427007929).abs() < 1e-6);
        // erfc(5) directly (no 1 - erf cancellation)
        let ec5 = ComplexSpecialFunctions::erfc(Complex::real(5.0), 60);
        assert!((ec5.re - 1.537e-12).abs() < 1e-12);
    }

    #[test]
    fn test_bessel_j_large_z() {
        // Asymptotic branch at z = 10 must agree with the series to the
        // leading-order accuracy (O(1/z) ≈ 10% at z = 10).
        let j_asymp = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(10.0), 50);
        let j_series =
            ComplexSpecialFunctions::bessel_j_series(Complex::zero(), Complex::real(10.0), 200);
        assert!((j_asymp.re - j_series.re).abs() < 0.01);

        // Negative real axis: asymptotic is invalid (branch cut) — the series
        // fallback must reproduce the even symmetry J_0(-10) = J_0(10).
        let jm = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(-10.0), 50);
        assert!((jm.re - j_series.re).abs() < 1e-6);

        // J_0 is even: J_0(-1) = J_0(1) through the series branch too
        let jp = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::one(), 30);
        let jn = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(-1.0), 30);
        assert!((jp.re - jn.re).abs() < 1e-12);
    }

    #[test]
    fn test_polylog_outside_disk_is_nan() {
        // |z| > 1 is not implemented: must be NaN, not silent garbage
        let v = ComplexSpecialFunctions::polylog(Complex::real(2.0), Complex::real(2.0), 100);
        assert!(v.is_nan());
        // |z| < 1 still works (Li_1(z) = -ln(1-z))
        let z = Complex::real(0.5);
        let v2 = ComplexSpecialFunctions::polylog(Complex::one(), z, 100);
        let expected = -(Complex::one() - z).ln();
        assert!((v2.re - expected.re).abs() < 0.1);
        assert!((v2.im - expected.im).abs() < 0.1);
    }
}
