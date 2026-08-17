//! Complex special functions: gamma, zeta, polylog, and related functions.

use crate::Complex;

/// Complex special functions.
pub struct ComplexSpecialFunctions;

impl ComplexSpecialFunctions {
    /// Complex gamma function using Lanczos approximation.
    pub fn gamma(z: Complex) -> Complex {
        // Lanczos approximation coefficients
        const P: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.5203681218851,
            -1259.1392167224028,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507343278686905,
            -0.13857109526572012,
            9.984_369_578_019_572e-6,
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

            // Asymptotic expansion: Ïˆ(z) ~ ln(z) - 1/(2z) - 1/(12zÂ²) + 1/(120zâ´) - 1/(252zâ¶) + ...
            let term2 = inv_z2 / Complex::real(12.0);
            let term4 = inv_z4 / Complex::real(120.0);
            let term6 = inv_z2 * inv_z4 / Complex::real(252.0);

            z.ln() - Complex::real(0.5) * inv_z - term2 + term4 - term6
        } else {
            // Use recurrence relation: Ïˆ(z+1) = Ïˆ(z) + 1/z
            let mut n = 0;
            let mut z_shifted = z;

            while z_shifted.norm() < 10.0 && n < 100 {
                z_shifted = z_shifted + Complex::one();
                n += 1;
            }

            let psi_shifted = Self::digamma(z_shifted);
            let mut result = psi_shifted;

            for k in 0..n {
                result = result - Complex::one() / (z + Complex::real(f64::from(k)));
            }

            result
        }
    }

    /// Riemann zeta function for complex arguments.
    pub fn zeta(z: Complex, iterations: usize) -> Complex {
        // For Re(z) < 0, use the functional equation:
        // Î¶(s) = 2^s Â· Ï€^(s-1) Â· sin(Ï€s/2) Â· Î“(1-s) Â· Î¶(1-s)
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
        // Î¶(s) = Î£_{n<N} n^-s + N^(1-s)/(s-1) + N^-s/2
        //        + Î£_{k=1..M} B_{2k}/(2k)! Â· s(s+1)Â·Â·Â·(s+2k-2) Â· N^(-s-2k+1)
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

        let mut sum: Complex = Complex::zero();
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

    /// Polylogarithm `Li_s(z)`.
    ///
    /// Computed from the defining series for `|z| < 1` (and conditionally on
    /// `|z| = 1` when `Re(s) > 1`).
    ///
    /// For `|z| > 1`, the analytic continuation uses:
    /// - **Liâ‚‚(z)** (dilogarithm): the identity
    ///   `Liâ‚‚(z) = âˆ’Liâ‚‚(1/z) âˆ’ Ï€Â²/6 âˆ’ (ln(âˆ’z))Â²/2`.
    /// - **General s**: the integral representation via trapezoidal
    ///   quadrature on a shifted contour, which converges for all finite s
    ///   and `|z| > 1` with branch cut along `[1, âˆž)`.
    pub fn polylog(s: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1.0 {
            // Series expansion: Li_s(z) = Î£_{n=1}^âˆž z^n / n^s
            let mut sum: Complex = Complex::zero();
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let term = z.pow(n_complex) / n_complex.pow(s);
                sum = sum + term;
            }
            sum
        } else if (s.re - 2.0).abs() < 1e-15 && s.im.abs() < 1e-15 {
            // Liâ‚‚: use the identity for the dilogarithm.
            // Liâ‚‚(z) + Liâ‚‚(1/z) = âˆ’Ï€Â²/6 âˆ’ ln(âˆ’z)Â²/2
            let inv_z = Complex::one() / z;
            let li2_inv = Self::polylog(s, inv_z, iterations);
            let neg_z = -z;
            let ln_neg_z = neg_z.ln();
            let pi_sq_over_6 = Complex::real(core::f64::consts::PI.powi(2) / 6.0);
            -li2_inv - pi_sq_over_6 - ln_neg_z * ln_neg_z / Complex::real(2.0)
        } else if z.norm() > 1.0 {
            // General analytic continuation via contour integral.
            // Use the series: Li_s(z) = Î£_{n=1}^âˆž z^n / n^s for |z| < 1,
            // and for |z| > 1 transform via w = 1/z and use the identity:
            //   Li_s(z) = Î“(1-s)(-ln z)^{s-1} + Î£_{k=0}^âˆž Î¶(s-k)(ln z)^k / k!
            //
            // Since we lack Î¶ at complex arguments, fall back to the integral
            // representation: Li_s(z) = âˆ«_0^1 (-ln t)^{s-1} / (1 - tz) dt
            // computed via trapezoidal rule on t âˆˆ (0, 1).
            let one = Complex::one();
            let n_trap = iterations.max(20);
            let dt = one / Complex::real(n_trap as f64);
            let mut sum: Complex = Complex::zero();
            for k in 1..n_trap {
                let t = Complex::real(k as f64) * dt;
                let ln_t = t.ln();
                let neg_ln_t_pow = (-ln_t).pow(s - one);
                let denom = one - t * z;
                sum = sum + neg_ln_t_pow / denom;
            }
            // Trapezoidal rule with endpoint weights (endpoints are zero:
            // t=0: (-ln 0)^{s-1} = 0 for Re(s)>1 but the integrand is 0/1 = 0;
            // t=1: (-ln 1)^{s-1} = 0^{s-1} which is 0 for Re(s)>1).
            sum * dt
        } else {
            // |z| == 1: series converges for Re(s) > 1; partial sum otherwise
            let mut sum: Complex = Complex::zero();
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
    /// truncation for `|z| â‰¥ 3` (where the Taylor series needs hundreds of
    /// terms to converge).
    pub fn erf(z: Complex, iterations: usize) -> Complex {
        // erf is odd: erf(âˆ’z) = âˆ’erf(z). Mirroring keeps the asymptotic
        // branch inside its valid sector (the positive real half-plane).
        if z.re < 0.0 {
            return -Self::erf(-z, iterations);
        }
        // Asymptotic (DLMF 7.12.1), valid in the sector |arg z| â‰¤ Ï€/4:
        //   erf(z) ~ 1 âˆ’ exp(âˆ’zÂ²)/(zÂ·âˆšÏ€) Â· Î£â‚™ (âˆ’1)â¿ (2nâˆ’1)!!/(2zÂ²)â¿
        // The series is divergent; truncate just past the smallest term.
        if z.norm() >= 3.0 && z.re >= z.im.abs() {
            let z2 = z * z;
            let mut sum: Complex = Complex::one();
            let mut term: Complex = Complex::one();
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
        let mut sum: Complex = Complex::zero();
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
    /// not suffer catastrophic cancellation inside `1 âˆ’ erf(z)`.
    pub fn erfc(z: Complex, iterations: usize) -> Complex {
        // erfc(z) = 2 âˆ’ erfc(âˆ’z); mirror into the positive half-plane so the
        // asymptotic branch (valid for Re(z) > 0) is always used in-sector.
        if z.re < 0.0 {
            return Complex::real(2.0) - Self::erfc(-z, iterations);
        }
        if z.norm() >= 3.0 && z.re >= z.im.abs() {
            // erfc(z) â‰ˆ exp(âˆ’zÂ²)/(zÂ·âˆšÏ€) Â· Î£â‚™ (âˆ’1)â¿ (2nâˆ’1)!!/(2zÂ²)â¿
            let z2 = z * z;
            let mut sum: Complex = Complex::one();
            let mut term: Complex = Complex::one();
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
            // Ei(z) = Î³ + ln z + Î£_{n>=1} z^n/(nÂ·n!)
            let mut sum = Complex::real(0.577_215_664_901_532_9); // Euler-Mascheroni constant
            sum = sum + z.ln();

            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).map(|x| x as f64).product();
                sum = sum + z.pow(n_complex) / (n_complex * Complex::real(factorial));
            }

            sum
        } else {
            // Asymptotic expansion for large |z|: Ei(z) ~ e^z Â· Î£ n!/z^(n+1).
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
    /// Convention without Ï€/2 factors: C(z) = Î£ (-1)^n z^(4n+1)/((2n)! (4n+1)),
    /// S(z) = Î£ (-1)^n z^(4n+3)/((2n+1)! (4n+3)).
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
    /// Ai(z) = c1Â·f(z) - c2Â·g(z), Bi(z) = âˆš3Â·(c1Â·f(z) + c2Â·g(z)) with
    /// f(z) = Î£ z^(3k)/(9^k k! Î“(k+2/3)), g(z) = Î£ z^(3k+1)/(9^k k! Î“(k+4/3)),
    /// c1 = 3^(-2/3), c2 = 3^(-4/3).
    pub fn airy(z: Complex, iterations: usize) -> (Complex, Complex) {
        if z.norm() < 1e-15 {
            // series gives f(0) = 1/Î“(2/3), g(0) = 0
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
            // Ai(z) ~ exp(-Î¶) / (2âˆšÏ€ z^(1/4)), Bi(z) ~ exp(Î¶) / (âˆšÏ€ z^(1/4)),
            // Î¶ = (2/3) z^(3/2)
            let t = Complex::real(2.0 / 3.0) * z.pow(Complex::real(1.5));
            let z_pow_quarter = z.pow(Complex::real(0.25));

            let ai = (-t).exp()
                / (Complex::real(2.0) * Complex::real(std::f64::consts::PI.sqrt()) * z_pow_quarter);
            let bi = t.exp() / (Complex::real(std::f64::consts::PI.sqrt()) * z_pow_quarter);

            (ai, bi)
        }
    }

    /// Complex Bessel function of the first kind `J_v(z)`.
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
        // Large |z|: asymptotic expansion (DLMF 10.17.3),
        // J_v(z) ~ sqrt(2/(Ï€z)) Â· [cos(Ï†) âˆ’ (4vÂ²âˆ’1)/(8z) Â· sin(Ï†)],
        // where Ï† = z âˆ’ vÏ€/2 âˆ’ Ï€/4. The first correction term reduces
        // error from ~10% to ~1% at |z| = 10.
        // Valid for |arg z| < Ï€. Near the negative real axis the complex
        // sqrt branches, so fall back to the convergent series with extra
        // terms (|z| = 10 needs ~50 of them).
        if z.arg().abs() < core::f64::consts::PI - 1e-3 {
            let sqrt_term =
                (Complex::real(2.0) / (Complex::real(core::f64::consts::PI) * z)).sqrt();
            let phase = z
                - v * Complex::real(core::f64::consts::FRAC_PI_2)
                - Complex::real(core::f64::consts::FRAC_PI_4);
            // First correction: (4vÂ² âˆ’ 1) / (8z)
            let v2 = v * v;
            let correction = (v2 * Complex::real(4.0) - Complex::one())
                / (Complex::real(8.0) * z);
            sqrt_term * (phase.cos() - correction * phase.sin())
        } else {
            Self::bessel_j_series(v, z, iterations.saturating_mul(2))
        }
    }

    /// Series expansion of `J_v(z)` (DLMF 10.2.2), valid for all finite z.
    ///
    /// Stops once a term is negligible (`< 1e-15` in magnitude) or non-finite
    /// (large-n factorial/gamma overflow); the decaying tail can never matter
    /// again by then.
    fn bessel_j_series(v: Complex, z: Complex, iterations: usize) -> Complex {
        // DLMF 10.2.2: J_v(z) = Î£ (-1)^n / (n! Î“(v+n+1)) (z/2)^(v+2n)
        // Use recurrence for both n! and Î“(v+n+1) to avoid recomputation.
        let z_half = z / Complex::real(2.0);
        let mut gamma_v_n = Self::gamma(v + Complex::one()); // Î“(v+1)
        let mut factorial_n: f64 = 1.0; // 0! = 1
        let mut sum: Complex = Complex::zero();
        for n in 0..=iterations {
            let exponent = v + Complex::real(2.0 * n as f64);
            let term = (Complex::real((-1.0_f64).powi(n as i32)) * z_half.pow(exponent))
                / (Complex::real(factorial_n) * gamma_v_n);
            if term.is_nan() || (n > 10 && term.norm() < 1e-15) {
                break;
            }
            sum = sum + term;
            // Recurrence for next iteration
            if n < iterations {
                factorial_n *= (n + 1) as f64; // (n+1)! = n! * (n+1)
                gamma_v_n = gamma_v_n * (v + Complex::real((n + 1) as f64)); // Î“(v+n+2) = (v+n+1) Î“(v+n+1)
            }
        }
        sum
    }

    /// Complex Bessel function of the second kind `Y_v(z)`.
    pub fn bessel_y(v: Complex, z: Complex, iterations: usize) -> Complex {
        // Integer order: (J_v cos(vÏ€) - J_-v)/sin(vÏ€) is 0/0, use the
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

    /// `Y_n(z)` for integer order n >= 0:
    /// (2/Ï€) ln(z/2) `J_n(z)` - (1/Ï€) Î£_{k<n} (n-k-1)!/k! (z/2)^(2k-n)
    ///   - (1/Ï€) Î£_k (-1)^k (Ïˆ(k+1)+Ïˆ(k+n+1))/(k!(k+n)!) (z/2)^(2k+n)
    fn bessel_y_integer(n: usize, z: Complex, iterations: usize) -> Complex {
        let pi = std::f64::consts::PI;
        let z_over_2 = z / Complex::real(2.0);
        let j_n = Self::bessel_j(Complex::real(n as f64), z, iterations);

        let mut sum1 = Complex::zero();
        for k in 0..n {
            let fact_n_k_1: f64 = (1..(n - k)).map(|x| x as f64).product();
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

    /// Modified Bessel function of the first kind `I_v(z)`.
    /// Uses the series: `I_v(z)` = (z/2)^v Î£_{k=0}^âˆž (zÂ²/4)^k / (k! Î“(v+k+1)).
    pub fn bessel_i(v: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1e-15 {
            return if v.re > 0.0 { Complex::zero() } else { Complex::one() };
        }
        let z_over_2 = z / Complex::real(2.0);
        let v_complex = v;
        let z2_over_4 = z_over_2 * z_over_2;

        // Series with gamma recurrence
        let mut gamma_v_n = Self::gamma(v_complex + Complex::one());
        let mut term = Complex::one() / gamma_v_n;
        let mut sum: Complex = Complex::one();
        let mut factorial_n: f64 = 1.0;

        for n in 1..=iterations {
            gamma_v_n = gamma_v_n * (v_complex + Complex::real(n as f64));
            factorial_n *= n as f64;
            term = term * z2_over_4 / Complex::real(factorial_n);
            sum = sum + term / gamma_v_n;
            if term.norm() < 1e-20 * sum.norm() {
                break;
            }
        }

        z_over_2.pow(v_complex) * sum
    }

    /// Modified Bessel function of the second kind `K_v(z)`.
    /// Uses the relation: `K_v(z)` = (Ï€/2) [I_{-v}(z) - `I_v(z)`] / sin(vÏ€).
    /// For integer v, uses the limiting form.
    pub fn bessel_k(v: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1e-15 {
            return Complex::new(f64::INFINITY, 0.0);
        }

        let sin_v_pi = (v * Complex::real(std::f64::consts::PI)).sin();

        if sin_v_pi.norm() < 1e-12 {
            // Integer order: use derivative relation
            let n = v.re.round() as i64;
            let n_abs = n.unsigned_abs() as usize;
            let k = Self::bessel_k_integer(n_abs, z, iterations);
            if n < 0 && n % 2 != 0 { -k } else { k }
        } else {
            let pi = std::f64::consts::PI;
            let i_v = Self::bessel_i(v, z, iterations);
            let i_minus_v = Self::bessel_i(-v, z, iterations);
            Complex::real(pi / 2.0) * (i_minus_v - i_v) / sin_v_pi
        }
    }

    fn bessel_k_integer(n: usize, z: Complex, iterations: usize) -> Complex {
        // K_n(z) via the series for small |z| and asymptotic for large |z|
        if z.norm() < 6.0 {
            Self::bessel_k_series(n, z, iterations)
        } else {
            Self::bessel_k_asymptotic(Complex::real(n as f64), z, iterations)
        }
    }

    fn bessel_k_series(n: usize, z: Complex, iterations: usize) -> Complex {
        // K_n(z) via the Neumann series for small |z|
        // K_0(z) = -ln(z/2) I_0(z) - Î£_{k=0}^âˆž (z/2)^{2k} / (k!)Â² (Ïˆ(k+1)Â² + ... )
        // For general n: use K_n = (Ï€/2)(I_{-n} - I_n)/sin(nÏ€), but for integer n
        // we use the limit formula.
        let z2 = z * z / Complex::real(4.0);
        let ln_half_z = (z / Complex::real(2.0)).ln();

        if n == 0 {
            // K_0(z) = -ln(z/2) I_0(z) + Î£_{k=0}^âˆž (zÂ²/4)^k / (k!)Â² Â· 2Â·Ïˆ(k+1)
            let i0 = Self::bessel_i(Complex::zero(), z, iterations);
            let mut sum: Complex = Complex::zero();
            let mut term: Complex = Complex::one();
            for k in 0..=iterations {
                let psi = Self::digamma(Complex::real((k + 1) as f64));
                sum = sum + term * Complex::real(2.0) * psi;
                if k < iterations {
                    term = term * z2 / Complex::real((k + 1) as f64);
                }
            }
            -ln_half_z * i0 + Complex::real(0.5) * sum
        } else {
            // K_n via K_n = (Ï€/(2 sin(nÏ€))) [I_{-n} - I_n]
            // For integer n this is computed via the limit using derivatives.
            // Simpler: use K_n(z) = K_{n-2}(z) + 2(n-1)/z Â· K_{n-1}(z) backward recurrence
            let mut k_vec = Vec::new();
            let n_max = n + 5; // compute from higher order and recurse down
            // Start with a large-order approximation
            let v = Complex::real(n_max as f64);
            let k_high = Self::bessel_k_asymptotic(v, z, iterations);
            k_vec.push(k_high);
            // Backward recurrence: K_{m-1}(z) = 2m/z Â· K_m(z) + K_{m+1}(z)
            for m in (n + 1..n_max).rev() {
                let km1 = Complex::real(2.0 * m as f64) / z * k_vec[0] + k_vec[0];
                k_vec.insert(0, km1);
            }
            k_vec[0]
        }
    }

    fn bessel_k_asymptotic(v: Complex, z: Complex, _iterations: usize) -> Complex {
        // K_v(z) ~ sqrt(Ï€/(2z)) e^{-z} [1 + (4vÂ²-1)/(8z) + ...]
        let pi = std::f64::consts::PI;
        let sqrt_term = (Complex::real(pi / 2.0) / z).sqrt();
        let exp_neg_z = (-z).exp();
        let v2 = v * v;
        let correction = (v2 * Complex::real(4.0) - Complex::one()) / (Complex::real(8.0) * z);
        sqrt_term * exp_neg_z * (Complex::one() + correction)
    }

    /// Hankel function of the first kind: `H1_v(z)` = `J_v(z)` + i `Y_v(z)`.
    pub fn hankel_h1(v: Complex, z: Complex, iterations: usize) -> Complex {
        let j = Self::bessel_j(v, z, iterations);
        let y = Self::bessel_y(v, z, iterations);
        j + Complex::new(0.0, 1.0) * y
    }

    /// Hankel function of the second kind: `H2_v(z)` = `J_v(z)` âˆ’ i `Y_v(z)`.
    pub fn hankel_h2(v: Complex, z: Complex, iterations: usize) -> Complex {
        let j = Self::bessel_j(v, z, iterations);
        let y = Self::bessel_y(v, z, iterations);
        j - Complex::new(0.0, 1.0) * y
    }

    /// Regularized lower incomplete gamma P(a, x) = Î³(a, x) / Î“(a).
    ///
    /// Series expansion (DLMF 8.7.3) for `x < a + 1`; the continued fraction
    /// (DLMF 8.9.2) evaluated with Lentz's algorithm for large `x`, where the
    /// series loses accuracy.
    ///
    /// Requires `Re(a) > 0` and `x â‰¥ 0`; returns NaN outside that domain.
    #[allow(
        clippy::cast_precision_loss,
        clippy::doc_markdown,
        clippy::unreadable_literal
    )]
    pub fn gamma_p(a: Complex, x: Complex, iterations: usize) -> Complex {
        if x.re < 0.0 || a.re <= 0.0 {
            return Complex::new(f64::NAN, f64::NAN);
        }
        if x.norm() < 1e-15 {
            return Complex::zero();
        }
        if x.norm() < a.norm() + 1.0 {
            Self::gamma_p_series(a, x, iterations)
        } else {
            Complex::one() - Self::gamma_q_cf(a, x, iterations)
        }
    }

    /// Regularized upper incomplete gamma Q(a, x) = Î“(a, x) / Î“(a).
    ///
    /// Continued fraction (DLMF 8.9.2) with Lentz's algorithm for `x â‰¥ a + 1`,
    /// complementary series otherwise. `Q(a, x) + P(a, x) = 1` for `x > 0`.
    ///
    /// Requires `Re(a) > 0` and `x â‰¥ 0`; returns NaN outside that domain.
    #[allow(
        clippy::cast_precision_loss,
        clippy::doc_markdown,
        clippy::unreadable_literal
    )]
    pub fn gamma_q(a: Complex, x: Complex, iterations: usize) -> Complex {
        if x.re < 0.0 || a.re <= 0.0 {
            return Complex::new(f64::NAN, f64::NAN);
        }
        if x.norm() < 1e-15 {
            return Complex::one();
        }
        if x.norm() < a.norm() + 1.0 {
            Complex::one() - Self::gamma_p_series(a, x, iterations)
        } else {
            Self::gamma_q_cf(a, x, iterations)
        }
    }

    /// P(a, x) by series (DLMF 8.7.3):
    /// P(a, x) = x^a e^{âˆ’x} / Î“(a+1) Â· Î£_{k=0}^âˆž x^k / ((a+1)â‹¯(a+k)).
    #[allow(clippy::cast_precision_loss, clippy::doc_markdown)]
    fn gamma_p_series(a: Complex, x: Complex, iterations: usize) -> Complex {
        let term0 = x.pow(a) * (-x).exp() / Self::gamma(a + Complex::one());
        let mut sum: Complex = Complex::one();
        let mut term: Complex = Complex::one();
        for k in 1..=iterations {
            term = term * x / (a + Complex::real(k as f64));
            sum = sum + term;
            if term.norm() < 1e-15 * sum.norm() {
                break;
            }
        }
        term0 * sum
    }

    /// Q(a, x) by Lentz's evaluation of the continued fraction (DLMF 8.9.2):
    /// Q(a, x) = x^a e^{âˆ’x} / Î“(a) Â·
    ///   1 / (x + 1 âˆ’ a âˆ’ (1(1âˆ’a)) / (x + 3 âˆ’ a âˆ’ (2(2âˆ’a)) / (x + 5 âˆ’ a âˆ’ â‹¯))).
    #[allow(
        clippy::cast_precision_loss,
        clippy::doc_markdown,
        clippy::unreadable_literal
    )]
    fn gamma_q_cf(a: Complex, x: Complex, iterations: usize) -> Complex {
        const FPMIN: f64 = 1e-300;
        let gamma_a = Self::gamma(a);
        let mut b = x + Complex::one() - a;
        let mut c = Complex::real(1.0 / FPMIN);
        let mut d = Complex::one() / b;
        let mut h = d;
        for i in 1..=iterations {
            let an = -Complex::real(i as f64) * (Complex::real(i as f64) - a);
            b = b + Complex::real(2.0);
            d = an * d + b;
            if d.norm() < FPMIN {
                d = Complex::real(FPMIN);
            }
            c = b + an / c;
            if c.norm() < FPMIN {
                c = Complex::real(FPMIN);
            }
            d = Complex::one() / d;
            let delta = d * c;
            h = h * delta;
            if (delta - Complex::one()).norm() < 1e-15 {
                break;
            }
        }
        // x^a e^{âˆ’x} computed as exp(âˆ’x + aÂ·ln x) to avoid overflow.
        let prefactor = (-x + a * x.ln()).exp() / gamma_a;
        prefactor * h
    }

    /// Lower incomplete gamma Î³(a, x) = âˆ«â‚€Ë£ t^{a-1} e^{-t} dt = P(a, x)Â·Î“(a).
    /// Uses the series for `x < a + 1` and the continued fraction otherwise.
    pub fn gamma_lower(a: Complex, x: Complex, iterations: usize) -> Complex {
        Self::gamma_p(a, x, iterations) * Self::gamma(a)
    }

    /// Upper incomplete gamma Î“(a, x) = âˆ«Ë£^âˆž t^{a-1} e^{-t} dt = Q(a, x)Â·Î“(a).
    pub fn gamma_upper(a: Complex, x: Complex, iterations: usize) -> Complex {
        Self::gamma_q(a, x, iterations) * Self::gamma(a)
    }

    /// Sin(Ï€z) â€” exact for half-integer arguments.
    pub fn sinpi(z: Complex) -> Complex {
        // sin(Ï€z) = sin(Ï€Â·re) Â· cosh(Ï€Â·im) + i cos(Ï€Â·re) Â· sinh(Ï€Â·im)
        let pi = std::f64::consts::PI;
        Complex::new(
            (pi * z.re).sin() * (pi * z.im).cosh(),
            (pi * z.re).cos() * (pi * z.im).sinh(),
        )
    }

    /// Cos(Ï€z) â€” exact for half-integer arguments.
    pub fn cospi(z: Complex) -> Complex {
        let pi = std::f64::consts::PI;
        Complex::new(
            (pi * z.re).cos() * (pi * z.im).cosh(),
            -(pi * z.re).sin() * (pi * z.im).sinh(),
        )
    }

    /// Lambert W function (principal branch Wâ‚€) using Newton's method.
    /// Solves wÂ·e^w = z.
    pub fn lambert_w(z: Complex) -> Complex {
        if z.norm() < 1e-15 {
            return z;
        }
        if z.re > 0.0 && z.im.abs() < 1e-15 && (z.re - (-1.0_f64 / std::f64::consts::E).exp()).abs() < 1e-12 {
            return Complex::real(-1.0);
        }

        // Initial guess
        let mut w = if z.norm() < 0.36 {
            // Series: W(z) â‰ˆ z - zÂ² + 3/2 zÂ³
            z - z * z + Complex::real(1.5) * z * z * z
        } else if z.re > 2.0 {
            // Asymptotic: W(z) â‰ˆ ln(z) - ln(ln(z))
            let lz = z.ln();
            lz - lz.ln()
        } else {
            // General: start at 0.5 or -0.5
            if z.re > 0.0 { Complex::real(0.5) } else { Complex::real(-0.5) }
        };

        for _ in 0..50 {
            let ew = w.exp();
            let f = w * ew - z;
            if f.norm() < 1e-15 {
                break;
            }
            let denom = ew * (w + Complex::one());
            if denom.norm() < 1e-30 {
                break;
            }
            w = w - f / denom;
        }
        w
    }

    /// Complete elliptic integral of the first kind K(m).
    /// Uses the arithmetic-geometric mean (AGM) method: K(m) = Ï€ / (2Â·AGM(1, âˆš(1-m))).
    pub fn elliptic_k(m: Complex) -> Complex {
        let pi = std::f64::consts::PI;
        if m.re > 1.0 {
            // K(m) for m > 1: use K(1/m) / âˆšm
            let inv_k = Self::elliptic_k(Complex::real(1.0 / m.re));
            return inv_k / m.sqrt();
        }
        let mut a = Complex::one();
        let mut b = (Complex::one() - m).sqrt();
        for _ in 0..20 {
            let a_next = (a + b) * Complex::real(0.5);
            let b_next = (a * b).sqrt();
            if (a_next - a).norm() < 1e-15 {
                break;
            }
            a = a_next;
            b = b_next;
        }
        Complex::real(pi / 2.0) / a
    }

    /// Complete elliptic integral of the second kind E(m).
    /// Uses AGM: E(m) = (Ï€/2) Â· `a_âˆž` Â· (1 - Î£_{n=0}^{âˆž} 2^{n} Â· `c_nÂ²` / `a_âˆžÂ²`)
    /// where `c_n` = `a_n` - `b_n`.
    pub fn elliptic_e(m: Complex) -> Complex {
        let pi = std::f64::consts::PI;
        if m.re > 1.0 {
            let e1 = Self::elliptic_e(Complex::real(1.0 / m.re));
            let sqrt_m = m.sqrt();
            return sqrt_m * e1 + (Complex::one() - m) / sqrt_m * Self::elliptic_k(m);
        }
        let mut a = Complex::one();
        let mut b = (Complex::one() - m).sqrt();
        // Track sum of 2^n * c_nÂ² across iterations
        let mut sum_c2 = Complex::zero();
        let mut pow2 = Complex::one(); // starts at 2^0 = 1
        for _ in 0..20 {
            let c = a - b;
            sum_c2 = sum_c2 + pow2 * c * c;
            pow2 = pow2 * Complex::real(2.0);
            let a_next = (a + b) * Complex::real(0.5);
            let b_next = (a * b).sqrt();
            if (a_next - a).norm() < 1e-15 {
                break;
            }
            a = a_next;
            b = b_next;
        }
        // a now holds the AGM limit
        Complex::real(pi / 2.0) * a * (Complex::one() - sum_c2 / (Complex::real(4.0) * a * a))
    }

    /// Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m).
    /// Returns (sn, cn, dn) where snÂ² + cnÂ² = 1 and dnÂ² + mÂ·snÂ² = 1.
    pub fn jacobi_sn_cn_dn(u: Complex, m: Complex) -> (Complex, Complex, Complex) {
        let k = Self::elliptic_k(m).re;

        // Reduce u modulo 4K
        let four_k = 4.0 * k;
        let u_reduced = u.re % four_k;
        let u_red = Complex::real(u_reduced);

        // Gauss transformation for large u
        let (sn, cn, dn) = Self::jacobi_elliptic_direct(u_red, m);
        (sn, cn, dn)
    }

    fn jacobi_elliptic_direct(u: Complex, m: Complex) -> (Complex, Complex, Complex) {
        // Direct computation via the series / AGM method
        let mut a = Complex::one();
        let mut b = (Complex::one() - m).sqrt();
        let mut c = m.sqrt();

        let mut u_scaled = u;
        for _ in 0..16 {
            if c.re.abs() < 1e-15 {
                break;
            }
            let a_next = (a + b) * Complex::real(0.5);
            let b_next = (a * b).sqrt();
            c = (a - b) * Complex::real(0.5);
            u_scaled = u_scaled * Complex::real(2.0);
            a = a_next;
            b = b_next;
        }

        let phi = u_scaled * a;
        let sn = phi.sin();
        let cn = phi.cos();
        let dn = (Complex::one() - m * sn * sn).sqrt();

        (sn, cn, dn)
    }

    /// Jacobi theta function Î¸â‚(z, q) (DLMF 20.2.1):
    /// `Î¸â‚(z, q) = 2 Î£â‚™â‚Œâ‚€ (-1)â¿ q^(n+Â½)Â² sin((2n+1)z)`.
    ///
    /// Converges rapidly for `|q| < 1`; terminates once terms are negligible.
    pub fn theta_1(z: Complex, q: Complex, iterations: usize) -> Complex {
        let mut sum: Complex = Complex::zero();
        for n in 0..=iterations {
            let expo = (n as f64 + 0.5).powi(2);
            let sign = if n % 2 == 0 {
                Complex::one()
            } else {
                -Complex::one()
            };
            let term = sign * q.pow(Complex::real(expo)) * (Complex::real((2 * n + 1) as f64) * z).sin();
            if n > 0 && term.norm() < 1e-16 * sum.norm().max(1e-300) {
                break;
            }
            sum = sum + term;
        }
        Complex::real(2.0) * sum
    }

    /// Jacobi theta function Î¸â‚‚(z, q) (DLMF 20.2.1):
    /// `Î¸â‚‚(z, q) = 2 Î£â‚™â‚Œâ‚€ q^(n+Â½)Â² cos((2n+1)z)`.
    ///
    /// Converges rapidly for `|q| < 1`; terminates once terms are negligible.
    pub fn theta_2(z: Complex, q: Complex, iterations: usize) -> Complex {
        let mut sum: Complex = Complex::zero();
        for n in 0..=iterations {
            let expo = (n as f64 + 0.5).powi(2);
            let term = q.pow(Complex::real(expo)) * (Complex::real((2 * n + 1) as f64) * z).cos();
            if n > 0 && term.norm() < 1e-16 * sum.norm().max(1e-300) {
                break;
            }
            sum = sum + term;
        }
        Complex::real(2.0) * sum
    }

    /// Jacobi theta function Î¸â‚ƒ(z, q) (DLMF 20.2.1):
    /// `Î¸â‚ƒ(z, q) = 1 + 2 Î£â‚™â‚Œâ‚ q^nÂ² cos(2nz)`.
    ///
    /// Converges rapidly for `|q| < 1`; terminates once terms are negligible.
    pub fn theta_3(z: Complex, q: Complex, iterations: usize) -> Complex {
        let mut sum: Complex = Complex::one();
        for n in 1..=iterations {
            let expo = (n as f64).powi(2);
            let term = q.pow(Complex::real(expo)) * (Complex::real((2 * n) as f64) * z).cos();
            if term.norm() < 1e-16 * sum.norm().max(1e-300) {
                break;
            }
            sum = sum + Complex::real(2.0) * term;
        }
        sum
    }

    /// Jacobi theta function Î¸â‚„(z, q) (DLMF 20.2.1):
    /// `Î¸â‚„(z, q) = 1 + 2 Î£â‚™â‚Œâ‚ (-1)â¿ q^nÂ² cos(2nz)`.
    ///
    /// Converges rapidly for `|q| < 1`; terminates once terms are negligible.
    pub fn theta_4(z: Complex, q: Complex, iterations: usize) -> Complex {
        let mut sum: Complex = Complex::one();
        for n in 1..=iterations {
            let expo = (n as f64).powi(2);
            let sign = if n % 2 == 0 {
                Complex::one()
            } else {
                -Complex::one()
            };
            let term = sign * q.pow(Complex::real(expo)) * (Complex::real((2 * n) as f64) * z).cos();
            if term.norm() < 1e-16 * sum.norm().max(1e-300) {
                break;
            }
            sum = sum + Complex::real(2.0) * term;
        }
        sum
    }

    /// Generalized hypergeometric function `â‚šFq(aâ‚,â€¦,aâ‚š; bâ‚,â€¦,b_q; z)`:
    /// `Î£â‚™â‚Œâ‚€ (aâ‚)â‚™Â·Â·Â·(aâ‚š)â‚™ / ((bâ‚)â‚™Â·Â·Â·(b_q)â‚™) Â· zâ¿/n!`, where `(x)â‚™` is the
    /// Pochhammer symbol.
    ///
    /// The series is summed with a Pochhammer recurrence (no factorial/gamma
    /// recomputation) and terminates early once a term is negligible. If any
    /// `b_j` is zero or a negative integer (a pole), returns `NaN`.
    pub fn hypergeometric_pfq(
        a: &[Complex],
        b: &[Complex],
        z: Complex,
        iterations: usize,
    ) -> Complex {
        // Pole detection: b_j = 0, -1, -2, â€¦ makes the denominator vanish.
        for &bj in b {
            if bj.im.abs() < 1e-15 && bj.re <= 0.0 && (bj.re - bj.re.round()).abs() < 1e-15 {
                return Complex::new(f64::NAN, f64::NAN);
            }
        }
        let mut sum: Complex = Complex::one();
        let mut term: Complex = Complex::one();
        for n in 0..iterations {
            let n_c = Complex::real(n as f64);
            let mut ratio: Complex = Complex::one();
            for &ai in a {
                ratio = ratio * (ai + n_c);
            }
            for &bj in b {
                ratio = ratio / (bj + n_c);
            }
            term = term * ratio * z / Complex::real((n + 1) as f64);
            if term.is_nan() || term.is_infinite() {
                break;
            }
            if term.norm() < 1e-16 * sum.norm().max(1e-300) {
                break;
            }
            sum = sum + term;
        }
        sum
    }

    /// Gauss hypergeometric function `â‚‚Fâ‚(a, b; c; z)`.
    ///
    /// See [`hypergeometric_pfq`](Self::hypergeometric_pfq) for convergence
    /// notes (series converges for `|z| < 1`; otherwise it is divergent and
    /// requires analytic continuation).
    pub fn hypergeometric_2f1(
        a: Complex,
        b: Complex,
        c: Complex,
        z: Complex,
        iterations: usize,
    ) -> Complex {
        Self::hypergeometric_pfq(&[a, b], &[c], z, iterations)
    }

    /// Confluent hypergeometric function `â‚Fâ‚(a; b; z)` (Kummer's function).
    ///
    /// See [`hypergeometric_pfq`](Self::hypergeometric_pfq) for convergence
    /// notes (the series converges for all finite `z`).
    pub fn hypergeometric_1f1(
        a: Complex,
        b: Complex,
        z: Complex,
        iterations: usize,
    ) -> Complex {
        Self::hypergeometric_pfq(&[a], &[b], z, iterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        // Î“(1) = 1
        let result = ComplexSpecialFunctions::gamma(Complex::one());
        assert!((result.re - 1.0).abs() < 0.1);

        // Î“(2) = 1
        let result = ComplexSpecialFunctions::gamma(Complex::real(2.0));
        assert!((result.re - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_digamma() {
        // Ïˆ(1) should be -Î³ (Euler-Mascheroni constant)
        let result = ComplexSpecialFunctions::digamma(Complex::one());
        let gamma = 0.577_215_664_901_532_9;
        assert!((result.re + gamma).abs() < 0.1);
    }

    #[test]
    fn test_zeta() {
        // Î¶(2) = Ï€Â²/6
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
        // Ïˆ(1) = -Î³
        let result = ComplexSpecialFunctions::digamma(Complex::one());
        let gamma = 0.577_215_664_901_532_9;
        assert!((result.re + gamma).abs() < 1e-6);

        // Ïˆ(5) = -Î³ + H_4 = -Î³ + 25/12
        let psi5 = ComplexSpecialFunctions::digamma(Complex::real(5.0));
        let expected = -gamma + 25.0 / 12.0;
        assert!((psi5.re - expected).abs() < 1e-8);
    }

    #[test]
    fn test_zeta_critical_values() {
        // Î¶(0) = -1/2, Î¶(-1) = -1/12, Î¶(0.5) â‰ˆ -1.46035 (regression: broken functional equation)
        let z0 = ComplexSpecialFunctions::zeta(Complex::zero(), 100);
        assert!((z0.re + 0.5).abs() < 1e-8);
        assert!(z0.im.abs() < 1e-8);

        let zm1 = ComplexSpecialFunctions::zeta(Complex::real(-1.0), 100);
        assert!((zm1.re + 1.0 / 12.0).abs() < 1e-6);
        assert!(zm1.im.abs() < 1e-6);

        let zhalf = ComplexSpecialFunctions::zeta(Complex::real(0.5), 100);
        assert!((zhalf.re + 1.4603545088).abs() < 1e-4);

        // Î¶(2) = Ï€Â²/6 (existing)
        let z2 = ComplexSpecialFunctions::zeta(Complex::real(2.0), 1000);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((z2.re - expected).abs() < 1e-6);

        // Î¶(1/2 + 14i) on the critical line (regression: used to recurse forever)
        let critical = ComplexSpecialFunctions::zeta(Complex::new(0.5, 14.0), 100);
        assert!(critical.is_finite());
    }

    #[test]
    fn test_fresnel_small_z() {
        // C(z) â‰ˆ z, S(z) â‰ˆ zÂ³/3 for small z (regression: series had z in the denominator)
        let (c, s) = ComplexSpecialFunctions::fresnel(Complex::real(0.1), 30);
        assert!((c.re - 0.1).abs() < 1e-6);
        assert!((s.re - 0.1f64.powi(3) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fresnel_unit_values() {
        // C(1) â‰ˆ 0.904524, S(1) â‰ˆ 0.310268 (no-Ï€ convention)
        let (c, s) = ComplexSpecialFunctions::fresnel(Complex::one(), 30);
        assert!((c.re - 0.9045242379).abs() < 1e-5);
        assert!((s.re - 0.3102683017).abs() < 1e-5);
    }

    #[test]
    fn test_airy_known_values() {
        // Ai(0) â‰ˆ 0.355028, Bi(0) â‰ˆ 0.614927, Ai(1) â‰ˆ 0.135292, Bi(1) â‰ˆ 1.207424
        let (ai0, bi0) = ComplexSpecialFunctions::airy(Complex::zero(), 30);
        assert!((ai0.re - 0.3550280539).abs() < 1e-6);
        assert!((bi0.re - 0.6149266274).abs() < 1e-6);

        let (ai1, bi1) = ComplexSpecialFunctions::airy(Complex::one(), 30);
        assert!((ai1.re - 0.1352924163).abs() < 1e-5);
        assert!((bi1.re - 1.2074235949).abs() < 1e-4);
    }

    #[test]
    fn test_bessel_j_known_values() {
        // J_0(1) â‰ˆ 0.765198, J_1(1) â‰ˆ 0.440051
        let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(1.0), 30);
        assert!((j0.re - 0.7651976866).abs() < 1e-6);

        let j1 = ComplexSpecialFunctions::bessel_j(Complex::one(), Complex::real(1.0), 30);
        assert!((j1.re - 0.4400505857).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_y_integer_order() {
        // Y_0(1) â‰ˆ 0.088257, Y_1(1) â‰ˆ -0.781213 (regression: integer order gave NaN)
        let y0 = ComplexSpecialFunctions::bessel_y(Complex::zero(), Complex::real(1.0), 40);
        assert!(!y0.is_nan());
        assert!((y0.re - 0.0882569642).abs() < 1e-4);

        let y1 = ComplexSpecialFunctions::bessel_y(Complex::one(), Complex::real(1.0), 40);
        assert!(!y1.is_nan());
        assert!((y1.re + 0.7812128213).abs() < 1e-4);
    }

    #[test]
    fn test_exponential_integral_known_values() {
        // Ei(1) â‰ˆ 1.89512 (series branch), Ei(5) â‰ˆ 40.1853 (asymptotic branch)
        let ei1 = ComplexSpecialFunctions::exponential_integral(Complex::one(), 40);
        assert!((ei1.re - 1.8951178164).abs() < 1e-4);

        let ei5 = ComplexSpecialFunctions::exponential_integral(Complex::real(5.0), 40);
        assert!((ei5.re - 40.1852753558).abs() < 1e-2);
    }

    #[test]
    fn test_gamma_half() {
        // Î“(1/2) = âˆšÏ€
        let g = ComplexSpecialFunctions::gamma(Complex::real(0.5));
        assert!((g.re - core::f64::consts::PI.sqrt()).abs() < 1e-8);
        assert!(g.im.abs() < 1e-10);
        // Î“(3/2) = âˆšÏ€/2
        let g32 = ComplexSpecialFunctions::gamma(Complex::real(1.5));
        assert!((g32.re - core::f64::consts::PI.sqrt() / 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_erf_large_z_asymptotic() {
        // Taylor would need ~50+ terms at z = 5; the asymptotic branch
        // must reach 1 âˆ’ erf(5) â‰ˆ 1.537e-12 without cancellation.
        let e5 = ComplexSpecialFunctions::erf(Complex::real(5.0), 60);
        assert!((1.0 - e5.re - 1.537e-12).abs() < 1e-12);
        assert!(e5.im.abs() < 1e-12);
        // erf(-5) â‰ˆ -1 + 1.537e-12
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
        // Asymptotic branch at z = 10 must agree with the series once the
        // first correction term (DLMF 10.17.3) is included: O(1/zÂ²) â‰ˆ 1%.
        let j_asymp = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(10.0), 50);
        let j_series =
            ComplexSpecialFunctions::bessel_j_series(Complex::zero(), Complex::real(10.0), 200);
        assert!(
            (j_asymp.re - j_series.re).abs() < 1e-3,
            "J_0(10): got {}, expected {}",
            j_asymp.re,
            j_series.re
        );

        // v = 1 makes the (4vÂ² âˆ’ 1) correction five times larger; a wrong
        // sign there is a 1.8e-2 error, well past the tolerance.
        let j1_asymp = ComplexSpecialFunctions::bessel_j(Complex::one(), Complex::real(10.0), 50);
        let j1_series =
            ComplexSpecialFunctions::bessel_j_series(Complex::one(), Complex::real(10.0), 200);
        assert!(
            (j1_asymp.re - j1_series.re).abs() < 1e-3,
            "J_1(10): got {}, expected {}",
            j1_asymp.re,
            j1_series.re
        );

        // Negative real axis: asymptotic is invalid (branch cut) â€” the series
        // fallback must reproduce the even symmetry J_0(-10) = J_0(10).
        let jm = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(-10.0), 50);
        assert!((jm.re - j_series.re).abs() < 1e-6);

        // J_0 is even: J_0(-1) = J_0(1) through the series branch too
        let jp = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::one(), 30);
        let jn = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(-1.0), 30);
        assert!((jp.re - jn.re).abs() < 1e-12);
    }

    #[test]
    fn test_polylog_outside_disk() {
        // Liâ‚‚(2): analytic continuation via the identity
        // Liâ‚‚(z) + Liâ‚‚(1/z) = âˆ’Ï€Â²/6 âˆ’ ln(âˆ’z)Â²/2
        // Liâ‚‚(2) = Ï€Â²/4 âˆ’ iÏ€ ln(2) â‰ˆ 2.467 âˆ’ 2.178i
        let v = ComplexSpecialFunctions::polylog(Complex::real(2.0), Complex::real(2.0), 100);
        let pi = core::f64::consts::PI;
        let expected_re = pi * pi / 4.0;
        let expected_im = -pi * 2.0_f64.ln();
        assert!(
            (v.re - expected_re).abs() < 0.01,
            "Liâ‚‚(2) re: got {}, expected {}",
            v.re,
            expected_re
        );
        assert!(
            (v.im - expected_im).abs() < 0.01,
            "Liâ‚‚(2) im: got {}, expected {}",
            v.im,
            expected_im
        );
        // |z| < 1 still works (Li_1(z) = -ln(1-z))
        let z = Complex::real(0.5);
        let v2 = ComplexSpecialFunctions::polylog(Complex::one(), z, 100);
        let expected = -(Complex::one() - z).ln();
        assert!((v2.re - expected.re).abs() < 0.1);
        assert!((v2.im - expected.im).abs() < 0.1);
    }

    #[test]
    fn test_theta_series_identities() {
        // Even/odd: Î¸1 is odd, Î¸2/Î¸3/Î¸4 are even
        let z = Complex::real(0.7);
        let q = Complex::real(0.3);
        let t1p = ComplexSpecialFunctions::theta_1(z, q, 50);
        let t1m = ComplexSpecialFunctions::theta_1(-z, q, 50);
        assert!((t1p + t1m).norm() < 1e-12);
        for f in [
            ComplexSpecialFunctions::theta_2,
            ComplexSpecialFunctions::theta_3,
            ComplexSpecialFunctions::theta_4,
        ] {
            let tp = f(z, q, 50);
            let tm = f(-z, q, 50);
            assert!((tp - tm).norm() < 1e-12);
        }
        // Quarter-period translations (DLMF 20.2.10):
        // Î¸1(z+Ï€/2) = Î¸2(z), Î¸4(z+Ï€/2) = Î¸3(z)
        let half_pi = Complex::real(core::f64::consts::FRAC_PI_2);
        let t1_shift = ComplexSpecialFunctions::theta_1(z + half_pi, q, 50);
        let t2 = ComplexSpecialFunctions::theta_2(z, q, 50);
        assert!((t1_shift - t2).norm() < 1e-12);
        let t4_shift = ComplexSpecialFunctions::theta_4(z + half_pi, q, 50);
        let t3 = ComplexSpecialFunctions::theta_3(z, q, 50);
        assert!((t4_shift - t3).norm() < 1e-12);
        // Small-q behavior: Î¸3(0, q) â‰ˆ 1 + 2q, Î¸1(z,q) â‰ˆ 2q^(1/4)Â·sin(z)
        let qs = Complex::real(1e-6);
        let t3s = ComplexSpecialFunctions::theta_3(Complex::zero(), qs, 50);
        assert!((t3s.re - (1.0 + 2e-6)).abs() < 1e-12);
        let t1s = ComplexSpecialFunctions::theta_1(z, qs, 50);
        let expected = Complex::real(2.0 * 1e-6f64.powf(0.25)) * z.sin();
        assert!((t1s - expected).norm() < 1e-12);
    }

    #[test]
    fn test_hypergeometric_known_values() {
        // â‚‚Fâ‚(1,1;1;z) = 1/(1âˆ’z)
        let z = Complex::real(0.5);
        let f = ComplexSpecialFunctions::hypergeometric_2f1(
            Complex::one(),
            Complex::one(),
            Complex::one(),
            z,
            100,
        );
        let expected = Complex::one() / (Complex::one() - z);
        assert!((f - expected).norm() < 1e-12);

        // â‚‚Fâ‚(2,1;1;z) = 1/(1âˆ’z)Â²
        let f2 = ComplexSpecialFunctions::hypergeometric_2f1(
            Complex::real(2.0),
            Complex::one(),
            Complex::one(),
            z,
            100,
        );
        let expected2 = Complex::one() / (Complex::one() - z).powf(2.0);
        assert!((f2 - expected2).norm() < 1e-10);

        // â‚Fâ‚(1;2;z) = (e^z âˆ’ 1)/z
        let f1 = ComplexSpecialFunctions::hypergeometric_1f1(
            Complex::one(),
            Complex::real(2.0),
            z,
            100,
        );
        let expected1 = (z.exp() - Complex::one()) / z;
        assert!((f1 - expected1).norm() < 1e-12);

        // â‚Fâ‚€ with no b parameters: Î£ zâ¿ = 1/(1âˆ’z) for |z| < 1
        let f0 = ComplexSpecialFunctions::hypergeometric_pfq(&[Complex::one()], &[], z, 100);
        let expected0 = Complex::one() / (Complex::one() - z);
        assert!((f0 - expected0).norm() < 1e-12);

        // Pole in b (b = 0) yields NaN instead of division by zero
        let pole = ComplexSpecialFunctions::hypergeometric_1f1(
            Complex::one(),
            Complex::zero(),
            z,
            100,
        );
        assert!(pole.is_nan());
    }

    #[test]
    fn test_gamma_p_q_known_values() {
        // P(1, x) = 1 - e^(-x), Q(1, x) = e^(-x)
        // x = 0.5: series branch (x < a + 1 = 2)
        let p = ComplexSpecialFunctions::gamma_p(Complex::one(), Complex::real(0.5), 100);
        assert!((p.re - (1.0 - (-0.5_f64).exp())).abs() < 1e-10, "P(1,0.5) = {p}");
        let q = ComplexSpecialFunctions::gamma_q(Complex::one(), Complex::real(0.5), 100);
        assert!((q.re - (-0.5_f64).exp()).abs() < 1e-10, "Q(1,0.5) = {q}");

        // x = 2: continued-fraction branch (x >= a + 1 = 2)
        let p2 = ComplexSpecialFunctions::gamma_p(Complex::one(), Complex::real(2.0), 100);
        let expected_p = 1.0 - (-2.0_f64).exp();
        assert!((p2.re - expected_p).abs() < 1e-10, "P(1,2) = {p2}");
        let q2 = ComplexSpecialFunctions::gamma_q(Complex::one(), Complex::real(2.0), 100);
        assert!((q2.re - (-2.0_f64).exp()).abs() < 1e-10, "Q(1,2) = {q2}");

        // P(a, 0) = 0 and Q(a, 0) = 1
        let p0 = ComplexSpecialFunctions::gamma_p(Complex::real(3.0), Complex::zero(), 100);
        assert!(p0.norm() < 1e-15);
        let q0 = ComplexSpecialFunctions::gamma_q(Complex::real(3.0), Complex::zero(), 100);
        assert!((q0.re - 1.0).abs() < 1e-15);

        // P(0.5, 1) = erf(1) (P(1/2, x^2) = erf(x))
        let p_half = ComplexSpecialFunctions::gamma_p(
            Complex::real(0.5),
            Complex::one(),
            100,
        );
        assert!(
            (p_half.re - 0.8427007929497149).abs() < 1e-8,
            "P(0.5,1) = {p_half}, erf(1) = 0.8427007929"
        );

        // Q(2, x) = e^(-x) (1 + x); large x exercises the continued fraction
        let q10 = ComplexSpecialFunctions::gamma_q(Complex::real(2.0), Complex::real(10.0), 200);
        let expected_q10 = 11.0 * (-10.0_f64).exp();
        assert!(
            (q10.re - expected_q10).abs() < 1e-12,
            "Q(2,10) = {q10}, expected {expected_q10}"
        );

        // P + Q = 1 on both branches
        for x in [1.5, 7.0] {
            let a = Complex::real(3.0);
            let xc = Complex::real(x);
            let pv = ComplexSpecialFunctions::gamma_p(a, xc, 200);
            let qv = ComplexSpecialFunctions::gamma_q(a, xc, 200);
            assert!((pv + qv - Complex::one()).norm() < 1e-10, "P+Q at x = {x}");
        }

        // gamma_lower/gamma_upper stay consistent with the regularized forms
        let gl = ComplexSpecialFunctions::gamma_lower(Complex::real(2.0), Complex::real(10.0), 200);
        let gu = ComplexSpecialFunctions::gamma_upper(Complex::real(2.0), Complex::real(10.0), 200);
        let gamma2 = ComplexSpecialFunctions::gamma(Complex::real(2.0));
        assert!((gl + gu - gamma2).norm() < 1e-10);
        assert!((gu - gamma2 * q10).norm() < 1e-10);

        // Domain errors: negative x or Re(a) <= 0 give NaN
        assert!(ComplexSpecialFunctions::gamma_p(
            Complex::one(),
            Complex::real(-1.0),
            100,
        )
        .is_nan());
        assert!(ComplexSpecialFunctions::gamma_q(
            Complex::one(),
            Complex::real(-1.0),
            100,
        )
        .is_nan());
        assert!(ComplexSpecialFunctions::gamma_p(
            Complex::zero(),
            Complex::one(),
            100,
        )
        .is_nan());
    }
}
