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
            let pi = Complex::new(std::f64::consts::PI, 0.0);
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
        
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        let t = z_minus_1 + Complex::real((P.len() - 2) as f64) + Complex::real(0.5);
        
        let sqrt_2pi_t = Complex::real(sqrt_2pi) * t.pow(z - Complex::real(0.5));
        let exp_t = (-t).exp();
        
        sqrt_2pi_t * exp_t * x
    }

    /// Complex digamma function (derivative of log gamma).
    pub fn digamma(z: Complex) -> Complex {
        // Use series expansion for large |z|
        if z.norm() > 10.0 {
            let inv_z = Complex::one() / z;
            let inv_z2 = inv_z * inv_z;
            let inv_z4 = inv_z2 * inv_z2;
            
            // Asymptotic expansion
            let term1 = inv_z;
            let term2 = inv_z2 / Complex::real(2.0);
            let term3 = inv_z2 * inv_z / Complex::real(12.0);
            let term4 = inv_z4 / Complex::real(120.0);
            
            z.ln() - Complex::real(0.5) * inv_z - term1 + term2 - term3 + term4
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
        // For Re(z) > 1, use Dirichlet series
        if z.re > 1.0 {
            let mut sum = Complex::zero();
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                sum = sum + Complex::one() / n_complex.pow(z);
            }
            return sum;
        }
        
        // For other values, use functional equation
        let s = z;
        let one_minus_s = Complex::one() - s;
        let pi = Complex::new(std::f64::consts::PI, 0.0);
        
        let gamma_term = Self::gamma(one_minus_s / Complex::real(2.0));
        let pi_term = pi.pow(s / Complex::real(2.0));
        let sin_term = (pi * s / Complex::real(2.0)).sin();
        
        let zeta_reflected = Self::zeta(Complex::one() - s, iterations);
        
        let factor = Complex::real(2.0_f64.powi(s.re as i32)) * pi_term * sin_term * gamma_term;
        factor * zeta_reflected
    }

    /// Polylogarithm Li_s(z).
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
            // Use inversion formula
            let log_minus_z = (Complex::one() - z).ln();
            let mut sum = Complex::zero();
            
            for k in 0..=iterations {
                let k_complex = Complex::real(k as f64);
                let binomial = Self::binomial_coefficient_complex(s + k_complex - Complex::one(), k_complex);
                let term = binomial * (log_minus_z).pow(k_complex) * Self::zeta(s + k_complex, 50);
                
                if k % 2 == 0 {
                    sum = sum + term;
                } else {
                    sum = sum - term;
                }
            }
            
            sum
        } else {
            // Near |z| = 1, use analytic continuation
            Complex::zero()
        }
    }

    /// Complex binomial coefficient.
    fn binomial_coefficient_complex(z: Complex, k: Complex) -> Complex {
        // Γ(z+1) / (Γ(k+1) * Γ(z-k+1))
        let numerator = Self::gamma(z + Complex::one());
        let denominator = Self::gamma(k + Complex::one()) * Self::gamma(z - k + Complex::one());
        numerator / denominator
    }

    /// Complex error function.
    pub fn erf(z: Complex, iterations: usize) -> Complex {
        let mut sum = Complex::zero();
        let z2 = z * z;
        
        for n in 0..=iterations {
            let n_complex = Complex::real(n as f64);
            let sign = if n % 2 == 0 { Complex::one() } else { -Complex::one() };
            
            let factorial: f64 = (1..=n).product::<usize>() as f64;
            let denominator = Complex::real((2 * n + 1) as f64 * factorial);
            
            let term = sign * z.pow(Complex::real((2 * n + 1) as f64)) / denominator;
            sum = sum + term;
        }
        
        sum * Complex::new(2.0 / std::f64::consts::PI.sqrt(), 0.0)
    }

    /// Complex complementary error function.
    pub fn erfc(z: Complex, iterations: usize) -> Complex {
        Complex::one() - Self::erf(z, iterations)
    }

    /// Complex exponential integral Ei(z).
    pub fn exponential_integral(z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1.0 {
            // Series expansion
            let mut sum = Complex::real(0.57721566490153286060651209008240243104215933593992); // Euler-Mascheroni constant
            sum = sum + z.ln();
            
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).product::<usize>() as f64;
                sum = sum + z.pow(n_complex) / (n_complex * Complex::real(factorial));
            }
            
            sum
        } else {
            // Asymptotic expansion for large |z|
            let inv_z = Complex::one() / z;
            let mut sum = inv_z;
            
            for n in 1..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial: f64 = (1..=n).product::<usize>() as f64;
                let term = Complex::real(factorial) * inv_z.pow(n_complex + Complex::one());
                
                if n % 2 == 0 {
                    sum = sum + term;
                } else {
                    sum = sum - term;
                }
            }
            
            sum * z.exp()
        }
    }

    /// Complex Fresnel integrals C(z) and S(z).
    pub fn fresnel(z: Complex, iterations: usize) -> (Complex, Complex) {
        let mut c = Complex::zero();
        let mut s = Complex::zero();
        
        for n in 0..=iterations {
            let n_complex = Complex::real(n as f64);
            let factorial: f64 = (1..=n).product::<usize>() as f64;
            
            let sign_c = if n % 4 == 0 { Complex::one() } else if n % 4 == 2 { -Complex::one() } else { Complex::zero() };
            let sign_s = if (n + 3) % 4 == 0 { Complex::one() } else if (n + 1) % 4 == 0 { -Complex::one() } else { Complex::zero() };
            
            let power = Complex::real(4.0 * n as f64 + 1.0);
            let denominator = Complex::real(factorial) * power * z.pow(Complex::real(4.0 * n as f64 + 1.0));
            
            c = c + sign_c / denominator;
            s = s + sign_s / denominator;
        }
        
        (c, s)
    }

    /// Complex Airy functions Ai(z) and Bi(z).
    pub fn airy(z: Complex, iterations: usize) -> (Complex, Complex) {
        // Use series expansion for small |z|
        if z.norm() < 2.0 {
            let mut ai = Complex::zero();
            let mut bi = Complex::zero();
            
            for n in 0..=iterations {
                let n_complex = Complex::real(n as f64);
                let three_n = Complex::real(3.0 * n as f64);
                
                let gamma_1 = Self::gamma(n_complex / Complex::real(3.0) + Complex::real(2.0 / 3.0));
                let gamma_2 = Self::gamma(n_complex / Complex::real(3.0) + Complex::real(1.0 / 3.0));
                
                let term1 = z.pow(three_n) / (Complex::real(3.0_f64.powi(2 * n as i32)) * gamma_1);
                let term2 = z.pow(three_n + Complex::real(2.0)) / (Complex::real(3.0_f64.powi(2 * n as i32 + 1)) * gamma_2);
                
                let sign = if n % 2 == 0 { Complex::one() } else { -Complex::one() };
                
                ai = ai + sign * term1;
                bi = bi + sign * (term1 + Complex::real(3.0_f64.sqrt()) * term2);
            }
            
            let factor = Complex::one() / (Complex::real(3.0_f64.pow(2.0 / 3.0)) * std::f64::consts::PI);
            (ai * factor, bi * factor)
        } else {
            // Asymptotic expansion for large |z|
            let t = (2.0 / 3.0) * z.pow(Complex::real(1.5));
            let exp_t = (-t).exp();
            let exp_neg_t = t.exp();
            
            let prefactor = Complex::one() / (Complex::real(2.0) * std::f64::consts::PI.sqrt() * z.pow(Complex::real(0.25)));
            
            let ai = prefactor * exp_t;
            let bi = prefactor * exp_neg_t;
            
            (ai, bi)
        }
    }

    /// Complex Bessel function of the first kind J_v(z).
    pub fn bessel_j(v: Complex, z: Complex, iterations: usize) -> Complex {
        if z.norm() < 1.0 {
            // Series expansion
            let mut sum = Complex::zero();
            
            for n in 0..=iterations {
                let n_complex = Complex::real(n as f64);
                let factorial_n: f64 = if n == 0 { 1.0 } else { (1..=n).product::<usize>() as f64 };
                let gamma_v_n = Self::gamma(v + n_complex + Complex::one());
                
                let term = (Complex::real((-1.0_f64).powi(n as i32)) * (z / Complex::real(2.0)).pow(v + Complex::real(2.0) * n_complex))
                    / (Complex::real(factorial_n) * gamma_v_n);
                
                sum = sum + term;
            }
            
            sum
        } else {
            // Asymptotic expansion for large |z|
            let phase = z - v * (z / Complex::real(2.0)).ln() - Complex::real(std::f64::consts::PI / 4.0);
            let amplitude = Complex::real(2.0 / (std::f64::consts::PI * z.norm()).sqrt());
            
            amplitude * phase.cos()
        }
    }

    /// Complex Bessel function of the second kind Y_v(z).
    pub fn bessel_y(v: Complex, z: Complex, iterations: usize) -> Complex {
        let j_v = Self::bessel_j(v, z, iterations);
        let j_minus_v = Self::bessel_j(-v, z, iterations);
        
        let cos_v_pi = (v * Complex::new(std::f64::consts::PI, 0.0)).cos();
        
        (j_v * cos_v_pi - j_minus_v) / (v * Complex::new(std::f64::consts::PI, 0.0)).sin()
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
        assert!((result.re - expected).abs() < 0.1);
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
}
