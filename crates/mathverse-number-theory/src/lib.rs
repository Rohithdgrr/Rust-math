#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! # mathverse-number-theory
//!
//! Production-grade number theory for the `MathVerse` ecosystem.
//!
//! All primality tests use **deterministic Miller-Rabin** for `u64`,
//! providing O(log³ n) performance with no `DoS` risk. Arithmetic operations
//! use checked arithmetic to prevent silent overflow.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`primes`] | Deterministic Miller-Rabin primality, sieve, segmented sieve, Goldbach, Mersenne, Lucas-Lehmer |
//! | [`factorization`] | Prime factorization, Pollard's Rho, divisors, σₖ, Möbius, Liouville |
//! | [`modular`] | Modular exponentiation (returns `Option`), inverse, CRT, linear congruence solver |
//! | [`totient`] | Euler φ, Carmichael λ, primitive roots, multiplicative order |
//! | [`quadratic_residue`] | Legendre/Jacobi symbols, Tonelli-Shanks, CRT combination |
//! | [`diophantine`] | Extended GCD, Pell equation (u128 + bigint), Kronecker symbol |
//! | [`continued_fraction`] | CF expansion, convergents, π/e/φ constants |
//! | [`advanced`] | Next/prev prime, radical, highly composite, perfect powers |
//!
//! ## Quick Start
//!
//! ```
//! use mathverse_number_theory::*;
//!
//! // Primality (fast Miller-Rabin)
//! assert!(is_prime(97));
//! assert!(is_prime(1_000_000_007));
//! assert!(!is_prime(1_000_000_007 * 2));
//!
//! // Primes up to N
//! let primes = sieve(30);
//! assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
//!
//! // Segmented sieve for large ranges
//! let big = segmented_sieve(1_000_000_000_000, 1_000_000_000_100);
//!
//! // Factorization
//! let factors = prime_factors(84);
//! assert_eq!(factors, vec![2, 2, 3, 7]);
//!
//! // Pollard's Rho for large composites
//! assert_eq!(factorize(91), vec![7, 13]);
//!
//! // Modular arithmetic (m=0 safely returns None)
//! assert_eq!(mod_pow(2, 10, 1000), Some(24));
//! assert_eq!(mod_pow(2, 10, 0), None);
//!
//! // Euler's totient
//! assert_eq!(euler_totient(36), 12);
//!
//! // Pell's equation x² - 2y² = 1
//! let (x, y) = pell_fundamental(2).unwrap();
//! assert_eq!(x * x - 2 * y * y, 1);
//! ```

pub mod advanced;
pub mod continued_fraction;
pub mod diophantine;
pub mod factorization;
pub mod modular;
pub mod primes;
pub mod quadratic_residue;
pub mod totient;

pub mod sympy_compat;

// Re-export core integer algorithms that are well-tested and correct.
pub use mathverse_core::integer::{gcd, is_square, isqrt, lcm};

// Re-export from local modules (canonical implementations).
pub use advanced::{
    highly_composite, kiuchi, next_prime, number_of_divisors, perfect_power, prev_prime,
    prime_factorization, radical, sum_of_divisors,
};
pub use continued_fraction::{
    approximant, cf_to_fraction, cf_to_value, continued_fraction, convergents, e_cf,
    golden_ratio_cf, pi_cf,
};
pub use diophantine::{
    euler_kronecker, extended_gcd, is_square_free, kronecker, pell_fundamental,
    quadratic_reciprocity, solve_linear_diophantine,
};
pub use factorization::{
    divisor_count, divisor_sum, divisors, factorize, is_abundant, is_deficient, is_perfect_number,
    liouville, mobius, pollard_rho, prime_factors, sigma_k,
};
pub use modular::{
    crt, linear_congruence_solvable, mod_add, mod_div, mod_inverse, mod_mul, mod_pow,
    mod_pow_unchecked, mod_sub, solve_linear_congruence,
};
pub use primes::{
    goldbach, is_prime, lucas_lehmer, mersenne_prime, miller_rabin, nth_prime, prime_between,
    prime_gap_after, prime_gap_containing, segmented_sieve, sieve, sieve_of_eratosthenes,
    twin_primes,
};
pub use quadratic_residue::{
    chinese_remainder_quadratic, is_quadratic_residue, jacobi, legendre, quadratic_residues,
    tonelli_shanks,
};
pub use totient::{
    carmichael, coprimes_up_to, euler_totient, euler_totient_sum, is_coprime, multiplicative_order,
    primitive_root,
};

#[cfg(feature = "bigint")]
pub use diophantine::pell_fundamental_big;
