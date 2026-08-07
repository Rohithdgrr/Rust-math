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
//! Production-grade number theory for the MathVerse ecosystem.
//!
//! All primality tests use **deterministic Miller-Rabin** for `u64`,
//! providing O(log³ n) performance with no DoS risk. Arithmetic operations
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

pub mod primes;
pub mod factorization;
pub mod modular;
pub mod totient;
pub mod quadratic_residue;
pub mod diophantine;
pub mod continued_fraction;
pub mod advanced;

pub mod sympy_compat;

// Re-export core integer algorithms that are well-tested and correct.
pub use mathverse_core::integer::{gcd, lcm, isqrt, is_square};

// Re-export from local modules (canonical implementations).
pub use primes::{
    is_prime, miller_rabin, sieve, sieve_of_eratosthenes, segmented_sieve,
    nth_prime, prime_between, twin_primes, goldbach,
    mersenne_prime, lucas_lehmer, prime_gap_after, prime_gap_containing,
};
pub use factorization::{
    prime_factors, factorize, pollard_rho, divisors, divisor_count,
    divisor_sum, sigma_k, mobius, liouville,
    is_perfect_number, is_abundant, is_deficient,
};
pub use modular::{
    mod_pow, mod_pow_unchecked, mod_inverse, mod_add, mod_sub, mod_mul,
    mod_div, crt, solve_linear_congruence, linear_congruence_solvable,
};
pub use totient::{
    euler_totient, euler_totient_sum, carmichael, is_coprime, coprimes_up_to,
    primitive_root, multiplicative_order,
};
pub use quadratic_residue::{
    legendre, jacobi, tonelli_shanks, quadratic_residues,
    is_quadratic_residue, chinese_remainder_quadratic,
};
pub use diophantine::{
    extended_gcd, solve_linear_diophantine, is_square_free,
    kronecker, quadratic_reciprocity, pell_fundamental, euler_kronecker,
};
pub use advanced::{
    next_prime, prev_prime, prime_factorization, sum_of_divisors,
    number_of_divisors, radical, kiuchi, highly_composite, perfect_power,
};
pub use continued_fraction::{
    continued_fraction, convergents, approximant,
    golden_ratio_cf, e_cf, pi_cf, cf_to_value, cf_to_fraction,
};

#[cfg(feature = "bigint")]
pub use diophantine::pell_fundamental_big;
