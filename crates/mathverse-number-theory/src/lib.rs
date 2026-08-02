//! # mathverse-number-theory
//!
//! Number theory for the MathVerse ecosystem.
//!
//! Provides:
//! - **Primes**: primality testing, sieve of Eratosthenes, prime generation
//! - **Factorization**: trial division, Pollard's rho
//! - **Modular arithmetic**: modular inverse, Chinese Remainder Theorem, modular exponentiation
//! - **Totient**: Euler's totient function, totient sum
//! - **Quadratic residues**: Legendre symbol, Tonelli-Shanks sqrt mod p
//! - **Diophantine**: extended GCD, linear Diophantine equation solver
//! - **Continued fractions**: convergents, best rational approximation
//! - **Advanced**: Fibonacci modular, Legendre/Jacobi symbols, Lucas sequences
//!
//! Re-exports core algorithms (`gcd`, `lcm`, `is_prime`, `mod_pow`, `sieve_of_eratosthenes`)
//! from `mathverse-core` for convenience.

pub mod primes;
pub mod factorization;
pub mod modular;
pub mod totient;
pub mod quadratic_residue;
pub mod diophantine;
pub mod continued_fraction;
pub mod advanced;

pub use mathverse_core::algorithms::{gcd, is_prime, lcm, mod_pow, sieve_of_eratosthenes};
pub use primes::*;
pub use factorization::*;
pub use modular::*;
pub use totient::*;
pub use advanced::*;
