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
