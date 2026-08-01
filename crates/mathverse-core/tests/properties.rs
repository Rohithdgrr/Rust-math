//! Property-based tests for mathverse-core.
//!
//! These tests verify mathematical invariants that must hold for all inputs.

use proptest::prelude::*;

proptest! {
    #[test]
    fn lerp_boundary(a in -1000i32..1000, b in -1000i32..1000) {
        let a = a as f64;
        let b = b as f64;
        let result_0 = mathverse_core::ops::lerp(a, b, 0.0);
        let result_1 = mathverse_core::ops::lerp(a, b, 1.0);
        prop_assert!((result_0 - a).abs() < 1e-10, "lerp(a,b,0) should equal a");
        prop_assert!((result_1 - b).abs() < 1e-10, "lerp(a,b,1) should equal b");
    }

    #[test]
    fn smoothstep_bounds(x in 0.0f64..=1.0) {
        let s = mathverse_core::ops::smoothstep(x);
        prop_assert!((0.0..=1.0).contains(&s), "smoothstep should be in [0,1], got {s}");
    }

    #[test]
    fn smoothstep_endpoints(_x in 0.0f64..=1.0) {
        let s0 = mathverse_core::ops::smoothstep(0.0_f64);
        let s1 = mathverse_core::ops::smoothstep(1.0_f64);
        prop_assert!(s0.abs() < 1e-12, "smoothstep(0) should be 0");
        prop_assert!((s1 - 1.0).abs() < 1e-12, "smoothstep(1) should be 1");
    }

    #[test]
    fn gcd_symmetry(a in 1u64..10000, b in 1u64..10000) {
        prop_assert_eq!(mathverse_core::algorithms::gcd(a, b), mathverse_core::algorithms::gcd(b, a));
    }

    #[test]
    fn gcd_divides(a in 1u64..1000, b in 1u64..1000) {
        let g = mathverse_core::algorithms::gcd(a, b);
        prop_assert_eq!(a % g, 0, "gcd should divide a");
        prop_assert_eq!(b % g, 0, "gcd should divide b");
    }

    #[test]
    fn lcm_identity(a in 1u64..100, b in 1u64..100) {
        let l = mathverse_core::algorithms::lcm(a, b);
        let g = mathverse_core::algorithms::gcd(a, b);
        // lcm(a,b) * gcd(a,b) == a * b (for small values that don't overflow)
        prop_assert_eq!(l * g, a * b);
    }

    #[test]
    fn is_prime_matches_sieve(n in 2u64..200) {
        let prime = mathverse_core::algorithms::is_prime(n);
        let sieve = mathverse_core::algorithms::sieve_of_eratosthenes(n as usize).contains(&n);
        prop_assert_eq!(prime, sieve);
    }

    #[test]
    fn factorial_positive(n in 0u64..21) {
        let f = mathverse_core::algorithms::factorial(n);
        prop_assert!(f > 0, "factorial should be positive for n >= 0");
        if n >= 2 {
            let f_prev = mathverse_core::algorithms::factorial(n - 1);
            prop_assert_eq!(f, f_prev * n as u128, "n! = (n-1)! * n");
        }
    }

    #[test]
    fn binomial_symmetry(n in 0u64..30, k in 0u64..30) {
        if k <= n {
            let b1 = mathverse_core::algorithms::binomial(n, k);
            let b2 = mathverse_core::algorithms::binomial(n, n - k);
            prop_assert_eq!(b1, b2, "C(n,k) should equal C(n,n-k)");
        }
    }

    #[test]
    fn wrap_in_range(x in -1000.0f64..1000.0, lo in 0.0f64..10.0, hi in 10.0f64..20.0) {
        let w = mathverse_core::ops::wrap(x, lo, hi);
        prop_assert!(w >= lo && w < hi, "wrap should be in [lo, hi), got {w}");
    }

    #[test]
    fn abs_always_nonneg(x in -1000.0f64..1000.0) {
        let a = x.abs();
        prop_assert!(a >= 0.0, "abs should be non-negative");
    }

    #[test]
    fn hypot2_pythagorean(a in 0.0f64..100.0, b in 0.0f64..100.0) {
        let h = mathverse_core::ops::hypot2(a, b);
        let expected = (a * a + b * b).sqrt();
        prop_assert!((h - expected).abs() < 1e-10, "hypot2 should match sqrt(a^2+b^2)");
    }

    #[test]
    fn div_by_gcd_coprime(a in 1u64..1000, b in 1u64..1000) {
        let g = mathverse_core::algorithms::gcd(a, b);
        let a_red = a / g;
        let b_red = b / g;
        prop_assert!(mathverse_core::algorithms::is_coprime(a_red, b_red),
            "a/gcd and b/gcd should be coprime");
    }

    #[test]
    fn digit_sum_bounded(n in 0u64..1000000) {
        let s = mathverse_core::algorithms::digit_sum(n);
        let dc = mathverse_core::algorithms::digit_count(n);
        // digit_sum <= 9 * digit_count
        prop_assert!(s <= 9 * dc as u64, "digit_sum should be bounded by 9 * digit_count");
    }

    #[test]
    fn fibonacci_growth(n in 2u64..50) {
        let f_n = mathverse_core::algorithms::fibonacci(n);
        let f_prev = mathverse_core::algorithms::fibonacci(n - 1);
        prop_assert!(f_n >= f_prev, "fibonacci should be non-decreasing");
    }

    #[test]
    fn euler_phi_upper_bound(n in 1u64..200) {
        let phi = mathverse_core::algorithms::euler_phi(n);
        prop_assert!(phi <= n, "phi(n) should be <= n");
        prop_assert!(phi >= 1, "phi(n) should be >= 1");
    }

    #[test]
    fn is_palindrome_reverse(n in 0u64..100000) {
        let p = mathverse_core::algorithms::is_palindrome(n);
        let rev = mathverse_core::algorithms::reverse_digits(n);
        prop_assert_eq!(p, n == rev, "is_palindrome should match n == reverse_digits(n)");
    }
}
