//! One-line imports for the whole core crate.
//!
//! ```
//! use mathverse_core::prelude::*;
//! assert_eq!(gcd(48, 18), 6);
//! ```

pub use crate::algorithms::{
    bell_number, bezout_coefficients, binomial, catalan_number,
    chinese_remainder, digit_count, digit_sum, double_factorial, euler_phi,
    extended_gcd, fermat_number, factorial, fibonacci, from_base, from_digits,
    gcd, gcd_n, is_abundant, is_armstrong, is_coprime, is_deficient,
    is_harshad, is_perfect_number, is_perfect_power, is_power_of,
    is_power_of_two, is_semiprime, is_smooth, is_square, is_squarefree,
    is_triangular, isqrt, isqrt_rem, lcm, lcm_n, liouville, lucas_number,
    mersenne_number, mobius, modular_inverse, next_prime, next_power_of_two,
    nth_prime, partition_number, pascal_triangle, permutation_count,
    prev_prime, primorial, prime_count, prime_factorization, prime_factors,
    radical, reverse_digits, segmented_sieve, sieve_of_eratosthenes,
    stirling_first, stirling_second, subfactorial, to_base, to_digits,
    tribonacci,
};
pub use crate::error::{MathError, MathResult};
pub use crate::ops::{
    abs, abs_sub, cbrt, clamp, copysign, deg_to_grad, deg_to_rad,
    distance, fract, grad_to_deg, grad_to_rad, hypot, hypot2, hypot3,
    integer_part, lerp, lerp_inv, map_range, nth_root, ping_pong,
    rad_to_deg, rad_to_grad, recip, repeat, sign, signum,
    smootherstep, smoothstep, trunc, wrap, wrap_angle, wrap_angle_positive,
    normalize,
};
pub use crate::precision::{
    abs_diff, almost_eq, almost_eq_rel, almost_eq_ulp,
    epsilon, is_close, is_subnormal, is_nan, is_infinite,
    next_float, prev_float, relative_diff, round_to, safe_div,
    significant_figures, ulp, EPS, F32_EPS,
};
pub use crate::traits::{Field, Num, Real, Signed};
pub use crate::constants::{
    APERY, CATALAN, CUBE_ROOT_2, CUBE_ROOT_3, DEG_TO_GRAD, DEG_TO_RAD, E,
    EULER_GAMMA, E_SQ, E_SQRT, FEIGENBAUM, FEIGENBAUM2, GRAD_TO_DEG, GRAD_TO_RAD,
    HALF_PI, INF, LN_10, LN_2, LN_PI, LOG10_E, LOG2_10, LOG2_E, LOG_2, LOG_10,
    NAN, PHI, PI, PI_CUBE, PI_SQ, QUARTER_PI, RAD_TO_DEG, SQRT_2, SQRT_2_INV,
    SQRT_3, SQRT_3_INV, SQRT_5, SQRT_6, SQRT_7, SQRT_LN4, SQRT_PI, SQRT_2PI,
    THIRD_PI, TAU,
};
