//! One-line imports for the whole core crate.
//!
//! ```
//! use mathverse_core::prelude::*;
//! assert_eq!(gcd(48, 18), 6);
//! ```

pub use crate::integer::{
    bell_number, bezout_coefficients, binomial, catalan_number, checked_binomial,
    checked_factorial, chinese_remainder, digit_count, digit_count_base, digit_sum,
    double_factorial, euler_phi, extended_gcd, factorial, fermat_number, fibonacci,
    from_base, from_digits, gcd, gcd_n, is_abundant, is_armstrong, is_coprime,
    is_cube, is_deficient, is_even, is_harshad, is_odd, is_palindrome,
    is_palindrome_base, is_perfect_number, is_perfect_power, is_power_of,
    is_power_of_two, is_semiprime, is_smooth, is_square, is_squarefree,
    is_triangular, isqrt, isqrt_rem, lcm, lcm_n, liouville, log2_ceil,
    log2_floor, log_base, lucas_number, mersenne_number, modular_inverse,
    mod_pow, mobius, nearest_power_of_two, next_power_of_two, next_prime,
    nth_prime, partition_number, pascal_triangle, permutation_count, popcount,
    prev_prime, primorial, prime_count, prime_factorization, prime_factors,
    radical, reverse_digits, reverse_digits_base, segmented_sieve,
    sieve_of_eratosthenes, stirling_first, stirling_second, subfactorial,
    sum_of_squares, to_base, to_digits, tribonacci,
};
pub use crate::error::{MathError, MathResult};
pub use crate::ops::{
    abs_sub, bilinear, cbrt, clamp, clamp01, clamp11, copysign, cumprod, cumsum,
    deg_to_grad, deg_to_rad, distance, dot_product, fract, grad_to_deg, grad_to_rad,
    hypot2, hypot3, integer_part, inv_lerp, is_between, is_sorted, lerp,
    lerp_angle, lerp_clamped, lerp_inv, map_range, max_value, mean, midpoint,
    min_value, normalize, nth_root, ping_pong, rad_to_deg, rad_to_grad,
    remap, remap_clamped, repeat, sign, smoothstep, smoothstep_between,
    smootherstep, snap, step, trunc, wrap, wrap_angle, wrap_angle_positive,
};
pub use crate::precision::{
    abs_diff, almost_eq, almost_eq_rel, almost_eq_ulp, ceil_to_multiple,
    epsilon, floor_to_multiple, is_close, is_subnormal, is_nan, is_infinite,
    next_float, prev_float, relative_diff, round_to, round_to_multiple,
    significant_figures, ulp, EPS, F32_EPS,
};
pub use crate::traits::{Field, Normed, Num, Real, Signed};
pub use crate::constants::{
    APERY, CATALAN, CUBE_ROOT_2, CUBE_ROOT_3, DEG_TO_GRAD, DEG_TO_RAD, E,
    EULER_GAMMA, E_INV, E_SQ, E_SQRT, FEIGENBAUM, FEIGENBAUM2, GRAD_TO_DEG, GRAD_TO_RAD,
    HALF_PI, INF, INV_PI, INV_SQRT_PI, LN_10, LN_2, LN_PI, LN_2PI, LOG10_2, LOG10_E,
    LOG2_10, LOG2_E, LOG_2, LOG_10, NAN, PHI, PI, PI_CUBE, PI_SQ, QUARTER_PI,
    RAD_TO_DEG, SILVER_RATIO, SQRT_2, SQRT_2_INV, SQRT_3, SQRT_3_INV, SQRT_5,
    SQRT_6, SQRT_7, SQRT_LN4, SQRT_PI, SQRT_2PI, THIRD_PI, TAU, TWO_SQRT_PI,
};
