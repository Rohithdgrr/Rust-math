//! One-line imports for the whole core crate.
//!
//! ```
//! use mathverse_core::prelude::*;
//! assert_eq!(gcd(48, 18), 6);
//! ```

pub use crate::algorithms::{
    bell_number, catalan_number, chinese_remainder, digit_count, digit_sum,
    double_factorial, euler_phi, fermat_number, fibonacci, from_base, from_digits,
    is_abundant, is_armstrong, is_coprime, is_deficient, is_harshad,
    is_perfect_number, is_perfect_power, is_power_of, is_power_of_two,
    is_semiprime, is_smooth, is_squarefree, is_triangular, liouville,
    lucas_number, mersenne_number, mobius, next_prime, next_power_of_two,
    nth_prime, partition_number, pascal_triangle, permutation_count,
    prev_prime, primorial, prime_count, prime_factorization, prime_factors,
    radical, reverse_digits, segmented_sieve, sieve_of_eratosthenes,
    stirling_first, stirling_second, subfactorial, to_base, to_digits,
    tribonacci,
};
pub use crate::integer::{
    bezout_coefficients, binomial, extended_gcd, factorial, gcd, gcd_n, is_square,
    isqrt, isqrt_rem, lcm, lcm_n, log2_ceil, modular_inverse, mod_pow,
};
pub use crate::error::{MathError, MathResult};
pub use crate::ops::{
    abs, abs_sub, cbrt, clamp, clamp01, clamp11, copysign, deg_to_grad, deg_to_rad,
    distance, fract, grad_to_deg, grad_to_rad, hypot, hypot2, hypot3,
    integer_part, inv_lerp, lerp, lerp_angle, lerp_inv, map_range, midpoint,
    normalize, nth_root, ping_pong, rad_to_deg, rad_to_grad, recip, remap,
    repeat, sign, signum, smoothstep, smoothstep_between, smootherstep, snap,
    step, trunc, wrap, wrap_angle, wrap_angle_positive,
};
pub use crate::precision::{
    abs_diff, almost_eq, almost_eq_rel, almost_eq_ulp, ceil_to_multiple,
    epsilon, floor_to_multiple, is_close, is_subnormal, is_nan, is_infinite,
    next_float, prev_float, relative_diff, round_to, round_to_multiple,
    safe_div, significant_figures, ulp, EPS, F32_EPS,
};
pub use crate::traits::{ComplexCore, Field, InnerProduct, Metric, Normed, Num, Real, RealField, Signed};
pub use crate::constants::{
    APERY, CATALAN, CUBE_ROOT_2, CUBE_ROOT_3, DEG_TO_GRAD, DEG_TO_RAD, E,
    EULER_GAMMA, E_INV, E_SQ, E_SQRT, FEIGENBAUM, FEIGENBAUM2, GRAD_TO_DEG, GRAD_TO_RAD,
    HALF_PI, INF, INV_PI, INV_SQRT_PI, LN_10, LN_2, LN_PI, LN_2PI, LOG10_2, LOG10_E,
    LOG2_10, LOG2_E, LOG_2, LOG_10, NAN, PHI, PI, PI_CUBE, PI_SQ, QUARTER_PI,
    RAD_TO_DEG, SILVER_RATIO, SQRT_2, SQRT_2_INV, SQRT_3, SQRT_3_INV, SQRT_5,
    SQRT_6, SQRT_7, SQRT_LN4, SQRT_PI, SQRT_2PI, THIRD_PI, TAU, TWO_SQRT_PI,
};
