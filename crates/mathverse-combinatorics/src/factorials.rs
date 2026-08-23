//! Factorial variants: standard, double, super, hyper, primorial, subfactorial, tetration.

/// Computes `n!` (factorial).
pub fn factorial(n: u64) -> u128 {
    (1..=u128::from(n)).product()
}

/// Double factorial `n!! = n * (n-2) * ...`.
#[must_use]
pub fn double_factorial(n: u64) -> u128 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut acc = 1u128;
    let mut i = if n.is_multiple_of(2) { 2 } else { 3 };
    while i <= n {
        acc *= u128::from(i);
        i += 2;
    }
    acc
}

/// Super-factorial: product of the first `n` factorials.
#[must_use]
pub fn super_factorial(n: u64) -> u128 {
    (1..=n).fold(1u128, |acc, i| acc * factorial(i))
}

/// Hyper-factorial: `1^1 * 2^2 * ... * n^n`.
#[must_use]
pub fn hyper_factorial(n: u64) -> u128 {
    (1..=n).fold(1u128, |acc, i| acc * u128::from(i).pow(i as u32))
}

/// Product of all primes up to and including `n`.
#[must_use]
pub fn primorial(n: u64) -> u128 {
    use mathverse_core::algorithms::is_prime;
    (2..=n).filter(|&p| is_prime(p)).map(u128::from).product()
}

/// Number of derangements (permutations with no fixed point) of `n` items.
#[must_use]
pub fn subfactorial(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    let (mut a, mut b) = (1u128, 0u128);
    for i in 2..=n {
        let t = (u128::from(i) - 1) * (a + b);
        a = b;
        b = t;
    }
    b
}

/// Tetration of height `n`: a power tower a^(a^(...)) evaluated top-down.
#[must_use]
pub fn tetration(a: u64, n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    let mut result = u128::from(a);
    for _ in 1..n {
        result = u128::from(a).pow(result as u32);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3_628_800);
    }

    #[test]
    fn double() {
        assert_eq!(double_factorial(0), 1);
        assert_eq!(double_factorial(5), 15);
        assert_eq!(double_factorial(6), 48);
    }

    #[test]
    fn super_and_hyper() {
        assert_eq!(super_factorial(3), 12);
        assert_eq!(hyper_factorial(3), 108);
    }

    #[test]
    fn prim() {
        assert_eq!(primorial(10), 2 * 3 * 5 * 7);
    }

    #[test]
    fn sub() {
        assert_eq!(subfactorial(0), 1);
        assert_eq!(subfactorial(4), 9);
    }
}
