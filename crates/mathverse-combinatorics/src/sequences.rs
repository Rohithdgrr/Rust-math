//! Integer sequences: Fibonacci, Lucas, Catalan, Tribonacci, Tetranacci, Collatz.

/// Computes the `n`th Fibonacci number (F(0)=0, F(1)=1).
pub fn fibonacci(n: u64) -> u128 {
    let n = n as usize;
    if n == 0 {
        return 0;
    }
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 0..n - 1 {
        let t = a + b;
        a = b;
        b = t;
    }
    b
}

/// nth Lucas number (L(0)=2, L(1)=1).
#[must_use]
pub fn lucas(n: u64) -> u128 {
    let n = n as usize;
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }
    let (mut a, mut b) = (2u128, 1u128);
    for _ in 2..=n {
        let t = a + b;
        a = b;
        b = t;
    }
    b
}

/// nth Catalan number: C(2n, n) / (n + 1).
#[must_use]
pub fn catalan(n: u64) -> u128 {
    use mathverse_core::algorithms::binomial;
    binomial(2 * n, n) / u128::from(n + 1)
}

/// nth Tribonacci number (T(0)=T(1)=T(2)=0, T(3)=1).
#[must_use]
pub fn tribonacci(n: u64) -> u128 {
    match n {
        0..=2 => 0,
        3 => 1,
        _ => {
            let (mut a, mut b, mut c) = (0u128, 0u128, 1u128);
            for _ in 3..=n {
                let t = a + b + c;
                a = b;
                b = c;
                c = t;
            }
            c
        }
    }
}

/// nth Tetranacci number (four-term recurrence, T(3)=1).
#[must_use]
pub fn tetranacci(n: u64) -> u128 {
    match n {
        0..=2 => 0,
        3 => 1,
        _ => {
            let (mut a, mut b, mut c, mut d) = (0u128, 0u128, 0u128, 1u128);
            for _ in 4..=n {
                let t = a + b + c + d;
                a = b;
                b = c;
                c = d;
                d = t;
            }
            d
        }
    }
}

/// Approximate nth Fibonacci number via Binet's closed-form formula.
#[must_use]
pub fn fibonacci_binet(n: u64) -> f64 {
    let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
    let psi = (1.0 - 5.0_f64.sqrt()) / 2.0;
    (phi.powi(n as i32) - psi.powi(n as i32)) / 5.0_f64.sqrt()
}

/// nth Fibonacci number modulo m, computed iteratively to avoid overflow.
#[must_use]
pub fn nth_fibonacci_mod(n: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let t = (a + b) % m;
        a = b;
        b = t;
    }
    a
}

/// Number of steps for n to reach 1 under the Collatz map
/// (n/2 when even, 3n+1 when odd).
#[must_use]
pub fn collatz_steps(mut n: u64) -> u64 {
    let mut steps = 0;
    while n != 1 {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            3 * n + 1
        };
        steps += 1;
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fib() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn lucas_seq() {
        assert_eq!(lucas(0), 2);
        assert_eq!(lucas(1), 1);
        assert_eq!(lucas(5), 11);
    }

    #[test]
    fn cat() {
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(4), 14);
        assert_eq!(catalan(5), 42);
    }

    #[test]
    fn higher_order() {
        assert_eq!(tribonacci(0), 0);
        assert_eq!(tribonacci(7), 13);
        assert_eq!(tetranacci(0), 0);
    }

    #[test]
    fn collatz() {
        assert_eq!(collatz_steps(6), 8);
        assert_eq!(collatz_steps(1), 0);
    }
}
