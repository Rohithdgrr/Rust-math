pub fn fibonacci(n: u64) -> u128 {
    let n = n as usize;
    if n == 0 { return 0; }
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 0..n - 1 { let t = a + b; a = b; b = t; }
    b
}

pub fn lucas(n: u64) -> u128 {
    let n = n as usize;
    if n == 0 { return 2; }
    if n == 1 { return 1; }
    let (mut a, mut b) = (2u128, 1u128);
    for _ in 2..=n { let t = a + b; a = b; b = t; }
    b
}

pub fn catalan(n: u64) -> u128 {
    use mathverse_core::algorithms::binomial;
    binomial(2 * n, n) / (n + 1) as u128
}

pub fn tribonacci(n: u64) -> u128 {
    match n { 0 => 0, 1 | 2 => 0, 3 => 1, _ => {
        let (mut a, mut b, mut c) = (0u128, 0u128, 1u128);
        for _ in 3..=n { let t = a + b + c; a = b; b = c; c = t; }
        c
    }}
}

pub fn tetranacci(n: u64) -> u128 {
    match n { 0 | 1 | 2 => 0, 3 => 1, _ => {
        let (mut a, mut b, mut c, mut d) = (0u128, 0u128, 0u128, 1u128);
        for _ in 4..=n { let t = a + b + c + d; a = b; b = c; c = d; d = t; }
        d
    }}
}

pub fn fibonacci_binet(n: u64) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let psi = (1.0 - 5.0_f64.sqrt()) / 2.0;
    (phi.powi(n as i32) - psi.powi(n as i32)) / 5.0_f64.sqrt()
}

pub fn nth_fibonacci_mod(n: u64, m: u64) -> u64 {
    if m == 1 { return 0; }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n { let t = (a + b) % m; a = b; b = t; }
    a
}

pub fn collatz_steps(mut n: u64) -> u64 {
    let mut steps = 0;
    while n != 1 { n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 }; steps += 1; }
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
