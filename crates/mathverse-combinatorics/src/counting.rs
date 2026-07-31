use mathverse_core::algorithms::binomial;

pub fn combinations(n: u64, k: u64) -> u128 {
    binomial(n, k)
}

pub fn permutations(n: u64, k: u64) -> u128 {
    if k > n { return 0; }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n { acc *= i as u128; }
    acc
}

pub fn permutations_with_repetition(n: u64, k: u64) -> u128 {
    (n as u128).pow(k as u32)
}

pub fn combinations_with_repetition(n: u64, k: u64) -> u128 {
    binomial(n + k - 1, k)
}

pub fn falling_factorial(n: u64, k: u64) -> u128 {
    if k > n { return 0; }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n { acc *= i as u128; }
    acc
}

pub fn rising_factorial(n: u64, k: u64) -> u128 {
    let mut acc: u128 = 1;
    for i in n..(n + k) { acc *= i as u128; }
    acc
}

pub fn multichoose(n: u64, k: u64) -> u128 {
    combinations_with_repetition(n, k)
}

pub fn arrangements(n: u64, k: u64) -> u128 {
    permutations(n, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(combinations(10, 3), 120);
        assert_eq!(permutations(5, 2), 20);
        assert_eq!(permutations(5, 0), 1);
        assert_eq!(permutations(3, 4), 0);
        assert_eq!(permutations_with_repetition(3, 2), 9);
        assert_eq!(combinations_with_repetition(3, 2), 6);
    }

    #[test]
    fn factorials() {
        assert_eq!(falling_factorial(5, 3), 60);
        assert_eq!(rising_factorial(3, 3), 60);
        assert_eq!(falling_factorial(5, 0), 1);
    }

    #[test]
    fn aliases() {
        assert_eq!(multichoose(3, 2), 6);
        assert_eq!(arrangements(5, 2), 20);
    }
}
