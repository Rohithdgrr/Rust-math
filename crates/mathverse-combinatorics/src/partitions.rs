pub fn partition(n: u64) -> u128 {
    let n = n as usize;
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in 1..=n {
        for j in i..=n { p[j] += p[j - i]; }
    }
    p[n]
}

pub fn partition_k(n: u64, k: u64) -> u128 {
    if k > n { return 0; }
    if n == 0 && k == 0 { return 1; }
    if k == 0 { return 0; }
    let n = n as usize;
    let k = k as usize;
    let mut dp = vec![vec![0u128; k + 1]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        for j in 1..=k.min(i) {
            dp[i][j] = dp[i - 1][j - 1] + dp[i - j][j];
        }
    }
    dp[n][k]
}

pub fn partition_into_distinct(n: u64) -> u128 {
    let n = n as usize;
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in 1..=n {
        for j in (i..=n).rev() { p[j] += p[j - i]; }
    }
    p[n]
}

pub fn partition_count_even_parts(n: u64) -> u128 {
    let n = n as usize;
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in (2..=n).step_by(2) {
        for j in i..=n { p[j] += p[j - i]; }
    }
    p[n]
}

pub fn partition_count_odd_parts(n: u64) -> u128 {
    let n = n as usize;
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in (1..=n).step_by(2) {
        for j in i..=n { p[j] += p[j - i]; }
    }
    p[n]
}

pub fn partitions_leq(n: u64, max_part: u64) -> u128 {
    let n = n as usize;
    let max = max_part as usize;
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in 1..=max.min(n) {
        for j in i..=n { p[j] += p[j - i]; }
    }
    p[n]
}

pub fn pentagonal(n: i64) -> i64 {
    n * (3 * n - 1) / 2
}

pub fn euler_partition_formula(n: u64) -> u128 {
    let n = n as i64;
    if n == 0 { return 1; }
    let mut result: i128 = 0;
    let mut k = 1i64;
    loop {
        let gk = pentagonal(k);
        if gk > n { break; }
        let sign = if k % 2 == 0 { -1 } else { 1 };
        result += sign as i128 * partition((n - gk) as u64) as i128;
        let gk_neg = pentagonal(-k);
        if gk_neg >= 0 && gk_neg <= n {
            result += sign as i128 * partition((n - gk_neg) as u64) as i128;
        }
        k += 1;
    }
    result as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_partitions() {
        assert_eq!(partition(0), 1);
        assert_eq!(partition(1), 1);
        assert_eq!(partition(2), 2);
        assert_eq!(partition(3), 3);
        assert_eq!(partition(4), 5);
        assert_eq!(partition(5), 7);
    }

    #[test]
    fn k_parts() {
        assert_eq!(partition_k(4, 2), 2);
        assert_eq!(partition_k(5, 3), 2);
    }

    #[test]
    fn distinct() {
        assert_eq!(partition_into_distinct(5), 3);
        assert_eq!(partition_into_distinct(7), 5);
    }

    #[test]
    fn even_odd() {
        assert_eq!(partition_count_even_parts(4), 2);
        assert_eq!(partition_count_odd_parts(4), 2);
    }

    #[test]
    fn euler() {
        assert_eq!(euler_partition_formula(5), 7);
    }
}
