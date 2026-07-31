pub fn stirling1_unsigned(n: u64, k: u64) -> u128 {
    if k > n { return 0; }
    let (n, k) = (n as usize, k as usize);
    let mut t = vec![vec![0u128; k + 1]; n + 1];
    t[0][0] = 1;
    for i in 1..=n {
        for j in 1..=k {
            t[i][j] = (i - 1) as u128 * t[i - 1][j] + t[i - 1][j - 1];
        }
    }
    t[n][k]
}

pub fn stirling2(n: u64, k: u64) -> u128 {
    if k > n { return 0; }
    let (n, k) = (n as usize, k as usize);
    let mut t = vec![vec![0u128; k + 1]; n + 1];
    t[0][0] = 1;
    for i in 1..=n {
        for j in 1..=k {
            t[i][j] = (j as u128) * t[i - 1][j] + t[i - 1][j - 1];
        }
    }
    t[n][k]
}

pub fn bell(n: u64) -> u128 {
    let n = n as usize;
    let mut t = vec![vec![0u128; n + 1]; n + 1];
    t[0][0] = 1;
    for i in 1..=n {
        t[i][0] = t[i - 1][i - 1];
        for j in 1..=i {
            t[i][j] = t[i][j - 1] + t[i - 1][j - 1];
        }
    }
    t[n][0]
}

pub fn derangements(n: u64) -> u128 {
    let n = n as usize;
    if n == 0 { return 1; }
    let (mut a, mut b) = (1u128, 0u128);
    for i in 2..=n {
        let t = (i as u128 - 1) * (b + a);
        a = b;
        b = t;
    }
    b
}

pub fn Lah_number(n: u64, k: u64) -> u128 {
    if k > n || n == 0 { return 0; }
    if k == 0 { return 0; }
    use mathverse_core::algorithms::binomial;
    mathverse_core::algorithms::factorial(n) / mathverse_core::algorithms::factorial(k) * binomial(n - 1, k - 1)
}

pub fn eulerian_number(n: u64, k: u64) -> u128 {
    if k >= n {
        if n == 0 && k == 0 { return 1; }
        return 0;
    }
    let (n, k) = (n as usize, k as usize);
    let mut dp = vec![0u128; n + 1];
    dp[1] = 1;
    for i in 2..=n {
        let mut next = vec![0u128; n + 1];
        for j in 1..i {
            next[j] = (j as u128 + 1) * dp[j] + (i as u128 - j as u128) * dp[j - 1];
        }
        dp = next;
    }
    dp[k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s1() {
        assert_eq!(stirling1_unsigned(5, 3), 35);
        assert_eq!(stirling1_unsigned(3, 3), 1);
        assert_eq!(stirling1_unsigned(3, 1), 2);
    }

    #[test]
    fn s2() {
        assert_eq!(stirling2(5, 3), 25);
        assert_eq!(stirling2(4, 2), 7);
        assert_eq!(stirling2(6, 6), 1);
    }

    #[test]
    fn bell_numbers() {
        assert_eq!(bell(0), 1);
        assert_eq!(bell(3), 5);
        assert_eq!(bell(4), 15);
    }

    #[test]
    fn derange() {
        assert_eq!(derangements(0), 1);
        assert_eq!(derangements(1), 0);
        assert_eq!(derangements(4), 9);
    }

    #[test]
    fn lah() {
        assert_eq!(Lah_number(4, 2), 36);
        assert_eq!(Lah_number(3, 3), 1);
    }
}
