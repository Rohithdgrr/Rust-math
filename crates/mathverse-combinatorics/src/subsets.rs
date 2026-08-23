//! Power set, subsets, Cartesian product, permutations, compositions, stars and bars.

/// Generates the power set of `{0, 1, ..., n-1}`.
pub fn power_set(n: usize) -> Vec<Vec<usize>> {
    (0..(1usize << n))
        .map(|mask| (0..n).filter(|i| mask & (1 << i) != 0).collect())
        .collect()
}

/// All size-`k` subsets of `{0, 1, ..., n-1}`.
#[must_use]
pub fn subsets_of_size(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn backtrack(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            backtrack(n, k, i + 1, current, result);
            current.pop();
        }
    }
    let mut result = Vec::new();
    if k > n {
        return result;
    }
    backtrack(n, k, 0, &mut Vec::new(), &mut result);
    result
}

/// Cartesian product of two index sets as pairs.
#[must_use]
pub fn cartesian_product(a: &[usize], b: &[usize]) -> Vec<(usize, usize)> {
    a.iter()
        .flat_map(|&x| b.iter().map(move |&y| (x, y)))
        .collect()
}

/// All permutations of `{0, 1, ..., n-1}` in lexicographic order.
#[must_use]
pub fn permutation_index(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut result = Vec::new();
    let mut current: Vec<usize> = (0..n).collect();
    result.push(current.clone());
    loop {
        let mut i = n - 1;
        while i > 0 && current[i - 1] > current[i] {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        let mut j = n - 1;
        while current[j] < current[i - 1] {
            j -= 1;
        }
        current.swap(i - 1, j);
        current[i..].reverse();
        result.push(current.clone());
    }
    result
}

/// Number of ways to write `n` as an ordered sum of `k` positive integers
/// (compositions).
#[must_use]
pub fn composition(n: u64, k: u64) -> u128 {
    use mathverse_core::algorithms::binomial;

    if k == 0 {
        return u128::from(n == 0);
    }
    binomial(n - 1, k - 1)
}

/// Number of ways to put `n` indistinguishable items into `k` distinguishable
/// boxes.
#[must_use]
pub fn stars_and_bars(n: u64, k: u64) -> u128 {
    use mathverse_core::algorithms::binomial;
    binomial(n + k - 1, k - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powerset() {
        let ps = power_set(3);
        assert_eq!(ps.len(), 8);
    }

    #[test]
    fn subsets() {
        let s = subsets_of_size(4, 2);
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn cartesian() {
        let cp = cartesian_product(&[0, 1], &[0, 1, 2]);
        assert_eq!(cp.len(), 6);
    }

    #[test]
    fn perms() {
        let p = permutation_index(3);
        assert_eq!(p.len(), 6);
        assert_eq!(p[0], vec![0, 1, 2]);
    }

    #[test]
    fn comp() {
        assert_eq!(composition(4, 2), 3);
        assert_eq!(stars_and_bars(3, 2), 4);
    }
}
