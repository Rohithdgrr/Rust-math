//! Elementary symmetric polynomials and Newton's identities.

const TOL: f64 = 1e-12;

/// Compute the elementary symmetric polynomials `e₁, e₂, …, eₙ` from a list
/// of variables `x₁, x₂, …, xₙ`.
///
/// `e_k` is the sum of all products of `k` distinct variables.
///
/// Elementary symmetric polynomial `e_k` of `values`.
///
/// `e_0 = 1`, `e_1 = Σxᵢ`, `e_2 = Σᵢ<ⱼ xᵢxⱼ`, …, `e_k = Σ` products of `k`
/// distinct elements.
///
/// ```
/// # use mathverse_algebra::symmetric::elementary_symmetric;
/// // e_2 of [1, 2, 3] = 1·2 + 1·3 + 2·3 = 11
/// assert_eq!(elementary_symmetric(&[1.0, 2.0, 3.0], 2), 11.0);
/// ```
pub fn elementary_symmetric(values: &[f64], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > values.len() {
        return 0.0;
    }
    // DP: dp[j] = e_j of values seen so far
    let mut dp = vec![0.0; k + 1];
    dp[0] = 1.0;
    for &v in values {
        for j in (1..=k).rev() {
            dp[j] += dp[j - 1] * v;
        }
    }
    dp[k]
}

/// Power sum `p_k = Σ xᵢᵏ`.
///
/// ```
/// # use mathverse_algebra::symmetric::power_sum;
/// assert_eq!(power_sum(&[1.0, 2.0, 3.0], 2), 14.0); // 1 + 4 + 9
/// ```
pub fn power_sum(values: &[f64], k: usize) -> f64 {
    values.iter().map(|v| v.powi(k as i32)).sum()
}

/// Newton's identities: given power sums `p_1, …, p_n`, compute the elementary
/// symmetric polynomials `e_1, …, e_n`.
///
/// Returns `e_0, e_1, …, e_n`.
///
/// ```
/// # use mathverse_algebra::symmetric::newtons_identities;
/// // Values [1, 2, 3]: p1=6, p2=14, p3=36
/// let e = newtons_identities(&[6.0, 14.0, 36.0]);
/// assert!((e[1] - 6.0).abs() < 1e-9); // e1 = 6
/// assert!((e[2] - 11.0).abs() < 1e-9); // e2 = 11
/// assert!((e[3] - 6.0).abs() < 1e-9); // e3 = 6

pub fn newtons_identities(power_sums: &[f64]) -> Vec<f64> {
    let n = power_sums.len();
    let mut e = vec![0.0; n + 1];
    e[0] = 1.0;
    for k in 1..=n {
        let mut s = 0.0;
        for i in 1..=k {
            s += e[k - i] * power_sums[i - 1];
        }
        e[k] = s / k as f64;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn elementary() {
        let v = [1.0, 2.0, 3.0];
        assert!(approx(elementary_symmetric(&v, 0), 1.0));
        assert!(approx(elementary_symmetric(&v, 1), 6.0));
        assert!(approx(elementary_symmetric(&v, 2), 11.0));
        assert!(approx(elementary_symmetric(&v, 3), 6.0));
        assert!(approx(elementary_symmetric(&v, 4), 0.0));
    }

    #[test]
    fn power_sums() {
        let v = [1.0, 2.0, 3.0];
        assert!(approx(power_sum(&v, 1), 6.0));
        assert!(approx(power_sum(&v, 2), 14.0));
        assert!(approx(power_sum(&v, 3), 36.0));
    }

    #[test]
    fn newton_id() {
        let e = newtons_identities(&[6.0, 14.0, 36.0]);
        assert!(approx(e[1], 6.0));
        assert!(approx(e[2], 11.0));
        assert!(approx(e[3], 6.0));
    }
}