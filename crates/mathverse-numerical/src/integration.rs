//! Numerical integration: Gaussian quadrature, Romberg, adaptive methods.


/// Gaussian quadrature with Legendre polynomials.
pub struct GaussianQuadrature;

impl GaussianQuadrature {
    /// Integrate using Gaussian-Legendre quadrature.
    ///
    /// An `n`-point rule is exact for polynomials of degree `2n − 1`. For
    /// `n ≤ 8` the classic tabulated nodes are used; for larger `n` the
    /// nodes and weights are computed with Newton–Raphson iteration on the
    /// Legendre polynomial, so every `n` yields a true Gaussian rule (there
    /// is no silent fallback to a lower-order formula).
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_numerical::integration::GaussianQuadrature;
    ///
    /// // ∫₀¹ x¹⁰ dx = 1/11, exact with n = 6 since 2·6 − 1 ≥ 10
    /// let result = GaussianQuadrature::integrate(&|x| x.powi(10), 0.0, 1.0, 6);
    /// assert!((result - 1.0 / 11.0).abs() < 1e-14);
    /// ```
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let (nodes, weights) = Self::legendre_nodes_weights(n);
        
        let mut sum = 0.0;
        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            let transformed = 0.5 * (b - a) * x + 0.5 * (a + b);
            sum += w * f(transformed);
        }
        
        0.5 * (b - a) * sum
    }

    /// Get Legendre polynomial nodes and weights on `[-1, 1]`.
    ///
    /// For `n ≤ 8` the classic tabulated rules are used; for larger `n`
    /// [`GaussianQuadrature::gauss_legendre_newton`] computes the rule.
    fn legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
        match n {
            0 => (Vec::new(), Vec::new()),
            1 => (vec![0.0], vec![2.0]),
            2 => (vec![-0.577_350_269_189_625_7, 0.577_350_269_189_625_7], vec![1.0, 1.0]),
            3 => (vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
                vec![0.555_555_555_555_555_6, 0.888_888_888_888_888_8, 0.555_555_555_555_555_6]),
            4 => (vec![
                    -0.861_136_311_594_052_6, -0.339_981_043_584_856_3,
                    0.339_981_043_584_856_3, 0.861_136_311_594_052_6,
                ],
                vec![
                    0.347_854_845_137_453_9, 0.652_145_154_862_546_1,
                    0.652_145_154_862_546_1, 0.347_854_845_137_453_9,
                ]),
            5 => (vec![
                    -0.906_179_845_938_664, -0.538_469_310_105_683_1, 0.0,
                    0.538_469_310_105_683_1, 0.906_179_845_938_664,
                ],
                vec![
                    0.236_926_885_056_189_1, 0.478_628_670_499_366_5,
                    0.568_888_888_888_888_9, 0.478_628_670_499_366_5,
                    0.236_926_885_056_189_1,
                ]),
            6 => (vec![
                    -0.932_469_514_203_152, -0.661_209_386_466_264_5,
                    -0.238_619_186_083_197, 0.238_619_186_083_197,
                    0.661_209_386_466_264_5, 0.932_469_514_203_152,
                ],
                vec![
                    0.171_324_492_379_170_4, 0.360_761_573_048_138_6,
                    0.467_913_934_572_691, 0.467_913_934_572_691,
                    0.360_761_573_048_138_6, 0.171_324_492_379_170_4,
                ]),
            7 => (vec![
                    -0.949_107_912_342_758_5, -0.741_531_185_599_394_5,
                    -0.405_845_151_377_397_2, 0.0, 0.405_845_151_377_397_2,
                    0.741_531_185_599_394_5, 0.949_107_912_342_758_5,
                ],
                vec![
                    0.129_484_966_168_869_7, 0.279_705_391_489_276_6,
                    0.381_830_050_505_118_9, 0.417_959_183_673_469_4,
                    0.381_830_050_505_118_9, 0.279_705_391_489_276_6,
                    0.129_484_966_168_869_7,
                ]),
            8 => (vec![
                    -0.960_289_856_497_536_3, -0.796_666_477_413_626_7,
                    -0.525_532_409_916_329, -0.183_434_642_495_649_8,
                    0.183_434_642_495_649_8, 0.525_532_409_916_329,
                    0.796_666_477_413_626_7, 0.960_289_856_497_536_3,
                ],
                vec![
                    0.101_228_536_290_376_3, 0.222_381_034_453_374_5,
                    0.313_706_645_877_887_3, 0.362_683_783_378_362,
                    0.362_683_783_378_362, 0.313_706_645_877_887_3,
                    0.222_381_034_453_374_5, 0.101_228_536_290_376_3,
                ]),
            _ => Self::gauss_legendre_newton(n),
        }
    }

    /// Gauss-Legendre nodes and weights for arbitrary `n` via Newton-Raphson
    /// root finding on `P_n` (the classical `gauleg` construction).
    ///
    /// The `i`-th root is seeded with `cos(π(i + 3/4)/(n + 1/2))`, which is
    /// accurate enough that iteration converges to machine precision in a
    /// handful of steps. Weights follow from
    /// `w = 2 / ((1 − x²) · P′ₙ(x))²`.
    fn gauss_legendre_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let half = n.div_ceil(2);

        for i in 0..half {
            // Initial guess for the i-th root of P_n (i-th from +1 downward).
            let mut z =
                (core::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();

            let (mut x, mut w) = (z, 0.0);
            for _ in 0..100 {
                // Evaluate P_n(z) by the three-term recurrence; P_{n-1} is
                // carried in `p_prev`.
                let (mut p, mut p_prev) = (1.0f64, 0.0f64);
                for j in 1..=n {
                    let mut p_next =
                        ((2 * j) as f64 - 1.0) * z * p - (j as f64 - 1.0) * p_prev;
                    p_next /= j as f64;
                    p_prev = p;
                    p = p_next;
                }
                // (z² − 1) P′_n(z) = n (z P_n(z) − P_{n−1}(z))
                let dp = n as f64 * (z * p - p_prev) / (z * z - 1.0);
                let dz = -p / dp;
                z += dz;
                x = z;
                w = 2.0 / ((1.0 - z * z) * dp * dp);
                if dz.abs() <= 1e-15 {
                    break;
                }
            }

            // Roots come in ± pairs; fill symmetric slots.
            nodes[i] = -x;
            weights[i] = w;
            nodes[n - 1 - i] = x;
            weights[n - 1 - i] = w;
        }

        (nodes, weights)
    }

    /// 2D Gaussian quadrature.
    pub fn integrate_2d(f: &dyn Fn(f64, f64) -> f64, a: f64, b: f64, c: f64, d: f64, n: usize) -> f64 {
        let (nodes, weights) = Self::legendre_nodes_weights(n);
        
        let mut sum = 0.0;
        for (&xi, &wi) in nodes.iter().zip(weights.iter()) {
            let x = 0.5 * (b - a) * xi + 0.5 * (a + b);
            
            for (&yj, &wj) in nodes.iter().zip(weights.iter()) {
                let y = 0.5 * (d - c) * yj + 0.5 * (c + d);
                sum += wi * wj * f(x, y);
            }
        }
        
        0.25 * (b - a) * (d - c) * sum
    }
}

/// Romberg integration (Richardson extrapolation).
pub struct RombergIntegration;

impl RombergIntegration {
    /// Integrate using Romberg method.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, max_levels: usize) -> f64 {
        let mut r = vec![vec![0.0; max_levels]; max_levels];
        
        // First column: trapezoidal rule with increasing subdivisions
        for k in 0..max_levels {
            let n = 2_usize.pow(k as u32);
            r[k][0] = Self::trapezoidal(f, a, b, n);
        }
        
        // Richardson extrapolation
        for j in 1..max_levels {
            for k in j..max_levels {
                let factor = 4.0_f64.powi(j as i32);
                r[k][j] = (factor * r[k][j - 1] - r[k - 1][j - 1]) / (factor - 1.0);
            }
        }
        
        r[max_levels - 1][max_levels - 1]
    }

    fn trapezoidal(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Adaptive Simpson's rule.
pub struct AdaptiveSimpson;

impl AdaptiveSimpson {
    /// Integrate with adaptive Simpson's rule.
    pub fn integrate(
        f: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_depth: usize,
    ) -> f64 {
        Self::adaptive_simpson_recursive(f, a, b, tolerance, max_depth, f(a), f(b), f(f64::midpoint(a, b)))
    }

    fn adaptive_simpson_recursive(
        f: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        depth: usize,
        fa: f64,
        fb: f64,
        fm: f64,
    ) -> f64 {
        let h = b - a;
        let m = f64::midpoint(a, b);
        
        let whole = h / 6.0 * (fa + 4.0 * fm + fb);
        
        let fl = f(f64::midpoint(a, m));
        let fr = f(f64::midpoint(m, b));
        
        let left = h / 12.0 * (fa + 4.0 * fl + fm);
        let right = h / 12.0 * (fm + 4.0 * fr + fb);
        
        if depth == 0 || (left + right - whole).abs() < 15.0 * tolerance {
            left + right + (left + right - whole) / 15.0
        } else {
            Self::adaptive_simpson_recursive(f, a, m, tolerance / 2.0, depth - 1, fa, fm, fl)
                + Self::adaptive_simpson_recursive(f, m, b, tolerance / 2.0, depth - 1, fm, fb, fr)
        }
    }
}

/// Monte Carlo integration.
pub struct MonteCarloIntegration;

impl MonteCarloIntegration {
    /// Integrate using Monte Carlo method.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, samples: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..samples {
            let x = a + (b - a) * rand::random::<f64>();
            let y = f(x);
            sum += y;
            sum_sq += y * y;
        }
        
        let mean = sum / samples as f64;
        let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
        let error = (variance / samples as f64).sqrt() * (b - a);
        
        (mean * (b - a), error)
    }

    /// 2D Monte Carlo integration.
    pub fn integrate_2d(f: &dyn Fn(f64, f64) -> f64, a: f64, b: f64, c: f64, d: f64, samples: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..samples {
            let x = a + (b - a) * rand::random::<f64>();
            let y = c + (d - c) * rand::random::<f64>();
            let z = f(x, y);
            sum += z;
            sum_sq += z * z;
        }
        
        let mean = sum / samples as f64;
        let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
        let error = (variance / samples as f64).sqrt() * (b - a) * (d - c);
        
        (mean * (b - a) * (d - c), error)
    }
}

/// Simpson's rule with fixed number of intervals.
pub struct SimpsonRule;

impl SimpsonRule {
    /// Integrate using Simpson's rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if !n.is_multiple_of(2) {
            return Self::trapezoidal(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * f(x);
        }
        
        sum * h / 3.0
    }

    fn trapezoidal(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Midpoint rule.
pub struct MidpointRule;

impl MidpointRule {
    /// Integrate using midpoint rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * h;
            sum += f(x);
        }
        
        sum * h
    }
}

/// Solve `A x = b` (square, non-singular) by Gaussian elimination with
/// partial pivoting. Used internally for quadrature weight construction.
fn solve_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        let pv = m[col][col];
        for r in col + 1..n {
            let factor = m[r][col] / pv;
            for c in col..n {
                m[r][c] -= factor * m[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut acc = rhs[r];
        for c in r + 1..n {
            acc -= m[r][c] * x[c];
        }
        x[r] = acc / m[r][r];
    }
    x
}

/// Boole's rule (4th order Newton-Cotes).
pub struct BooleRule;

impl BooleRule {
    /// Integrate using Boole's rule.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if !n.is_multiple_of(4) {
            return SimpsonRule::integrate(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        
        for i in 0..n / 4 {
            let x0 = a + (4 * i) as f64 * h;
            let x1 = a + (4 * i + 1) as f64 * h;
            let x2 = a + (4 * i + 2) as f64 * h;
            let x3 = a + (4 * i + 3) as f64 * h;
            let x4 = a + (4 * i + 4) as f64 * h;
            
            sum += 7.0 * f(x0) + 32.0 * f(x1) + 12.0 * f(x2) + 32.0 * f(x3) + 7.0 * f(x4);
        }
        
        sum * 2.0 * h / 45.0
    }
}

/// Clenshaw-Curtis quadrature.
pub struct ClenshawCurtis;

impl ClenshawCurtis {
    /// Integrate using Clenshaw-Curtis quadrature.
    ///
    /// Uses the `n` Chebyshev-extrema nodes `x_k = cos(πk/(n−1))` and the
    /// standard Clenshaw-Curtis weights (exact for polynomials up to degree
    /// `n−1`). The weights are computed from the discrete cosine transform
    /// of the moment sequence `2/(1−k²)` for even `k`.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        assert!(n >= 2, "Clenshaw-Curtis requires at least 2 nodes");
        let m = n - 1;
        let mut nodes = vec![0.0; n];
        for i in 0..n {
            nodes[i] = (core::f64::consts::PI * i as f64 / m as f64).cos();
        }
        let weights = Self::cc_weights(n);

        let mut sum = 0.0;
        for i in 0..n {
            let x = 0.5 * (b - a) * nodes[i] + 0.5 * (a + b);
            sum += weights[i] * f(x);
        }

        0.5 * (b - a) * sum
    }

    /// Clenshaw-Curtis weights for `n` Chebyshev-extrema nodes.
    ///
    /// The weights are the unique solution of the moment system
    /// `Σ_j w_j x_j^k = ∫₋₁¹ x^k dx` for `k = 0..n−1`, i.e. the rule is exact
    /// for all polynomials of degree `< n`. Solving the Vandermonde system
    /// directly avoids transcription errors in closed-form weight formulas
    /// and is numerically stable for the modest `n` used in practice.
    fn cc_weights(n: usize) -> Vec<f64> {
        let mut a = vec![vec![0.0; n]; n];
        let mut b = vec![0.0; n];
        for i in 0..n {
            let x = (core::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            let mut pow = 1.0;
            for k in 0..n {
                a[i][k] = pow;
                pow *= x;
            }
            // ∫₋₁¹ x^i dx = 0 for odd i, 2/(i+1) for even i.
            b[i] = if i % 2 == 0 { 2.0 / (i as f64 + 1.0) } else { 0.0 };
        }
        // Solve Aᵀ w = b (the system is V·w = b with V[i][k] = x_i^k;
        // transpose since we index rows by node).
        let mut at = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                at[i][j] = a[j][i];
            }
        }
        solve_system(&at, &b)
    }
}

/// Double exponential integration (tanh-sinh quadrature).
pub struct DoubleExponential;

impl DoubleExponential {
    /// Integrate using double exponential (tanh-sinh) quadrature.
    ///
    /// Maps `t ∈ [-L, L]` onto `[a, b]` via `x = (a+b)/2 + (b−a)/2 · tanh(π/2·sinh t)`
    /// and applies the trapezoidal rule with step `h`. The quadrature
    /// converges geometrically for integrands with endpoint singularities.
    pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = 1.0 / n as f64;
        // Cover t ∈ [-L, L] with L ≈ 4: beyond that, tanh(π/2·sinh t) is
        // indistinguishable from ±1 and the jacobian underflows to 0.
        let steps = (4.0 / h) as usize;
        let mut sum = 0.0;

        for k in -(steps as isize)..=(steps as isize) {
            let t = k as f64 * h;
            let phi = core::f64::consts::PI / 2.0 * t.sinh();
            let x = f64::midpoint(b, a) + (b - a) / 2.0 * phi.tanh();

            // dx/dt = (b−a)/2 · d/dt[tanh(φ(t))]
            //       = (b−a)/2 · φ'(t) · sech²(φ) with φ'(t) = π/2·cosh t
            let jacobian = (b - a) / 2.0
                * (core::f64::consts::PI / 2.0 * t.cosh())
                * (1.0 - phi.tanh().powi(2));

            sum += jacobian * f(x);
        }

        sum * h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_quadrature() {
        let result = GaussianQuadrature::integrate(&|x| x * x, -1.0, 1.0, 5);
        assert!((result - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_quadrature_high_order_is_true_gauss_rule() {
        // n > 8 previously degraded to a trapezoidal-like fallback; now it
        // must remain a genuine degree-(2n−1)-exact Gaussian rule.
        // ∫₋₁¹ x¹⁰ dx = 2/11 — exact for n ≥ 6, and certainly for n = 12.
        let result = GaussianQuadrature::integrate(&|x| x.powi(10), -1.0, 1.0, 12);
        assert!((result - 2.0 / 11.0).abs() < 1e-13);

        // ∫₋₁¹ x²⁰ dx = 2/21 — needs degree-40 exactness, i.e. n ≥ 21 > 8.
        let result = GaussianQuadrature::integrate(&|x| x.powi(20), -1.0, 1.0, 21);
        assert!((result - 2.0 / 21.0).abs() < 1e-12);

        // Weights sum to the length of [-1, 1] and the rule is symmetric.
        let (nodes, weights) = GaussianQuadrature::gauss_legendre_newton(16);
        let total: f64 = weights.iter().sum();
        assert!((total - 2.0).abs() < 1e-12);
        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            assert!(w >= 0.0 && x.abs() <= 1.0);
        }
        let mirrored: Vec<f64> = nodes.iter().rev().copied().collect();
        for (a, b) in nodes.iter().zip(mirrored.iter()) {
            assert!((a + b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gaussian_2d() {
        let result = GaussianQuadrature::integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5);
        assert!((result - 0.25).abs() < 1e-8);
    }

    #[test]
    fn test_romberg() {
        let result = RombergIntegration::integrate(&|x| x * x, 0.0, 1.0, 8);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_simpson() {
        let result = AdaptiveSimpson::integrate(&|x| x * x, 0.0, 1.0, 1e-10, 20);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo() {
        let (result, error) = MonteCarloIntegration::integrate(&|x| x * x, 0.0, 1.0, 10000);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
        assert!(error < 0.01);
    }

    #[test]
    fn test_simpson_rule() {
        let result = SimpsonRule::integrate(&|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_rule() {
        let result = MidpointRule::integrate(&|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_boole_rule() {
        let result = BooleRule::integrate(&|x| x * x, 0.0, 1.0, 8);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clenshaw_curtis() {
        let result = ClenshawCurtis::integrate(&|x| x * x, -1.0, 1.0, 10);
        assert!((result - 2.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_double_exponential() {
        let result = DoubleExponential::integrate(&|x| x * x, 0.0, 1.0, 10);
        assert!((result - 1.0 / 3.0).abs() < 1e-6);
    }
}
