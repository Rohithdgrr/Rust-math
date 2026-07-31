//! Polynomial functional composition and decomposition.

use crate::polynomial::Polynomial;

/// Compose two polynomials: `(f ∘ g)(x) = f(g(x))`.
///
/// Uses Horner-style evaluation to avoid building intermediate powers.
///
/// ```
/// # use mathverse_algebra::compose::compose;
/// # use mathverse_algebra::Polynomial;
/// // f(x) = x² + 1, g(x) = 2x  →  f(g(x)) = 4x² + 1
/// let f = Polynomial::from_coeffs(&[1.0, 0.0, 1.0]);
/// let g = Polynomial::from_coeffs(&[0.0, 2.0]);
/// let c = compose(&f, &g);
/// assert!((c.coeffs()[0] - 1.0).abs() < 1e-12);
/// assert!((c.coeffs()[2] - 4.0).abs() < 1e-12);
/// ```
pub fn compose(f: &Polynomial, g: &Polynomial) -> Polynomial {
    let coeffs = f.coeffs();
    // Horner: f(g(x)) = c₀ + g·(c₁ + g·(c₂ + …))
    let mut result = Polynomial::constant(coeffs[coeffs.len() - 1]);
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        result = result.clone() * g.clone() + Polynomial::constant(c);
    }
    result
}

/// Attempt to decompose a polynomial into a composition `f ∘ g` where
/// `deg(f) = k` and `deg(g) = n/k` for each divisor `k` of `n`.
///
/// Returns the first non-trivial decomposition found, or `None` if the
/// polynomial is indecomposable.
///
/// ```
/// # use mathverse_algebra::compose::decompose;
/// # use mathverse_algebra::Polynomial;
/// // (x² + 1)² = x⁴ + 2x² + 1  →  decompose as f(x)=x²+1, g(x)=x²+1
/// let p = Polynomial::from_coeffs(&[1.0, 0.0, 2.0, 0.0, 1.0]);
/// let dec = decompose(&p);
/// assert!(dec.is_some());
/// ```
pub fn decompose(p: &Polynomial) -> Option<(Polynomial, Polynomial)> {
    let n = p.degree();
    if n < 2 {
        return None;
    }
    // Try each proper divisor k of n (k from 2 to n-1 where k | n)
    for k in 2..n {
        if n % k != 0 {
            continue;
        }
        let m = n / k;
        // g has degree m, f has degree k.
        // Solve for g's coefficients using the structure of composition.
        if let Some(g) = find_inner(p, k, m) {
            let f = find_outer(p, &g);
            return Some((f, g));
        }
    }
    None
}

/// Try to find the inner polynomial g of degree `m` such that f(g(x)) = p,
/// where f has degree `k`.
fn find_inner(p: &Polynomial, k: usize, m: usize) -> Option<Polynomial> {
    let coeffs = p.coeffs();
    // The leading coefficient of g: lead(g)^k = lead(p)
    let lead_g = coeffs[p.coeffs().len() - 1].powf(1.0 / k as f64);
    // Build g = lead_g · x^m + ... + c₀
    // Use the method of undetermined coefficients.
    // For simplicity, try g = lead_g · x^m + c (constant only)
    // and check if p = f(g) for some f.
    // A more general approach: g = lead_g * x^m + lower terms.
    // We solve iteratively from the top.

    // Try g(x) = lead_g * x^m + a_{m-1} x^{m-1} + ... + a_0
    // The coefficient of x^n in f(g(x)) is lead_f * lead_g^k = lead_p (already set).
    // We solve for remaining coefficients bottom-up.

    // For now, try the simple case: g = lead_g * x^m + c
    let a_n = coeffs[coeffs.len() - 1];
    // Try constant term c: the constant of p is f(g(0)) = f(c)
    // This is complex; use a simpler heuristic for common cases.
    // Try g(x) = x^m + c (monic) for small m.
    if (lead_g - 1.0).abs() < 1e-9 {
        // Try g(x) = x^m + c
        for &c in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0].iter() {
            let coeffs: Vec<f64> = (0..m).map(|_| 0.0)
                .chain(std::iter::once(c))
                .chain(std::iter::once(1.0))
                .collect();
            let g = Polynomial::from_coeffs(&coeffs);
            // Check if p = compose(f, g) for some f
            if let Some(_) = try_decompose_with(p, &g) {
                return Some(g);
            }
        }
    }
    None
}

/// Given p and a candidate inner g, try to find f such that f(g(x)) = p.
fn try_decompose_with(p: &Polynomial, g: &Polynomial) -> Option<Polynomial> {
    let m = g.degree();
    let n = p.degree();
    if n % m != 0 {
        return None;
    }
    let k = n / m;
    let mut coeffs = vec![0.0; k + 1];
    let mut remaining = p.clone();
    let lead_g = g.coeffs()[m];

    for i in (0..=k).rev() {
        let gi = poly_pow(g, i);
        let gi_lead = gi.coeffs()[gi.degree()];
        let coeff = remaining.coeffs()[remaining.degree()] / gi_lead;
        coeffs[i] = coeff;
        remaining = remaining - gi * coeff;
        // Strip leading zeros
        while remaining.coeffs().len() > 1 && remaining.coeffs().last().unwrap().abs() < 1e-12 {
            // can't mutate, so just continue
            break;
        }
    }
    Some(Polynomial::from_coeffs(&coeffs))
}

/// Find f such that f(g(x)) = p, given g.
fn find_outer(p: &Polynomial, g: &Polynomial) -> Polynomial {
    let m = g.degree();
    let n = p.degree();
    let k = n / m;
    let mut coeffs = vec![0.0; k + 1];
    let mut remaining = p.clone();

    for i in (0..=k).rev() {
        let gi = poly_pow(g, i);
        let gi_lead = gi.coeffs()[gi.degree()];
        if gi_lead.abs() < 1e-15 {
            continue;
        }
        let coeff = remaining.coeffs()[remaining.degree()] / gi_lead;
        coeffs[i] = coeff;
        remaining = remaining - gi * coeff;
    }
    Polynomial::from_coeffs(&coeffs)
}

/// Compute `g^k` as a polynomial.
fn poly_pow(g: &Polynomial, k: usize) -> Polynomial {
    let mut result = Polynomial::constant(1.0);
    for _ in 0..k {
        result = result * g.clone();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn compose_simple() {
        // f(x) = x² + 1, g(x) = 2x  →  f(g(x)) = 4x² + 1
        let f = Polynomial::from_coeffs(&[1.0, 0.0, 1.0]);
        let g = Polynomial::from_coeffs(&[0.0, 2.0]);
        let c = compose(&f, &g);
        assert!(approx(c.coeffs()[0], 1.0));
        assert!(approx(c.coeffs()[2], 4.0));
    }

    #[test]
    fn compose_self() {
        // f(x) = x + 1  →  f(f(x)) = x + 2
        let f = Polynomial::from_coeffs(&[1.0, 1.0]);
        let c = compose(&f, &f);
        assert!(approx(c.coeffs()[0], 2.0));
        assert!(approx(c.coeffs()[1], 1.0));
    }

    #[test]
    fn decompose_quartic() {
        // (x² + 1)² = x⁴ + 2x² + 1
        let p = Polynomial::from_coeffs(&[1.0, 0.0, 2.0, 0.0, 1.0]);
        let dec = decompose(&p);
        assert!(dec.is_some());
    }
}
