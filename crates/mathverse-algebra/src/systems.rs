//! Solving 2x2 and 3x3 systems of linear equations via elimination.

use crate::{AlgebraError, Result, TOL};

/// Solve a 2x2 system using elimination:
/// `a1*x + b1*y = c1`, `a2*x + b2*y = c2`.
///
/// Returns `Some((x, y))` or `None` if singular.
///
/// ```
/// # use mathverse_algebra::systems::solve_2x2;
/// // x + y = 3, 2x - y = 0 -> x = 1, y = 2
/// let sol = solve_2x2(1.0, 1.0, 3.0, 2.0, -1.0, 0.0).unwrap();
/// assert!((sol.0 - 1.0).abs() < 1e-12);
/// assert!((sol.1 - 2.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn solve_2x2(a1: f64, b1: f64, c1: f64, a2: f64, b2: f64, c2: f64) -> Result<(f64, f64)> {
    let det = a1 * b2 - a2 * b1;
    if det.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let x = (c1 * b2 - c2 * b1) / det;
    let y = (a1 * c2 - a2 * c1) / det;
    Ok((x, y))
}

/// Solve a 3x3 system using Gaussian elimination with partial pivoting.
///
/// `a1*x + b1*y + c1*z = d1`, etc.
///
/// Returns `Some((x, y, z))` or `None` if singular.
///
/// ```
/// # use mathverse_algebra::systems::solve_3x3;
/// let sol = solve_3x3(
///     1.0, 1.0, 1.0, 6.0,
///     2.0, 3.0, 1.0, 11.0,
///     1.0, 2.0, 3.0, 14.0,
/// ).unwrap();
/// assert!((sol.0 - 1.0).abs() < 1e-9);
/// assert!((sol.1 - 2.0).abs() < 1e-9);
/// assert!((sol.2 - 3.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn solve_3x3(
    a1: f64, b1: f64, c1: f64, d1: f64,
    a2: f64, b2: f64, c2: f64, d2: f64,
    a3: f64, b3: f64, c3: f64, d3: f64,
) -> Result<(f64, f64, f64)> {
    let mut m = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2],
        [a3, b3, c3, d3],
    ];

    // Forward elimination with partial pivoting.
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
            .unwrap();
        if m[pivot][col].abs() < TOL {
            return Err(AlgebraError::Singular);
        }
        if pivot != col {
            m.swap(col, pivot);
        }
        for row in (col + 1)..3 {
            let factor = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }

    // Back-substitution.
    let z = m[2][3] / m[2][2];
    let y = (m[1][3] - m[1][2] * z) / m[1][1];
    let x = (m[0][3] - m[0][1] * y - m[0][2] * z) / m[0][0];
    Ok((x, y, z))
}

/// Solve a system of 2 linear equations (matrix form).
/// `[a b; c d] * [x; y] = [e; f]`.
#[must_use]
pub fn solve_matrix_2x2(
    a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
) -> Result<(f64, f64)> {
    solve_2x2(a, b, e, c, d, f)
}

/// Solve a system of 3 linear equations (matrix form).
#[must_use]
pub fn solve_matrix_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
    j: f64, k: f64, l: f64,
) -> Result<(f64, f64, f64)> {
    solve_3x3(a, b, c, j, d, e, f, k, g, h, i, l)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn system_2x2() {
        let sol = solve_2x2(1.0, 1.0, 3.0, 2.0, -1.0, 0.0).unwrap();
        assert!(approx(sol.0, 1.0));
        assert!(approx(sol.1, 2.0));
    }

    #[test]
    fn system_2x2_singular() {
        assert_eq!(
            solve_2x2(1.0, 2.0, 3.0, 2.0, 4.0, 6.0),
            Err(AlgebraError::Singular)
        );
    }

    #[test]
    fn system_3x3() {
        let sol = solve_3x3(
            1.0, 1.0, 1.0, 6.0,
            2.0, 3.0, 1.0, 11.0,
            1.0, 2.0, 3.0, 14.0,
        )
        .unwrap();
        assert!(approx(sol.0, 1.0));
        assert!(approx(sol.1, 2.0));
        assert!(approx(sol.2, 3.0));
    }

    #[test]
    fn system_3x3_singular() {
        assert_eq!(
            solve_3x3(
                1.0, 2.0, 3.0, 4.0,
                2.0, 4.0, 6.0, 8.0,
                1.0, 1.0, 1.0, 3.0,
            ),
            Err(AlgebraError::Singular)
        );
    }

    #[test]
    fn matrix_form() {
        let sol = solve_matrix_2x2(1.0, 1.0, 2.0, -1.0, 3.0, 0.0).unwrap();
        assert!(approx(sol.0, 1.0));
        assert!(approx(sol.1, 2.0));
    }
}
