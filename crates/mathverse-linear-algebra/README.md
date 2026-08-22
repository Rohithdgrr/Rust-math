# MathVerse Linear Algebra

[![Crates.io](https://img.shields.io/crates/v/mathverse-linear-algebra.svg)](https://crates.io/crates/mathverse-linear-algebra)
[![docs.rs](https://docs.rs/mathverse-linear-algebra/badge.svg)](https://docs.rs/mathverse-linear-algebra)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Lightweight linear algebra — decompositions, solvers, norms, eigenvalues, and more. Built on the shared `mathverse_matrix::Matrix` type (flat row-major `Vec<f64>` storage) so results interoperate with the rest of the MathVerse ecosystem.

---

## Features

- LU, QR, and Cholesky decompositions
- Forward/back substitution solvers
- Fixed-size solvers (2×2, 3×3) via Cramer's rule
- Gaussian elimination with partial pivoting
- Least squares via normal equations
- Matrix norms: L1, L∞, Frobenius, spectral (L2)
- Singular value computation
- Condition number estimation
- Power iteration for dominant eigenvalue
- Analytical 2×2 eigenvalue computation

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `decomposition` | LU, QR, Cholesky decompositions; 2×2 eigenvalue; power iteration |
| `solve` | LU solve, QR solve, 2×2/3×3 Cramer, Gaussian elimination, least squares |
| `norm` | L1, L∞, Frobenius, spectral norms; singular values; condition number |
| `eigen` | Re-exports from decomposition |
| `inverse` | Re-exports from solve |
| `rank` | Re-exports from norm |
| `least_squares` | Re-exports from solve |

---

## ASCII Art: Decompositions

```
LU Decomposition
================
A = L U

    A               L (unit lower)    U (upper)
┌           ┐   ┌           ┐   ┌           ┐
│ 2  1      │   │ 1  0      │   │ 2  1      │
│ 1  3      │ = │ 0.5 1     │ × │ 0  2.5    │
└           ┘   └           ┘   └           ┘

    Forward sub:  Ly = b     O(n²)
    Back sub:     Ux = y     O(n²)
    Total solve:  O(n³) decomposition + O(n²) per RHS


QR Decomposition (Gram-Schmidt)
================
A = Q R

    Q columns are orthonormal basis of col(A)
    R encodes coefficients of A's columns in Q's basis


Cholesky Decomposition
======================
A = L L^T    (symmetric positive definite)

    A is SPD ⟺ Cholesky succeeds
    det(A) = det(L)² = Π l_ii²
```

---

## Installation

```toml
[dependencies]
mathverse-linear-algebra = "0.1"
```

---

## Quick Start

```rust
use mathverse_linear_algebra::*;
use mathverse_matrix::Matrix;

fn main() {
    let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]).unwrap();
    let b = vec![5.0, 7.0];

    // Solve via Gaussian elimination
    let x = solve_gauss(&a, &b).unwrap();
    println!("x = {:?}", x);  // [1.6, 1.8]

    // Or via LU decomposition (with pivoting)
    let (l, u, perm) = lu_decompose(&a).unwrap();
    let x = solve_lu(&l, &u, &perm, &b);
    println!("x = {:?}", x);  // [1.6, 1.8]

    // Norms
    println!("‖A‖₁   = {:.4}", norm_1(&a));
    println!("‖A‖∞   = {:.4}", norm_inf(&a));
    println!("‖A‖_F  = {:.4}", norm_frobenius(&a));
}
```

---

## Per-Module Documentation

### Decomposition (`decomposition`)

| Function | Description |
|---|---|
| `lu_decompose(a)` | LU decomposition with partial pivoting, returns `(L, U, perm)` |
| `qr_decompose(a)` | QR decomposition (Gram-Schmidt), returns `(Q, R)` |
| `cholesky(a)` | Cholesky decomposition, returns `L` where `A = LL^T` |
| `eigenvalue_2x2(a)` | Analytical eigenvalue computation for 2×2 matrices |
| `power_iteration(a, max_iter, tol)` | Dominant eigenvalue + eigenvector via power iteration |

---

### Solve (`solve`)

| Function | Description |
|---|---|
| `solve_lu(l, u, perm, b)` | Solve `Ax = b` via LU: forward sub `Ly = Pb`, back sub `Ux = y` |
| `solve_qr(q, r, b)` | Solve `Ax = b` via QR: `Q^T b` then `Rx = Q^T b` |
| `solve_2x2(a, b)` | 2×2 Cramer's rule |
| `solve_3x3(a, b)` | 3×3 Cramer's rule |
| `solve_gauss(a, b)` | Gaussian elimination with partial pivoting |
| `ls_solve(a, b)` | Least squares via normal equations: `(A^T A)x = A^T b` |
| `residual_norm(a, x, b)` | Compute `‖Ax - b‖₂` |

---

### Norm (`norm`)

| Function | Formula |
|---|---|
| `norm_1(a)` | `max_j Σ_i |a_ij|` (maximum column sum) |
| `norm_inf(a)` | `max_i Σ_j |a_ij|` (maximum row sum) |
| `norm_frobenius(a)` | `√(Σ_ij |a_ij|²)` |
| `norm_2(a)` | `σ_max(A)` (spectral norm) |
| `singular_values(a)` | `[σ₁, σ₂, ...]` |
| `condition_number(a)` | `σ_max / σ_min` |
| `matrix_norm(a, p)` | General Lp matrix norm |

---

## Roadmap

- QR with column pivoting for rank-revealing
- Eigenvalue decomposition for general (non-symmetric) matrices
- Iterative solvers (CG, GMRES)
- Sparse matrix support
- Parallel decompositions for large systems

---

## License

MIT — see [LICENSE](LICENSE).
