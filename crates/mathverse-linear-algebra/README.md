# mathverse-linear-algebra

**Lightweight linear algebra: decompositions, solvers, norms, eigenvalues, and more.**

`mathverse-linear-algebra` is a dependency-free linear algebra library built on
`Vec<Vec<f64>>` matrices. It wraps `mathverse-matrix` for heavy lifting while
exposing a simple, functional API for everyday linear algebra tasks.

## Features

- LU, QR, and Cholesky decompositions
- Forward/back substitution solvers
- Fixed-size solvers (2x2, 3x3) via Cramer's rule
- Gaussian elimination with partial pivoting
- Least squares via normal equations
- Matrix norms: L1, L∞, Frobenius, spectral (L2)
- Singular value computation
- Condition number estimation
- Power iteration for dominant eigenvalue
- Analytical 2x2 eigenvalue computation

## Module Overview

| Module | Description |
|---|---|
| `decomposition` | LU, QR, Cholesky decompositions; 2x2 eigenvalue; power iteration |
| `solve` | LU solve, QR solve, 2x2/3x3 Cramer, Gaussian elimination, least squares |
| `norm` | L1, L∞, Frobenius, spectral norms; singular values; condition number |
| `eigen` | (Re-exports from decomposition) |
| `inverse` | (Re-exports from solve) |
| `rank` | (Re-exports from norm) |
| `least_squares` | (Re-exports from solve) |

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

    A (m×n)       Q (m×n)        R (n×n)
┌           ┐   ┌           ┐   ┌           ┐
│           │ = │           │ × │           │
│  columns  │   │ orthonorm │   │ upper tri │
│           │   │ Q^T Q = I │   │           │
└           ┘   └           ┘   └           ┘

    Q columns are orthonormal basis of col(A)
    R encodes coefficients of A's columns in Q's basis


Cholesky Decomposition
======================
A = L L^T    (symmetric positive definite)

    A               L (lower tri)
┌           ┐   ┌           ┐
│ 4  2      │   │ 2  0      │
│ 2  5      │ = │ 1  2      │
└           ┘   └           ┘

    A is SPD ⟺ Cholesky succeeds
    det(A) = det(L)² = Π l_ii²
```

## Installation

```toml
[dependencies]
mathverse-linear-algebra = { path = "../mathverse-linear-algebra" }
```

## Quick Start

```rust
use mathverse_linear_algebra::*;

fn main() {
    let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let b = vec![5.0, 7.0];

    // Solve via Gaussian elimination
    let x = solve_gauss(&a, &b).unwrap();
    println!("x = {:?}", x);  // [1.6, 1.8]

    // Or via LU decomposition
    let (l, u) = lu_decompose(&a).unwrap();
    let x = solve_lu(&l, &u, &b);
    println!("x = {:?}", x);  // [1.6, 1.8]

    // Norms
    println!("‖A‖₁   = {:.4}", norm_1(&a));
    println!("‖A‖∞   = {:.4}", norm_inf(&a));
    println!("‖A‖_F  = {:.4}", norm_frobenius(&a));
}
```

## Per-Module Documentation

### Decomposition (`decomposition`)

```rust
use mathverse_linear_algebra::*;

// LU decomposition: A = LU (no pivoting)
let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
let (l, u) = lu_decompose(&a).unwrap();

// QR decomposition: A = QR (Gram-Schmidt)
let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let (q, r) = qr_decompose(&a).unwrap();

// Cholesky: A = LL^T (symmetric positive definite)
let a = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
let l = cholesky(&a).unwrap();

// 2x2 eigenvalue (analytical)
let eigs = eigenvalue_2x2([[2.0, 1.0], [1.0, 2.0]]);
// [3.0, 1.0]

// Power iteration → dominant eigenvalue + eigenvector
let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
let (v, lambda) = power_iteration(&a, 100, 1e-10).unwrap();
// lambda ≈ 3.618 (dominant eigenvalue)
```

**Formula:**

```
LU:  L[i][j] = (A[i][j] - Σ_k L[i][k]U[k][j]) / U[j][j]  (i > j)
     U[i][j] = A[i][j] - Σ_k L[i][k]U[k][j]              (i ≤ j)

QR:  v = a_j - Σ_{i<j} (q_i · a_j) q_i
     r[j][j] = ‖v‖
     q_j = v / r[j][j]

Power iteration:  v_{k+1} = Av_k / ‖Av_k‖,  λ = v^T A v
```

### Solve (`solve`)

```rust
use mathverse_linear_algebra::*;

// LU solve: Ly = b then Ux = y
let (l, u) = lu_decompose(&a).unwrap();
let x = solve_lu(&l, &u, &[5.0, 7.0]);
// x ≈ [1.6, 1.8]

// QR solve: Q^T b then Rx = Q^T b
let (q, r) = qr_decompose(&a).unwrap();
let x = solve_qr(&q, &r, &[5.0, 7.0]);

// 2x2 Cramer's rule
let x = solve_2x2([[2.0, 1.0], [1.0, 3.0]], [5.0, 7.0]);
// Some([1.6, 1.8])

// 3x3 Cramer's rule
let x = solve_3x3(
    [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]],
    [5.0, 7.0, 6.0],
);

// Gaussian elimination with partial pivoting
let x = solve_gauss(&a, &[5.0, 7.0]).unwrap();

// Least squares: min ‖Ax - b‖₂  via normal equations (A^T A)x = A^T b
let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
let x = ls_solve(&a, &[1.0, 2.0, 3.5]).unwrap();
// x ≈ [1.083, 2.083]
```

### Norm (`norm`)

```rust
use mathverse_linear_algebra::*;

let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

norm_1(&a);            // 6.0  (max column sum)
norm_inf(&a);          // 7.0  (max row sum)
norm_frobenius(&a);    // √(1+4+9+16) ≈ 5.477
norm_2(&a);            // spectral norm (σ_max)
singular_values(&a);   // [σ₁, σ₂, ...]
condition_number(&a);  // σ_max / σ_min
matrix_norm(&a, 1.0);  // L1 norm
matrix_norm(&a, 2.0);  // spectral norm
```

**Formulas:**

```
‖A‖₁   = max_j Σ_i |a_ij|          (maximum column sum)
‖A‖∞   = max_i Σ_j |a_ij|          (maximum row sum)
‖A‖_F  = √(Σ_ij |a_ij|²)          (Frobenius)
‖A‖₂   = σ_max(A)                   (spectral)
κ(A)    = σ_max / σ_min             (condition number)
```

## Future Scope

- Full matrix inverse computation
- QR with column pivoting for rank-revealing
- Eigenvalue decomposition for general (non-symmetric) matrices
- Iterative solvers (CG, GMRES)
- Sparse matrix support
- Parallel decompositions for large systems

## License

MIT OR Apache-2.0
