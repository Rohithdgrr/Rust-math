# MathVerse Matrix

[![Crates.io](https://img.shields.io/crates/v/mathverse-matrix.svg)](https://crates.io/crates/mathverse-matrix)
[![docs.rs](https://docs.rs/mathverse-matrix/badge.svg)](https://docs.rs/mathverse-matrix)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Dense and sparse matrix operations over `f64` with row-major storage — decompositions, solvers, norms, eigenvalues, and more.

---

## Features

- Dense `Matrix` type with row-major `Vec<f64>` storage
- Sparse matrices: COO, CSR, CSC, Diagonal formats
- Decompositions: LU, QR (Householder), Cholesky, SVD (one-sided Jacobi), Schur, LDL, Polar
- Eigenvalue solvers: symmetric (Jacobi), general (QR iteration), generalized (Ax = λBx)
- Matrix functions: exp, log, sqrt, sin, cos, sinh, cosh, element-wise ops
- Norms: Frobenius, L1, L∞, spectral, nuclear, Schatten-p, induced-p
- Condition number analysis and sensitivity
- Rank computation (SVD, QR, randomized, energy-based)
- Moore-Penrose pseudoinverse (SVD, normal equations, Tikhonov regularization)
- Kronecker and Hadamard products, tensor operations
- Low-rank approximation (truncated SVD, randomized SVD)
- Positive definiteness tests, Sylvester/Lyapunov/Riccati equation solvers
- Block matrix operations, banded/tridiagonal/Toeplitz/circulant matrices
- Iterative solvers: Conjugate Gradient, GMRES, Jacobi, Gauss-Seidel, SOR
- Matrix calculus: gradient, Jacobian, Hessian, automatic differentiation
- Least squares: QR, normal equations, SVD, weighted, constrained, non-negative, total least squares
- Deterministic RNG for reproducible tests

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `decompositions` | LU, Cholesky, QR (Householder), SVD (Jacobi), symmetric eigen (Jacobi) |
| `sparse` | COO sparse matrix: triplet storage, sparse matvec |
| `sparse_formats` | CSR, CSC, COO, Diagonal formats and conversions |
| `norms` | Frobenius, L1, L∞, spectral, nuclear, Schatten-p, induced-p norms |
| `condition` | Condition number (spectral, Frobenius, L1, L∞), sensitivity analysis |
| `rank` | Rank via SVD, QR, Gaussian elimination, randomized estimation |
| `pseudoinverse` | Moore-Penrose pseudoinverse, Tikhonov, truncated SVD |
| `functions` | Matrix exp, log, sqrt, sin, cos, element-wise functions |
| `kronecker` | Kronecker product, sum, Khatri-Rao, tensor operations |
| `hadamard` | Hadamard (element-wise) product, power, division, comparisons |
| `power` | Matrix power (integer, rational), commutator, Cayley-Hamilton |
| `schur` | Schur decomposition (QR iteration), Sylvester equation solver |
| `eigen_general` | General eigenvalue decomposition (QR with shifts) |
| `eigen_generalized` | Generalized eigenvalue problem Ax = λBx |
| `lowrank` | Low-rank approximation (truncated SVD, randomized SVD) |
| `positivedefinite` | Positive (semi-)definiteness tests, nearest PD matrix |
| `ldl` | LDL decomposition, Bunch-Kaufman for indefinite matrices |
| `banded` | Banded, tridiagonal, diagonal, Toeplitz, circulant matrices |
| `block` | Block matrix operations, block LU/Cholesky, Schur complement |
| `leastsquares` | QR, normal equations, SVD, weighted, constrained, NNLS, TLS |
| `polar` | Polar decomposition, orthogonal Procrustes, nearest orthogonal |
| `equations` | Sylvester, Lyapunov, Stein, Riccati equation solvers |
| `iterative` | CG, GMRES, Jacobi, Gauss-Seidel, SOR with preconditioners |
| `calculus` | Gradient, Jacobian, Hessian, auto-diff, gradient descent, BFGS |
| `rng` | Deterministic xorshift64* RNG |

---

## ASCII Art: Matrix Decompositions

```
LU Decomposition (Partial Pivoting)
================
P A = L U

    P               A               L               U
┌       ┐   ┌           ┐   ┌           ┐   ┌           ┐
│ 0 1 0 │   │ 1  2  3   │   │ 1  0  0   │   │ 6  5  4   │
│ 1 0 0 │ × │ 6  5  4   │ = │ 1/6 1 0  │ × │ 0 5/6 2/3 │
│ 0 0 1 │   │ 2  1  0   │   │ 1/3 0 1  │   │ 0 0   -2  │
└       ┘   └           ┘   └           ┘   └           ┘

QR Decomposition (Householder)
================
A = Q R

    A (m×n)           Q (m×m)           R (m×n)
┌           ┐   ┌               ┐   ┌           ┐
│           │   │               │   │ r11 r12 r13│
│  m rows   │ = │  orthogonal   │ × │  0  r22 r23│
│           │   │  Q^T Q = I    │   │  0   0  r33│
└           ┘   └               ┘   └           ┘

SVD (Singular Value Decomposition)
================
A = U Σ V^T

    σ1 ≥ σ2 ≥ σ3 ≥ ... ≥ 0 (singular values, descending)
    rank(A) = #{ σ_i > tolerance }

Cholesky Decomposition
======================
A = L L^T    (A must be symmetric positive definite)
    det(A) = product(diag(L))^2
```

---

## Installation

```toml
[dependencies]
mathverse-matrix = "0.1"
```

---

## Quick Start

```rust
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a matrix
    let a = Matrix::from_rows(&[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 10.0],
    ])?;

    // Determinant
    println!("det(A) = {}", a.det()?);

    // LU decomposition
    let lu = a.lu()?;
    println!("LU sign = {}", lu.sign);

    // Solve Ax = b
    let b = Vector::new(vec![1.0, 2.0, 3.0]);
    let x = a.solve(&b)?;
    println!("x = {:?}", x.data);

    // SVD
    let svd = a.svd()?;
    println!("singular values = {:?}", svd.s);

    // Eigenvalues of symmetric matrix
    let m = Matrix::from_rows(&[
        &[2.0, 1.0],
        &[1.0, 2.0],
    ])?;
    let (vals, vecs) = m.eigen_symmetric()?;
    println!("eigenvalues = {:?}", vals);

    Ok(())
}
```

---

## Per-Module Documentation

### Decompositions (`decompositions`)

```
LU:  P A = L U          O(n³)   general square matrices
QR:  A = Q R            O(mn²)  any m×n matrix (Householder)
Cholesky: A = L L^T      O(n³/3) symmetric positive definite only
SVD:  A = U Σ V^T        O(mn²)  any m×n matrix (one-sided Jacobi)
Eigen: A V = V Λ         O(n³)  symmetric matrices only (Jacobi)
```

---

### Sparse Matrices (`sparse`, `sparse_formats`)

| Format | Best For | Storage (nnz) |
|---|---|---|
| COO | Assembly | 3 × nnz |
| CSR | Row access | 2 × nnz + rows |
| CSC | Col access | 2 × nnz + cols |
| Diagonal | Diag matrices | n |

---

### Condition Number (`condition`)

```
κ(A)                  Classification
─────────────────────────────────────
< 10                   well-conditioned
10 ≤ κ < 100          moderate
100 ≤ κ < 1000        ill-conditioned
1000 ≤ κ < 10^10      severely ill
≥ 10^10               singular

Error bound: ||Δx||/||x|| ≤ κ(A) × (||ΔA||/||A|| + ||Δb||/||b||)
```

---

### Norms (`norms`)

```
||A||_F = √(Σ |a_ij|²)
||A||_1  = max_j Σ_i |a_ij|        (maximum column sum)
||A||_∞  = max_i Σ_j |a_ij|        (maximum row sum)
||A||_2  = σ_max(A)                (spectral norm)
||A||_*  = Σ σ_i                   (nuclear/trace norm)
```

---

### Equation Solvers (`equations`)

Sylvester (`AX + XB = C`), Lyapunov (`AX + XA^T = Q`), Stein (`X - AXB = C`), Riccati (`A^T X + XA - XBR⁻¹B^T X + Q = 0`).

---

### Iterative Solvers (`iterative`)

Conjugate Gradient (SPD), GMRES (general), Jacobi, Gauss-Seidel, SOR — with preconditioner support.

---

### Matrix Calculus (`calculus`)

```rust
use mathverse_matrix::calculus::{MatrixCalculus, AutoDiff, GradientOptimization};

// Gradient of scalar function
let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let grad = MatrixCalculus::gradient(&f, &[1.0, 2.0], 1e-6);
// [2.0, 4.0]

// Jacobian of vector function
let f = |x: &[f64]| vec![x[0] + x[1], x[0] * x[1]];
let jac = MatrixCalculus::jacobian(&f, &[2.0, 3.0], 1e-6);
// [[1, 1], [3, 2]]
```

---

## Roadmap

- GPU-accelerated operations via `wgpu` or `cuda` backend
- Parallel decompositions with rayon for large matrices
- Sparse direct solvers (SuperLU-style)
- Matrix function via contour integral (Cauchy integral formula)
- Multi-threaded blocked algorithms for cache efficiency
- FFI bindings to LAPACK/BLAS for high-performance backends

---

## License

MIT — see [LICENSE](LICENSE).
