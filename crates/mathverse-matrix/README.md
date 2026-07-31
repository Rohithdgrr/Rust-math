# mathverse-matrix

**Dense and sparse matrix operations over `f64` with row-major storage.**

`mathverse-matrix` provides a comprehensive, zero-panic matrix library for Rust.
Dimension mismatches and singular matrices return `MathError` -- nothing panics on
user input.

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

## Module Overview

| Module | Description |
|---|---|
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

    L has unit diagonal (1s on diagonal)
    U is upper triangular
    P is row permutation matrix
    det(A) = sign × product(diag(U))


QR Decomposition (Householder)
=================
A = Q R

    A (m×n)           Q (m×m)           R (m×n)
┌           ┐   ┌               ┐   ┌           ┐
│           │   │               │   │ r11 r12 r13│
│  m rows   │ = │  orthogonal   │ × │  0  r22 r23│
│           │   │  Q^T Q = I    │   │  0   0  r33│
└           ┘   └               ┘   └           ┘
                              ↑
                   Householder reflectors
                   H_k = I - 2 v_k v_k^T


SVD (Singular Value Decomposition)
=================================
A = U Σ V^T

    A (m×n)     U (m×n)    Σ (n×n)    V^T (n×n)
┌         ┐   ┌       ┐   ┌         ┐   ┌         ┐
│         │ = │       │ × │ σ1  0  0│ × │         │
│  m×n    │   │ ortho │   │  0 σ2  0│   │  ortho  │
│         │   │ cols  │   │  0  0 σ3│   │         │
└         ┘   └       ┘   └         ┘   └         ┘

    σ1 ≥ σ2 ≥ σ3 ≥ ... ≥ 0 (singular values, descending)
    rank(A) = #{ σ_i > tolerance }


Cholesky Decomposition
======================
A = L L^T    (A must be symmetric positive definite)

    A               L (lower tri)    L^T (upper tri)
┌           ┐   ┌           ┐   ┌           ┐
│ 4  1  1   │   │ 2  0  0   │   │ 2  0.5 0.5│
│ 1  3  2   │ = │ 0.5 1.5 0 │ × │ 0  1.5 0.6│
│ 1  2  5   │   │ 0.5 1.3 1 │   │ 0  0   1.3│
└           ┘   └           ┘   └           ┘

    det(A) = product(diag(L))^2
    Positive definite iff Cholesky succeeds
```

## Installation

```toml
[dependencies]
mathverse-matrix = { path = "../mathverse-matrix" }
```

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

## Per-Module Documentation

### Decompositions (`decompositions`)

LU, Cholesky, QR, SVD, and symmetric eigen decomposition.

```
LU:  P A = L U          O(n³)   general square matrices
QR:  A = Q R            O(mn²)  any m×n matrix (Householder)
Cholesky: A = L L^T      O(n³/3) symmetric positive definite only
SVD:  A = U Σ V^T        O(mn²)  any m×n matrix (one-sided Jacobi)
Eigen: A V = V Λ         O(n³)  symmetric matrices only (Jacobi)
```

```rust
use mathverse_matrix::{Matrix, decompositions::{Lu, Qr, Svd}};

let a = Matrix::from_rows(&[&[4.0, 3.0], &[6.0, 3.0]])?;

// LU
let lu = a.lu()?;
// lu.l × lu.u reconstructs P×A

// QR
let qr = a.qr()?;
// qr.q is orthogonal, qr.r is upper triangular

// SVD
let svd = a.svd()?;
// svd.u × diag(svd.s) × svd.vt == a
println!("σ = {:?}", svd.s);  // descending singular values

// Cholesky (symmetric positive definite)
let spd = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]])?;
let l = spd.cholesky()?;
// l × l^T == spd
```

**Use cases:** Solving linear systems, matrix inversion, least squares, rank determination, image compression (SVD truncation), covariance analysis (Cholesky).

### Sparse Matrices (`sparse`, `sparse_formats`)

Coordinate list (COO), Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), and Diagonal formats.

```rust
use mathverse_matrix::sparse::SparseMatrix;
use mathverse_vector::Vector;

let mut s = SparseMatrix::new(3, 3);
s.add(0, 0, 1.0)?;
s.add(1, 1, 2.0)?;
s.add(2, 2, 3.0)?;

let v = Vector::new(vec![1.0, 1.0, 1.0]);
let result = s.mul_vec(&v)?;  // [1.0, 2.0, 3.0]

// Convert to dense
let dense = s.to_dense();
```

```
Sparse format comparison:
┌──────────┬──────────────┬─────────────────┐
│ Format   │ Best For     │ Storage (nnz)   │
├──────────┼──────────────┼─────────────────┤
│ COO      │ Assembly     │ 3 × nnz         │
│ CSR      │ Row access   │ 2 × nnz + rows  │
│ CSC      │ Col access   │ 2 × nnz + cols  │
│ Diagonal │ Diag matrices│ n               │
└──────────┴──────────────┴─────────────────┘
```

### Norms (`norms`)

```rust
use mathverse_matrix::norms::MatrixNorms;
use mathverse_matrix::Matrix;

let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]])?;

MatrixNorms::frobenius(&m);   // √(1+4+9+16) = 5.477
MatrixNorms::l1(&m);          // max column sum = 6
MatrixNorms::linf(&m);        // max row sum = 7
MatrixNorms::spectral(&m)?;   // σ_max (largest singular value)
MatrixNorms::nuclear(&m)?;    // Σ σ_i
```

**Formulas:**

```
||A||_F = √(Σ |a_ij|²)
||A||_1  = max_j Σ_i |a_ij|        (maximum column sum)
||A||_∞  = max_i Σ_j |a_ij|        (maximum row sum)
||A||_2  = σ_max(A)                (spectral norm)
||A||_*  = Σ σ_i                   (nuclear/trace norm)
```

### Condition Number (`condition`)

```rust
use mathverse_matrix::condition::{ConditionNumber, ConditioningClassification};

let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 1.000001]])?;
let κ = ConditionNumber::spectral(&a)?;
// κ ≈ 4,000,000 → severely ill-conditioned

println!("{}", ConditioningClassification::classify(κ));
// "severely ill-conditioned"
```

```
Condition number classification:
┌────────────────────────┬─────────────────┐
│ κ(A)                  │ Classification  │
├────────────────────────┼─────────────────┤
│ < 10                   │ well-conditioned│
│ 10 ≤ κ < 100          │ moderate        │
│ 100 ≤ κ < 1000        │ ill-conditioned │
│ 1000 ≤ κ < 10^10      │ severely ill    │
│ ≥ 10^10               │ singular        │
└────────────────────────┴─────────────────┘

Error bound: ||Δx||/||x|| ≤ κ(A) × (||ΔA||/||A|| + ||Δb||/||b||)
```

### Rank (`rank`)

```rust
use mathverse_matrix::rank::MatrixRank;

let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]])?;
let rank = MatrixRank::compute(&m, 1e-10)?;  // 1 (rank-deficient)

let i = Matrix::identity(3);
let rank = MatrixRank::compute(&i, 1e-10)?;  // 3 (full rank)
```

### Pseudoinverse (`pseudoinverse`)

Moore-Penrose pseudoinverse via SVD: `A⁺ = V Σ⁺ Uᵀ`.

```rust
use mathverse_matrix::pseudoinverse::{Pseudoinverse, PseudoinverseApplications};

// Overdetermined: least squares
let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]])?;
let b = mathverse_vector::Vector::new(vec![1.0, 2.0, 2.0]);
let x = PseudoinverseApplications::solve(&a, &b)?;

// Tikhonov regularization (damped least squares)
let x_reg = PseudoinverseApplications::tikhonov(&a, &b, 0.1)?;
```

### Matrix Functions (`functions`)

```rust
use mathverse_matrix::functions::{MatrixExponential, MatrixSquareRoot, MatrixFunctions};

let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]])?;

// exp(A) — diagonal → exp on diagonal
let exp_a = MatrixExponential::compute(&a)?;

// sqrt(A)
let sqrt_a = MatrixSquareRoot::compute(&a, 1e-10)?;

// Element-wise: |A|, sign(A), floor, ceil, round
let abs_a = MatrixFunctions::abs(&a);

// A^n
let a3 = MatrixFunctions::power(&a, 3)?;
```

### Kronecker & Hadamard Products (`kronecker`, `hadamard`)

```rust
use mathverse_matrix::kronecker::KroneckerProduct;
use mathverse_matrix::hadamard::HadamardProduct;

let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]])?;
let b = Matrix::from_rows(&[&[0.0, 5.0], &[6.0, 7.0]])?;

// Kronecker: A ⊗ B  (4×4 result)
let kron = KroneckerProduct::compute(&a, &b);

// Hadamard: A ∘ B  (element-wise)
let had = HadamardProduct::compute(&a, &b)?;
// [[1×0, 2×5], [3×6, 4×7]] = [[0, 10], [18, 28]]
```

### Low-Rank Approximation (`lowrank`)

```rust
use mathverse_matrix::lowrank::LowRankApprox;

// Truncated SVD: A ≈ U_k Σ_k V_k^T
let approx = LowRankApprox::truncated_svd(&large_matrix, 10)?;
println!("rank: {}, error: {}", approx.rank, approx.error);

// Energy-based: keep 95% of energy
let approx = LowRankApprox::energy_based(&large_matrix, 0.95)?;

// Randomized SVD for large matrices
let approx = LowRankApprox::randomized_svd(&large_matrix, 10, 5)?;
```

### Positive Definiteness (`positivedefinite`)

```rust
use mathverse_matrix::positivedefinite::{PositiveDefinite, DefinitenessClassification};

let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]])?;
assert!(PositiveDefinite::is_positive_definite(&a));

println!("{}", DefinitenessClassification::classify(&a, 1e-10));
// "positive definite"
```

### Equation Solvers (`equations`)

Sylvester (`AX + XB = C`), Lyapunov (`AX + XA^T = Q`), Stein (`X - AXB = C`),
Riccati (`A^T X + XA - XBR⁻¹B^T X + Q = 0`).

```rust
use mathverse_matrix::equations::{SylvesterEquation, LyapunovEquation, RiccatiEquation};

let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]])?;
let b = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]])?;
let c = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]])?;

// Solve AX + XB = C
let x = SylvesterEquation::solve(&a, &b, &c)?;
```

### Iterative Solvers (`iterative`)

```rust
use mathverse_matrix::iterative::{ConjugateGradient, Gmres, Jacobi};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]])?;
let b = Vector::new(vec![1.0, 2.0]);

// Conjugate Gradient (SPD matrices)
let result = ConjugateGradient::solve(&a, &b, 100, 1e-10)?;
// result.solution, result.converged, result.iterations

// GMRES (general matrices)
let result = Gmres::solve(&a, &b, 20, 100, 1e-10)?;
```

### Block, Banded, and Special Matrices (`block`, `banded`)

```rust
use mathverse_matrix::banded::{TridiagonalMatrix, ToeplitzMatrix};
use mathverse_matrix::block::{BlockMatrix, BlockOperations};

// Tridiagonal: O(n) solve via Thomas algorithm
let t = TridiagonalMatrix::new(&[2.0, 2.0, 2.0], &[-1.0, -1.0], &[-1.0, -1.0]);
let x = t.solve(&b)?;

// Block matrix
let blocks = vec![a, b, c, d];
let block_mat = BlockMatrix::new(blocks, 2, 2)?;
let full = block_mat.to_full();
```

### Least Squares (`leastsquares`)

```rust
use mathverse_matrix::leastsquares::LeastSquares;

let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]])?;
let b = mathverse_vector::Vector::new(vec![1.0, 2.0, 2.0]);

let result = LeastSquares::qr_solve(&a, &b)?;
// result.solution, result.residuals, result.residual_norm, result.rank

// Non-negative least squares
let result = LeastSquares::non_negative(&a, &b, 100)?;
```

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

// Gradient descent
let (x, iters, _) = GradientOptimization::gradient_descent(
    &f, &grad, &[0.0, 0.0], 0.1, 100, 1e-10
);
```

## Future Scope

- GPU-accelerated operations via `wgpu` or `cuda` backend
- Parallel decompositions with rayon for large matrices
- Sparse direct solvers (SuperLU-style)
- Eigenvalue solvers for non-symmetric general matrices (QR with shifts)
- Matrix function via contour integral (Cauchy integral formula)
- Band matrix solvers (LAPACK-style banded routines)
- Multi-threaded blocked algorithms for cache efficiency
- FFI bindings to LAPACK/BLAS for high-performance backends

## License

MIT OR Apache-2.0
