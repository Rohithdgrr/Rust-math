# mathverse-vector

> Lightweight dense vector operations — arithmetic, norms, geometry, linear algebra, statistics, and distance metrics. All `f64`, zero required dependencies beyond `mathverse-core`.

[![MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)

## Features

- **Vector arithmetic** — add, subtract, scale, dot product, cross product, Hadamard, outer product, lerp
- **Norms** — L0, L1, L2 (Euclidean), Lp, L∞, L(-∞)
- **Geometry** — angle between vectors, projection, rejection, triple scalar product, Gram-Schmidt
- **Linear algebra** — matrix-vector multiply, 2×2 and 3×3 determinants, rank, orthogonality test
- **Statistics** — mean, variance, std deviation, covariance, Pearson correlation
- **Distance metrics** — Euclidean, Manhattan, Chebyshev, cosine, Mahalanobis, Minkowski
- **Utilities** — zeros, ones, linspace, random, argmax, argmin, clip, reverse
- Optional `simd` (SSE2/NEON via `wide`) and `parallel` (Rayon) acceleration for the O(n) reductions — `dot`, `sum`, `mean`, magnitudes, and distances

## Module Overview

| Module | Purpose | Key Functions |
|---|---|---|
| `operations` | Core vector arithmetic | `add`, `sub`, `scale`, `dot`, `cross`, `magnitude`, `normalize`, `hadamard`, `outer`, `negate`, `add_scalar`, `lerp` |
| `norms` | Vector norms | `l1`, `l2`, `lp`, `linf`, `l0`, `l_neg_inf` |
| `geometry` | Vector geometry | `angle`, `distance`, `project`, `reject`, `triple_product`, `gram_schmidt` |
| `linear_algebra` | Matrix operations | `mat_vec_mul`, `det2x2`, `det3x3`, `rank`, `is_orthogonal` |
| `statistics` | Descriptive statistics | `mean`, `variance`, `std_dev`, `covariance`, `correlation` |
| `distance` | Distance metrics | `euclidean`, `manhattan`, `chebyshev`, `cosine`, `mahalanobis`, `minkowski` |
| `utils` | Vector creation & queries | `zeros`, `ones`, `linspace`, `random`, `argmax`, `argmin`, `max`, `min`, `sum`, `prod`, `clip`, `reverse` |

## Installation

```toml
[dependencies]
mathverse-vector = { path = "../mathverse-vector" }
```

With optional features:

```toml
[dependencies]
mathverse-vector = { path = "../mathverse-vector", features = ["simd", "parallel"] }
```

- `simd` — safe 128-bit SIMD lanes for the O(n) reductions (SSE2 on x86-64,
  NEON on AArch64; scalar fallback elsewhere). No `unsafe` in the crate.
- `parallel` — the same reductions use Rayon parallel iterators once inputs
  exceed ~4096 elements; smaller inputs keep the scalar path.

## Quick Start

```rust
use mathverse_vector::*;

fn main() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    // Dot product
    println!("dot = {}", dot(&a, &b));
    // dot = 32

    // Euclidean norm
    println!("‖a‖ = {}", l2(&a));
    // ‖a‖ = 3.7417

    // Angle between vectors
    println!("angle = {:.4} rad", angle(&a, &b));
    // angle = 0.2257 rad

    // Euclidean distance
    println!("dist = {}", euclidean(&a, &b));
    // dist = 5.1962

    // Mean & std dev
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    println!("mean={:.1}, std={:.1}", mean(&data), std_dev(&data));
    // mean=5.0, std=2.0

    // Linear interpolation
    let mid = lerp(&a, &b, 0.5);
    println!("lerp(a,b,0.5) = {mid:?}");
    // lerp(a,b,0.5) = [2.5, 3.5, 4.5]
}
```

## Module Documentation

### Vector Operations (`operations`)

```
  a = [a₁, a₂, ..., aₙ]
  b = [b₁, b₂, ..., bₙ]

  add:  a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]
  sub:  a - b = [a₁-b₁, a₂-b₂, ..., aₙ-bₙ]
  scale: s·a  = [s·a₁, s·a₂, ..., s·aₙ]

  dot:  a · b = Σ aᵢbᵢ          (scalar)
  cross: a × b (3D only)        (vector)
         = [a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁]
```

| Function | Formula | Input → Output |
|---|---|---|
| `add(a, b)` | `a + b` | `&[f64], &[f64]` → `Vec<f64>` |
| `sub(a, b)` | `a - b` | `&[f64], &[f64]` → `Vec<f64>` |
| `scale(v, s)` | `s · v` | `&[f64], f64` → `Vec<f64>` |
| `dot(a, b)` | `Σ aᵢbᵢ` | `&[f64], &[f64]` → `f64` |
| `cross(a, b)` | `a × b` | `&[f64;3], &[f64;3]` → `Vec<f64>` |
| `magnitude(v)` | `√(Σ vᵢ²)` | `&[f64]` → `f64` |
| `normalize(v)` | `v / ‖v‖` | `&[f64]` → `Vec<f64>` |
| `hadamard(a, b)` | `a ⊙ b` (element-wise ×) | `&[f64], &[f64]` → `Vec<f64>` |
| `outer(a, b)` | `a ⊗ b` (rank-1 matrix) | `&[f64], &[f64]` → `Vec<Vec<f64>>` |
| `lerp(a, b, t)` | `a + t(b - a)` | `&[f64], &[f64], f64` → `Vec<f64>` |

```rust
use mathverse_vector::operations::*;

assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);

let c = cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
assert!((c[2] - 1.0).abs() < 1e-10); // z-axis

let n = normalize(&[3.0, 4.0]);
assert!((n[0] - 0.6).abs() < 1e-10 && (n[1] - 0.8).abs() < 1e-10);
```

---

### Norms (`norms`)

```
  ‖v‖₀ = count of nonzero elements
  ‖v‖₁ = Σ |vᵢ|
  ‖v‖₂ = √(Σ vᵢ²)         (Euclidean)
  ‖v‖ₚ = (Σ |vᵢ|ᵖ)^(1/p)  (general)
  ‖v‖∞ = max |vᵢ|
  ‖v‖₋∞ = min |vᵢ|
```

```rust
use mathverse_vector::norms::*;

let v = [3.0, -4.0, 0.0, 5.0];
assert_eq!(l0(&v), 3);                    // 3 nonzero elements
assert!((l1(&v) - 12.0).abs() < 1e-10);  // 3+4+0+5
assert!((l2(&v) - 7.0711).abs() < 0.001);
assert!((linf(&v) - 5.0).abs() < 1e-10);
```

**Common norm relationships:**

```
  ‖v‖∞ ≤ ‖v‖₂ ≤ ‖v‖₁ ≤ √n · ‖v‖₂
```

---

### Vector Geometry (`geometry`)

```
  angle(a,b) = arccos( (a·b) / (‖a‖·‖b‖) )

  project(a,b) = (a·b / b·b) · b      (projection of a onto b)
  reject(a,b)  = a - project(a,b)      (perpendicular component)

  triple_product(a,b,c) = a · (b × c)  (signed volume of parallelepiped)
```

| Function | Formula |
|---|---|
| `angle(a, b)` | `arccos(a·b / (‖a‖·‖b‖))` |
| `project(a, b)` | `(a·b / b·b) · b` |
| `reject(a, b)` | `a - project(a, b)` |
| `triple_product(a,b,c)` | `a · (b × c)` |
| `gram_schmidt(vecs)` | In-place orthonormalization |

```rust
use mathverse_vector::geometry::*;
use std::f64::consts::FRAC_PI_2;

// Angle between x and y axes
let a = angle(&[1.0, 0.0], &[0.0, 1.0]);
assert!((a - FRAC_PI_2).abs() < 1e-10);

// Projection: project [3,4] onto [1,0] → [3,0]
let p = project(&[3.0, 4.0], &[1.0, 0.0]);
assert!((p[0] - 3.0).abs() < 1e-10 && (p[1]).abs() < 1e-10);

// Gram-Schmidt: orthonormalize two vectors
let mut vecs = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
gram_schmidt(&mut vecs);
// vecs is now orthonormal
```

---

### Linear Algebra (`linear_algebra`)

```
  Matrix × Vector:          Determinant 2×2:
  ┌           ┐             │ a  b │
  │ a  b │   │ x │          │ c  d │ = ad - bc
  │ c  d │ × │ y │
  └           ┘             Determinant 3×3:
                            | a b c |
  result = [ax+by, cx+dy]   | d e f | = a(ei-fh) - b(di-fg) + c(dh-eg)
                            | g h i |
```

| Function | Description |
|---|---|
| `mat_vec_mul(mat, v)` | Matrix-vector multiplication |
| `det2x2(m)` | 2×2 determinant |
| `det3x3(m)` | 3×3 determinant |
| `rank(vectors)` | Matrix rank via Gaussian elimination |
| `is_orthogonal(vecs, tol)` | Pairwise orthogonality test |

```rust
use mathverse_vector::linear_algebra::*;

let mat = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let v = vec![5.0, 6.0];
assert_eq!(mat_vec_mul(&mat, &v), vec![17.0, 39.0]);

let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]];
assert!((det3x3(&m) - (-27.0)).abs() < 1e-10);

// Rank of 2 independent vectors in R³
let vecs = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
assert_eq!(rank(&vecs), 2);
```

---

### Statistics (`statistics`)

```
  mean(x̄)    = (1/n) Σ xᵢ
  variance(σ²) = (1/n) Σ (xᵢ - x̄)²
  std_dev(σ)  = √variance
  covariance  = (1/n) Σ (aᵢ - ā)(bᵢ - b̄)
  correlation = covariance(a,b) / (σₐ · σᵦ)
```

| Function | Formula |
|---|---|
| `mean(v)` | `Σvᵢ / n` |
| `variance(v)` | `Σ(vᵢ - μ)² / n` |
| `std_dev(v)` | `√variance` |
| `covariance(a, b)` | `Σ(aᵢ - ā)(bᵢ - b̄) / n` |
| `correlation(a, b)` | `cov(a,b) / (σₐσᵦ)` |

```rust
use mathverse_vector::statistics::*;

let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
assert!((mean(&data) - 5.0).abs() < 1e-10);
assert!((std_dev(&data) - 2.0).abs() < 1e-10);

// Perfect correlation
let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![2.0, 4.0, 6.0, 8.0];
assert!((correlation(&a, &b) - 1.0).abs() < 1e-10);
```

---

### Distance Metrics (`distance`)

```
  Euclidean:   d(a,b) = √(Σ (aᵢ - bᵢ)²)
  Manhattan:   d(a,b) = Σ |aᵢ - bᵢ|
  Chebyshev:   d(a,b) = max |aᵢ - bᵢ|
  Cosine:      d(a,b) = 1 - (a·b)/(‖a‖·‖b‖)
  Mahalanobis: d(a,b) = √((a-b)ᵀ S⁻¹ (a-b))
  Minkowski:   d(a,b) = (Σ |aᵢ - bᵢ|ᵖ)^(1/p)
```

| Function | Metric | Formula |
|---|---|---|
| `euclidean(a, b)` | L2 distance | `√(Σ(aᵢ-bᵢ)²)` |
| `manhattan(a, b)` | L1 / taxicab | `Σ\|aᵢ-bᵢ\|` |
| `chebyshev(a, b)` | L∞ / chessboard | `max\|aᵢ-bᵢ\|` |
| `cosine(a, b)` | Cosine distance | `1 - cos(θ)` |
| `mahalanobis(a, b, S⁻¹)` | Mahalanobis | `√(dᵀ S⁻¹ d)` |
| `minkowski(a, b, p)` | Lp distance | `(Σ\|aᵢ-bᵢ\|ᵖ)^(1/p)` |

```
  Distance visual (2D):

  Euclidean     Manhattan      Chebyshev
  ╲             │              ─────
   ╲            │             │     │
    ╲           │             │     │
     ╲          │             ─────
  ────•────   ──•──         ──•──
```

```rust
use mathverse_vector::distance::*;

let a = [0.0, 0.0];
let b = [3.0, 4.0];

assert!((euclidean(&a, &b) - 5.0).abs() < 1e-10);
assert!((manhattan(&a, &b) - 7.0).abs() < 1e-10);
assert!((chebyshev(&a, &b) - 4.0).abs() < 1e-10);
assert!((cosine(&a, &b) - 1.0).abs() < 1e-10); // orthogonal → distance = 1

// Minkowski with p=1 is Manhattan
assert!((minkowski(&a, &b, 1.0) - 7.0).abs() < 1e-10);
// Minkowski with p=2 is Euclidean
assert!((minkowski(&a, &b, 2.0) - 5.0).abs() < 1e-10);
```

**Use cases:** k-NN classifiers, clustering, similarity search, anomaly detection.

---

### Utilities (`utils`)

| Function | Description |
|---|---|
| `zeros(n)` | `vec![0.0; n]` |
| `ones(n)` | `vec![1.0; n]` |
| `linspace(start, end, n)` | `n` evenly spaced values from `start` to `end` |
| `random(n, min, max)` | Pseudo-random vector in `[min, max)` |
| `argmax(v)` | Index of maximum value |
| `argmin(v)` | Index of minimum value |
| `max(v)` / `min(v)` | Maximum / minimum value |
| `sum(v)` / `prod(v)` | Sum / product of elements |
| `clip(v, min, max)` | Clamp all elements in-place |
| `reverse(v)` | Reversed copy |

```rust
use mathverse_vector::utils::*;

assert_eq!(zeros(3), vec![0.0, 0.0, 0.0]);
assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);

let mut v = vec![0.0, 5.0, 10.0, 15.0];
clip(&mut v, 2.0, 8.0);
assert_eq!(v, vec![2.0, 5.0, 8.0, 8.0]);
```

## Future Scope

- [ ] Sparse vector support (`SparseVec` with CSR-like storage)
- [ ] Generic `Real` trait support (like trigonometry crate)
- [ ] Matrix operations (inverse, eigenvalues, SVD)
- [ ] Complex number vector support
- [ ] `no_std` support with `alloc` feature

## License

MIT OR Apache-2.0
