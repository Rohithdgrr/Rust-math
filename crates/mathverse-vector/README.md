# MathVerse Vector

[![Crates.io](https://img.shields.io/crates/v/mathverse-vector.svg)](https://crates.io/crates/mathverse-vector)
[![docs.rs](https://docs.rs/mathverse-vector/badge.svg)](https://docs.rs/mathverse-vector)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Lightweight dense vector operations — arithmetic, norms, geometry, linear algebra, statistics, and distance metrics.

---

## Features

- **Vector arithmetic** — add, subtract, scale, dot product, cross product, Hadamard, outer product, lerp
- **Norms** — L0, L1, L2 (Euclidean), Lp, L∞, L(-∞)
- **Geometry** — angle between vectors, projection, rejection, triple scalar product, Gram-Schmidt
- **Linear algebra** — matrix-vector multiply, 2×2 and 3×3 determinants, rank, orthogonality test
- **Statistics** — mean, variance, std deviation, covariance, Pearson correlation
- **Distance metrics** — Euclidean, Manhattan, Chebyshev, cosine, Mahalanobis, Minkowski
- **Utilities** — zeros, ones, linspace, random, argmax, argmin, clip, reverse
- Optional `simd` (SSE2/NEON via `wide`) and `parallel` (Rayon) acceleration for O(n) reductions

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `operations` | Core vector arithmetic: `add`, `sub`, `scale`, `dot`, `cross`, `magnitude`, `normalize`, `hadamard`, `outer`, `negate`, `lerp` |
| `norms` | Vector norms: `l0`, `l1`, `l2`, `lp`, `linf`, `l_neg_inf` |
| `geometry` | Vector geometry: `angle`, `distance`, `project`, `reject`, `triple_product`, `gram_schmidt` |
| `linear_algebra` | Matrix operations: `mat_vec_mul`, `det2x2`, `det3x3`, `rank`, `is_orthogonal` |
| `statistics` | Descriptive statistics: `mean`, `variance`, `std_dev`, `covariance`, `correlation` |
| `distance` | Distance metrics: `euclidean`, `manhattan`, `chebyshev`, `cosine`, `mahalanobis`, `minkowski` |
| `utils` | Vector creation & queries: `zeros`, `ones`, `linspace`, `random`, `argmax`, `argmin`, `max`, `min`, `sum`, `prod`, `clip`, `reverse` |

---

## Installation

```toml
[dependencies]
mathverse-vector = "0.1"
```

With optional features:

```toml
mathverse-vector = { version = "0.1", features = ["simd", "parallel"] }
```

- `simd` — safe 128-bit SIMD lanes for O(n) reductions (SSE2 on x86-64, NEON on AArch64; scalar fallback elsewhere)
- `parallel` — Rayon parallel iterators for inputs exceeding ~4096 elements

---

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

---

## Module Documentation

### Vector Operations (`operations`)

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

---

### Norms (`norms`)

| Norm | Formula |
|---|---|
| `l0(v)` | Count of nonzero elements |
| `l1(v)` | `Σ |vᵢ|` |
| `l2(v)` | `√(Σ vᵢ²)` (Euclidean) |
| `lp(v, p)` | `(Σ |vᵢ|ᵖ)^(1/p)` |
| `linf(v)` | `max |vᵢ|` |
| `l_neg_inf(v)` | `min |vᵢ|` |

**Common norm relationships:**

```
‖v‖∞ ≤ ‖v‖₂ ≤ ‖v‖₁ ≤ √n · ‖v‖₂
```

---

### Vector Geometry (`geometry`)

| Function | Formula |
|---|---|
| `angle(a, b)` | `arccos(a·b / (‖a‖·‖b‖))` |
| `project(a, b)` | `(a·b / b·b) · b` |
| `reject(a, b)` | `a - project(a, b)` |
| `triple_product(a,b,c)` | `a · (b × c)` |
| `gram_schmidt(vecs)` | In-place orthonormalization |

---

### Linear Algebra (`linear_algebra`)

| Function | Description |
|---|---|
| `mat_vec_mul(mat, v)` | Matrix-vector multiplication |
| `det2x2(m)` | 2×2 determinant |
| `det3x3(m)` | 3×3 determinant |
| `rank(vectors)` | Matrix rank via Gaussian elimination |
| `is_orthogonal(vecs, tol)` | Pairwise orthogonality test |

---

### Statistics (`statistics`)

| Function | Formula |
|---|---|
| `mean(v)` | `Σvᵢ / n` |
| `variance(v)` | `Σ(vᵢ - μ)² / n` |
| `std_dev(v)` | `√variance` |
| `covariance(a, b)` | `Σ(aᵢ - ā)(bᵢ - b̄) / n` |
| `correlation(a, b)` | `cov(a,b) / (σₐσᵦ)` |

---

### Distance Metrics (`distance`)

| Function | Metric | Formula |
|---|---|---|
| `euclidean(a, b)` | L2 distance | `√(Σ(aᵢ-bᵢ)²)` |
| `manhattan(a, b)` | L1 / taxicab | `Σ|aᵢ-bᵢ|` |
| `chebyshev(a, b)` | L∞ / chessboard | `max|aᵢ-bᵢ|` |
| `cosine(a, b)` | Cosine distance | `1 - cos(θ)` |
| `mahalanobis(a, b, S⁻¹)` | Mahalanobis | `√(dᵀ S⁻¹ d)` |
| `minkowski(a, b, p)` | Lp distance | `(Σ|aᵢ-bᵢ|ᵖ)^(1/p)` |

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

---

## Roadmap

- [ ] Sparse vector support (`SparseVec` with CSR-like storage)
- [ ] Generic `Real` trait support (like trigonometry crate)
- [ ] Matrix operations (inverse, eigenvalues, SVD)
- [ ] Complex number vector support
- [ ] `no_std` support with `alloc` feature

---

## License

MIT — see [LICENSE](LICENSE).
