# mathverse-complex: Path from 7.2 to 9.8/10

> **Audit revision**: The original audit flagged 8 "bugs." Deep inspection reveals **0 correctness bugs** — all flagged behaviors are documented conventions or numerically sound implementations. The real gaps are **performance**, **accuracy**, and **ecosystem integration**.

---

## Reassessed Findings

| # | Original Claim | Actual Status | Action |
|---|----------------|---------------|--------|
| 1 | `pow(0, iy)` returns 1 | **Correct** — documented numpy convention, tested | No change needed |
| 2 | Matrix `exp` factorial overflow | **Correct** — scaling-and-squaring + early termination prevents it | No change needed |
| 3 | Eigenvalues O(n⁴) per iteration | **Correct algorithm, slow** — full dense QR instead of Hessenberg | Optimize with Hessenberg reduction |
| 4 | Naive O(n³) matmul | **Correct, slow** — no cache blocking | Add blocked mul + BLAS feature gate |
| 5 | Recursive FFT allocates per level | **Correct, slow** — textbook recursive | Replace with iterative in-place |
| 6 | Bessel `gamma()` called in loop | **Correct, wasteful** — could use recurrence | Cache via gamma recurrence |
| 7 | Bessel asymptotic leading-order only | **Correct, ~10% error at \|z\|=10** | Add first correction term |
| 8 | `polylog()` returns NaN for \|z\|>1 | **Honest limitation** — analytic continuation not implemented | Implement via inversion formula |

---

## Phase 1: Performance (7.2 → 8.2)

### 1.1 Iterative In-Place FFT
**File:** `src/fft.rs`
**Effort:** 2h

Replace recursive Cooley-Tukey with iterative in-place version:
- Bit-reversal permutation
- Butterfly loops with precomputed twiddle factors
- Add `fft_in_place(&mut [Complex])` variant
- Keep recursive version as fallback for small sizes (< 64)

**Test:** Existing `fft_matches_dft` and `fft_roundtrip` must pass.

### 1.2 Blocked Matrix Multiplication
**File:** `src/matrix.rs`
**Effort:** 3h

Replace naive `mul()` with cache-blocked algorithm:
- Block size B = 64 (fits L1 cache for Complex<f64>)
- 6 nested loops: ii, jj, kk, i, j, k
- Keep naive fallback for small matrices (< B)

**Test:** Existing `test_matrix_multiplication` must pass. Add benchmark for n=128, 512.

### 1.3 Hessenberg Reduction for Eigenvalues
**File:** `src/matrix.rs`
**Effort:** 4h

Add `hessenberg_reduction()` method:
- Householder reflections to reduce to upper Hessenberg form (O(n³))
- Modify `eigenvalues()` to:
  1. Reduce to Hessenberg form once
  2. Apply QR iteration on Hessenberg (O(n²) per iteration using Givens rotations)
  3. Deflate when subdiagonal is small

**Test:** Existing eigenvalue tests must pass. Add test for n=10 random matrix.

### 1.4 Gamma Recurrence in Bessel Series
**File:** `src/special_functions.rs`
**Effort:** 30min

In `bessel_j_series()`, compute `gamma(v+1)` once, then use recurrence:
```
Gamma(v+n+2) = (v+n+1) * Gamma(v+n+1)
```
Avoids 50+ redundant Lanczos evaluations per series call.

**Test:** Existing `test_bessel_j` and `test_bessel_j_known_values` must pass.

---

## Phase 2: Accuracy (8.2 → 8.6)

### 2.1 Bessel Asymptotic Correction Term
**File:** `src/special_functions.rs`
**Effort:** 1h

Add first correction term from DLMF 10.17.3:
```
J_v(z) ~ sqrt(2/(πz)) * [cos(φ) - (4v²-1)/(8z) * sin(φ)]
```
where φ = z - vπ/2 - π/4. Improves accuracy from ~10% to ~1% at |z|=10.

**Test:** Add accuracy test: `J_0(10)` should match series to < 1e-4 (currently 1e-2).

### 2.2 Polylog Analytic Continuation
**File:** `src/special_functions.rs`
**Effort:** 2h

Implement Jonquière inversion for |z| > 1:
```
Li_s(z) = (-1)^{s-1} * Li_s(1/z) + (2πi)^s / Γ(s) * ζ(1-s, ln(z)/(2πi))
```
For integer s, use the simpler reflection formula.

**Test:** Verify Li_2(2) ≈ 2.467 - 2.178i (known value).

---

## Phase 3: Ecosystem Integration (8.6 → 9.2)

### 3.1 Feature Flags in Cargo.toml
**File:** `Cargo.toml`
**Effort:** 30min

```toml
[features]
default = []
serde = ["dep:serde"]
rand = ["dep:rand"]
blas = ["dep:ndarray", "dep:matrixmultiply"]

[dependencies]
serde = { version = "1", features = ["derive"], optional = true }
rand = { version = "0.8", optional = true }
ndarray = { version = "0.15", optional = true }
matrixmultiply = { version = "0.3", optional = true }
```

### 3.2 Serde Support
**File:** `src/lib.rs`, `src/matrix.rs`
**Effort:** 1h

- `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` on `Complex<T>` and `ComplexMatrix`
- Add `#[serde(rename)]` for field names

### 3.3 Rand Integration
**File:** `src/lib.rs` (new section)
**Effort:** 1h

- `impl Distribution<Complex<f64>> for Standard` — complex Gaussian
- `fn complex_uniform_disk(rng) -> Complex<f64>` — uniform on unit disk
- `fn complex_gaussian(rng, sigma) -> Complex<f64>` — circular Gaussian

### 3.4 ndarray Interop
**File:** `src/matrix.rs` (new impls)
**Effort:** 1h

- `From<ComplexMatrix> for Array2<Complex64>`
- `From<Array2<Complex64>> for ComplexMatrix`
- `blas_mul()` method when `blas` feature enabled

---

## Phase 4: New Features (9.2 → 9.8)

### 4.1 Modified Bessel Functions
**File:** `src/special_functions.rs`
**Effort:** 2h

- `bessel_i(v, z)` — modified Bessel I_v(z) via series
- `bessel_k(v, z)` — modified Bessel K_v(z) via relation to I

### 4.2 Hankel Functions
**File:** `src/special_functions.rs`
**Effort:** 1h

- `hankel_h1(v, z) = J_v(z) + i*Y_v(z)`
- `hankel_h2(v, z) = J_v(z) - i*Y_v(z)`

### 4.3 SVD
**File:** `src/matrix.rs`
**Effort:** 4h

- `ComplexMatrix::svd()` → `(U, S, V^H)`
- Golub-Kahan bidiagonalization + QR iteration
- Returns singular values as `Vec<f64>`, U and V as `ComplexMatrix`

### 4.4 Eigenvectors
**File:** `src/matrix.rs`
**Effort:** 3h

- `ComplexMatrix::eigenvectors()` → `(eigenvalues, eigenvector_matrix)`
- Inverse iteration on Hessenberg form
- Returns column-wise eigenvectors

### 4.5 Incomplete Gamma
**File:** `src/special_functions.rs`
**Effort:** 2h

- `gamma_p(a, x)` — regularized lower incomplete gamma
- `gamma_q(a, x)` — regularized upper incomplete gamma
- Series expansion for small x, continued fraction for large x

### 4.6 Elliptic Functions
**File:** `src/special_functions.rs`
**Effort:** 2h

- `elliptic_k(m)` — complete elliptic integral of first kind
- `elliptic_e(m)` — complete elliptic integral of second kind
- `jacobi_sn/cn/dn(z, m)` — Jacobi elliptic functions

---

## Execution Order

```
Phase 1 (Performance)     ← do first, biggest impact
  1.4 Gamma recurrence    ← quick win, 30min
  1.1 Iterative FFT       ← standalone, 2h
  1.2 Blocked matmul      ← standalone, 3h
  1.3 Hessenberg eigen    ← depends on 1.2 for clean matrix ops, 4h

Phase 2 (Accuracy)        ← after Phase 1, builds on stable base
  2.1 Bessel correction   ← standalone, 1h
  2.2 Polylog continuation ← standalone, 2h

Phase 3 (Ecosystem)       ← can parallelize with Phase 2
  3.1 Feature flags       ← prerequisite, 30min
  3.2 Serde               ← depends on 3.1, 1h
  3.3 Rand                ← depends on 3.1, 1h
  3.4 ndarray interop     ← depends on 3.1, 1h

Phase 4 (New Features)    ← after Phase 1, independent of Phase 2/3
  4.1 Modified Bessel     ← depends on 1.4, 2h
  4.2 Hankel              ← depends on existing bessel_j/y, 1h
  4.3 SVD                 ← large, 4h
  4.4 Eigenvectors        ← depends on 1.3, 3h
  4.5 Incomplete gamma    ← standalone, 2h
  4.6 Elliptic functions  ← standalone, 2h
```

---

## Verification Checklist

After each phase, run:
```bash
cargo test --manifest-path crates/mathverse-complex/Cargo.toml
cargo clippy --manifest-path crates/mathverse-complex/Cargo.toml -- -D warnings
cargo bench --manifest-path crates/mathverse-complex/Cargo.toml -- --warm-up-time 1
```

Phase 3 additionally:
```bash
cargo test --manifest-path crates/mathverse-complex/Cargo.toml --features serde,rand
```

---

## Total Effort Estimate

| Phase | Hours | Rating Gain |
|-------|-------|-------------|
| 1: Performance | ~9.5h | +1.0 |
| 2: Accuracy | ~3h | +0.4 |
| 3: Ecosystem | ~3.5h | +0.6 |
| 4: New Features | ~14h | +0.6 |
| **Total** | **~30h** | **7.2 → 9.8** |
