# MathVerse Phase-wise Plan

Execution plan: phases in dependency order, each with scope, deliverables, and acceptance criteria. Releases map to phases per ROADMAP.md.

## Phase 1 — Foundation (v0.1)

**Scope:** `mathverse-core` crate, workspace skeleton.

**Deliverables:**
- Cargo workspace with `mathverse-core`
- Numeric traits (built on std/native conventions)
- Error taxonomy with rich messages
- Constants, precision utilities
- Feature flag layout (`std`, `simd`, `parallel`)
- CI (unit tests, clippy, fmt)

**Acceptance:** 95% coverage on core; `no_std` build passes; every public item documented with example.

## Phase 2 — Basic Mathematics (v0.1–v0.2)

**Scope:** arithmetic, algebra, trigonometry, geometry.

**Deliverables:**
- `mathverse-arithmetic`: all basic operations with generic numeric types
- `mathverse-algebra`: polynomials, quadratic/cubic solvers, factorization
- `mathverse-trigonometry`: trig + hyperbolic + inverse + conversions
- `mathverse-geometry`: 2D/3D shapes with transforms and queries

**Acceptance:** accuracy tests against known reference values; edge cases (division by zero, negative roots) return typed errors, not panics.

## Phase 3 — Linear Algebra (v0.3)

**Scope:** matrix, vector, tensor, decompositions.

**Deliverables:**
- `mathverse-linear-algebra`, `mathverse-matrix`, `mathverse-vector`
- Dense matrix ops: determinant, inverse
- Decompositions: LU, QR, SVD, Cholesky
- Eigenvalues/eigenvectors
- Sparse matrix support
- SIMD feature for hot paths

**Acceptance:** numerical stability tests (ill-conditioned matrices); round-trip tests on decompositions (A = QR, etc.).

## Phase 4 — Calculus (v0.4)

**Scope:** differentiation, integration, vector calculus.

**Deliverables:**
- `mathverse-calculus`: analytic derivatives, numerical integration
- Vector calculus: gradient, curl, divergence, Laplacian
- Numerical derivative fallbacks

**Acceptance:** derivative accuracy vs. symbolic references; integration convergence tests.

## Phase 5 — Probability & Statistics (v0.5)

**Scope:** probability, statistics, complex.

**Deliverables:**
- `mathverse-probability`: distributions, Bayes, Monte Carlo, Markov chains
- `mathverse-statistics`: descriptive + inferential
- `mathverse-complex`: complex arithmetic

**Acceptance:** distribution moments match closed forms within tolerance; statistical tests validated against known datasets.

## Phase 6 — Numerical & Optimization (v0.6)

**Scope:** numerical methods, optimization, equations, discrete, number theory, combinatorics, graph.

**Deliverables:**
- `mathverse-numerical`: root finding, RK integrators, interpolation
- `mathverse-optimization`: GD, SGD, Adam, RMSProp, annealing, genetic
- `mathverse-equations`
- `mathverse-number-theory`, `mathverse-combinatorics`, `mathverse-graph`

**Acceptance:** root finders converge on benchmark functions; optimizers reach known minima; convergence criteria configurable. — ✅ complete (all tests + clippy green)

## Phase 7 — Signal Processing (v0.7)

**Scope:** transforms, signal.

**Deliverables:**
- `mathverse-transforms`: FFT, DCT, wavelets
- `mathverse-signal`: FIR/IIR, convolution, correlation

**Acceptance:** FFT round-trip (inverse ≈ identity) within tolerance; filter frequency responses verified. — ✅ complete (all tests + clippy green)

## Phase 8 — Image & Vision (v0.8)

**Scope:** image processing, computer vision.

**Deliverables:**
- `mathverse-image`: kernels, blur, Sobel, Canny, histogram, morphology, transforms
- `mathverse-vision`: camera matrix, homography, epipolar, features, optical flow
- `mathverse-graphics`: 2D/3D graphics math

**Acceptance:** reference-image tests; homography round-trip on synthetic point sets. — ✅ complete (all tests + clippy green)

## Phase 9 — AI/ML (v0.9)

**Scope:** mathverse-ai.

**Deliverables:**
- Activations, losses, metrics
- Tensor ops (broadcasting, matmul, normalization)
- Optimizers (SGD, Adam, AdamW)
- Attention math (QKV, scaled dot product, rotary embeddings, positional encoding)

**Acceptance:** known-value tests (e.g., softmax sums to 1); gradient checks; metric correctness on labeled samples.

## Phase 10 — Domain Applications (v1.0)

**Scope:** physics, finance, symbolic, units, plot, prelude.

**Deliverables:**
- `mathverse-physics`, `mathverse-finance`
- `mathverse-symbolic`: expression trees, symbolic derivatives, LaTeX
- `mathverse-units`: compile-time dimensional analysis
- `mathverse-plot`: SVG/HTML/terminal output backends
- `mathverse-prelude` finalization

**Acceptance:** dimensional-analysis errors caught at compile time; plot output renders in all backends; prelude exposes unified API.

## Cross-cutting (every phase)

- Docs per standard (see API-GUIDELINES.md)
- Benchmarks in `benches/` per crate
- Property tests for generic code
- Fuzz targets for parsers (symbolic, units, plot)
- Cross-platform CI incl. `wasm32`
