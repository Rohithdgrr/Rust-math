# Architecture

## Purpose

`mathverse-complex` provides complex-number mathematics for the MathVerse
ecosystem: a generic `Complex<T>` core type plus analysis, special functions,
matrix algebra, FFT, and polynomial root-finding — all with Python
(`cmath`/`numpy`) naming parity.

## Core Components

```text
┌──────────────────────────────────────────────────────────┐
│                         lib.rs                           │
│   Complex<T: RealFull = f64>  •  C32/C64 aliases          │
│   mandelbrot helpers   •  module re-exports              │
├────────────┬────────────┬────────────┬───────────────────┤
│ analysis   │ special_   │ matrix     │ fft • polynomial  │
│ (Complex   │ functions  │ (Complex   │ (radix-2  (Durand- │
│  analysis) │ gamma,zeta │  Matrix)   │  FFT)    Kerner)  │
│            │ erf,bessel │ LU,QR,eig  │                   │
└────────────┴────────────┴────────────┴───────────────────┘
        │                │                │
        └────────────────┴────────────────┘
                    mathverse-core
                  (RealFull trait)
```

## Module Boundaries

| Module | Responsibility | Depends on |
|--------|----------------|------------|
| `lib.rs` | `Complex<T>` arithmetic & transcendentals, Mandelbrot | `mathverse-core::traits::RealFull` |
| `analysis` | Contour integrals, derivatives, residues, conformal maps | `Complex` (f64) |
| `special_functions` | Gamma, zeta, polylog, erf, Bessel, Airy, Fresnel | `Complex` (f64) |
| `matrix` | Complex matrix algebra, decompositions, eigenvalues | `Complex`, `mathverse_core::error` |
| `fft` | Radix-2 FFT/IFFT | `Complex` (f64) |
| `polynomial` | Horner evaluation, Durand–Kerner roots | `Complex<T>` |

## Data Flow

1. **Scalar path**: `Complex<T>` methods operate component-wise on `T: RealFull`
   (conversion, powers, transcendental, trigonometric, hyperbolic, float-class).
   Division goes through `recip` using Smith's overflow-safe algorithm.
2. **Vector path**: `fft::fft` splits input into even/odd sub-problems
   (Cooley–Tukey) and combines with twiddle factors; `ifft` reuses `fft` via
   conjugation.
3. **Matrix path**: `ComplexMatrix` (row-major `Vec<Complex>`) is the workhorse;
   `lu_decomposition` → `solve`/`inverse`/`determinant`; `qr_decomposition` →
   `eigenvalues` (QR iteration with Wilkinson shifts); matrix `exp`/`ln` use
   Taylor series with convergence checks.
4. **Root path**: `polynomial_roots` starts from `n` equally-spaced seed roots
   and iterates the Weierstrass correction until every root update is below
   tolerance.

## Error Flow

- **Scalar functions** never panic: division by zero and `ln(0)` yield
  `NaN`/`inf` components, mirroring std float semantics.
- **Matrix ops** return `mathverse_core::error::MathResult`
  (`MathError::DimensionMismatch`, `Singular`, `NotConverged`, …) instead of
  panicking. `get`/`set` follow slice-indexing (panic on OOB) for hot paths,
  while `try_get`/`try_set` return `Option`/`Result`.
- **FFT** asserts power-of-two input length (documented panic, `# Panics`).
- **Polynomial roots** return the best estimate after `max_iterations` even if
  the tolerance was not reached.

## Dependency Graph

```text
mathverse-prelude ──► mathverse-complex ──► mathverse-core
mathverse-plot     ──► mathverse-complex
mathverse-signal   ──► mathverse-complex
mathverse-transforms ─► mathverse-complex
```

`mathverse-complex` has exactly **one** runtime dependency (`mathverse-core`);
dev-dependencies add `criterion` for benchmarks only.

## Design Decisions

- **Genericity via default type param** — `Complex<T: RealFull = f64>` keeps
  source compatibility (`Complex` still means f64) while enabling `Complex<f32>`.
- **numpy/cmath parity** — `phase`, `to_polar`, `rect`, `is_close`, `real`,
  `imag` names map 1:1 onto Python equivalents.
- **`0⁰ = 1`** — documented combinatorial convention (matches numpy complex power).
- **Conventions** — `#![forbid(unsafe_code)]`, `missing_docs = warn`,
  clippy pedantic (workspace lints).
