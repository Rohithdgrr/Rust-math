# MathVerse API Guidelines

Standards every public API item must meet before merge.

## Naming Conventions

- Crates: `mathverse-<domain>` (snake_case crate names).
- Modules: lowercase, `snake_case`.
- Types: `UpperCamelCase`.
- Functions/methods: `snake_case`.
- Domain verbs are consistent: `area()`, `perimeter()`, `volume()`, `distance()`, `intersect()`, `solve()`, `integrate()`, `differentiate()`, `transform()`.
- Angle functions take radians by default; `*_deg` variants provided.

## Error Handling

- One error enum per crate, implementing `std::error::Error` and usable under `no_std`.
- Errors carry context: which inputs were invalid and why.
- Never panic on user input. Panics reserved for programmer errors (`assert`/`unreachable`).
- Fallible ops return `Result`; pure math with no failure mode returns the value directly.

## Generics

- Math operations are generic over numeric traits from `mathverse-core`.
- Prefer `T: Real`/`T: Field` style bounds; never pin to `f64` in public APIs unless the algorithm requires it (and say so in docs).
- Generic functions must work for `f32`, `f64`, and (where applicable) fixed-point/arbitrary-precision types.

## Feature Flags

Every crate uses consistent flags:

| Flag | Effect |
|---|---|
| `std` (default) | Enables std; disabling gives `no_std` |
| `simd` | SIMD-accelerated hot paths |
| `parallel` | Rayon-based parallelism |
| `serde` | Serialization derives |

Unstable/experimental APIs live behind `experimental` flags.

## Builder Patterns

Complex configuration (e.g., plot figures, optimizers, filters) uses builder patterns:

```rust
let opt = AdamBuilder::new(1e-3)
    .beta1(0.9)
    .beta2(0.999)
    .build();
```

Builders validate on `build()` and return `Result`.

## Documentation Standard

Every public item includes, in order:

1. **Mathematical definition** — what it computes
2. **Formula** — rendered in plain text or LaTeX-compatible notation
3. **Derivation** — when appropriate, and only when it clarifies behavior
4. **Complexity** — Big-O notation
5. **Numerical stability notes** — cancellation, overflow, ill-conditioning
6. **References** — sources (papers, textbooks, standards)
7. **Examples** — runnable `# Examples` doc-tests, at least one per function

Every public module includes an overview with usage examples and related modules.

## Testing Requirements

- Unit tests: every function, happy path + edge cases
- Property tests: generic numeric code (identity laws, commutativity where applicable)
- Numerical accuracy: compare against closed-form references with documented tolerances
- Doc-tests: all examples must compile and run (`cargo test --doc` green)
- Coverage target: 95%+ overall, enforced per crate

## Benchmarks

- Hot paths get `criterion` benches in `benches/`
- Benchmarks cover: scalar, SIMD, and parallel variants where applicable
- Keep a baseline; regressions block merge

## API Stability

- `mathverse-core` traits are stable after v0.1 — additive changes only.
- Breaking changes require a major version bump and a migration note.
- New domains may land as pre-1.0; public APIs still follow semver.
