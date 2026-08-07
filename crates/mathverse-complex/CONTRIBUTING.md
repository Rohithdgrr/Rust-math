# Contributing to MathVerse Complex

Thanks for your interest in contributing! This crate is part of the
[MathVerse](https://github.com/Rohithdgrr/Rust-math) workspace, so a few
conventions apply workspace-wide.

## Getting Started

1. **Fork & clone** the repository.
2. **Build** — the crate is part of a Cargo workspace:

   ```bash
   cargo build -p mathverse-complex
   ```

3. **Run the tests**:

   ```bash
   cargo test -p mathverse-complex
   ```

## Local Checks

Run all of these before opening a PR. CI runs the same checks.

```bash
cargo fmt --all -- --check     # formatting
cargo clippy -p mathverse-complex --all-targets -- -D warnings
cargo test -p mathverse-complex
cargo doc -p mathverse-complex --no-deps
```

Notes:

- `cargo clippy` must be **warning-free** for this crate (workspace lints are
  already strict: `unsafe_code = forbid`, `missing_docs = warn`, clippy pedantic).
- Benchmarks live in `benches/benchmark.rs`; keep them green with
  `cargo bench -p mathverse-complex --no-run`.

## Coding Standards

- **No `unsafe`** — the crate forbids it at the workspace level.
- **Document everything** — every public item needs a doc comment; include
  `# Panics`, `# Errors`, and math formulas where relevant.
- **Match the ecosystem style** — method names follow `cmath`/`numpy`
  conventions for Python parity (`phase`, `rect`, `is_close`, …).
- **Numerical rigor** — prefer numerically stable formulas (Smith division,
  algebraic `sqrt`, asymptotic expansions with optimal truncation). When you add
  a numerically-sensitive function, add a regression test for the boundary case
  that motivated it.
- **No external runtime dependencies** without discussion — the crate currently
  depends only on `mathverse-core`.

## Testing

- Unit tests live in `#[cfg(test)] mod tests` next to the code.
- Public-API tests live in `tests/integration.rs`.
- Add a test for every new public function, and a *regression* test for every bug fix.
- For numerically approximate results, prefer asserting against a tolerance
  derived from the error estimate rather than a magic constant.

## Submitting a PR

1. Create a branch from `master` with a descriptive name (`fix/lu-pivot`,
   `feat/fft2`, …).
2. Make your changes with clear, atomic commits.
3. Run the local checks above.
4. Update `CHANGELOG.md` under `[Unreleased]` (or the next version heading).
5. Open a PR against `master` describing what changed and why.
6. A maintainer will review; expect at least one round of feedback on numerical
   or API-design details.

## Review Process

- Reviewers check correctness, numerical stability, API ergonomics, docs, and tests.
- Changes that alter the public API should justify the break in the PR
  description (semver impact).
- Small, focused PRs review faster than large ones.

## License

By contributing you agree that your contributions are licensed under the
[MIT OR Apache-2.0](LICENSE.md) license.
