# MSRV Policy

MathVerse is a workspace of ~31 crates. Every crate depends (directly or
transitively) on `mathverse-core`, so the workspace's Minimum Supported Rust
Version (MSRV) is effectively determined by `mathverse-core`.

## Current MSRV: Rust 1.87

The MSRV is declared centrally:

```toml
# Cargo.toml (workspace root)
[workspace.package]
rust-version = "1.87"
```

and propagated to every member via:

```toml
# crates/*/Cargo.toml
rust-version.workspace = true
```

All 31 workspace members use `rust-version.workspace = true`. This makes the
constraint visible downstream: `cargo` rejects builds on older toolchains for
any crate that is a direct dependency of a project pulling MathVerse in.

## What the version pin protects

Rust 1.87 was chosen because it is the first release to expose the APIs and
const-eval capabilities the workspace relies on:

- `f64::round_ties_even` (stabilized 1.80)
- const-generic array length expressions in generic contexts (incremental,
  fully available by 1.87-era compilers)
- `Option::is_some_and` / `Option::is_none_or` (1.70 / 1.82)

Nothing in the workspace uses `nightly` features.

## When the MSRV changes

Policy:

1. **Raise conservatively.** Bump the MSRV only when a dependency or an
   ecosystem convention forces it, or when a new language feature provides a
   measurable correctness/ergonomics win for the crate that uses it.
2. **Bump centrally.** Edit `[workspace.package] rust-version` once. The
   `rust-version.workspace = true` members inherit the change automatically.
3. **Document the rationale.** Record the reason in the commit message and, for
   notable bumps, in this file.
4. **Stay on stable.** The workspace never requires `nightly`; do not pin an
   MSRV that implies one.

## Supported targets

- `std` targets: all Tier 1 platforms (tested in CI on `ubuntu-latest`).
- `no_std` targets: any target with `core` + `alloc` (e.g. bare-metal, WASM).
  `no_std` builds require the `libm` feature on `mathverse-core`:
  `default-features = false, features = ["libm"]`. This path is exercised by a
  dedicated CI job (see `.github/workflows/ci.yml`).

## Downstream guidance

If you embed MathVerse in a project, match or exceed the workspace MSRV
(i.e. compile with Rust >= 1.87) and, for embedded targets, remember to enable
the `libm` feature:

```toml
[dependencies]
mathverse-core = { version = "0.1", default-features = false, features = ["libm"] }
```
