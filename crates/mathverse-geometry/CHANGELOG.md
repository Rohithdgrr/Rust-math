# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-07

### Added
- `error::GeometryError` centralized error type.
- Production-grade CI pipeline (`.github/workflows/ci.yml`).
- Integration tests (`tests/integration.rs`).
- Benchmark (`benches/benchmark.rs`).
- Example (`examples/basic.rs`).
- Architecture docs (`docs/architecture.md`).
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`.

### Changed
- `Cargo.toml` upgraded with `rust-version`, `categories`, `docs.rs` metadata, features (`default`, `full`), `thiserror` dependency, release profile tuning.
- `lib.rs` exposes `pub mod error`.
