# Contributing

## Setup

```bash
rustc --version  # 1.87+
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check
```

## Coding Standards
- All public items must have doc comments (`missing_docs = warn` in workspace).
- No `unsafe` code (`unsafe_code = forbid` in workspace).
- Use `thiserror` for errors; prefer `#[derive(Error, Debug)]`.
- Keep functions under 50 lines where practical.

## Submitting PRs
1. Fork the repo.
2. Create a feature branch.
3. Run full CI locally: `cargo test`, `cargo clippy`, `cargo fmt --check`.
4. Update `CHANGELOG.md` and `docs/` if behavior changes.
5. Open PR with description linking issue (if any).

## Review Process
Maintainers review within 3 business days. All PRs must pass CI before merge.
