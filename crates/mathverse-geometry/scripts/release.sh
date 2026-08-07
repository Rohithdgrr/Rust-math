#!/bin/bash
set -euo pipefail

VERSION=$(grep '^version =' Cargo.toml | cut -d'"' -f2)
echo "Releasing $VERSION"

cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check
cargo audit || true

echo "Release $VERSION ready. Run: cargo publish --all-features"
