#!/bin/bash
set -e
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo fmt -- --check
