#!/bin/bash
set -e
cargo test
cargo fmt -- --check
cargo clippy -- -D warnings
cargo audit
cargo deny check
