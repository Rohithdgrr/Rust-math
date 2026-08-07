#!/bin/bash
set -e
cargo install cargo-llvm-cov || true
cargo llvm-cov --all-features --summary-only
