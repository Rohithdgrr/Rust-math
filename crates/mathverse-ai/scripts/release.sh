#!/bin/bash
set -e
VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d '"' -f2)
echo "Releasing $VERSION"
cargo publish --all-features
