#!/bin/bash
set -e
cargo llvm-cov --html --output-dir coverage
