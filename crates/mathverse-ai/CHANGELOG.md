# Changelog

## 0.1.1

- Added: CI pipeline, tests, benchmarks, docs
- Fixed: Format and lint compliance

## Unreleased

- Changed: `Tensor.shape`/`Tensor.data` are no longer `pub`; use `shape()`,
  `data()`, `data_mut()`, `as_slice()`, `into_data()`, `into_parts()`
- Changed: autograd rewritten around an explicit `ComputationGraph` — the
  global thread-local graph and free functions were removed; graphs are now
  independently owned and multiple can coexist
- Changed: `gather`/`scatter_add` accept `&[usize]` indices instead of an
  f64 index tensor
- Changed: optimizer `step` methods panic on params/grads length mismatch
  instead of silently returning in release builds
- Changed: `DataLoader::new`/`with_seed` return `MathResult<Self>` and
  validate inputs (non-scalar, equal lengths, non-zero batch size)
- Changed: `Dropout::new`/`with_seed` assert dropout probability is in `[0, 1)`
- Fixed: `BatchNorm` training variance uses the numerically stable two-pass
  algorithm (constant inputs no longer produce NaN)
- Fixed: `cosine_embedding_loss` no longer skips zero-norm samples, which
  skewed the loss mean
