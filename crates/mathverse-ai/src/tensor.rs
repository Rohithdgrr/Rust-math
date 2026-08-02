//! N-dimensional tensor with row-major layout, broadcasting, and math ops.

use std::fmt;
use mathverse_core::error::{MathError, MathResult};

/// N-dimensional tensor with row-major (C-contiguous) data.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape in row-major order.
    pub shape: Vec<usize>,
    /// Flat row-major data buffer.
    pub data: Vec<f64>,
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={}, ", self.shape, self.data.len())?;
        if self.data.len() <= 8 {
            write!(f, "data={:?})", self.data)?;
        } else {
            let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
            write!(f, "min={:.4}, max={:.4}, mean={:.4}, first5={:?})", min, max, mean, &self.data[..5])?;
        }
        Ok(())
    }
}

/// Advance an xorshift64 state; return uniform in [0, 1).
fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    if x == 0 { x = 0xDEAD_BEEF_CAFE_1234; }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f64) / (u64::MAX as f64).max(1e-30)
}

impl Tensor {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a tensor from shape + data. Validates numel matches.
    pub fn new(shape: &[usize], data: &[f64]) -> MathResult<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Self { shape: shape.to_vec(), data: data.to_vec() })
    }

    /// Create from shape + owned data.
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> MathResult<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Self { shape: shape.to_vec(), data })
    }

    /// Zero-filled tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![0.0; numel] }
    }

    /// One-filled tensor.
    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![1.0; numel] }
    }

    /// Fill with a constant value.
    pub fn full(shape: &[usize], val: f64) -> Self {
        let numel: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![val; numel] }
    }

    /// 0-dimensional scalar.
    pub fn scalar(val: f64) -> Self {
        Self { shape: vec![], data: vec![val] }
    }

    /// 1-D range: [start, stop) with step. Negative steps are supported.
    pub fn arange(start: f64, stop: f64, step: f64) -> MathResult<Self> {
        if step == 0.0 {
            return Err(MathError::InvalidArgument("arange: step must be non-zero"));
        }
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < stop {
                data.push(v);
                v += step;
            }
        } else {
            while v > stop {
                data.push(v);
                v += step;
            }
        }
        Ok(Self { shape: vec![data.len()], data })
    }

    /// 1-D linspace.
    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        if n == 0 {
            return Self { shape: vec![0], data: vec![] };
        }
        if n == 1 {
            return Self { shape: vec![1], data: vec![start] };
        }
        let step = (end - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self { shape: vec![n], data }
    }

    /// Pseudo-random normal via xorshift64. Uses persistent thread-local state
    /// so successive calls produce different values.
    pub fn randn(shape: &[usize]) -> Self {
        use std::cell::Cell;
        // 0 is the "uninitialized" sentinel; first call advances to a non-zero state.
        thread_local! { static S: Cell<u64> = const { Cell::new(0xDEAD_BEEF_CAFE_1234) }; }
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|_| {
            S.with(|s| {
                let mut x = s.get();
                if x == 0 { x = 0xDEAD_BEEF_CAFE_1234; }
                let u1 = xorshift(&mut x);
                let u2 = xorshift(&mut x);
                s.set(x);
                (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
        }).collect();
        Self { shape: shape.to_vec(), data }
    }

    /// Pseudo-random normal with explicit seed. Uses a local RNG state so it
    /// does not interfere with the shared thread-local state used by [`randn`].
    pub fn randn_seeded(shape: &[usize], seed: u64) -> Self {
        let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed };
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|_| {
            let u1 = xorshift(&mut state);
            let u2 = xorshift(&mut state);
            (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }).collect();
        Self { shape: shape.to_vec(), data }
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    /// Number of dimensions.
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Total number of elements.
    pub fn numel(&self) -> usize { self.data.len() }

    /// True if 0-dimensional.
    pub fn is_scalar(&self) -> bool { self.shape.is_empty() }

    /// Compute strides from shape (row-major).
    pub fn strides(&self) -> Vec<usize> {
        let nd = self.shape.len();
        let mut strides = vec![1usize; nd];
        for i in (0..nd.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// Multi-index to flat offset.
    pub fn index_to_flat(&self, coords: &[usize]) -> MathResult<usize> {
        if coords.len() != self.shape.len() {
            return Err(MathError::DimensionMismatch);
        }
        let strides = self.strides();
        let mut flat = 0;
        for (i, (c, s)) in coords.iter().zip(&strides).enumerate() {
            if *c >= self.shape[i] {
                return Err(MathError::OutOfRange);
            }
            flat += c * s;
        }
        Ok(flat)
    }

    /// Flat offset to multi-index.
    pub fn flat_to_index(&self, flat: usize) -> MathResult<Vec<usize>> {
        let strides = self.strides();
        let mut remaining = flat;
        let mut coords = vec![0usize; self.shape.len()];
        for i in 0..self.shape.len() {
            coords[i] = remaining / strides[i];
            remaining %= strides[i];
        }
        Ok(coords)
    }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------

    /// Get element at multi-index.
    pub fn get(&self, coords: &[usize]) -> MathResult<f64> {
        let flat = self.index_to_flat(coords)?;
        Ok(self.data[flat])
    }

    /// Get element at flat index.
    pub fn get_flat(&self, idx: usize) -> f64 {
        assert!(idx < self.data.len(), "get_flat: index {idx} out of bounds (len {})", self.data.len());
        self.data[idx]
    }

    /// Set element at multi-index.
    pub fn set(&mut self, coords: &[usize], val: f64) -> MathResult<()> {
        let flat = self.index_to_flat(coords)?;
        self.data[flat] = val;
        Ok(())
    }

    /// Set element at flat index.
    pub fn set_flat(&mut self, idx: usize, val: f64) {
        assert!(idx < self.data.len(), "set_flat: index {idx} out of bounds (len {})", self.data.len());
        self.data[idx] = val;
    }

    /// Borrow data as slice.
    pub fn as_slice(&self) -> &[f64] { &self.data }

    /// Consume into Vec.
    pub fn to_vec(self) -> Vec<f64> { self.data }

    // -----------------------------------------------------------------------
    // Shape manipulation (data-cloning views)
    // -----------------------------------------------------------------------

    /// Reshape (numel must match).
    pub fn reshape(&self, new_shape: &[usize]) -> MathResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Self { shape: new_shape.to_vec(), data: self.data.clone() })
    }

    /// Flatten to 1-D.
    pub fn flatten(&self) -> Self {
        Self { shape: vec![self.numel()], data: self.data.clone() }
    }

    /// 2-D transpose.
    pub fn transpose(&self) -> MathResult<Self> {
        if self.shape.len() != 2 {
            return Err(MathError::InvalidArgument("transpose requires 2-D tensor"));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0; self.numel()];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = self.data[i * cols + j];
            }
        }
        Ok(Self { shape: vec![cols, rows], data: out })
    }

    /// General axis permutation.
    pub fn permute(&self, axes: &[usize]) -> MathResult<Self> {
        if axes.len() != self.shape.len() {
            return Err(MathError::DimensionMismatch);
        }
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let mut out = vec![0.0; self.numel()];
        let old_strides = self.strides();
        let new_strides = {
            let mut s = vec![1usize; axes.len()];
            for i in (0..axes.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * new_shape[i + 1];
            }
            s
        };
        #[allow(clippy::needless_range_loop)]
        for flat in 0..self.numel() {
            let mut old_flat = 0;
            for (i, &a) in axes.iter().enumerate() {
                let coord = flat / new_strides[i] % new_shape[i];
                old_flat += coord * old_strides[a];
            }
            out[flat] = self.data[old_flat];
        }
        Ok(Self { shape: new_shape, data: out })
    }

    /// Remove size-1 dimensions (or a specific axis).
    pub fn squeeze(&self, axis: Option<usize>) -> Self {
        let shape: Vec<usize> = match axis {
            Some(a) => self.shape.iter().enumerate()
                .filter(|&(i, &s)| i != a || s != 1)
                .map(|(_, &s)| s)
                .collect(),
            None => self.shape.iter().copied().filter(|&s| s != 1).collect(),
        };
        let shape = if shape.is_empty() { vec![1] } else { shape };
        Self { shape, data: self.data.clone() }
    }

    /// Add a size-1 dimension at axis.
    pub fn unsqueeze(&self, axis: usize) -> MathResult<Self> {
        if axis > self.shape.len() {
            return Err(MathError::InvalidArgument("unsqueeze: axis out of range"));
        }
        let mut shape = self.shape.clone();
        shape.insert(axis, 1);
        Ok(Self { shape, data: self.data.clone() })
    }

    /// Expand to a target shape (broadcast semantics, clones data).
    pub fn broadcast_to(&self, target_shape: &[usize]) -> MathResult<Self> {
        let _ = broadcast_shapes(&self.shape, target_shape)?;
        let out_numel: usize = target_shape.iter().product();
        let mut out = vec![0.0; out_numel];
        let target_strides = {
            let mut s = vec![1usize; target_shape.len()];
            for i in (0..target_shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * target_shape[i + 1];
            }
            s
        };
        #[allow(clippy::needless_range_loop)]
        for flat in 0..out_numel {
            let mut coords = vec![0usize; target_shape.len()];
            let mut rem = flat;
            for i in 0..target_shape.len() {
                coords[i] = rem / target_strides[i];
                rem %= target_strides[i];
            }
            let src_coords: Vec<usize> = if coords.len() >= self.shape.len() {
                let offset = coords.len() - self.shape.len();
                coords[offset..].iter().zip(&self.shape)
                    .map(|(&c, &s)| if s == 1 { 0 } else { c.min(s - 1) })
                    .collect()
            } else {
                let offset = self.shape.len() - coords.len();
                coords.iter().zip(&self.shape[offset..])
                    .map(|(&c, &s)| if s == 1 { 0 } else { c.min(s - 1) })
                    .collect()
            };
            let src_flat = self.index_to_flat(&src_coords)?;
            out[flat] = self.data[src_flat];
        }
        Ok(Self { shape: target_shape.to_vec(), data: out })
    }

    // -----------------------------------------------------------------------
    // Element-wise arith (broadcast-aware)
    // -----------------------------------------------------------------------

    /// Element-wise add with broadcasting.
    pub fn add(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        Ok(Tensor { shape: target, data })
    }

    /// Element-wise sub.
    pub fn sub(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect();
        Ok(Tensor { shape: target, data })
    }

    /// Element-wise mul.
    pub fn mul(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
        Ok(Tensor { shape: target, data })
    }

    /// Element-wise div (safe: replaces zero with epsilon, preserving sign).
    pub fn div(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| {
            let denom = if y.abs() < f64::EPSILON { f64::EPSILON } else { *y };
            x / denom
        }).collect();
        Ok(Tensor { shape: target, data })
    }

    /// Add a scalar to every element.
    #[must_use]
    pub fn add_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x + s).collect() }
    }

    /// Subtract a scalar from every element.
    #[must_use]
    pub fn sub_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x - s).collect() }
    }

    /// Multiply every element by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x * s).collect() }
    }

    /// Divide every element by a scalar (denominator clamped to `f64::EPSILON`, preserving sign).
    #[must_use]
    pub fn div_scalar(&self, s: f64) -> Tensor {
        let denom = if s.abs() < f64::EPSILON { f64::EPSILON } else { s };
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x / denom).collect() }
    }

    /// Negate every element.
    #[must_use]
    pub fn neg(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| -x).collect() }
    }

    /// Absolute value of every element.
    #[must_use]
    pub fn abs(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.abs()).collect() }
    }

    /// Square root of every element.
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.sqrt()).collect() }
    }

    /// Exponential of every element.
    #[must_use]
    pub fn exp(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.exp()).collect() }
    }

    /// Natural logarithm of every element.
    #[must_use]
    pub fn ln(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.ln()).collect() }
    }

    /// Raise every element to a floating-point exponent.
    #[must_use]
    pub fn powf(&self, e: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.powf(e)).collect() }
    }

    /// Clip values to [lo, hi].
    #[must_use]
    pub fn clip(&self, lo: f64, hi: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.clamp(lo, hi)).collect() }
    }

    // -----------------------------------------------------------------------
    // Reduction ops
    // -----------------------------------------------------------------------

    /// Sum all elements.
    pub fn sum(&self) -> f64 { self.data.iter().sum() }

    /// Sum along axis (removes that axis).
    pub fn sum_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce(self, axis, |vals| vals.iter().sum())
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 { self.sum() / self.numel() as f64 }

    /// Mean along axis.
    pub fn mean_axis(&self, axis: usize) -> MathResult<Tensor> {
        let n = self.shape[axis] as f64;
        axis_reduce(self, axis, |vals| vals.iter().sum::<f64>() / n)
    }

    /// Max of all elements.
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }

    /// Max along axis.
    pub fn max_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce(self, axis, |vals| vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }

    /// Min of all elements.
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }

    /// Min along axis.
    pub fn min_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce(self, axis, |vals| vals.iter().cloned().fold(f64::INFINITY, f64::min))
    }

    /// Argmax along axis (returns flat index per slice).
    pub fn argmax_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce_idx(self, axis, |vals| {
            vals.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0)
        })
    }

    /// Argmin along axis.
    pub fn argmin_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce_idx(self, axis, |vals| {
            vals.iter().enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0)
        })
    }

    // -----------------------------------------------------------------------
    // Matrix multiply
    // -----------------------------------------------------------------------

    /// Matrix multiply: 2-D @ 2-D or batched 3-D @ 3-D.
    pub fn matmul(&self, other: &Tensor) -> MathResult<Tensor> {
        match (self.shape.len(), other.shape.len()) {
            (2, 2) => matmul_2d(self, other),
            (3, 3) => matmul_batched(self, other),
            _ => Err(MathError::InvalidArgument("matmul requires 2-D or 3-D tensors")),
        }
    }

    // -----------------------------------------------------------------------
    // Normalization
    // -----------------------------------------------------------------------

    /// Layer normalization (over last dimension).
    pub fn layer_norm(&self, eps: f64) -> Tensor {
        let nd = self.shape.len();
        let d = self.shape[nd - 1];
        let mut out = self.data.clone();
        let outer: usize = self.shape[..nd.saturating_sub(1)].iter().product();
        for i in 0..outer {
            let start = i * d;
            let slice = &self.data[start..start + d];
            let mu: f64 = slice.iter().sum::<f64>() / d as f64;
            let var: f64 = slice.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..d {
                out[start + j] = (self.data[start + j] - mu) * inv;
            }
        }
        Tensor { shape: self.shape.clone(), data: out }
    }

    /// Batch normalization (over first dimension).
    ///
    /// For 2-D tensors `[N, features]`, normalizes each feature across the batch.
    /// For 4-D tensors `[N, C, H, W]`, normalizes per channel across `N, H, W`
    /// (standard image batch norm behavior).
    pub fn batch_norm(&self, eps: f64) -> MathResult<Tensor> {
        if self.shape.len() < 2 {
            return Err(MathError::InvalidArgument("batch_norm requires >= 2-D tensor"));
        }
        let batch = self.shape[0];
        let mut out = self.data.clone();

        if self.shape.len() == 4 {
            // [N, C, H, W] — per-channel normalization
            let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
            let hw = h * w;
            let spatial_count = (n * h * w) as f64;
            for ch in 0..c {
                let mut sum = 0.0;
                let mut sum2 = 0.0;
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let v = self.data[ni * c * hw + ch * hw + hi * w + wi];
                            sum += v;
                            sum2 += v * v;
                        }
                    }
                }
                let mu = sum / spatial_count;
                let var = sum2 / spatial_count - mu * mu;
                let inv = 1.0 / (var + eps).sqrt();
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let idx = ni * c * hw + ch * hw + hi * w + wi;
                            out[idx] = (self.data[idx] - mu) * inv;
                        }
                    }
                }
            }
        } else {
            // Generic: normalize each flat position across batch
            let feature_size: usize = self.shape[1..].iter().product();
            let per_sample = self.numel() / batch;
            for f in 0..feature_size {
                let mut sum = 0.0;
                let mut sum2 = 0.0;
                for b in 0..batch {
                    let idx = b * per_sample + f;
                    let v = self.data[idx];
                    sum += v;
                    sum2 += v * v;
                }
                let mu = sum / batch as f64;
                let var = sum2 / batch as f64 - mu * mu;
                let inv = 1.0 / (var + eps).sqrt();
                for b in 0..batch {
                    let idx = b * per_sample + f;
                    out[idx] = (self.data[idx] - mu) * inv;
                }
            }
        }
        Ok(Tensor { shape: self.shape.clone(), data: out })
    }

    /// RMS normalization: x / sqrt(mean(x²) + eps) along last dim.
    pub fn rms_norm(&self, eps: f64) -> Tensor {
        let nd = self.shape.len();
        let d = self.shape[nd - 1];
        let mut out = self.data.clone();
        let outer: usize = self.shape[..nd.saturating_sub(1)].iter().product();
        for i in 0..outer {
            let start = i * d;
            let slice = &self.data[start..start + d];
            let rms = (slice.iter().map(|x| x * x).sum::<f64>() / d as f64 + eps).sqrt();
            for j in 0..d {
                out[start + j] = self.data[start + j] / rms;
            }
        }
        Tensor { shape: self.shape.clone(), data: out }
    }

    // -----------------------------------------------------------------------
    // Advanced ops
    // -----------------------------------------------------------------------

    /// Element-wise where: condition ? a : b.
    pub fn where_tensor(condition: &Tensor, a: &Tensor, b: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&broadcast_shapes(&condition.shape, &a.shape)?, &b.shape)?;
        let c = condition.broadcast_to(&target)?;
        let av = a.broadcast_to(&target)?;
        let bv = b.broadcast_to(&target)?;
        let data: Vec<f64> = c.data.iter().zip(&av.data).zip(&bv.data)
            .map(|((&cond, &av), &bv)| if cond > 0.0 { av } else { bv })
            .collect();
        Ok(Tensor { shape: target, data })
    }

    /// Gather along axis: selects from `self` using indices.
    /// `indices` has same shape as output, values are indices along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `MathError::OutOfRange` if any index is outside `[0, axis_size)`.
    pub fn gather(&self, axis: usize, indices: &Tensor) -> MathResult<Tensor> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("gather: axis out of range")); }
        let mut out_data = Vec::with_capacity(indices.numel());
        let axis_size = self.shape[axis];
        let _outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let iouter: usize = indices.shape[..axis].iter().product();
        let iinner: usize = indices.shape[axis + 1..].iter().product();
        for io in 0..iouter {
            for k in 0..indices.shape[axis] {
                for ii in 0..iinner {
                    let idx = io * indices.shape[axis] * iinner + k * iinner + ii;
                    let gi = indices.data[idx];
                    if gi < 0.0 || gi.fract() != 0.0 || gi as usize >= axis_size {
                        return Err(MathError::OutOfRange);
                    }
                    let gather_idx = gi as usize;
                    let src_flat = io * axis_size * inner + gather_idx * inner + ii;
                    out_data.push(self.data[src_flat]);
                }
            }
        }
        let out_shape = indices.shape.clone();
        Ok(Tensor { shape: out_shape, data: out_data })
    }

    /// Scatter add: adds `src` into a zero tensor at positions given by `indices`.
    ///
    /// # Errors
    ///
    /// Returns `MathError::OutOfRange` if any index is outside `[0, axis_size)`.
    pub fn scatter_add(&self, axis: usize, indices: &Tensor, src: &Tensor) -> MathResult<Tensor> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("scatter_add: axis out of range")); }
        let mut out = self.clone();
        let axis_size = self.shape[axis];
        let _outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let iouter: usize = indices.shape[..axis].iter().product();
        let iinner: usize = indices.shape[axis + 1..].iter().product();
        for io in 0..iouter {
            for k in 0..indices.shape[axis] {
                for ii in 0..iinner {
                    let idx = io * indices.shape[axis] * iinner + k * iinner + ii;
                    let si = indices.data[idx];
                    if si < 0.0 || si.fract() != 0.0 || si as usize >= axis_size {
                        return Err(MathError::OutOfRange);
                    }
                    let scatter_idx = si as usize;
                    let src_flat = io * src.shape[axis] * inner + k * inner + ii;
                    let dst_flat = io * axis_size * inner + scatter_idx * inner + ii;
                    out.data[dst_flat] += src.data[src_flat];
                }
            }
        }
        Ok(out)
    }

    /// Top-k along axis: returns (values, indices) tensors.
    pub fn topk(&self, k: usize, axis: usize) -> MathResult<(Tensor, Tensor)> {
        if axis >= self.shape.len() || k > self.shape[axis] {
            return Err(MathError::InvalidArgument("topk: invalid axis or k"));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut val_data = Vec::with_capacity(outer * k * inner);
        let mut idx_data = Vec::with_capacity(outer * k * inner);
        for io in 0..outer {
            for ii in 0..inner {
                let mut pairs: Vec<(f64, usize)> = (0..axis_size).map(|a| {
                    let flat = io * axis_size * inner + a * inner + ii;
                    (self.data[flat], a)
                }).collect();
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                #[allow(clippy::needless_range_loop)]
                for ki in 0..k {
                    val_data.push(pairs[ki].0);
                    idx_data.push(pairs[ki].1 as f64);
                }
            }
        }
        let mut out_shape = self.shape.clone();
        out_shape[axis] = k;
        let idx_shape = out_shape.clone();
        Ok((Tensor { shape: out_shape, data: val_data }, Tensor { shape: idx_shape, data: idx_data }))
    }

    /// Concatenate tensors along an axis.
    pub fn concat(tensors: &[Tensor], axis: usize) -> MathResult<Tensor> {
        if tensors.is_empty() { return Err(MathError::InvalidArgument("concat: empty input")); }
        let ndim = tensors[0].shape.len();
        if axis >= ndim { return Err(MathError::InvalidArgument("concat: axis out of range")); }
        let mut out_shape = tensors[0].shape.clone();
        let mut total_axis = 0;
        for t in tensors {
            if t.shape.len() != ndim { return Err(MathError::DimensionMismatch); }
            for (i, (a, b)) in out_shape.iter().zip(&t.shape).enumerate() {
                if i != axis && a != b { return Err(MathError::DimensionMismatch); }
            }
            total_axis += t.shape[axis];
        }
        out_shape[axis] = total_axis;
        let outer: usize = tensors[0].shape[..axis].iter().product();
        let inner: usize = tensors[0].shape[axis + 1..].iter().product();
        let mut out_data = Vec::with_capacity(outer * total_axis * inner);
        for io in 0..outer {
            for t in tensors {
                let asize = t.shape[axis];
                for a in 0..asize {
                    for ii in 0..inner {
                        let flat = io * asize * inner + a * inner + ii;
                        out_data.push(t.data[flat]);
                    }
                }
            }
        }
        Ok(Tensor { shape: out_shape, data: out_data })
    }

    /// Split into chunks along axis. Last chunk absorbs any remainder.
    pub fn split(&self, num_chunks: usize, axis: usize) -> MathResult<Vec<Tensor>> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("split: axis out of range")); }
        if num_chunks == 0 { return Err(MathError::InvalidArgument("split: num_chunks must be > 0")); }
        let axis_len = self.shape[axis];
        let chunk_size = axis_len / num_chunks;
        let remainder = axis_len % num_chunks;
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(num_chunks);
        let mut offset = 0;
        for c in 0..num_chunks {
            let cur_size = if c == num_chunks - 1 { chunk_size + remainder } else { chunk_size };
            let mut chunk_data = Vec::with_capacity(outer * cur_size * inner);
            for io in 0..outer {
                for a in 0..cur_size {
                    let src_a = offset + a;
                    for ii in 0..inner {
                        let flat = io * axis_len * inner + src_a * inner + ii;
                        chunk_data.push(self.data[flat]);
                    }
                }
            }
            let mut shape = self.shape.clone();
            shape[axis] = cur_size;
            result.push(Tensor { shape, data: chunk_data });
            offset += cur_size;
        }
        Ok(result)
    }

    /// Dot product (flattened). Errors if the tensors have different sizes.
    pub fn dot(&self, other: &Tensor) -> MathResult<f64> {
        if self.numel() != other.numel() {
            return Err(MathError::DimensionMismatch);
        }
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Cross product (3-vectors only).
    pub fn cross(&self, other: &Tensor) -> MathResult<Tensor> {
        if self.numel() != 3 || other.numel() != 3 {
            return Err(MathError::InvalidArgument("cross: requires 3-element vectors"));
        }
        Ok(Tensor {
            shape: vec![3],
            data: vec![
                self.data[1] * other.data[2] - self.data[2] * other.data[1],
                self.data[2] * other.data[0] - self.data[0] * other.data[2],
                self.data[0] * other.data[1] - self.data[1] * other.data[0],
            ],
        })
    }

    /// Clamp (alias for clip).
    #[must_use]
    pub fn clamp(&self, lo: f64, hi: f64) -> Tensor { self.clip(lo, hi) }

    /// Variance of all elements.
    #[must_use]
    pub fn var(&self) -> f64 {
        let m = self.mean();
        self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.numel() as f64
    }

    /// Variance along axis.
    pub fn var_axis(&self, axis: usize) -> MathResult<Tensor> {
        let n = self.shape[axis] as f64;
        let mean = self.mean_axis(axis)?;
        // broadcast mean back to self shape for subtraction
        let mean_expanded = mean.broadcast_to(&self.shape)?;
        let diff = self.sub(&mean_expanded)?;
        let sq = diff.mul(&diff)?;
        axis_reduce(&sq, axis, |vals| vals.iter().sum::<f64>() / n)
    }

    /// Standard deviation along axis.
    pub fn std_axis(&self, axis: usize) -> MathResult<Tensor> {
        let v = self.var_axis(axis)?;
        Ok(v.sqrt())
    }

    /// Sort along axis (ascending), returns (sorted_values, original_indices).
    pub fn sort(&self, axis: usize) -> MathResult<(Tensor, Tensor)> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("sort: axis out of range")); }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut val_data = Vec::with_capacity(self.numel());
        let mut idx_data = Vec::with_capacity(self.numel());
        for io in 0..outer {
            for ii in 0..inner {
                let mut pairs: Vec<(f64, usize)> = (0..axis_size).map(|a| {
                    let flat = io * axis_size * inner + a * inner + ii;
                    (self.data[flat], a)
                }).collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                for &(v, idx) in &pairs {
                    val_data.push(v);
                    idx_data.push(idx as f64);
                }
            }
        }
        Ok((Tensor { shape: self.shape.clone(), data: val_data }, Tensor { shape: self.shape.clone(), data: idx_data }))
    }

    /// Unique elements (sorted, deduplicated). Exact equality only —
    /// near-but-not-equal values are kept.
    #[must_use]
    pub fn unique(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.data.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.dedup();
        v
    }

    /// Count non-zero elements.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        self.data.iter().filter(|&&x| x.abs() > 1e-15).count()
    }

    /// Any (true if any element > 0).
    #[must_use]
    pub fn any(&self) -> bool { self.data.iter().any(|&x| x > 0.0) }

    /// All (true if all elements > 0).
    #[must_use]
    pub fn all(&self) -> bool { self.data.iter().all(|&x| x > 0.0) }
}

// ---------------------------------------------------------------------------
// Broadcasting helper: NumPy trailing-dim rules
// ---------------------------------------------------------------------------

/// Compute the broadcasted shape of `a` and `b` using NumPy trailing-dimension rules.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> MathResult<Vec<usize>> {
    let nd = a.len().max(b.len());
    let mut result = vec![0usize; nd];
    for i in 0..nd {
        let da = if i + (nd - a.len()) < nd { a[i + (nd - a.len())] } else { 1 };
        let db = if i + (nd - b.len()) < nd { b[i + (nd - b.len())] } else { 1 };
        if da == db || da == 1 || db == 1 {
            result[i] = da.max(db);
        } else {
            return Err(MathError::DimensionMismatch);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal: 2-D matmul
// ---------------------------------------------------------------------------

fn matmul_2d(a: &Tensor, b: &Tensor) -> MathResult<Tensor> {
    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k1 != k2 { return Err(MathError::DimensionMismatch); }
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k1 {
            let av = a.data[i * k1 + p];
            for j in 0..n {
                data[i * n + j] += av * b.data[p * n + j];
            }
        }
    }
    Ok(Tensor { shape: vec![m, n], data })
}

fn matmul_batched(a: &Tensor, b: &Tensor) -> MathResult<Tensor> {
    let (batch, m, k1) = (a.shape[0], a.shape[1], a.shape[2]);
    let (_, k2, n) = (b.shape[0], b.shape[1], b.shape[2]);
    if k1 != k2 { return Err(MathError::DimensionMismatch); }
    let mut data = vec![0.0; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for p in 0..k1 {
                let av = a.data[bi * m * k1 + i * k1 + p];
                for j in 0..n {
                    data[bi * m * n + i * n + j] += av * b.data[bi * k1 * n + p * n + j];
                }
            }
        }
    }
    Ok(Tensor { shape: vec![batch, m, n], data })
}

// ---------------------------------------------------------------------------
// Internal: axis reduction
// ---------------------------------------------------------------------------

fn axis_reduce<F>(t: &Tensor, axis: usize, f: F) -> MathResult<Tensor>
where F: Fn(&[f64]) -> f64
{
    if axis >= t.shape.len() {
        return Err(MathError::InvalidArgument("axis out of range"));
    }
    let mut out_shape = t.shape.clone();
    let axis_size = out_shape[axis];
    out_shape.remove(axis);

    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let mut out_data = Vec::with_capacity(outer * inner);

    for i in 0..outer {
        for j in 0..inner {
            let vals: Vec<f64> = (0..axis_size)
                .map(|k| t.data[i * axis_size * inner + k * inner + j])
                .collect();
            out_data.push(f(&vals));
        }
    }

    if out_shape.is_empty() { out_shape.push(1); }
    Ok(Tensor { shape: out_shape, data: out_data })
}

fn axis_reduce_idx<F>(t: &Tensor, axis: usize, f: F) -> MathResult<Tensor>
where F: Fn(&[f64]) -> usize
{
    if axis >= t.shape.len() {
        return Err(MathError::InvalidArgument("axis out of range"));
    }
    let mut out_shape = t.shape.clone();
    let axis_size = out_shape[axis];
    out_shape.remove(axis);

    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let mut out_data = Vec::with_capacity(outer * inner);

    for i in 0..outer {
        for j in 0..inner {
            let vals: Vec<f64> = (0..axis_size)
                .map(|k| t.data[i * axis_size * inner + k * inner + j])
                .collect();
            out_data.push(f(&vals) as f64);
        }
    }

    if out_shape.is_empty() { out_shape.push(1); }
    Ok(Tensor { shape: out_shape, data: out_data })
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f64 = 1e-9;

    #[test]
    fn constructors() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));

        let t = Tensor::ones(&[3]);
        assert_eq!(t.data, vec![1.0, 1.0, 1.0]);

        let t = Tensor::arange(0.0, 4.0, 1.0).unwrap();
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0]);

        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape, vec![5]);
        assert!((t.data[2] - 0.5).abs() < E);

        let t = Tensor::arange(5.0, 0.0, -1.0).unwrap();
        assert_eq!(t.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        assert!(Tensor::arange(0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn randn_is_standard_normal() {
        // Box-Muller regression: ~99.9% of a standard normal is within [-3.5, 3.5],
        // and sample variance must be ~1 (the broken u2 range gave |x| < ~1.3e-9).
        let t = Tensor::randn_seeded(&[200_000], 12345);
        let n = t.numel() as f64;
        let mean: f64 = t.data.iter().sum::<f64>() / n;
        let var: f64 = t.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean.abs()) < 0.02, "mean {mean} deviates from 0");
        assert!((var - 1.0).abs() < 0.05, "variance {var} deviates from 1");
        assert!(t.data.iter().any(|x| x.abs() > 1.0), "no samples beyond 1 sigma");
    }

    #[test]
    fn reshape_and_flatten() {
        let t = Tensor::arange(0.0, 6.0, 1.0).unwrap().reshape(&[2, 3]).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!((t.get(&[1, 2]).unwrap() - 5.0).abs() < E);
        let f = t.flatten();
        assert_eq!(f.shape, vec![6]);
    }

    #[test]
    fn transpose_2d() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t2 = t.transpose().unwrap();
        assert_eq!(t2.shape, vec![3, 2]);
        assert!((t2.get(&[0, 1]).unwrap() - 4.0).abs() < E);
    }

    #[test]
    fn broadcast_add() {
        let a = Tensor::new(&[2, 1], &[1.0, 2.0]).unwrap();
        let b = Tensor::new(&[1, 3], &[10.0, 20.0, 30.0]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert!((c.get(&[0, 0]).unwrap() - 11.0).abs() < E);
        assert!((c.get(&[1, 2]).unwrap() - 32.0).abs() < E);
    }

    #[test]
    fn matmul_2d_test() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        // [1,2,3]·[7,9,11] = 7+18+33 = 58
        assert!((c.get(&[0, 0]).unwrap() - 58.0).abs() < E);
        // [4,5,6]·[8,10,12] = 32+50+72 = 154
        assert!((c.get(&[1, 1]).unwrap() - 154.0).abs() < E);
    }

    #[test]
    fn reductions() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!((t.sum() - 21.0).abs() < E);
        assert!((t.mean() - 3.5).abs() < E);
        let s = t.sum_axis(0).unwrap();
        assert_eq!(s.shape, vec![3]);
        assert!((s.get(&[0]).unwrap() - 5.0).abs() < E); // 1+4
        let s2 = t.sum_axis(1).unwrap();
        assert_eq!(s2.shape, vec![2]);
        assert!((s2.get(&[0]).unwrap() - 6.0).abs() < E); // 1+2+3
    }

    #[test]
    fn normalization() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let ln = t.layer_norm(1e-5);
        // Each row should have mean ≈ 0
        let m0: f64 = (0..3).map(|j| ln.data[j]).sum::<f64>() / 3.0;
        assert!(m0.abs() < 1e-5);
    }

    #[test]
    fn argmax_test() {
        let t = Tensor::new(&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]).unwrap();
        let am = t.argmax_axis(1).unwrap();
        assert!((am.data[0] - 1.0).abs() < E); // argmax of [1,5,3] = 1
        assert!((am.data[1] - 2.0).abs() < E); // argmax of [4,2,6] = 2
    }

    #[test]
    fn clip_test() {
        let t = Tensor::new(&[4], &[1.0, 5.0, -3.0, 2.0]).unwrap();
        let c = t.clip(-1.0, 3.0);
        assert_eq!(c.data, vec![1.0, 3.0, -1.0, 2.0]);
    }

    #[test]
    fn batch_norm_test() {
        let t = Tensor::new(&[4, 2], &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]).unwrap();
        let bn = t.batch_norm(1e-5).unwrap();
        // Per-feature mean should be ~0
        let m0: f64 = (0..4).map(|i| bn.data[i * 2]).sum::<f64>() / 4.0;
        assert!(m0.abs() < 1e-5);
    }

    #[test]
    fn rms_norm_test() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rn = t.rms_norm(1e-8);
        // RMS of each row should be ~1
        let rms0: f64 = (0..3).map(|j| rn.data[j].powi(2)).sum::<f64>() / 3.0;
        assert!((rms0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_3d_batched() {
        let a = Tensor::new(&[2, 2, 3], &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]).unwrap();
        let b = Tensor::new(&[2, 3, 2], &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2, 2]);
        // Batch 0: [1,2,3]·[1,3,5]=1+6+15=22
        assert!((c.get(&[0, 0, 0]).unwrap() - 22.0).abs() < E);
    }

    #[test]
    fn permute_test() {
        let t = Tensor::new(&[2, 3, 4], &(0..24).map(|x| x as f64).collect::<Vec<_>>()).unwrap();
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape, vec![4, 2, 3]);
        // original [0,1,2] = 2 → permuted [2,0,0] should be 2
        assert!((p.get(&[2, 0, 0]).unwrap() - 2.0).abs() < E);
    }

    #[test]
    fn squeeze_unsqueeze() {
        let t = Tensor::zeros(&[1, 3, 1]);
        let s = t.squeeze(None);
        assert_eq!(s.shape, vec![3]);
        let u = s.unsqueeze(1).unwrap();
        assert_eq!(u.shape, vec![3, 1]);
    }
}













