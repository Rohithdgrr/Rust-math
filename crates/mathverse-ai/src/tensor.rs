//! N-dimensional tensor with row-major layout, broadcasting, and math ops.

use mathverse_core::error::{MathError, MathResult};

/// N-dimensional tensor with row-major (C-contiguous) data.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
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

    /// 1-D range: [start, stop) with step.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        assert!(step > 0.0, "step must be positive");
        let mut data = Vec::new();
        let mut v = start;
        while v < stop {
            data.push(v);
            v += step;
        }
        Self { shape: vec![data.len()], data }
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

    /// Pseudo-random normal via xorshift64 (no external dep).
    pub fn randn(shape: &[usize]) -> Self {
        use std::cell::Cell;
        thread_local! { static S: Cell<u64> = Cell::new(0xDEAD_BEEF_CAFE_1234); }
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|_| {
            S.with(|s| {
                let mut x = s.get();
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.set(x);
                // Box-Muller: uniform → normal
                let u1 = (x as f64) / (u64::MAX as f64).max(1e-30);
                let u2 = ((x >> 32) as f64) / (u64::MAX as f64).max(1e-30);
                (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
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
        for (c, s) in coords.iter().zip(&strides) {
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
    pub fn unsqueeze(&self, axis: usize) -> Self {
        assert!(axis <= self.shape.len(), "axis out of range");
        let mut shape = self.shape.clone();
        shape.insert(axis, 1);
        Self { shape, data: self.data.clone() }
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

    /// Element-wise div.
    pub fn div(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x / y).collect();
        Ok(Tensor { shape: target, data })
    }

    pub fn add_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x + s).collect() }
    }

    pub fn sub_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x - s).collect() }
    }

    pub fn mul_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x * s).collect() }
    }

    pub fn div_scalar(&self, s: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x / s).collect() }
    }

    pub fn neg(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| -x).collect() }
    }

    pub fn abs(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.abs()).collect() }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.sqrt()).collect() }
    }

    pub fn exp(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.exp()).collect() }
    }

    pub fn ln(&self) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.ln()).collect() }
    }

    pub fn powf(&self, e: f64) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x.powf(e)).collect() }
    }

    /// Clip values to [lo, hi].
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
            vals.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
        })
    }

    /// Argmin along axis.
    pub fn argmin_axis(&self, axis: usize) -> MathResult<Tensor> {
        axis_reduce_idx(self, axis, |vals| {
            vals.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
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
    pub fn batch_norm(&self, eps: f64) -> MathResult<Tensor> {
        if self.shape.len() < 2 {
            return Err(MathError::InvalidArgument("batch_norm requires >= 2-D tensor"));
        }
        let batch = self.shape[0];
        let feature_size: usize = self.shape[1..].iter().product();
        let mut out = self.data.clone();
        let total = self.numel();
        let per_sample = total / batch;
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
}

// ---------------------------------------------------------------------------
// Broadcasting helper: NumPy trailing-dim rules
// ---------------------------------------------------------------------------

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

        let t = Tensor::arange(0.0, 4.0, 1.0);
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0]);

        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape, vec![5]);
        assert!((t.data[2] - 0.5).abs() < E);
    }

    #[test]
    fn reshape_and_flatten() {
        let t = Tensor::arange(0.0, 6.0, 1.0).reshape(&[2, 3]).unwrap();
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
        let u = s.unsqueeze(1);
        assert_eq!(u.shape, vec![3, 1]);
    }
}
