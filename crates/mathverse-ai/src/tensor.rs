//! N-dimensional tensor with row-major layout, broadcasting, and math ops.

use std::fmt;
use mathverse_core::error::{MathError, MathResult};

/// N-dimensional tensor with row-major (C-contiguous) data.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape in row-major order.
    pub(crate) shape: Vec<usize>,
    /// Flat row-major data buffer.
    pub(crate) data: Vec<f64>,
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

/// Advance a SplitMix64 state; return uniform in [0, 1).
///
/// SplitMix64 has better statistical quality than the previously used
/// xorshift64* (it passes BigCrush with a good mixing function) while
/// remaining dependency-free and trivial to seed. It is still **not**
/// cryptographically secure — use a dedicated CSPRNG for security contexts.
fn splitmix64(state: &mut u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw a fresh, well-mixed seed from a global atomic counter.
///
/// Gives each thread a distinct [`Tensor::randn`] stream without requiring an
/// external entropy source. Reproducible seeding via [`Tensor::randn_seeded`]
/// is unaffected. Not cryptographically secure.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // Skip the zero state (splitmix64 would still mix it, but starting at 1
    // avoids any degeneracy) and mix the counter so adjacent threads get
    // well-separated streams.
    let mut n = COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    splitmix64(&mut n)
}

/// Uniform sample in [0, 1) from a SplitMix64 stream.
fn uniform01(state: &mut u64) -> f64 {
    // 53 bits of entropy, matching the precision of an f64 mantissa.
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard-normal sample via the polar (Marsaglia) method.
///
/// Uses a single uniform draw per call and is more numerically stable than
/// Box-Muller when `u1` is extremely small.
fn normal_sample(state: &mut u64) -> f64 {
    loop {
        let u1 = uniform01(state);
        let u2 = uniform01(state);
        let x = 2.0 * u1 - 1.0;
        let y = 2.0 * u2 - 1.0;
        let r2 = x * x + y * y;
        if r2 > 0.0 && r2 <= 1.0 {
            // Marsaglia polar method: z = x * sqrt(-2 ln r² / r²)
            return x * (-2.0 * r2.ln() / r2).sqrt();
        }
    }
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

    /// Pseudo-random standard normal via SplitMix64 + Marsaglia polar.
    ///
    /// Uses persistent thread-local state so successive calls produce
    /// different values. Each thread derives a distinct seed from a global
    /// counter on first use, so parallel callers never share an identical
    /// stream. For reproducible results use [`randn_seeded`].
    pub fn randn(shape: &[usize]) -> Self {
        use std::cell::Cell;
        thread_local! {
            // `None` marks an unseeded stream; the first call draws a seed.
            static S: Cell<Option<u64>> = const { Cell::new(None) };
        }
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|_| {
            S.with(|s| {
                let mut x = match s.get() {
                    Some(seed) => seed,
                    None => {
                        let seed = fresh_seed();
                        s.set(Some(seed));
                        seed
                    }
                };
                let v = normal_sample(&mut x);
                s.set(Some(x));
                v
            })
        }).collect();
        Self { shape: shape.to_vec(), data }
    }

    /// Pseudo-random standard normal with explicit seed. Uses a local RNG
    /// state so it does not interfere with the shared thread-local state used
    /// by [`randn`].
    pub fn randn_seeded(shape: &[usize], seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = (0..numel).map(|_| normal_sample(&mut state)).collect();
        Self { shape: shape.to_vec(), data }
    }

    /// Standard-normal random with crypto-secure RNG seeding.
    ///
    /// Uses `getrandom::fill` to obtain cryptographically secure randomness
    /// for seeding the internal SplitMix64-derived RNG. The generation logic
    /// is identical to [`randn_seeded`], but the initial seed is derived from
    /// an OS-provided CSPRNG rather than a predictable counter.
    #[cfg(feature = "secure-rng")]
    pub fn randn_secure(shape: &[usize]) -> MathResult<Self> {
        use getrandom::fill;
        let numel: usize = shape.iter().product();

        // Seed the RNG with cryptographically secure randomness
        let mut seed: u64 = 0;
        let mut buf = [0u8; 8];
        fill(&mut buf)?;
        seed = u64::from_le_bytes(buf);

        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let data: Vec<f64> = (0..numel).map(|_| normal_sample(&mut state)).collect();

        Ok(Self { shape: shape.to_vec(), data })
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    /// Returns the tensor shape as a slice (e.g. `&[2, 3]`).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions (rank).
    #[must_use]
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

    /// Get element at flat index, returning an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::OutOfRange`] if `idx >= self.numel()`.
    pub fn get_flat_checked(&self, idx: usize) -> MathResult<f64> {
        self.data.get(idx).copied().ok_or(MathError::OutOfRange)
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

    /// Set element at flat index, returning an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::OutOfRange`] if `idx >= self.numel()`.
    pub fn set_flat_checked(&mut self, idx: usize, val: f64) -> MathResult<()> {
        let slot = self
            .data
            .get_mut(idx)
            .ok_or(MathError::OutOfRange)?;
        *slot = val;
        Ok(())
    }

    /// Borrow data as slice.
    pub fn as_slice(&self) -> &[f64] { &self.data }

    /// Borrow the flat row-major data buffer.
    #[must_use]
    pub fn data(&self) -> &[f64] { &self.data }

    /// Mutably borrow the flat row-major data buffer.
    pub fn data_mut(&mut self) -> &mut [f64] { &mut self.data }

    /// Consume into Vec (alias of [`Tensor::to_vec`]).
    #[must_use]
    pub fn into_data(self) -> Vec<f64> { self.data }

    /// Consume into `(shape, data)` parts.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, Vec<f64>) { (self.shape, self.data) }

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

    /// Element-wise div (safe: returns NaN for zero denominators instead of corrupting sign).
    pub fn div(&self, other: &Tensor) -> MathResult<Tensor> {
        let target = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&target)?;
        let b = other.broadcast_to(&target)?;
        let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| {
            if *y == 0.0 { f64::NAN } else { x / y }
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

    /// Divide every element by a scalar (returns NaN for zero denominator).
    #[must_use]
    pub fn div_scalar(&self, s: f64) -> Tensor {
        let denom = if s == 0.0 { f64::NAN } else { s };
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
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            sum += self.data[ni * c * hw + ch * hw + hi * w + wi];
                        }
                    }
                }
                let mu = sum / spatial_count;
                // Two-pass variance (E[(x-μ)²]) — numerically stable; the
                // one-pass E[x²]-μ² form can go negative and produce NaN.
                let mut var = 0.0;
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let d = self.data[ni * c * hw + ch * hw + hi * w + wi] - mu;
                            var += d * d;
                        }
                    }
                }
                var /= spatial_count;
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
                for b in 0..batch {
                    let idx = b * per_sample + f;
                    sum += self.data[idx];
                }
                let mu = sum / batch as f64;
                // Two-pass variance for numerical stability.
                let mut var = 0.0;
                for b in 0..batch {
                    let idx = b * per_sample + f;
                    let d = self.data[idx] - mu;
                    var += d * d;
                }
                var /= batch as f64;
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
    /// Gather along `axis`: selects slices of `self` using integer indices.
    ///
    /// The result has the same shape as `self` except `shape[axis]` becomes
    /// `indices.len()`; element `[.., k, ..]` of the output is
    /// `self[.., indices[k], ..]`.
    pub fn gather(&self, axis: usize, indices: &[usize]) -> MathResult<Tensor> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("gather: axis out of range")); }
        let axis_size = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        for &gi in indices {
            if gi >= axis_size { return Err(MathError::OutOfRange); }
        }
        let mut out_data = Vec::with_capacity(indices.len() * outer * inner);
        for io in 0..outer {
            for &gather_idx in indices {
                for ii in 0..inner {
                    let src_flat = io * axis_size * inner + gather_idx * inner + ii;
                    out_data.push(self.data[src_flat]);
                }
            }
        }
        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();
        Tensor::new(&out_shape, &out_data)
    }

    /// Scatter add: adds `src` into a zero tensor at positions given by `indices`.
    ///
    /// # Errors
    ///
    /// Returns `MathError::OutOfRange` if any index is outside `[0, axis_size)`.
    /// Scatter-add along `axis`: adds `src` slices of `self` at the given
    /// integer indices.
    ///
    /// `src` must have the same shape as `self` except `shape[axis]` equals
    /// `indices.len()`; `out[.., indices[k], ..] += src[.., k, ..]`.
    /// Overlapping indices accumulate (true add, not overwrite).
    pub fn scatter_add(&self, axis: usize, indices: &[usize], src: &Tensor) -> MathResult<Tensor> {
        if axis >= self.shape.len() { return Err(MathError::InvalidArgument("scatter_add: axis out of range")); }
        let axis_size = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        for &si in indices {
            if si >= axis_size { return Err(MathError::OutOfRange); }
        }
        let mut expected = self.shape.clone();
        expected[axis] = indices.len();
        if src.shape != expected { return Err(MathError::DimensionMismatch); }
        let mut out = self.clone();
        for io in 0..outer {
            for (k, &scatter_idx) in indices.iter().enumerate() {
                for ii in 0..inner {
                    let src_flat = io * indices.len() * inner + k * inner + ii;
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

    /// Stack tensors along a **new** axis (torch `stack`, NumPy `np.stack`).
    ///
    /// All input tensors must share the same shape. The output shape is the
    /// input shape with `axis` inserted and sized to the number of tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor list is empty, shapes differ, or `axis`
    /// is out of range for the output.
    pub fn stack(tensors: &[Tensor], axis: usize) -> MathResult<Tensor> {
        if tensors.is_empty() {
            return Err(MathError::InvalidArgument("stack: empty input"));
        }
        let ndim = tensors[0].shape.len();
        if axis > ndim {
            return Err(MathError::InvalidArgument("stack: axis out of range"));
        }
        for t in &tensors[1..] {
            if t.shape != tensors[0].shape {
                return Err(MathError::DimensionMismatch);
            }
        }
        let outer: usize = tensors[0].shape[..axis].iter().product();
        let suffix: usize = tensors[0].shape[axis..].iter().product();
        let mut data = Vec::with_capacity(tensors[0].numel() * tensors.len());
        for o in 0..outer {
            for t in tensors {
                let start = o * suffix;
                data.extend_from_slice(&t.data[start..start + suffix]);
            }
        }
        let mut shape = tensors[0].shape.clone();
        shape.insert(axis, tensors.len());
        Ok(Tensor { shape, data })
    }

    /// Repeat the tensor `repeats[i]` times along dimension `i` (torch
    /// `Tensor.repeat`, NumPy `np.tile`). Fewer repeats than dimensions are
    /// left-aligned with `1`s. Tensors with zero-size dimensions yield an
    /// empty result (matching NumPy) instead of dividing by zero.
    ///
    /// # Errors
    ///
    /// Returns an error if more repeat counts are given than dimensions.
    pub fn repeat(&self, repeats: &[usize]) -> MathResult<Tensor> {
        if repeats.len() > self.shape.len() {
            return Err(MathError::InvalidArgument("repeat: too many repeat counts"));
        }
        let mut reps = vec![1usize; self.shape.len() - repeats.len()];
        reps.extend_from_slice(repeats);
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(&reps)
            .map(|(s, r)| s * r)
            .collect();
        let out_numel: usize = out_shape.iter().product();
        if out_numel == 0 || self.numel() == 0 {
            return Ok(Tensor {
                shape: out_shape,
                data: Vec::new(),
            });
        }
        let strides = self.strides();
        let out_strides = {
            let mut s = vec![1usize; out_shape.len()];
            for i in (0..out_shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * out_shape[i + 1];
            }
            s
        };
        let mut data = Vec::with_capacity(out_numel);
        #[allow(clippy::needless_range_loop)]
        for flat in 0..out_numel {
            let mut src_flat = 0;
            for i in 0..out_shape.len() {
                let coord = flat / out_strides[i] % out_shape[i];
                let src_dim = self.shape[i];
                src_flat += (if src_dim == 0 { 0 } else { coord % src_dim }) * strides[i];
            }
            data.push(self.data[src_flat]);
        }
        Ok(Tensor { shape: out_shape, data })
    }

    /// Reverse the tensor along the given dimensions (torch `flip`,
    /// NumPy `np.flip`).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is out of range.
    pub fn flip(&self, dims: &[usize]) -> MathResult<Tensor> {
        for &d in dims {
            if d >= self.shape.len() {
                return Err(MathError::InvalidArgument("flip: dimension out of range"));
            }
        }
        let strides = self.strides();
        let mut data = vec![0.0; self.numel()];
        #[allow(clippy::needless_range_loop)]
        for flat in 0..self.numel() {
            let mut src_flat = 0;
            for i in 0..self.shape.len() {
                let coord = flat / strides[i] % self.shape[i];
                let mapped = if dims.contains(&i) {
                    self.shape[i] - 1 - coord
                } else {
                    coord
                };
                src_flat += mapped * strides[i];
            }
            data[flat] = self.data[src_flat];
        }
        Ok(Tensor { shape: self.shape.clone(), data })
    }

    /// Numerically stable log-sum-exp along an axis (torch `logsumexp`).
    ///
    /// Computes `m + ln(Σ exp(x − m))` with `m = max(x)` along the axis,
    /// which avoids overflow for large inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of range.
    pub fn logsumexp(&self, axis: usize) -> MathResult<Tensor> {
        let maxv = self.max_axis(axis)?;
        // Re-insert a size-1 dimension at `axis` so the max tensor can be
        // broadcast back against `self` for the stable shift x − m. A
        // rank-1 tensor reduces to a scalar (stored as `[1]`), which is
        // already broadcastable and needs no insertion.
        let maxv_keep = if self.shape.len() == 1 {
            maxv.clone()
        } else {
            let mut keep_shape = maxv.shape.clone();
            keep_shape.insert(axis, 1);
            maxv.reshape(&keep_shape)?
        };
        let maxv_expanded = maxv_keep.broadcast_to(&self.shape)?;
        let shifted = self.sub(&maxv_expanded)?;
        let exps = shifted.exp();
        let sum_exp = axis_reduce(&exps, axis, |vals| vals.iter().sum())?;
        // Where max == −inf (all inputs −inf), exp(−inf − (−inf)) = NaN;
        // torch/NumPy define logsumexp of all −inf as −inf. Repair those.
        let mut out = sum_exp.ln().add(&maxv)?;
        for (i, &m) in maxv.data.iter().enumerate() {
            if m == f64::NEG_INFINITY {
                out.data[i] = f64::NEG_INFINITY;
            }
        }
        Ok(out)
    }

    /// Cumulative sum along an axis (torch `cumsum`, NumPy `np.cumsum`).
    /// The output has the same shape as the input.
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of range.
    pub fn cumsum_axis(&self, axis: usize) -> MathResult<Tensor> {
        if axis >= self.shape.len() {
            return Err(MathError::InvalidArgument("cumsum: axis out of range"));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut data = self.data.clone();
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0;
                for k in 0..axis_size {
                    let flat = o * axis_size * inner + k * inner + i;
                    acc += self.data[flat];
                    data[flat] = acc;
                }
            }
        }
        Ok(Tensor { shape: self.shape.clone(), data })
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
        if axis >= self.shape.len() {
            return Err(MathError::InvalidArgument("var_axis: axis out of range"));
        }
        let n = self.shape[axis] as f64;
        let mean = self.mean_axis(axis)?;
        // Re-insert a size-1 dim at `axis` so the mean can be broadcast back
        // against `self` for the centered subtraction (skip for rank-1, where
        // the reduced mean is already `[1]` and broadcastable).
        let mean_keep = if self.shape.len() == 1 {
            mean.clone()
        } else {
            let mut keep_shape = mean.shape.clone();
            keep_shape.insert(axis, 1);
            mean.reshape(&keep_shape)?
        };
        let mean_expanded = mean_keep.broadcast_to(&self.shape)?;
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
///
/// Dimensions are aligned from the right: a shape `[2]` against `[2, 3]` is
/// treated as `[1, 2]`. Incompatible shapes return
/// [`MathError::DimensionMismatch`] rather than panicking.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> MathResult<Vec<usize>> {
    let nd = a.len().max(b.len());
    let mut result = vec![0usize; nd];
    for i in 0..nd {
        let da = dim_at(a, i, nd);
        let db = dim_at(b, i, nd);
        if da == db || da == 1 || db == 1 {
            result[i] = da.max(db);
        } else {
            return Err(MathError::DimensionMismatch);
        }
    }
    Ok(result)
}

/// Returns the dimension of `shape` at broadcast position `i` of a `nd`-dim
/// frame (right-aligned), or `1` when `shape` has no such dimension.
fn dim_at(shape: &[usize], i: usize, nd: usize) -> usize {
    let offset = nd - shape.len();
    if i < offset { 1 } else { shape[i - offset] }
}

// ---------------------------------------------------------------------------
// Internal: 2-D matmul
// ---------------------------------------------------------------------------

/// Cache-blocked ikj matrix multiply. Blocking keeps the working set of `B`
/// tiles resident in L1, which is typically 1.5–3× faster than the naive
/// triple loop for large matrices while staying numerically identical.
fn matmul_2d(a: &Tensor, b: &Tensor) -> MathResult<Tensor> {
    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k1 != k2 { return Err(MathError::DimensionMismatch); }
    let mut data = vec![0.0; m * n];
    // Tile sizes chosen to fit comfortably in a typical 32–64 KiB L1 cache.
    const BK: usize = 64;
    const BJ: usize = 64;
    for i0 in (0..m).step_by(BK) {
        let imax = (i0 + BK).min(m);
        for p0 in (0..k1).step_by(BK) {
            let pmax = (p0 + BK).min(k1);
            for j0 in (0..n).step_by(BJ) {
                let jmax = (j0 + BJ).min(n);
                for i in i0..imax {
                    let arow = &a.data[i * k1..i * k1 + k1];
                    let drow = &mut data[i * n + j0..i * n + jmax];
                    for p in p0..pmax {
                        let av = arow[p];
                        if av == 0.0 {
                            continue;
                        }
                        let brow = &b.data[p * n + j0..p * n + jmax];
                        for (d, bv) in drow.iter_mut().zip(brow) {
                            *d += av * bv;
                        }
                    }
                }
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
        let aslice = &a.data[bi * m * k1..(bi + 1) * m * k1];
        let bslice = &b.data[bi * k1 * n..(bi + 1) * k1 * n];
        let out = &mut data[bi * m * n..(bi + 1) * m * n];
        for i in 0..m {
            let arow = &aslice[i * k1..(i + 1) * k1];
            let drow = &mut out[i * n..(i + 1) * n];
            for p in 0..k1 {
                let av = arow[p];
                if av == 0.0 {
                    continue;
                }
                let brow = &bslice[p * n..(p + 1) * n];
                for (d, bv) in drow.iter_mut().zip(brow) {
                    *d += av * bv;
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
        // Marsaglia polar regression: sample mean ≈ 0, variance ≈ 1.
        let t = Tensor::randn_seeded(&[200_000], 12345);
        let n = t.numel() as f64;
        let mean: f64 = t.data.iter().sum::<f64>() / n;
        let var: f64 = t.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean.abs()) < 0.02, "mean {mean} deviates from 0");
        assert!((var - 1.0).abs() < 0.05, "variance {var} deviates from 1");
        assert!(t.data.iter().any(|x| x.abs() > 1.0), "no samples beyond 1 sigma");
    }

    #[test]
    fn randn_threads_get_distinct_streams() {
        // Two threads must not share the identical (constant-seeded) stream.
        let h1 = std::thread::spawn(|| Tensor::randn(&[512]).data);
        let h2 = std::thread::spawn(|| Tensor::randn(&[512]).data);
        let a = h1.join().unwrap();
        let b = h2.join().unwrap();
        assert_ne!(a, b, "threads must not share a single RNG stream");
    }

    #[test]
    fn randn_seeded_is_reproducible() {
        let a = Tensor::randn_seeded(&[100], 42);
        let b = Tensor::randn_seeded(&[100], 42);
        assert_eq!(a.data, b.data);
        let c = Tensor::randn_seeded(&[100], 43);
        assert_ne!(a.data, c.data);
    }

    #[test]
    fn batch_norm_constant_input_is_stable() {
        // A constant channel would produce NaN variance under the naive
        // one-pass formula (E[x²] - μ² can be negative); it must stay finite.
        let t = Tensor::new(&[2, 3], &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        let bn = t.batch_norm(1e-5).unwrap();
        assert!(bn.data.iter().all(|x| x.is_finite()));
        // Normalized output of a constant input is 0 (mean-centered).
        assert!(bn.data.iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn batch_norm_4d_per_channel() {
        // [N=2, C=2, H=1, W=2]: channel 0 constant, channel 1 varying.
        let t = Tensor::new(&[2, 2, 1, 2], &[
            1.0, 1.0, 2.0, 4.0,
            1.0, 1.0, 6.0, 8.0,
        ]).unwrap();
        let bn = t.batch_norm(1e-5).unwrap();
        // Channel 0 (constant) → ~0; channel 1 mean ≈ 0 over N,H,W.
        let ch0: Vec<f64> = (0..2).map(|n| bn.data[n * 4]).collect();
        assert!(ch0.iter().all(|x| x.abs() < 1e-5));
        let ch1_mean: f64 = (0..2).map(|n| bn.data[n * 4 + 1]).sum::<f64>() / 2.0;
        assert!(ch1_mean.abs() < 1e-5);
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

    #[test]
    fn stack_along_new_axis() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let s = Tensor::stack(&[a.clone(), b.clone()], 0).unwrap();
        assert_eq!(s.shape, vec![2, 2, 2]);
        assert!((s.get(&[0, 0, 0]).unwrap() - 1.0).abs() < E);
        assert!((s.get(&[1, 1, 1]).unwrap() - 8.0).abs() < E);

        let s1 = Tensor::stack(&[a.clone(), b.clone()], 1).unwrap();
        assert_eq!(s1.shape, vec![2, 2, 2]);
        assert!((s1.get(&[0, 0, 0]).unwrap() - 1.0).abs() < E);
        assert!((s1.get(&[0, 1, 0]).unwrap() - 5.0).abs() < E);
    }

    #[test]
    fn stack_mismatched_shapes_error() {
        let a = Tensor::new(&[2], &[1.0, 2.0]).unwrap();
        let b = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
        assert!(Tensor::stack(&[a, b], 0).is_err());
        assert!(Tensor::stack(&[], 0).is_err());
    }

    #[test]
    fn repeat_tiles() {
        let t = Tensor::new(&[2], &[1.0, 2.0]).unwrap();
        let r = t.repeat(&[3]).unwrap();
        assert_eq!(r.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(r.shape, vec![6]);

        let m = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let r2 = m.repeat(&[2, 2]).unwrap();
        assert_eq!(r2.shape, vec![4, 4]);
        // Top-left quadrant unchanged; top-right repeats column 0.
        assert!((r2.get(&[0, 2]).unwrap() - 1.0).abs() < E);
        assert!((r2.get(&[2, 0]).unwrap() - 1.0).abs() < E);
    }

    #[test]
    fn repeat_zero_dim_does_not_panic() {
        // Regression: shape with a zero-size dim used to panic on `% 0`.
        let t = Tensor::new(&[0, 3], &[]).unwrap();
        let r = t.repeat(&[2, 2]).unwrap();
        assert_eq!(r.shape, vec![0, 6]);
        assert!(r.data.is_empty());
    }

    #[test]
    fn flip_reverses_axis() {
        let t = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let f = t.flip(&[0]).unwrap();
        assert_eq!(f.data, vec![4.0, 3.0, 2.0, 1.0]);

        let m = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let f1 = m.flip(&[1]).unwrap();
        assert!((f1.get(&[0, 0]).unwrap() - 3.0).abs() < E);
        let f0 = m.flip(&[0]).unwrap();
        assert!((f0.get(&[0, 0]).unwrap() - 4.0).abs() < E);
    }

    #[test]
    fn logsumexp_stability() {
        // Large values would overflow a naive exp; logsumexp must stay finite.
        let t = Tensor::new(&[2, 3], &[1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0]).unwrap();
        let l = t.logsumexp(1).unwrap();
        assert!(l.data.iter().all(|x| x.is_finite()));
        // logsumexp([1000,1001,1002]) ≈ 1002 + ln(1 + e^-1 + e^-2)
        let expected = 1002.0 + (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((l.data[0] - expected).abs() < 1e-9);
        let l0 = t.logsumexp(0).unwrap();
        assert_eq!(l0.shape, vec![3]);
    }

    #[test]
    fn logsumexp_all_neg_inf() {
        // Regression: all −inf inputs must yield −inf, not NaN.
        let t = Tensor::new(&[2], &[f64::NEG_INFINITY, f64::NEG_INFINITY]).unwrap();
        let l = t.logsumexp(0).unwrap();
        assert_eq!(l.data[0], f64::NEG_INFINITY);
    }

    #[test]
    fn cumsum_along_axis() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = t.cumsum_axis(1).unwrap();
        assert_eq!(c.data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
        let c0 = t.cumsum_axis(0).unwrap();
        assert_eq!(c0.data, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
        assert!(t.cumsum_axis(5).is_err());
    }

    #[test]
    fn broadcast_shapes_right_aligns() {
        // NumPy right-alignment semantics.
        assert_eq!(broadcast_shapes(&[3], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
        // Regression: previously panicked with index-out-of-bounds instead
        // of returning an error. `[2]` right-aligns to `[1, 2]` vs `[2, 3]`,
        // which is incompatible (2 vs 3).
        assert!(broadcast_shapes(&[2], &[2, 3]).is_err());
        assert!(broadcast_shapes(&[2, 3], &[2, 2]).is_err());
        assert!(broadcast_shapes(&[4], &[2, 3]).is_err());
    }

    #[test]
    fn checked_flat_accessors() {
        let mut t = Tensor::zeros(&[2, 2]);
        t.set_flat_checked(0, 7.0).unwrap();
        assert!((t.get_flat_checked(0).unwrap() - 7.0).abs() < E);
        assert!(t.get_flat_checked(99).is_err());
        assert!(t.set_flat_checked(99, 1.0).is_err());
    }

    #[test]
    fn blocked_matmul_matches_reference() {
        // Cross-check the cache-blocked path against a scalar triple loop.
        let a = Tensor::new(&[5, 7], &(0..35).map(|x| x as f64).collect::<Vec<_>>()).unwrap();
        let b = Tensor::new(&[7, 4], &(0..28).map(|x| (x as f64) * 0.5 - 1.0).collect::<Vec<_>>()).unwrap();
        let c = a.matmul(&b).unwrap();
        // Reference: element (2,3) = Σ_k a[2,k]·b[k,3]
        let mut ref_val = 0.0;
        for k in 0..7 {
            ref_val += a.data[2 * 7 + k] * b.data[k * 4 + 3];
        }
        assert!((c.data[2 * 4 + 3] - ref_val).abs() < 1e-9);
    }

    #[test]
    fn gather_integer_indices() {
        // [2, 3]: rows [1, 2, 3] and [4, 5, 6]; gather row index 1 twice.
        let x = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = x.gather(0, &[1, 1]).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
        assert!(x.gather(0, &[2]).is_err());
    }

    #[test]
    fn scatter_add_integer_indices() {
        let x = Tensor::zeros(&[2, 2]);
        let src = Tensor::new(&[3, 2], &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let out = x.scatter_add(0, &[0, 1, 0], &src).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        // Row 0 receives src rows 0 and 2; row 1 receives src row 1.
        assert_eq!(out.as_slice(), &[4.0, 4.0, 2.0, 2.0]);
        assert!(x.scatter_add(0, &[5], &src).is_err());
        let bad_src = Tensor::new(&[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        assert!(x.scatter_add(0, &[0], &bad_src).is_err());
    }
}













