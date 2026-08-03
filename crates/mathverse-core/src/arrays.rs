//! Const-generic fixed-size numeric arrays.
//!
//! A `no_std`-friendly, zero-dependency alternative to heap-backed vectors for
//! embedded and fixed-size workloads. Backed by a plain `[T; N]`, all methods
//! are `const`-compatible where possible and require only `core`.
//!
//! ```text
//!            ┌───────────────────────────┐
//!            │  Array<T, const N: usize> │
//!            └───────────────────────────┘
//!                  │ T: Num
//!            ┌─────┴─────┐
//!            │ dot/sum   │  elementwise via core ops
//!            └───────────┘
//!                  │ T: Real
//!            ┌─────┴─────┐
//!            │ l1/l2/∞   │  norms, normalize
//!            └───────────┘
//! ```

use crate::traits::{Num, Real};
use core::ops::{Add, Div, Mul, Sub};

/// Fixed-size numeric vector backed by `[T; N]`.
///
/// Unlike a heap-backed `Vec`, the length is part of the type, so no allocation
/// or bounds checks are needed and the whole value can live on the stack or in a
/// static. Works in `no_std` environments.
///
/// # Example
/// ```
/// use mathverse_core::arrays::Array;
///
/// let a = Array::from([1.0f64, 2.0, 3.0]);
/// let b = Array::ones();
/// let c = a + b;
/// assert_eq!(c, Array::from([2.0, 3.0, 4.0]));
/// assert!((a.dot(&b) - 6.0).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Array<T, const N: usize> {
    data: [T; N],
}

impl<T: Num, const N: usize> Default for Array<T, N> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<T: Num, const N: usize> Array<T, N> {
    /// Construct from a raw array.
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Build from a mapping function `f(i)`.
    ///
    /// ```
    /// use mathverse_core::arrays::Array;
    /// let a = Array::from_fn(|i| i as f64 * 2.0);
    /// assert_eq!(a, Array::from([0.0, 2.0, 4.0]));
    /// ```
    pub fn from_fn<F: FnMut(usize) -> T>(mut f: F) -> Self {
        let mut data = [T::zero(); N];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = f(i);
        }
        Self { data }
    }

    /// All elements set to `T::zero()`.
    ///
    /// ```
    /// use mathverse_core::arrays::Array;
    /// assert_eq!(Array::<f64, 3>::zeros(), Array::from([0.0; 3]));
    /// ```
    pub fn zeros() -> Self {
        Self::from_fn(|_| T::zero())
    }

    /// All elements set to `T::one()`.
    pub fn ones() -> Self {
        Self::from_fn(|_| T::one())
    }

    /// The backing slice (read-only).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The backing mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get element `i` (panics if out of bounds, same as slice indexing).
    pub fn get(&self, i: usize) -> T {
        self.data[i]
    }

    /// Set element `i` in place.
    pub fn set(&mut self, i: usize, value: T) {
        self.data[i] = value;
    }

    /// Element at index `i` without panicking.
    pub fn try_get(&self, i: usize) -> Option<T> {
        self.data.get(i).copied()
    }

    /// Sum of all elements.
    pub fn sum(&self) -> T {
        let mut acc = T::zero();
        for &x in &self.data {
            acc = acc + x;
        }
        acc
    }

    /// Dot (scalar) product with another array of the same length.
    ///
    /// ```
    /// use mathverse_core::arrays::Array;
    /// let a = Array::from([1.0f64, 2.0, 3.0]);
    /// let b = Array::from([4.0, 5.0, 6.0]);
    /// assert!((a.dot(&b) - 32.0).abs() < 1e-12);
    /// ```
    pub fn dot(&self, other: &Self) -> T {
        let mut acc = T::zero();
        for i in 0..N {
            acc = acc + self.data[i] * other.data[i];
        }
        acc
    }

    /// Element-wise apply of `f`.
    pub fn map<U: Num, F: FnMut(T) -> U>(&self, mut f: F) -> Array<U, N> {
        Array::from_fn(|i| f(self.data[i]))
    }

    /// Zip with another array and combine with `f`.
    pub fn zip_with<U: Num, V: Num, F: FnMut(T, U) -> V>(
        &self,
        other: &Array<U, N>,
        mut f: F,
    ) -> Array<V, N> {
        Array::from_fn(|i| f(self.data[i], other.data[i]))
    }

    /// Element-wise `a * b`.
    pub fn component_mul(&self, other: &Self) -> Self {
        self.zip_with(other, |x, y| x * y)
    }

    /// Element-wise `a / b`.
    pub fn component_div(&self, other: &Self) -> Self
    where
        T: Div<Output = T>,
    {
        self.zip_with(other, |x, y| x / y)
    }

    /// Cumulative sum (prefix sum).
    ///
    /// ```
    /// use mathverse_core::arrays::Array;
    /// let a = Array::from([1.0f64, 2.0, 3.0]);
    /// assert_eq!(a.cumulative_sum(), Array::from([1.0, 3.0, 6.0]));
    /// ```
    pub fn cumulative_sum(&self) -> Self {
        let mut acc = T::zero();
        let mut out = [T::zero(); N];
        for (i, slot) in out.iter_mut().enumerate() {
            acc = acc + self.data[i];
            *slot = acc;
        }
        Self { data: out }
    }

    /// Reverse the elements.
    pub fn reversed(&self) -> Self {
        let mut out = [T::zero(); N];
        for i in 0..N {
            out[i] = self.data[N - 1 - i];
        }
        Self { data: out }
    }

    /// True when every element compares equal to the corresponding one in `other`.
    pub fn all_eq(&self, other: &Self) -> bool
    where
        T: PartialEq,
    {
        self.data == other.data
    }
}

impl<T: Num, const N: usize> Array<T, N> {
    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        self.zip_with(other, |x, y| x + y)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        self.zip_with(other, |x, y| x - y)
    }

    /// Scalar multiply: `self * s`.
    pub fn scalar_mul(&self, s: T) -> Self {
        self.map(|x| x * s)
    }

    /// Scalar addition: `self + s`.
    pub fn scalar_add(&self, s: T) -> Self {
        self.map(|x| x + s)
    }

    /// Divide every element by `s`.
    pub fn scalar_div(&self, s: T) -> Self
    where
        T: Div<Output = T>,
    {
        self.map(|x| x / s)
    }
}

impl<T: Real, const N: usize> Array<T, N> {
    /// Euclidean (L2) norm.
    ///
    /// ```
    /// use mathverse_core::arrays::Array;
    /// let a = Array::from([3.0f64, 4.0]);
    /// assert!((a.l2_norm() - 5.0).abs() < 1e-12);
    /// ```
    pub fn l2_norm(&self) -> T {
        self.dot(self).sqrt()
    }

    /// Manhattan (L1) norm.
    pub fn l1_norm(&self) -> T {
        let mut acc = T::zero();
        for &x in &self.data {
            acc = acc + x.abs();
        }
        acc
    }

    /// Maximum (L∞ / Chebyshev) norm.
    pub fn max_norm(&self) -> T {
        let mut acc = T::zero();
        for &x in &self.data {
            let ax = x.abs();
            if ax > acc {
                acc = ax;
            }
        }
        acc
    }

    /// Normalize to unit length; returns `None` for a zero vector.
    pub fn try_normalize(&self) -> Option<Self> {
        let n = self.l2_norm();
        if n == T::zero() {
            None
        } else {
            Some(self.scalar_div(n))
        }
    }

    /// Normalize to unit length; a zero vector stays zero.
    pub fn normalize(&self) -> Self {
        self.try_normalize().unwrap_or_else(Self::zeros)
    }

    /// Mean of the elements.
    pub fn mean(&self) -> T {
        self.sum() / T::from_f64(N as f64)
    }
}

impl<T, const N: usize> From<[T; N]> for Array<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> From<Array<T, N>> for [T; N] {
    fn from(arr: Array<T, N>) -> Self {
        arr.data
    }
}

impl<T, const N: usize> AsRef<[T]> for Array<T, N> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T, const N: usize> core::ops::Index<usize> for Array<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Num, const N: usize> Add for Array<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(&self, &rhs)
    }
}

impl<T: Num, const N: usize> Sub for Array<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(&self, &rhs)
    }
}

impl<T: Num, const N: usize> Mul<T> for Array<T, N> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

impl<T: Num + Div<Output = T>, const N: usize> Div<T> for Array<T, N> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.scalar_div(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let a = Array::from([1.0f64, 2.0, 3.0]);
        let b = Array::from([4.0, 5.0, 6.0]);
        assert_eq!(a + b, Array::from([5.0, 7.0, 9.0]));
        assert_eq!(a.dot(&b), 32.0);
        assert_eq!(a.sum(), 6.0);
    }

    #[test]
    fn const_generic_sizes() {
        let a: Array<u32, 4> = Array::ones();
        assert_eq!(a.sum(), 4);
        let b: Array<u32, 8> = Array::zeros();
        assert_eq!(b.sum(), 0);
    }

    #[test]
    fn norms() {
        let a = Array::from([3.0f64, 4.0]);
        assert!((a.l2_norm() - 5.0).abs() < 1e-12);
        assert_eq!(a.l1_norm(), 7.0);
        assert_eq!(a.max_norm(), 4.0);
        assert!((a.try_normalize().unwrap().l2_norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn int_arrays() {
        let a = Array::from([1i32, 2, 3]);
        let b = Array::from([10, 20, 30]);
        assert_eq!(a.component_mul(&b), Array::from([10, 40, 90]));
        assert_eq!(a.cumulative_sum(), Array::from([1, 3, 6]));
    }

    #[test]
    fn normalize_zero() {
        let z: Array<f64, 3> = Array::zeros();
        assert!(z.try_normalize().is_none());
        assert_eq!(z.normalize(), z);
    }
}
