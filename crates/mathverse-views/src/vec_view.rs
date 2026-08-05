//! Zero-copy borrowed vector view.

use core::ops::{Index, Range, RangeFrom, RangeInclusive};
use alloc::vec::Vec;

/// A borrowed, zero-copy view into a `f64` slice.
///
/// `VecView` is a lightweight wrapper around `&[f64]` that provides
/// the same API surface as an owned vector without any allocation.
#[derive(Debug, Clone, Copy)]
pub struct VecView<'a> {
    data: &'a [f64],
}

impl<'a> VecView<'a> {
    /// Create a view over a slice.
    pub fn new(data: &'a [f64]) -> Self {
        Self { data }
    }

    /// View over a subvector: `data[start..end]`.
    pub fn slice(&self, range: Range<usize>) -> VecView<'a> {
        VecView {
            data: &self.data[range],
        }
    }

    /// View over `data[start..]`.
    pub fn slice_from(&self, start: usize) -> VecView<'a> {
        VecView {
            data: &self.data[start..],
        }
    }

    /// View over `data[..=end]`.
    pub fn slice_to(&self, end: usize) -> VecView<'a> {
        VecView {
            data: &self.data[..=end],
        }
    }

    /// Length of the view.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index.
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Get a reference to the underlying slice.
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Copy elements into an owned `Vec<f64>`.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.to_vec()
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        self.sum() / self.len() as f64
    }

    /// Maximum element.
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum element.
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Dot product with another view.
    pub fn dot(&self, other: &VecView<'a>) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize to unit vector.
    pub fn normalized(&self) -> Vec<f64> {
        let n = self.norm();
        self.data.iter().map(|&x| x / n).collect()
    }

    /// Iterate over elements.
    pub fn iter(&self) -> core::slice::Iter<'a, f64> {
        self.data.iter()
    }

    /// Strided view: every `step`-th element.
    pub fn stride(&self, step: usize) -> Vec<f64> {
        self.data.iter().step_by(step).copied().collect()
    }

    /// Reverse view (copies into owned Vec).
    pub fn reversed(&self) -> Vec<f64> {
        self.data.iter().rev().copied().collect()
    }
}

impl<'a> From<&'a [f64]> for VecView<'a> {
    fn from(data: &'a [f64]) -> Self {
        Self::new(data)
    }
}

impl<'a> Index<usize> for VecView<'a> {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl<'a> Index<Range<usize>> for VecView<'a> {
    type Output = [f64];

    fn index(&self, range: Range<usize>) -> &[f64] {
        &self.data[range]
    }
}

impl<'a> Index<RangeFrom<usize>> for VecView<'a> {
    type Output = [f64];

    fn index(&self, range: RangeFrom<usize>) -> &[f64] {
        &self.data[range]
    }
}

impl<'a> Index<RangeInclusive<usize>> for VecView<'a> {
    type Output = [f64];

    fn index(&self, range: RangeInclusive<usize>) -> &[f64] {
        &self.data[range]
    }
}

impl<'a> IntoIterator for VecView<'a> {
    type Item = f64;
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter().copied()
    }
}

/// Owned vector created from a view.
impl<'a> From<VecView<'a>> for Vec<f64> {
    fn from(v: VecView<'a>) -> Self {
        v.data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = VecView::new(&data);
        assert_eq!(v.len(), 5);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn slice_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = VecView::new(&data);
        let sub = v.slice(1..4);
        assert_eq!(sub.len(), 3);
        assert_eq!(sub[0], 2.0);
        assert_eq!(sub[2], 4.0);
    }

    #[test]
    fn stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = VecView::new(&data);
        assert!((v.mean() - 3.0).abs() < 1e-12);
        assert_eq!(v.max(), 5.0);
        assert_eq!(v.min(), 1.0);
        assert!((v.norm() - (55.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let va = VecView::new(&a);
        let vb = VecView::new(&b);
        assert!((va.dot(&vb) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn stride_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = VecView::new(&data);
        assert_eq!(v.stride(2), vec![1.0, 3.0, 5.0]);
        assert_eq!(v.stride(3), vec![1.0, 4.0]);
    }

    #[test]
    fn iter() {
        let data = vec![1.0, 2.0, 3.0];
        let v = VecView::new(&data);
        let collected: Vec<f64> = v.iter().copied().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
    }
}
