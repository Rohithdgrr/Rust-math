use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::dtype::DType;
use crate::null::NullBitmap;

/// A named, typed, nullable column of data — the fundamental building block.
///
/// A `Series<T>` holds:
/// - A `name` (column label)
/// - A `data` vector of values
/// - An optional `NullBitmap` tracking which values are null
///
/// When `T` is `f64`, `i64`, `bool`, or `String`, the series can participate
/// in arithmetic, comparison, and statistical operations.
#[derive(Debug, Clone)]
pub struct Series<T> {
    name: String,
    data: Vec<T>,
    validity: Option<NullBitmap>,
}

impl<T> Series<T> {
    /// Creates a new series from raw data (no nulls).
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<T>) -> Self {
        Self {
            name: name.into(),
            data,
            validity: None,
        }
    }

    /// Creates a series with an explicit null bitmap.
    #[must_use]
    pub fn with_validity(name: impl Into<String>, data: Vec<T>, validity: NullBitmap) -> Self {
        debug_assert_eq!(
            data.len(),
            validity.len(),
            "data and validity must have the same length"
        );
        Self {
            name: name.into(),
            data,
            validity: if validity.all_valid_flag() {
                None
            } else {
                Some(validity)
            },
        }
    }

    /// Returns the series name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the underlying data vector.
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data vector.
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Consumes the series and returns the underlying data vector.
    #[must_use]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Returns the dtype of this series (must be implemented by concrete types).
    #[must_use]
    pub fn dtype(&self) -> DType
    where
        T: SeriesDType,
    {
        T::DTYPE
    }

    /// Returns a reference to the validity bitmap, if any nulls exist.
    #[must_use]
    pub fn validity(&self) -> Option<&NullBitmap> {
        self.validity.as_ref()
    }

    /// Returns `true` if the element at `pos` is null.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.len()`.
    #[must_use]
    pub fn is_null(&self, pos: usize) -> bool {
        assert!(
            pos < self.len(),
            "index {pos} out of bounds for length {}",
            self.len()
        );
        self.validity
            .as_ref()
            .map_or(false, |bm| bm.is_null(pos))
    }

    /// Returns the number of null elements.
    #[must_use]
    pub fn null_count(&self) -> usize {
        self.validity
            .as_ref()
            .map_or(0, |bm| bm.null_count())
    }

    /// Returns the number of non-null elements.
    #[must_use]
    pub fn non_null_count(&self) -> usize {
        self.len() - self.null_count()
    }

    /// Sets the element at `pos` to null.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.len()`.
    pub fn set_null(&mut self, pos: usize) {
        assert!(
            pos < self.len(),
            "index {pos} out of bounds for length {}",
            self.len()
        );
        if self.validity.is_none() {
            let mut bm = NullBitmap::all_valid(self.len());
            bm.set_null(pos);
            self.validity = Some(bm);
        } else {
            self.validity.as_mut().unwrap().set_null(pos);
        }
    }

    /// Sets the element at `pos` to valid (non-null).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.len()`.
    pub fn set_valid(&mut self, pos: usize) {
        if let Some(ref mut bm) = self.validity {
            bm.set_valid(pos);
            if bm.all_valid_flag() {
                self.validity = None;
            }
        }
    }

    /// Returns `true` if the series has no null values.
    #[must_use]
    pub fn has_no_nulls(&self) -> bool {
        self.validity.is_none() || self.validity.as_ref().map_or(false, |bm| bm.all_valid_flag())
    }

    /// Returns a new series with nulls removed (elements shifted left).
    #[must_use]
    pub fn drop_nulls(&self) -> Self
    where
        T: Clone,
    {
        let (data, validity) = self.filter_mask(&self.valid_mask());
        Self {
            name: self.name.clone(),
            data,
            validity,
        }
    }

    /// Returns the element at `pos`, or `None` if null.
    #[must_use]
    pub fn get(&self, pos: usize) -> Option<&T> {
        assert!(
            pos < self.len(),
            "index {pos} out of bounds for length {}",
            self.len()
        );
        if self.is_null(pos) {
            None
        } else {
            Some(&self.data[pos])
        }
    }

    /// Sets the value at `pos`.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.len()`.
    pub fn set(&mut self, pos: usize, value: T) {
        assert!(
            pos < self.len(),
            "index {pos} out of bounds for length {}",
            self.len()
        );
        self.data[pos] = value;
        self.set_valid(pos);
    }

    /// Returns a slice of the series from `start` to `end` (exclusive).
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Self
    where
        T: Clone,
    {
        assert!(
            start <= end && end <= self.len(),
            "slice bounds [{start}, {end}) out of range for length {}",
            self.len()
        );
        Self {
            name: self.name.clone(),
            data: self.data[start..end].to_vec(),
            validity: self.validity.as_ref().map(|bm| {
                let bits = bm.as_bytes()[start / 8..=end / 8].to_vec();
                NullBitmap::from_bytes(bits, end - start)
            }),
        }
    }

    /// Returns the first `n` elements.
    #[must_use]
    pub fn head(&self, n: usize) -> Self
    where
        T: Clone,
    {
        let n = n.min(self.len());
        self.slice(0, n)
    }

    /// Returns the last `n` elements.
    #[must_use]
    pub fn tail(&self, n: usize) -> Self
    where
        T: Clone,
    {
        let n = n.min(self.len());
        self.slice(self.len() - n, self.len())
    }

    /// Returns an iterator over `(index, Option<&T>)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&T>)> {
        self.data
            .iter()
            .enumerate()
            .map(move |(i, v)| (i, Some(v).filter(|_| !self.is_null(i))))
    }

    /// Creates a new series by applying a function to each element.
    #[must_use]
    pub fn map<U, F>(&self, f: F) -> Series<U>
    where
        F: Fn(&T) -> U,
        U: Clone,
    {
        Series {
            name: self.name.clone(),
            data: self.data.iter().map(f).collect(),
            validity: self.validity.clone(),
        }
    }

    /// Renames the series, returning a new series with the given name.
    #[must_use]
    pub fn rename(&self, name: impl Into<String>) -> Self
    where
        T: Clone,
    {
        Self {
            name: name.into(),
            data: self.data.clone(),
            validity: self.validity.clone(),
        }
    }

    /// Renames the series in place.
    pub fn rename_mut(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Returns the indices of non-null values.
    #[must_use]
    pub fn non_null_indices(&self) -> Vec<usize> {
        if self.has_no_nulls() {
            (0..self.len()).collect()
        } else {
            (0..self.len())
                .filter(|&i| !self.is_null(i))
                .collect()
        }
    }

    /// Returns a boolean vector where `true` means the element is non-null.
    #[must_use]
    pub fn valid_mask(&self) -> Vec<bool> {
        if self.has_no_nulls() {
            vec![true; self.len()]
        } else {
            (0..self.len()).map(|i| !self.is_null(i)).collect()
        }
    }

    /// Filters elements by a boolean mask, returning (data, validity).
    fn filter_mask(&self, mask: &[bool]) -> (Vec<T>, Option<NullBitmap>)
    where
        T: Clone,
    {
        debug_assert_eq!(mask.len(), self.len());
        let mut data = Vec::with_capacity(mask.iter().filter(|&&m| m).count());
        let mut has_nulls = false;
        let mut validities = Vec::with_capacity(data.capacity());

        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                data.push(self.data[i].clone());
                let is_valid = !self.is_null(i);
                if is_valid {
                    validities.push(false);
                } else {
                    has_nulls = true;
                    validities.push(true);
                }
            }
        }

        let validity = if has_nulls {
            Some(NullBitmap::from_bools(&validities))
        } else {
            None
        };

        (data, validity)
    }
}

/// Trait mapping Rust types to their `DType`.
pub trait SeriesDType {
    /// The `DType` for this Rust type.
    const DTYPE: DType;
}

impl SeriesDType for f64 {
    const DTYPE: DType = DType::Float64;
}

impl SeriesDType for f32 {
    const DTYPE: DType = DType::Float32;
}

impl SeriesDType for i64 {
    const DTYPE: DType = DType::Int64;
}

impl SeriesDType for i32 {
    const DTYPE: DType = DType::Int32;
}

impl SeriesDType for bool {
    const DTYPE: DType = DType::Bool;
}

impl SeriesDType for String {
    const DTYPE: DType = DType::Utf8;
}

impl<T: fmt::Display + SeriesDType> fmt::Display for Series<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Series(\"{}\" [{}]): ", self.name, T::DTYPE.name())?;
        let max_display = 5;
        let len = self.len();
        for i in 0..len.min(max_display) {
            if i > 0 {
                write!(f, ", ")?;
            }
            if self.is_null(i) {
                write!(f, "null")?;
            } else {
                write!(f, "{}", self.data[i])?;
            }
        }
        if len > max_display {
            write!(f, ", ... ({len} total)")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_series() {
        let s = Series::new("prices", vec![1.0, 2.0, 3.0]);
        assert_eq!(s.name(), "prices");
        assert_eq!(s.len(), 3);
        assert!(s.has_no_nulls());
        assert_eq!(s.null_count(), 0);
    }

    #[test]
    fn series_with_nulls() {
        let mut s = Series::new("x", vec![1.0, 2.0, 3.0]);
        s.set_null(1);
        assert!(s.is_null(1));
        assert!(!s.has_no_nulls());
        assert_eq!(s.null_count(), 1);
    }

    #[test]
    fn series_drop_nulls() {
        let mut s = Series::new("x", vec![1.0, 2.0, 3.0, 4.0]);
        s.set_null(1);
        s.set_null(3);
        let cleaned = s.drop_nulls();
        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned.data(), &[1.0, 3.0]);
        assert!(cleaned.has_no_nulls());
    }

    #[test]
    fn series_head_tail() {
        let s = Series::new("x", vec![1, 2, 3, 4, 5]);
        assert_eq!(s.head(3).data(), &[1, 2, 3]);
        assert_eq!(s.tail(2).data(), &[4, 5]);
    }

    #[test]
    fn series_map() {
        let s = Series::new("x", vec![1.0, 2.0, 3.0]);
        let doubled = s.map(|&v| v * 2.0);
        assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
        assert_eq!(doubled.name(), "x");
    }
}
