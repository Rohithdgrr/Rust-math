use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::errors::{DataFrameError, DataFrameResult};

/// Row index for a DataFrame.
///
/// An index maps logical row positions to labels. It can be:
/// - `Range`: implicit 0..n (most memory-efficient)
/// - `Int64`: explicit i64 labels
/// - `Labels`: explicit string labels
/// - `Categorical`: fixed set of string categories with integer codes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// Implicit range: `start`, `start+step`, ..., `start + (len-1)*step`.
    Range {
        /// Starting value.
        start: i64,
        /// Step between values.
        step: i64,
        /// Number of elements.
        len: usize,
    },
    /// Explicit i64 labels.
    Int64(Vec<i64>),
    /// Explicit string labels.
    Labels(Vec<String>),
    /// Categorical index: a fixed set of category labels with per-row
    /// integer codes referencing into the categories vector.
    Categorical {
        /// The category labels, stored in order.
        categories: Vec<String>,
        /// Per-row integer codes (indices into `categories`).
        codes: Vec<i64>,
    },
    /// Hierarchical multi-level index. Each level is a vector of
    /// `String` labels, all of the same length. Levels are ordered
    /// outermost-first.
    MultiIndex(Vec<Vec<String>>),
}

impl Index {
    /// Creates a default integer range index: 0, 1, 2, ..., len-1.
    #[must_use]
    pub fn default_range(len: usize) -> Self {
        Self::Range {
            start: 0,
            step: 1,
            len,
        }
    }

    /// Creates a range index with custom start and step.
    #[must_use]
    pub fn range(start: i64, step: i64, len: usize) -> Self {
        Self::Range { start, step, len }
    }

    /// Creates an index from string labels.
    #[must_use]
    pub fn labels(labels: Vec<String>) -> Self {
        Self::Labels(labels)
    }

    /// Creates an index from i64 labels.
    #[must_use]
    pub fn int64(labels: Vec<i64>) -> Self {
        Self::Int64(labels)
    }

    /// Creates a categorical index from category labels and integer codes.
    ///
    /// # Errors
    ///
    /// Returns an error if any code is out of range for the given categories.
    pub fn categorical(categories: Vec<String>, codes: Vec<i64>) -> DataFrameResult<Self> {
        for (i, &code) in codes.iter().enumerate() {
            if code < 0 || code >= categories.len() as i64 {
                return Err(DataFrameError::InvalidOperation(alloc::format!(
                    "categorical code {code} at position {i} is out of range for {} categories",
                    categories.len()
                )));
            }
        }
        Ok(Self::Categorical { categories, codes })
    }

    /// Creates a categorical index by inferring categories from string labels.
    ///
    /// Each unique string becomes a category; the codes are assigned in
    /// sorted order of the categories (matching pandas `Categorical` with
    /// `ordered=False`).
    #[must_use]
    pub fn categorical_from_labels(labels: Vec<String>) -> Self {
        let mut cats: Vec<String> = labels.iter().cloned().collect();
        cats.sort();
        cats.dedup();
        let codes: Vec<i64> = labels
            .iter()
            .map(|label| cats.binary_search(label).unwrap_or(0) as i64)
            .collect();
        Self::Categorical {
            categories: cats,
            codes,
        }
    }

    /// Returns the number of elements in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Range { len, .. } => *len,
            Self::Int64(v) => v.len(),
            Self::Labels(v) => v.len(),
            Self::Categorical { codes, .. } => codes.len(),
            Self::MultiIndex(levels) => levels.first().map_or(0, |l| l.len()),
        }
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the label at the given position as an `i64`.
    ///
    /// For string labels, returns the position as i64.
    /// For categorical labels, returns the category code.
    #[must_use]
    pub fn get_int(&self, pos: usize) -> DataFrameResult<i64> {
        self.bounds_check(pos)?;
        match self {
            Self::Range { start, step, .. } => Ok(start + pos as i64 * step),
            Self::Int64(v) => Ok(v[pos]),
            Self::Labels(_) => Ok(pos as i64),
            Self::Categorical { codes, .. } => Ok(codes[pos]),
            Self::MultiIndex(_) => Ok(pos as i64),
        }
    }

    /// Returns the label at the given position as a string.
    ///
    /// For numeric labels, formats them as strings.
    /// For categorical labels, returns the category name.
    #[must_use]
    pub fn get_str(&self, pos: usize) -> DataFrameResult<String> {
        self.bounds_check(pos)?;
        match self {
            Self::Range { start, step, .. } => {
                let val = start + pos as i64 * step;
                Ok(alloc::format!("{val}"))
            }
            Self::Int64(v) => Ok(alloc::format!("{}", v[pos])),
            Self::Labels(v) => Ok(v[pos].clone()),
            Self::Categorical { categories, codes } => {
                let code = codes[pos];
                if code < 0 || code >= categories.len() as i64 {
                    Ok(alloc::format!("<invalid code {code}>"))
                } else {
                    Ok(categories[code as usize].clone())
                }
            }
            Self::MultiIndex(levels) => {
                let parts: alloc::vec::Vec<String> = levels
                    .iter()
                    .map(|level| level[pos].clone())
                    .collect();
                Ok(parts.join(", "))
            }
        }
    }

    /// Returns `true` if all labels are unique.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        match self {
            Self::Range { step, len, .. } => *step != 0 || *len <= 1,
            Self::Int64(v) => {
                let mut sorted = v.clone();
                sorted.sort_unstable();
                sorted.windows(2).all(|w| w[0] != w[1])
            }
            Self::Labels(v) => {
                let mut sorted = v.clone();
                sorted.sort();
                sorted.windows(2).all(|w| w[0] != w[1])
            }
            Self::Categorical { codes, .. } => {
                let mut sorted = codes.clone();
                sorted.sort_unstable();
                sorted.windows(2).all(|w| w[0] != w[1])
            }
            Self::MultiIndex(levels) => {
                if levels.is_empty() {
                    return true;
                }
                let len = levels[0].len();
                for i in 0..len {
                    for j in (i + 1)..len {
                        let mut same = true;
                        for level in levels {
                            if level[i] != level[j] {
                                same = false;
                                break;
                            }
                        }
                        if same {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    /// Creates a new index by selecting positions.
    #[must_use]
    pub fn select(&self, positions: &[usize]) -> DataFrameResult<Self> {
        match self {
            Self::Range { start, step, .. } => {
                let labels: Vec<i64> = positions
                    .iter()
                    .map(|&p| start + p as i64 * step)
                    .collect();
                Ok(Self::Int64(labels))
            }
            Self::Int64(v) => {
                let labels: DataFrameResult<Vec<i64>> =
                    positions.iter().map(|&p| v.get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: v.len() })).collect();
                Ok(Self::Int64(labels?))
            }
            Self::Labels(v) => {
                let labels: DataFrameResult<Vec<String>> =
                    positions.iter().map(|&p| v.get(p).cloned().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: v.len() })).collect();
                Ok(Self::Labels(labels?))
            }
            Self::Categorical { categories, codes } => {
                let new_codes: DataFrameResult<Vec<i64>> =
                    positions.iter().map(|&p| codes.get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: codes.len() })).collect();
                Ok(Self::Categorical {
                    categories: categories.clone(),
                    codes: new_codes?,
                })
            }
            Self::MultiIndex(levels) => {
                let new_levels: DataFrameResult<Vec<Vec<String>>> = levels
                    .iter()
                    .map(|level| {
                        positions
                            .iter()
                            .map(|&p| level.get(p).cloned().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: level.len() }))
                            .collect()
                    })
                    .collect();
                Ok(Self::MultiIndex(new_levels?))
            }
        }
    }

    /// Returns the category labels for a categorical index, or `None` if
    /// this is not a categorical index.
    #[must_use]
    pub fn categories(&self) -> Option<&[String]> {
        match self {
            Self::Categorical { categories, .. } => Some(categories),
            _ => None,
        }
    }

    /// Returns the integer codes for a categorical index, or `None` if
    /// this is not a categorical index.
    #[must_use]
    pub fn codes(&self) -> Option<&[i64]> {
        match self {
            Self::Categorical { codes, .. } => Some(codes),
            _ => None,
        }
    }

    fn bounds_check(&self, pos: usize) -> DataFrameResult<()> {
        let len = self.len();
        if pos >= len {
            Err(DataFrameError::IndexOutOfBounds { index: pos, length: len })
        } else {
            Ok(())
        }
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::default_range(0)
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range { start, step, len } => {
                write!(f, "Index(range({start}, +{step}, len={len}))")
            }
            Self::Int64(v) => {
                write!(f, "Index([")?;
                for (i, &val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val}")?;
                }
                write!(f, "])")
            }
            Self::Labels(v) => {
                write!(f, "Index([")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "`{val}`")?;
                }
                write!(f, "])")
            }
            Self::Categorical { categories, codes } => {
                write!(f, "CategoricalIndex([")?;
                for (i, &code) in codes.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if code >= 0 && (code as usize) < categories.len() {
                        write!(f, "`{}`", categories[code as usize])?;
                    } else {
                        write!(f, "<invalid>")?;
                    }
                }
                write!(f, "], categories=[")?;
                for (i, cat) in categories.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "`{cat}`")?;
                }
                write!(f, "])")
            }
            Self::MultiIndex(levels) => {
                write!(f, "MultiIndex([")?;
                if let Some(first_level) = levels.first() {
                    for i in 0..first_level.len() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "(")?;
                        for (j, level) in levels.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "`{}`", level[i])?;
                        }
                        write!(f, ")")?;
                    }
                }
                write!(f, "])")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_basic() {
        let idx = Index::categorical(
            vec!["low".into(), "med".into(), "high".into()],
            vec![0, 2, 1, 0, 2],
        )
        .unwrap();
        assert_eq!(idx.len(), 5);
        assert_eq!(idx.get_str(0).unwrap(), "low");
        assert_eq!(idx.get_str(1).unwrap(), "high");
        assert_eq!(idx.get_str(2).unwrap(), "med");
        assert!(idx.categories().is_some());
        assert_eq!(idx.categories().unwrap().len(), 3);
    }

    #[test]
    fn categorical_from_labels() {
        let labels = vec![
            "cherry".into(),
            "apple".into(),
            "banana".into(),
            "apple".into(),
        ];
        let idx = Index::categorical_from_labels(labels);
        // Categories sorted: ["apple", "banana", "cherry"]
        assert_eq!(idx.get_str(0).unwrap(), "cherry");
        assert_eq!(idx.get_str(1).unwrap(), "apple");
        assert_eq!(idx.get_str(2).unwrap(), "banana");
        assert_eq!(idx.get_str(3).unwrap(), "apple");
    }

    #[test]
    fn categorical_select() {
        let idx = Index::categorical(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0, 1, 2, 1],
        )
        .unwrap();
        let selected = idx.select(&[2, 0]).unwrap();
        assert_eq!(selected.get_str(0).unwrap(), "c");
        assert_eq!(selected.get_str(1).unwrap(), "a");
    }

    #[test]
    fn categorical_is_unique() {
        let idx = Index::categorical(
            vec!["a".into(), "b".into()],
            vec![0, 1],
        )
        .unwrap();
        assert!(idx.is_unique());
        let idx2 = Index::categorical(
            vec!["a".into(), "b".into()],
            vec![0, 0],
        )
        .unwrap();
        assert!(!idx2.is_unique());
    }

    #[test]
    fn categorical_out_of_range_errors() {
        let result = Index::categorical(
            vec!["a".into()],
            vec![0, 5],
        );
        assert!(result.is_err());
    }

    #[test]
    fn multi_index_basic() {
        let idx = Index::MultiIndex(vec![
            vec!["A".into(), "A".into(), "B".into(), "B".into()],
            vec!["one".into(), "two".into(), "one".into(), "two".into()],
        ]);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.get_str(0).unwrap(), "A, one");
        assert_eq!(idx.get_str(3).unwrap(), "B, two");
    }

    #[test]
    fn multi_index_is_unique() {
        let idx = Index::MultiIndex(vec![
            vec!["A".into(), "A".into()],
            vec!["one".into(), "two".into()],
        ]);
        assert!(idx.is_unique());
        let idx2 = Index::MultiIndex(vec![
            vec!["A".into(), "A".into()],
            vec!["one".into(), "one".into()],
        ]);
        assert!(!idx2.is_unique());
    }

    #[test]
    fn multi_index_select() {
        let idx = Index::MultiIndex(vec![
            vec!["A".into(), "B".into(), "C".into()],
            vec!["x".into(), "y".into(), "z".into()],
        ]);
        let selected = idx.select(&[2, 0]).unwrap();
        assert_eq!(selected.get_str(0).unwrap(), "C, z");
        assert_eq!(selected.get_str(1).unwrap(), "A, x");
    }
}
