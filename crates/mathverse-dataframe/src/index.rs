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

    /// Returns the number of elements in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Range { len, .. } => *len,
            Self::Int64(v) => v.len(),
            Self::Labels(v) => v.len(),
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
    #[must_use]
    pub fn get_int(&self, pos: usize) -> DataFrameResult<i64> {
        self.bounds_check(pos)?;
        match self {
            Self::Range { start, step, .. } => Ok(start + pos as i64 * step),
            Self::Int64(v) => Ok(v[pos]),
            Self::Labels(_) => Ok(pos as i64),
        }
    }

    /// Returns the label at the given position as a string.
    ///
    /// For numeric labels, formats them as strings.
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
        }
    }
}
