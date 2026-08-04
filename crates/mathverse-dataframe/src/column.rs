use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::dtype::DType;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

/// A type- erased column that can hold any supported `Series<T>`.
#[derive(Debug, Clone)]
pub enum AnyColumn {
    /// 64-bit floating-point column.
    Float64(Series<f64>),
    /// 32-bit floating-point column.
    Float32(Series<f32>),
    /// 64-bit signed integer column.
    Int64(Series<i64>),
    /// 32-bit signed integer column.
    Int32(Series<i32>),
    /// Boolean column.
    Bool(Series<bool>),
    /// String column.
    Utf8(Series<String>),
}

impl AnyColumn {
    /// Returns the column name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Float64(s) => s.name(),
            Self::Float32(s) => s.name(),
            Self::Int64(s) => s.name(),
            Self::Int32(s) => s.name(),
            Self::Bool(s) => s.name(),
            Self::Utf8(s) => s.name(),
        }
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(s) => s.len(),
            Self::Float32(s) => s.len(),
            Self::Int64(s) => s.len(),
            Self::Int32(s) => s.len(),
            Self::Bool(s) => s.len(),
            Self::Utf8(s) => s.len(),
        }
    }

    /// Returns `true` if the column is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the `DType` of this column.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Float64(_) => DType::Float64,
            Self::Float32(_) => DType::Float32,
            Self::Int64(_) => DType::Int64,
            Self::Int32(_) => DType::Int32,
            Self::Bool(_) => DType::Bool,
            Self::Utf8(_) => DType::Utf8,
        }
    }

    /// Returns `true` if the element at `pos` is null.
    #[must_use]
    pub fn is_null(&self, pos: usize) -> bool {
        match self {
            Self::Float64(s) => s.is_null(pos),
            Self::Float32(s) => s.is_null(pos),
            Self::Int64(s) => s.is_null(pos),
            Self::Int32(s) => s.is_null(pos),
            Self::Bool(s) => s.is_null(pos),
            Self::Utf8(s) => s.is_null(pos),
        }
    }

    /// Returns the number of null values.
    #[must_use]
    pub fn null_count(&self) -> usize {
        match self {
            Self::Float64(s) => s.null_count(),
            Self::Float32(s) => s.null_count(),
            Self::Int64(s) => s.null_count(),
            Self::Int32(s) => s.null_count(),
            Self::Bool(s) => s.null_count(),
            Self::Utf8(s) => s.null_count(),
        }
    }

    /// Sets the element at `pos` to null.
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.len()`.
    pub fn set_null(&mut self, pos: usize) {
        match self {
            Self::Float64(s) => s.set_null(pos),
            Self::Float32(s) => s.set_null(pos),
            Self::Int64(s) => s.set_null(pos),
            Self::Int32(s) => s.set_null(pos),
            Self::Bool(s) => s.set_null(pos),
            Self::Utf8(s) => s.set_null(pos),
        }
    }

    /// Renames the column in place.
    pub fn rename_mut(&mut self, name: impl Into<String>) {
        match self {
            Self::Float64(s) => s.rename_mut(name),
            Self::Float32(s) => s.rename_mut(name),
            Self::Int64(s) => s.rename_mut(name),
            Self::Int32(s) => s.rename_mut(name),
            Self::Bool(s) => s.rename_mut(name),
            Self::Utf8(s) => s.rename_mut(name),
        }
    }

    /// Returns a new column with the given name, cloned data, and same validity.
    #[must_use]
    pub fn with_name(&self, name: impl Into<String>) -> Self {
        let mut col = self.clone();
        col.rename_mut(name);
        col
    }

    /// Returns a reference to the f64 series, or an error if the type doesn't match.
    #[must_use]
    pub fn as_f64(&self) -> DataFrameResult<&Series<f64>> {
        match self {
            Self::Float64(s) => Ok(s),
            other => Err(DataFrameError::TypeMismatch {
                expected: "f64",
                actual: other.dtype().name(),
            }),
        }
    }

    /// Returns a reference to the i64 series, or an error if the type doesn't match.
    #[must_use]
    pub fn as_i64(&self) -> DataFrameResult<&Series<i64>> {
        match self {
            Self::Int64(s) => Ok(s),
            other => Err(DataFrameError::TypeMismatch {
                expected: "i64",
                actual: other.dtype().name(),
            }),
        }
    }

    /// Returns a reference to the bool series, or an error if the type doesn't match.
    #[must_use]
    pub fn as_bool(&self) -> DataFrameResult<&Series<bool>> {
        match self {
            Self::Bool(s) => Ok(s),
            other => Err(DataFrameError::TypeMismatch {
                expected: "bool",
                actual: other.dtype().name(),
            }),
        }
    }

    /// Returns a reference to the string series, or an error if the type doesn't match.
    #[must_use]
    pub fn as_utf8(&self) -> DataFrameResult<&Series<String>> {
        match self {
            Self::Utf8(s) => Ok(s),
            other => Err(DataFrameError::TypeMismatch {
                expected: "str",
                actual: other.dtype().name(),
            }),
        }
    }

    /// Returns a reference to the inner series for downcasting.
    ///
    /// This is used with `core::any::Any::downcast_ref` to get a typed
    /// reference to the inner `Series<T>`.
    #[must_use]
    pub fn as_any(&self) -> &dyn core::any::Any {
        match self {
            Self::Float64(s) => s,
            Self::Float32(s) => s,
            Self::Int64(s) => s,
            Self::Int32(s) => s,
            Self::Bool(s) => s,
            Self::Utf8(s) => s,
        }
    }

    /// Collects the f64 values into a new series, converting from this column's type.
    ///
    /// Integer types are cast to f64. Bool is cast to 0.0/1.0. Utf8 is not convertible.
    #[must_use]
    pub fn to_f64(&self) -> DataFrameResult<Series<f64>> {
        let name = self.name().to_string();
        let data = match self {
            Self::Float64(s) => s.data().clone(),
            Self::Float32(s) => s.data().iter().map(|&v| f64::from(v)).collect(),
            Self::Int64(s) => s.data().iter().map(|&v| v as f64).collect(),
            Self::Int32(s) => s.data().iter().map(|&v| f64::from(v)).collect(),
            Self::Bool(s) => s.data().iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
            Self::Utf8(_) => {
                return Err(DataFrameError::InvalidOperation(
                    "cannot convert Utf8 to f64".to_string(),
                ))
            }
        };
        Ok(Series::new(name, data))
    }

    /// Selects rows by position, returning a new `AnyColumn`.
    #[must_use]
    pub fn select_rows(&self, positions: &[usize]) -> DataFrameResult<Self> {
        match self {
            Self::Float64(s) => {
                let data: DataFrameResult<Vec<f64>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Float64(Series::new(s.name(), data?)))
            }
            Self::Float32(s) => {
                let data: DataFrameResult<Vec<f32>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Float32(Series::new(s.name(), data?)))
            }
            Self::Int64(s) => {
                let data: DataFrameResult<Vec<i64>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Int64(Series::new(s.name(), data?)))
            }
            Self::Int32(s) => {
                let data: DataFrameResult<Vec<i32>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Int32(Series::new(s.name(), data?)))
            }
            Self::Bool(s) => {
                let data: DataFrameResult<Vec<bool>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).copied().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Bool(Series::new(s.name(), data?)))
            }
            Self::Utf8(s) => {
                let data: DataFrameResult<Vec<String>> = positions
                    .iter()
                    .map(|&p| s.data().get(p).cloned().ok_or(DataFrameError::IndexOutOfBounds { index: p, length: s.len() }))
                    .collect();
                Ok(Self::Utf8(Series::new(s.name(), data?)))
            }
        }
    }

    /// Casts this column to a different dtype.
    #[must_use]
    pub fn cast(&self, target: DType) -> DataFrameResult<Self> {
        match target {
            DType::Float64 => Ok(Self::Float64(self.to_f64()?)),
            DType::Int64 => match self {
                Self::Int64(s) => Ok(Self::Int64(s.clone())),
                Self::Float64(s) => {
                    let data: Vec<i64> = s.data().iter().map(|&v| v as i64).collect();
                    Ok(Self::Int64(Series::new(s.name(), data)))
                }
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to i64",
                    other.dtype()
                ))),
            },
            DType::Bool => match self {
                Self::Bool(s) => Ok(Self::Bool(s.clone())),
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to bool",
                    other.dtype()
                ))),
            },
            DType::Utf8 => match self {
                Self::Utf8(s) => Ok(Self::Utf8(s.clone())),
                Self::Int64(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    Ok(Self::Utf8(Series::new(s.name(), data)))
                }
                Self::Float64(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    Ok(Self::Utf8(Series::new(s.name(), data)))
                }
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to str",
                    other.dtype()
                ))),
            },
            _ => Err(DataFrameError::InvalidOperation(alloc::format!(
                "casting to {target} is not supported"
            ))),
        }
    }
}

impl fmt::Display for AnyColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_display = 5;
        let len = self.len();
        write!(f, "Column(\"{}\" [{}], len={len}): ", self.name(), self.dtype())?;
        for i in 0..len.min(max_display) {
            if i > 0 {
                write!(f, ", ")?;
            }
            if self.is_null(i) {
                write!(f, "null")?;
            } else {
                match self {
                    Self::Float64(s) => write!(f, "{}", s.data()[i])?,
                    Self::Float32(s) => write!(f, "{}", s.data()[i])?,
                    Self::Int64(s) => write!(f, "{}", s.data()[i])?,
                    Self::Int32(s) => write!(f, "{}", s.data()[i])?,
                    Self::Bool(s) => write!(f, "{}", s.data()[i])?,
                    Self::Utf8(s) => write!(f, "`{}`", s.data()[i])?,
                }
            }
        }
        if len > max_display {
            write!(f, ", ...")?;
        }
        Ok(())
    }
}

impl From<Series<f64>> for AnyColumn {
    fn from(s: Series<f64>) -> Self {
        Self::Float64(s)
    }
}

impl From<Series<f32>> for AnyColumn {
    fn from(s: Series<f32>) -> Self {
        Self::Float32(s)
    }
}

impl From<Series<i64>> for AnyColumn {
    fn from(s: Series<i64>) -> Self {
        Self::Int64(s)
    }
}

impl From<Series<i32>> for AnyColumn {
    fn from(s: Series<i32>) -> Self {
        Self::Int32(s)
    }
}

impl From<Series<bool>> for AnyColumn {
    fn from(s: Series<bool>) -> Self {
        Self::Bool(s)
    }
}

impl From<Series<String>> for AnyColumn {
    fn from(s: Series<String>) -> Self {
        Self::Utf8(s)
    }
}

impl From<Vec<f64>> for AnyColumn {
    fn from(v: Vec<f64>) -> Self {
        Self::Float64(Series::new("", v))
    }
}

impl From<Vec<i64>> for AnyColumn {
    fn from(v: Vec<i64>) -> Self {
        Self::Int64(Series::new("", v))
    }
}

impl From<Vec<bool>> for AnyColumn {
    fn from(v: Vec<bool>) -> Self {
        Self::Bool(Series::new("", v))
    }
}

impl From<Vec<String>> for AnyColumn {
    fn from(v: Vec<String>) -> Self {
        Self::Utf8(Series::new("", v))
    }
}
