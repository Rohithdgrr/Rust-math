use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::dtype::DType;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::null::NullBitmap;
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
    /// Date column (days since Unix epoch).
    Date(Series<i64>),
    /// DateTime column (microseconds since Unix epoch).
    DateTime(Series<i64>),
    /// Duration column (microseconds).
    Duration(Series<i64>),
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
            Self::Date(s) => s.name(),
            Self::DateTime(s) => s.name(),
            Self::Duration(s) => s.name(),
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
            Self::Date(s) => s.len(),
            Self::DateTime(s) => s.len(),
            Self::Duration(s) => s.len(),
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
            Self::Date(_) => DType::Date,
            Self::DateTime(_) => DType::DateTime,
            Self::Duration(_) => DType::Duration,
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
            Self::Date(s) => s.is_null(pos),
            Self::DateTime(s) => s.is_null(pos),
            Self::Duration(s) => s.is_null(pos),
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
            Self::Date(s) => s.null_count(),
            Self::DateTime(s) => s.null_count(),
            Self::Duration(s) => s.null_count(),
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
            Self::Date(s) => s.set_null(pos),
            Self::DateTime(s) => s.set_null(pos),
            Self::Duration(s) => s.set_null(pos),
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
            Self::Date(s) => s.rename_mut(name),
            Self::DateTime(s) => s.rename_mut(name),
            Self::Duration(s) => s.rename_mut(name),
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

    /// Returns the string representation of the value at `pos`.
    ///
    /// For `Utf8` columns this is the string itself. For numeric types
    /// the value is formatted. For `Date`/`DateTime`/`Duration` the
    /// raw i64 is shown.
    ///
    /// # Errors
    ///
    /// Returns `IndexOutOfBounds` if `pos >= len`.
    pub fn get_str(&self, pos: usize) -> DataFrameResult<String> {
        if pos >= self.len() {
            return Err(DataFrameError::IndexOutOfBounds { index: pos, length: self.len() });
        }
        match self {
            Self::Float64(s) => Ok(alloc::format!("{}", s.data()[pos])),
            Self::Float32(s) => Ok(alloc::format!("{}", s.data()[pos])),
            Self::Int64(s) => Ok(alloc::format!("{}", s.data()[pos])),
            Self::Int32(s) => Ok(alloc::format!("{}", s.data()[pos])),
            Self::Bool(s) => Ok(alloc::format!("{}", s.data()[pos])),
            Self::Utf8(s) => Ok(s.data()[pos].clone()),
            Self::Date(s) | Self::DateTime(s) | Self::Duration(s) => {
                Ok(alloc::format!("{}", s.data()[pos]))
            }
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
            Self::Date(s) => s,
            Self::DateTime(s) => s,
            Self::Duration(s) => s,
        }
    }

    /// Collects the f64 values into a new series, converting from this column's type.
    ///
    /// Integer types are cast to f64. Bool is cast to 0.0/1.0. Utf8 is not convertible.
    /// Null positions are preserved in the output series.
    #[must_use]
    pub fn to_f64(&self) -> DataFrameResult<Series<f64>> {
        let name = self.name().to_string();
        let data = match self {
            Self::Float64(s) => s.data().to_vec(),
            Self::Float32(s) => s.data().iter().map(|&v| f64::from(v)).collect(),
            Self::Int64(s) => s.data().iter().map(|&v| v as f64).collect(),
            Self::Int32(s) => s.data().iter().map(|&v| f64::from(v)).collect(),
            Self::Bool(s) => s.data().iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
            Self::Date(s) | Self::DateTime(s) | Self::Duration(s) => {
                s.data().iter().map(|&v| v as f64).collect()
            }
            Self::Utf8(_) => {
                return Err(DataFrameError::InvalidOperation(
                    "cannot convert Utf8 to f64".to_string(),
                ))
            }
        };
        let mut series = Series::new(name, data);
        for i in 0..self.len() {
            if self.is_null(i) {
                series.set_null(i);
            }
        }
        Ok(series)
    }

    /// Selects rows by position, returning a new `AnyColumn` with the same
    /// validity (null) pattern as the source rows.
    #[must_use]
    pub fn select_rows(&self, positions: &[usize]) -> DataFrameResult<Self> {
        if let Some(&p) = positions.iter().find(|&&p| p >= self.len()) {
            return Err(DataFrameError::IndexOutOfBounds {
                index: p,
                length: self.len(),
            });
        }
        let gather = |s: &Series<f64>| -> DataFrameResult<Vec<f64>> {
            positions
                .iter()
                .map(|&p| {
                    s.data()
                        .get(p)
                        .copied()
                        .ok_or(DataFrameError::IndexOutOfBounds {
                            index: p,
                            length: s.len(),
                        })
                })
                .collect()
        };
        let nulls: Vec<bool> = positions.iter().map(|&p| self.is_null(p)).collect();
        let validity = NullBitmap::from_bools(&nulls);
        match self {
            Self::Float64(s) => Ok(Self::Float64(Series::with_validity(
                s.name(),
                gather(s)?,
                validity,
            ))),
            Self::Float32(s) => {
                let data: DataFrameResult<Vec<f32>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .copied()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Float32(Series::with_validity(s.name(), data?, validity)))
            }
            Self::Int64(s) => {
                let data: DataFrameResult<Vec<i64>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .copied()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Int64(Series::with_validity(s.name(), data?, validity)))
            }
            Self::Int32(s) => {
                let data: DataFrameResult<Vec<i32>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .copied()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Int32(Series::with_validity(s.name(), data?, validity)))
            }
            Self::Bool(s) => {
                let data: DataFrameResult<Vec<bool>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .copied()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Bool(Series::with_validity(s.name(), data?, validity)))
            }
            Self::Utf8(s) => {
                let data: DataFrameResult<Vec<String>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .cloned()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Utf8(Series::with_validity(s.name(), data?, validity)))
            }
            Self::Date(s) | Self::DateTime(s) | Self::Duration(s) => {
                let data: DataFrameResult<Vec<i64>> = positions
                    .iter()
                    .map(|&p| {
                        s.data()
                            .get(p)
                            .copied()
                            .ok_or(DataFrameError::IndexOutOfBounds {
                                index: p,
                                length: s.len(),
                            })
                    })
                    .collect();
                Ok(Self::Date(Series::with_validity(s.name(), data?, validity)))
            }
        }
    }

    /// Casts this column to a different dtype.
    #[must_use]
    pub fn cast(&self, target: DType) -> DataFrameResult<Self> {
        match target {
            DType::Float64 => Ok(Self::Float64(self.to_f64()?)),
            DType::Float32 => match self {
                Self::Float32(s) => Ok(Self::Float32(s.clone())),
                Self::Float64(s) => {
                    let data: Vec<f32> = s.data().iter().map(|&v| v as f32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Float32(Series::with_validity(s.name(), data, validity)))
                }
                Self::Int32(s) => {
                    let data: Vec<f32> = s.data().iter().map(|&v| v as f32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Float32(Series::with_validity(s.name(), data, validity)))
                }
                Self::Int64(s) => {
                    let data: Vec<f32> = s.data().iter().map(|&v| v as f32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Float32(Series::with_validity(s.name(), data, validity)))
                }
                Self::Bool(s) => {
                    let data: Vec<f32> = s.data().iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Float32(Series::with_validity(s.name(), data, validity)))
                }
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to f32",
                    other.dtype()
                ))),
            },
            DType::Int64 => match self {
                Self::Int64(s) => Ok(Self::Int64(s.clone())),
                Self::Float64(s) => {
                    let data: Vec<i64> = s.data().iter().map(|&v| v as i64).collect();
                    Ok(Self::Int64(Series::new(s.name(), data)))
                }
                Self::Float32(s) => {
                    let data: Vec<i64> = s.data().iter().map(|&v| v as i64).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Int64(Series::with_validity(s.name(), data, validity)))
                }
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to i64",
                    other.dtype()
                ))),
            },
            DType::Int32 => match self {
                Self::Int32(s) => Ok(Self::Int32(s.clone())),
                Self::Float64(s) => {
                    let data: Vec<i32> = s.data().iter().map(|&v| v as i32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Int32(Series::with_validity(s.name(), data, validity)))
                }
                Self::Float32(s) => {
                    let data: Vec<i32> = s.data().iter().map(|&v| v as i32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Int32(Series::with_validity(s.name(), data, validity)))
                }
                Self::Int64(s) => {
                    let data: Vec<i32> = s.data().iter().map(|&v| v as i32).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Int32(Series::with_validity(s.name(), data, validity)))
                }
                other => Err(DataFrameError::InvalidOperation(alloc::format!(
                    "cannot cast {} to i32",
                    other.dtype()
                ))),
            },
            DType::Bool => match self {
                Self::Bool(s) => Ok(Self::Bool(s.clone())),
                Self::Utf8(s) => {
                    let data: Vec<bool> = s.data().iter().map(|v| !v.is_empty()).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Bool(Series::with_validity(s.name(), data, validity)))
                }
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
                Self::Int32(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
                }
                Self::Float32(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
                }
                Self::Bool(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
                }
                Self::Date(s) | Self::DateTime(s) | Self::Duration(s) => {
                    let data: Vec<String> = s.data().iter().map(|v| alloc::format!("{v}")).collect();
                    let validity = crate::null::NullBitmap::from_bools(
                        &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                    );
                    Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
                }
            },
            _ => Err(DataFrameError::InvalidOperation(alloc::format!(
                "casting to {target} is not supported"
            ))),
        }
    }

    /// Returns a boolean column indicating whether each string value
    /// contains the given `pattern` substring. Non-Utf8 columns return an
    /// error. Null positions produce `null` in the output.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("words",
    ///     vec!["hello".into(), "world".into(), "hi".into()]));
    /// let result = col.str_contains("lo").unwrap();
    /// // result is a Bool column: [true, false, false]
    /// ```
    #[must_use]
    pub fn str_contains(&self, pattern: &str) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<bool> = s.data().iter().map(|v| v.contains(pattern)).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Bool(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_contains is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a boolean column indicating whether each string value
    /// starts with the given `prefix`. Non-Utf8 columns return an error.
    /// Null positions produce `null` in the output.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("items",
    ///     vec!["apple".into(), "banana".into(), "avocado".into()]));
    /// let result = col.str_startswith("a").unwrap();
    /// // result: [true, false, true]
    /// ```
    #[must_use]
    pub fn str_startswith(&self, prefix: &str) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<bool> = s.data().iter().map(|v| v.starts_with(prefix)).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Bool(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_startswith is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a boolean column indicating whether each string value
    /// ends with the given `suffix`. Non-Utf8 columns return an error.
    /// Null positions produce `null` in the output.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("files",
    ///     vec!["data.csv".into(), "readme.md".into(), "out.csv".into()]));
    /// let result = col.str_endswith(".csv").unwrap();
    /// // result: [true, false, true]
    /// ```
    #[must_use]
    pub fn str_endswith(&self, suffix: &str) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<bool> = s.data().iter().map(|v| v.ends_with(suffix)).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Bool(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_endswith is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with all string values converted to
    /// lowercase. Non-Utf8 columns return an error. Null positions are
    /// preserved.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("mixed",
    ///     vec!["Hello".into(), "WORLD".into()]));
    /// let result = col.str_to_lowercase().unwrap();
    /// // result: ["hello", "world"]
    /// ```
    #[must_use]
    pub fn str_to_lowercase(&self) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| v.to_lowercase()).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_to_lowercase is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with all string values converted to
    /// uppercase. Non-Utf8 columns return an error. Null positions are
    /// preserved.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("mixed",
    ///     vec!["hello".into(), "WORLD".into()]));
    /// let result = col.str_to_uppercase().unwrap();
    /// // result: ["HELLO", "WORLD"]
    /// ```
    #[must_use]
    pub fn str_to_uppercase(&self) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| v.to_uppercase()).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_to_uppercase is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with leading/trailing whitespace
    /// stripped from each string value. Null positions are preserved.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("padded",
    ///     vec!["  hello  ".into(), "world".into(), "  !  ".into()]));
    /// let result = col.str_strip().unwrap();
    /// // result: ["hello", "world", "!"]
    /// ```
    #[must_use]
    pub fn str_strip(&self) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| {
                    let trimmed = v.trim();
                    if trimmed.len() == v.len() {
                        v.clone()
                    } else {
                        trimmed.to_string()
                    }
                }).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_strip is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with each string value replaced. Only
    /// the first `count` occurrences are replaced (or all if `count` is
    /// `None`). Null positions are preserved.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("text",
    ///     vec!["aabb".into(), "abab".into()]));
    /// let result = col.str_replace("ab", "x", None).unwrap();
    /// // result: ["axb", "xx"]
    /// ```
    #[must_use]
    pub fn str_replace(&self, from: &str, to: &str, count: Option<usize>) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| {
                    match count {
                        Some(n) => {
                            let mut result = String::new();
                            let mut remaining = v.as_str();
                            let mut replaced = 0;
                            while let Some(idx) = remaining.find(from) {
                                if replaced >= n {
                                    result.push_str(remaining);
                                    remaining = "";
                                    break;
                                }
                                result.push_str(&remaining[..idx]);
                                result.push_str(to);
                                remaining = &remaining[idx + from.len()..];
                                replaced += 1;
                            }
                            result.push_str(remaining);
                            result
                        }
                        None => v.replace(from, to),
                    }
                }).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_replace is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with each string value split by the
    /// delimiter, keeping only the `index`-th part (0-based). If the
    /// index is out of range for a particular row, the output is null.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("csv",
    ///     vec!["a,b,c".into(), "x,y".into()]));
    /// let result = col.str_split(",", 1).unwrap();
    /// // result: ["b", "y"]
    /// ```
    #[must_use]
    pub fn str_split(&self, delimiter: &str, index: usize) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let mut data: Vec<String> = Vec::with_capacity(s.len());
                let mut validity_bools: Vec<bool> = Vec::with_capacity(s.len());
                for i in 0..s.len() {
                    if s.is_null(i) {
                        data.push(String::new());
                        validity_bools.push(true);
                    } else {
                        let parts: Vec<&str> = s.data()[i].split(delimiter).collect();
                        if index < parts.len() {
                            data.push(parts[index].to_string());
                            validity_bools.push(false);
                        } else {
                            data.push(String::new());
                            validity_bools.push(true);
                        }
                    }
                }
                let validity = crate::null::NullBitmap::from_bools(&validity_bools);
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_split is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a boolean column indicating whether each string value
    /// matches the given `pattern` (substring containment, same as
    /// `str_contains`). Provided for pandas `str.match()` compat.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("x",
    ///     vec!["abc".into(), "def".into()]));
    /// let result = col.str_matches("bc").unwrap();
    /// // result: [true, false]
    /// ```
    #[must_use]
    pub fn str_matches(&self, pattern: &str) -> DataFrameResult<Self> {
        self.str_contains(pattern)
    }

    /// Returns an `Int64` column with the length (in bytes) of each
    /// string value. Null positions produce null in the output.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("x",
    ///     vec!["hi".into(), "hello".into()]));
    /// let result = col.str_len().unwrap();
    /// // result is Int64: [2, 5]
    /// ```
    #[must_use]
    pub fn str_len(&self) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<i64> = s.data().iter().map(|v| v.len() as i64).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Int64(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_len is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with each string value repeated `n`
    /// times. Null positions are preserved.
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    /// let col = AnyColumn::Utf8(Series::new("x",
    ///     vec!["ab".into(), "c".into()]));
    /// let result = col.str_repeat(3).unwrap();
    /// // result: ["ababab", "ccc"]
    /// ```
    #[must_use]
    pub fn str_repeat(&self, n: usize) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| v.repeat(n)).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_repeat is not supported on {}",
                other.dtype()
            ))),
        }
    }

    /// Returns a new Utf8 column with each string padded on the left
    /// with `pad_char` to reach `width` total bytes. Strings already
    /// at or past `width` are unchanged.
    #[must_use]
    pub fn str_pad_left(&self, width: usize, pad_char: char) -> DataFrameResult<Self> {
        match self {
            Self::Utf8(s) => {
                let data: Vec<String> = s.data().iter().map(|v| {
                    if v.len() >= width {
                        v.clone()
                    } else {
                        let pad: String = core::iter::repeat(pad_char)
                            .take(width - v.len())
                            .collect();
                        alloc::format!("{pad}{v}")
                    }
                }).collect();
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Utf8(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "str_pad_left is not supported on {}",
                other.dtype()
            ))),
        }
    }

    // ── ML / Data Science column methods ──────────────────────────

    /// Returns a new column with all values clamped to `[min, max]`.
    /// Null positions remain null.
    #[must_use]
    pub fn clip(&self, min: f64, max: f64) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|v| v.clamp(min, max)).collect();
        let validity = crate::null::NullBitmap::from_bools(
            &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
        );
        Ok(Self::Float64(Series::with_validity(s.name(), data, validity)))
    }

    /// Returns a new column with ranks assigned. `method` is one of:
    /// `"dense"`, `"min"`, `"max"`, `"average"`.
    /// Null positions receive null in the output.
    #[must_use]
    pub fn rank(&self, method: &str) -> DataFrameResult<Self> {
        let vals = self.valid_f64()?;
        if vals.is_empty() {
            return Ok(Self::Float64(Series::new(self.name(), Vec::new())));
        }
        // Pair (value, original_index) and sort by value.
        let mut pairs: Vec<(f64, usize)> = vals.iter().copied().zip(0..vals.len()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Assign ranks per method.
        let mut ranks = vec![0.0f64; vals.len()];
        match method {
            "dense" => {
                let mut rank = 1.0f64;
                let mut prev = f64::NEG_INFINITY;
                for &(v, orig_idx) in &pairs {
                    if v != prev {
                        rank += if prev == f64::NEG_INFINITY { 0.0 } else { 1.0 };
                        prev = v;
                    }
                    ranks[orig_idx] = rank;
                }
            }
            "min" => {
                let mut i = 0;
                while i < pairs.len() {
                    let mut j = i;
                    while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                        j += 1;
                    }
                    let rank = (i + 1) as f64;
                    for k in i..j {
                        ranks[pairs[k].1] = rank;
                    }
                    i = j;
                }
            }
            "max" => {
                let mut i = 0;
                while i < pairs.len() {
                    let mut j = i;
                    while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                        j += 1;
                    }
                    let rank = j as f64;
                    for k in i..j {
                        ranks[pairs[k].1] = rank;
                    }
                    i = j;
                }
            }
            "average" => {
                let mut i = 0;
                while i < pairs.len() {
                    let mut j = i;
                    while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                        j += 1;
                    }
                    let avg = ((i + 1) + j) as f64 / 2.0;
                    for k in i..j {
                        ranks[pairs[k].1] = avg;
                    }
                    i = j;
                }
            }
            other => {
                return Err(DataFrameError::InvalidOperation(alloc::format!(
                    "unknown rank method '{other}', expected dense|min|max|average"
                )));
            }
        }

        // Build output with original null positions.
        let out_data: Vec<f64> = ranks;
        let validity = crate::null::NullBitmap::from_bools(
            &(0..self.len()).map(|i| self.is_null(i)).collect::<Vec<_>>(),
        );
        Ok(Self::Float64(Series::with_validity(self.name(), out_data, validity)))
    }

    /// Returns a boolean column: `true` where the value is an outlier
    /// by the IQR method (outside `k * IQR` from Q1/Q3).
    #[must_use]
    pub fn outliers_iqr(&self, k: f64) -> DataFrameResult<Self> {
        let vals = self.valid_f64()?;
        if vals.len() < 4 {
            // Not enough data for IQR.
            let flags = vec![false; self.len()];
            return Ok(Self::Bool(Series::new(self.name(), flags)));
        }
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = sorted.len();
        let q1 = sorted[n / 4];
        let q3 = sorted[(n * 3) / 4];
        let iqr = q3 - q1;
        let lower = q1 - k * iqr;
        let upper = q3 + k * iqr;
        let s = self.to_f64()?;
        let flags: Vec<bool> = s.data().iter().enumerate().map(|(i, &v)| {
            if s.is_null(i) { false } else { v < lower || v > upper }
        }).collect();
        Ok(Self::Bool(Series::new(self.name(), flags)))
    }

    /// Returns a new Float64 column with z-score standardized values
    /// (μ=0, σ=1). Null positions remain null.
    #[must_use]
    pub fn zscore(&self) -> DataFrameResult<Self> {
        let vals = self.valid_f64()?;
        if vals.is_empty() {
            return Ok(Self::Float64(Series::new(self.name(), Vec::new())));
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();
        if std == 0.0 {
            // All values identical → zeros.
            let data = vec![0.0f64; self.len()];
            let validity = crate::null::NullBitmap::from_bools(
                &(0..self.len()).map(|i| self.is_null(i)).collect::<Vec<_>>(),
            );
            return Ok(Self::Float64(Series::with_validity(self.name(), data, validity)));
        }
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|v| (v - mean) / std).collect();
        let validity = crate::null::NullBitmap::from_bools(
            &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
        );
        Ok(Self::Float64(Series::with_validity(s.name(), data, validity)))
    }

    /// Returns a new Float64 column scaled to `[0, 1]` (min-max) or
    /// `[-1, 1]` (max-abs). `method` is `"minmax"` or `"maxabs"`.
    /// Null positions remain null.
    #[must_use]
    pub fn normalize(&self, method: &str) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        match method {
            "minmax" => {
                let vals: Vec<f64> = s.data().iter().enumerate()
                    .filter(|(i, _)| !s.is_null(*i))
                    .map(|(_, &v)| v)
                    .collect();
                if vals.is_empty() {
                    return Ok(Self::Float64(Series::new(s.name(), vec![0.0; s.len()])));
                }
                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                let data: Vec<f64> = if range == 0.0 {
                    vec![0.0; s.len()]
                } else {
                    s.data().iter().map(|v| (v - min) / range).collect()
                };
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Float64(Series::with_validity(s.name(), data, validity)))
            }
            "maxabs" => {
                let max_abs = s.data().iter().enumerate()
                    .filter(|(i, _)| !s.is_null(*i))
                    .map(|(_, &v)| v.abs())
                    .fold(0.0f64, f64::max);
                let data: Vec<f64> = if max_abs == 0.0 {
                    vec![0.0; s.len()]
                } else {
                    s.data().iter().map(|v| v / max_abs).collect()
                };
                let validity = crate::null::NullBitmap::from_bools(
                    &(0..s.len()).map(|i| s.is_null(i)).collect::<Vec<_>>(),
                );
                Ok(Self::Float64(Series::with_validity(s.name(), data, validity)))
            }
            other => Err(DataFrameError::InvalidOperation(alloc::format!(
                "unknown normalize method '{other}', expected minmax|maxabs"
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
                    Self::Date(s) => write!(f, "Date({})", s.data()[i])?,
                    Self::DateTime(s) => write!(f, "DateTime({})", s.data()[i])?,
                    Self::Duration(s) => write!(f, "Duration({})", s.data()[i])?,
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
