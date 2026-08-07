use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

impl AnyColumn {
    /// Adds another column element-wise (both must be numeric and cast to f64).
    #[must_use]
    pub fn add(&self, other: &AnyColumn) -> DataFrameResult<AnyColumn> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot add columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<f64> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(AnyColumn::Float64(Series::new(a.name(), data)))
    }

    /// Subtracts another column element-wise (both must be numeric, cast to f64).
    #[must_use]
    pub fn sub(&self, other: &AnyColumn) -> DataFrameResult<AnyColumn> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot subtract columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<f64> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x - y)
            .collect();
        Ok(AnyColumn::Float64(Series::new(a.name(), data)))
    }

    /// Multiplies another column element-wise (both must be numeric, cast to f64).
    #[must_use]
    pub fn mul(&self, other: &AnyColumn) -> DataFrameResult<AnyColumn> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot multiply columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<f64> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x * y)
            .collect();
        Ok(AnyColumn::Float64(Series::new(a.name(), data)))
    }

    /// Divides another column element-wise (both must be numeric, cast to f64).
    #[must_use]
    pub fn div(&self, other: &AnyColumn) -> DataFrameResult<AnyColumn> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot divide columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<f64> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x / y)
            .collect();
        Ok(AnyColumn::Float64(Series::new(a.name(), data)))
    }

    /// Adds a scalar to every element (cast to f64).
    #[must_use]
    pub fn add_scalar(&self, scalar: f64) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x + scalar).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Subtracts a scalar from every element (cast to f64).
    #[must_use]
    pub fn sub_scalar(&self, scalar: f64) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x - scalar).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Multiplies every element by a scalar (cast to f64).
    #[must_use]
    pub fn mul_scalar(&self, scalar: f64) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x * scalar).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Divides every element by a scalar (cast to f64).
    #[must_use]
    pub fn div_scalar(&self, scalar: f64) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x / scalar).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Returns `true` where `self > other`, element-wise.
    #[must_use]
    pub fn gt(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot compare columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<bool> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x > y)
            .collect();
        Ok(Series::new(a.name(), data))
    }

    /// Returns `true` where `self >= other`, element-wise.
    #[must_use]
    pub fn gte(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot compare columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<bool> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x >= y)
            .collect();
        Ok(Series::new(a.name(), data))
    }

    /// Returns `true` where `self < other`, element-wise.
    #[must_use]
    pub fn lt(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot compare columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<bool> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x < y)
            .collect();
        Ok(Series::new(a.name(), data))
    }

    /// Returns `true` where `self <= other`, element-wise.
    #[must_use]
    pub fn lte(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot compare columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<bool> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| x <= y)
            .collect();
        Ok(Series::new(a.name(), data))
    }

    /// Returns `true` where `self == other`, element-wise.
    #[must_use]
    pub fn eq(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let a = self.to_f64()?;
        let b = other.to_f64()?;
        if a.len() != b.len() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "cannot compare columns of lengths {} and {}",
                    a.len(),
                    b.len()
                ),
            });
        }
        let data: Vec<bool> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(x, y)| (x - y).abs() < f64::EPSILON)
            .collect();
        Ok(Series::new(a.name(), data))
    }

    /// Returns `true` where `self != other`, element-wise.
    #[must_use]
    pub fn neq(&self, other: &AnyColumn) -> DataFrameResult<Series<bool>> {
        let result = self.eq(other)?;
        let data: Vec<bool> = result.data().iter().map(|x| !x).collect();
        Ok(Series::new(result.name(), data))
    }

    /// Compares against a scalar: returns `self > scalar`.
    #[must_use]
    pub fn gt_scalar(&self, scalar: f64) -> DataFrameResult<Series<bool>> {
        let s = self.to_f64()?;
        let data: Vec<bool> = s.data().iter().map(|x| *x > scalar).collect();
        Ok(Series::new(s.name(), data))
    }

    /// Compares against a scalar: returns `self < scalar`.
    #[must_use]
    pub fn lt_scalar(&self, scalar: f64) -> DataFrameResult<Series<bool>> {
        let s = self.to_f64()?;
        let data: Vec<bool> = s.data().iter().map(|x| *x < scalar).collect();
        Ok(Series::new(s.name(), data))
    }

    /// Negates every element (cast to f64).
    #[must_use]
    pub fn neg(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| -x).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Absolute value of every element (cast to f64).
    #[must_use]
    pub fn abs(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x.abs()).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }
}
