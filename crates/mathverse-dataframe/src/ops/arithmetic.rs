use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

impl AnyColumn {
    /// Adds another column element-wise (both must be numeric and cast to f64).
    pub fn add(&self, other: &Self) -> DataFrameResult<Self> {
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
        Ok(Self::Float64(Series::new(a.name(), data)))
    }

    /// Subtracts another column element-wise (both must be numeric, cast to f64).
    pub fn sub(&self, other: &Self) -> DataFrameResult<Self> {
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
        Ok(Self::Float64(Series::new(a.name(), data)))
    }

    /// Multiplies another column element-wise (both must be numeric, cast to f64).
    pub fn mul(&self, other: &Self) -> DataFrameResult<Self> {
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
        Ok(Self::Float64(Series::new(a.name(), data)))
    }

    /// Divides another column element-wise (both must be numeric, cast to f64).
    pub fn div(&self, other: &Self) -> DataFrameResult<Self> {
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
        Ok(Self::Float64(Series::new(a.name(), data)))
    }

    /// Adds a scalar to every element (cast to f64).
    pub fn add_scalar(&self, scalar: f64) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x + scalar).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }

    /// Subtracts a scalar from every element (cast to f64).
    pub fn sub_scalar(&self, scalar: f64) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x - scalar).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }

    /// Multiplies every element by a scalar (cast to f64).
    pub fn mul_scalar(&self, scalar: f64) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x * scalar).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }

    /// Divides every element by a scalar (cast to f64).
    pub fn div_scalar(&self, scalar: f64) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x / scalar).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }

    /// Returns `true` where `self > other`, element-wise.
    pub fn gt(&self, other: &Self) -> DataFrameResult<Series<bool>> {
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
    pub fn gte(&self, other: &Self) -> DataFrameResult<Series<bool>> {
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
    pub fn lt(&self, other: &Self) -> DataFrameResult<Series<bool>> {
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
    pub fn lte(&self, other: &Self) -> DataFrameResult<Series<bool>> {
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
    pub fn eq(&self, other: &Self) -> DataFrameResult<Series<bool>> {
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
    pub fn neq(&self, other: &Self) -> DataFrameResult<Series<bool>> {
        let result = self.eq(other)?;
        let data: Vec<bool> = result.data().iter().map(|x| !x).collect();
        Ok(Series::new(result.name(), data))
    }

    /// Compares against a scalar: returns `self > scalar`.
    pub fn gt_scalar(&self, scalar: f64) -> DataFrameResult<Series<bool>> {
        let s = self.to_f64()?;
        let data: Vec<bool> = s.data().iter().map(|x| *x > scalar).collect();
        Ok(Series::new(s.name(), data))
    }

    /// Compares against a scalar: returns `self < scalar`.
    pub fn lt_scalar(&self, scalar: f64) -> DataFrameResult<Series<bool>> {
        let s = self.to_f64()?;
        let data: Vec<bool> = s.data().iter().map(|x| *x < scalar).collect();
        Ok(Series::new(s.name(), data))
    }

    /// Negates every element (cast to f64).
    pub fn neg(&self) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| -x).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }

    /// Absolute value of every element (cast to f64).
    pub fn abs(&self) -> DataFrameResult<Self> {
        let s = self.to_f64()?;
        let data: Vec<f64> = s.data().iter().map(|x| x.abs()).collect();
        Ok(Self::Float64(Series::new(s.name(), data)))
    }
}
