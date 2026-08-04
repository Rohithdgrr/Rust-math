use crate::column::AnyColumn;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

impl AnyColumn {
    /// Returns the sum of all non-null values.
    #[must_use]
    pub fn sum(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        Ok(s.data().iter().sum())
    }

    /// Returns the mean of all non-null values.
    #[must_use]
    pub fn mean(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        let data = s.data();
        if data.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        let sum: f64 = data.iter().sum();
        Ok(sum / data.len() as f64)
    }

    /// Returns the variance of all non-null values (sample variance, ddof=1).
    #[must_use]
    pub fn var(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        let data = s.data();
        if data.len() < 2 {
            return Err(DataFrameError::InvalidOperation(
                "variance requires at least 2 values".to_string(),
            ));
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        Ok(var)
    }

    /// Returns the standard deviation of all non-null values (sample std, ddof=1).
    #[must_use]
    pub fn std(&self) -> DataFrameResult<f64> {
        self.var().map(|v| v.sqrt())
    }

    /// Returns the minimum value.
    #[must_use]
    pub fn min(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        s.data()
            .iter()
            .copied()
            .reduce(f64::min)
            .ok_or(DataFrameError::EmptyDataFrame)
    }

    /// Returns the maximum value.
    #[must_use]
    pub fn max(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        s.data()
            .iter()
            .copied()
            .reduce(f64::max)
            .ok_or(DataFrameError::EmptyDataFrame)
    }

    /// Returns the median value.
    #[must_use]
    pub fn median(&self) -> DataFrameResult<f64> {
        let s = self.to_f64()?;
        let mut data: Vec<f64> = s.data().to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = data.len();
        if n == 0 {
            return Err(DataFrameError::EmptyDataFrame);
        }
        if n % 2 == 0 {
            Ok((data[n / 2 - 1] + data[n / 2]) / 2.0)
        } else {
            Ok(data[n / 2])
        }
    }

    /// Returns the quantile at the given percentile (0.0 - 1.0).
    #[must_use]
    pub fn quantile(&self, q: f64) -> DataFrameResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DataFrameError::InvalidOperation(
                "quantile must be between 0.0 and 1.0".to_string(),
            ));
        }
        let s = self.to_f64()?;
        let mut data: Vec<f64> = s.data().to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = data.len();
        if n == 0 {
            return Err(DataFrameError::EmptyDataFrame);
        }
        let idx = q * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            Ok(data[lo])
        } else {
            let frac = idx - lo as f64;
            Ok(data[lo] * (1.0 - frac) + data[hi] * frac)
        }
    }

    /// Returns the count of non-null values.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            Self::Float64(s) => s.non_null_count(),
            Self::Float32(s) => s.non_null_count(),
            Self::Int64(s) => s.non_null_count(),
            Self::Int32(s) => s.non_null_count(),
            Self::Bool(s) => s.non_null_count(),
            Self::Utf8(s) => s.non_null_count(),
        }
    }

    /// Returns (min, max, mean, std, count) as a summary tuple.
    #[must_use]
    pub fn describe_numeric(&self) -> DataFrameResult<(f64, f64, f64, f64, f64)> {
        Ok((self.min()?, self.max()?, self.mean()?, self.std()?, self.count() as f64))
    }

    /// Returns a new series containing the cumulative sum.
    #[must_use]
    pub fn cumsum(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let mut running = 0.0;
        let data: Vec<f64> = s.data().iter().map(|x| { running += x; running }).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Returns a new series containing the cumulative product.
    #[must_use]
    pub fn cumprod(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let mut running = 1.0;
        let data: Vec<f64> = s.data().iter().map(|x| { running *= x; running }).collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Returns a new series containing the first differences (diff[i] = x[i] - x[i-1]).
    #[must_use]
    pub fn diff(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if data.is_empty() {
            return Ok(AnyColumn::Float64(Series::new(s.name(), vec![])));
        }
        let mut result = Vec::with_capacity(data.len());
        result.push(f64::NAN); // first diff is undefined
        for i in 1..data.len() {
            result.push(data[i] - data[i - 1]);
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Returns a new series containing the percentage change.
    #[must_use]
    pub fn pct_change(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if data.is_empty() {
            return Ok(AnyColumn::Float64(Series::new(s.name(), vec![])));
        }
        let mut result = Vec::with_capacity(data.len());
        result.push(f64::NAN);
        for i in 1..data.len() {
            if data[i - 1] == 0.0 {
                result.push(f64::NAN);
            } else {
                result.push((data[i] - data[i - 1]) / data[i - 1]);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the mean.
    #[must_use]
    pub fn rolling_mean(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if window == 0 || window > data.len() {
            return Err(DataFrameError::InvalidOperation(alloc::format!(
                "window size {window} is invalid for length {}",
                data.len()
            )));
        }
        let mut result = Vec::with_capacity(data.len());
        // First (window-1) elements are NaN
        for _ in 0..window - 1 {
            result.push(f64::NAN);
        }
        // Sliding window sum
        let mut wsum: f64 = data[..window].iter().sum();
        result.push(wsum / window as f64);
        for i in window..data.len() {
            wsum += data[i] - data[i - window];
            result.push(wsum / window as f64);
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the sum.
    #[must_use]
    pub fn rolling_sum(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if window == 0 || window > data.len() {
            return Err(DataFrameError::InvalidOperation(alloc::format!(
                "window size {window} is invalid for length {}",
                data.len()
            )));
        }
        let mut result = Vec::with_capacity(data.len());
        for _ in 0..window - 1 {
            result.push(f64::NAN);
        }
        let mut wsum: f64 = data[..window].iter().sum();
        result.push(wsum);
        for i in window..data.len() {
            wsum += data[i] - data[i - window];
            result.push(wsum);
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the min.
    #[must_use]
    pub fn rolling_min(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if window == 0 || window > data.len() {
            return Err(DataFrameError::InvalidOperation(alloc::format!(
                "window size {window} is invalid for length {}",
                data.len()
            )));
        }
        let mut result = Vec::with_capacity(data.len());
        for _ in 0..window - 1 {
            result.push(f64::NAN);
        }
        for i in window - 1..data.len() {
            let wmin = data[i + 1 - window..=i]
                .iter()
                .copied()
                .reduce(f64::min)
                .unwrap_or(f64::NAN);
            result.push(wmin);
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the max.
    #[must_use]
    pub fn rolling_max(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if window == 0 || window > data.len() {
            return Err(DataFrameError::InvalidOperation(alloc::format!(
                "window size {window} is invalid for length {}",
                data.len()
            )));
        }
        let mut result = Vec::with_capacity(data.len());
        for _ in 0..window - 1 {
            result.push(f64::NAN);
        }
        for i in window - 1..data.len() {
            let wmax = data[i + 1 - window..=i]
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(f64::NAN);
            result.push(wmax);
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }
}
