use alloc::format;
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec;
use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::dataframe::DataFrame;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::math;
use crate::series::Series;

impl AnyColumn {
    /// Returns the non-null f64 values of this column, cast from the column's
    /// native type. Null positions are skipped, so callers never see
    /// placeholder values.
    #[must_use]
    pub fn valid_f64(&self) -> DataFrameResult<Vec<f64>> {
        let s = self.to_f64()?;
        Ok(s.iter()
            .filter_map(|(_, v)| v.copied())
            .collect())
    }

    /// Returns the sum of all non-null values (empty sum = 0).
    #[must_use]
    pub fn sum(&self) -> DataFrameResult<f64> {
        Ok(self.valid_f64()?.iter().sum())
    }

    /// Returns the mean of all non-null values.
    #[must_use]
    pub fn mean(&self) -> DataFrameResult<f64> {
        let vals = self.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        Ok(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Returns the sample variance (ddof=1) of non-null values using the
    /// numerically stable Welford online algorithm (single pass, no
    /// catastrophic cancellation for large-magnitude data).
    #[must_use]
    pub fn var(&self) -> DataFrameResult<f64> {
        let vals = self.valid_f64()?;
        if vals.len() < 2 {
            return Err(DataFrameError::InvalidOperation(
                "variance requires at least 2 non-null values".to_string(),
            ));
        }
        let (_, m2) = welford(&vals);
        Ok(m2 / (vals.len() - 1) as f64)
    }

    /// Returns the sample standard deviation (ddof=1) of non-null values.
    #[must_use]
    pub fn std(&self) -> DataFrameResult<f64> {
        self.var().map(math::sqrt)
    }

    /// Returns the minimum non-null value.
    #[must_use]
    pub fn min(&self) -> DataFrameResult<f64> {
        self.valid_f64()?
            .into_iter()
            .reduce(f64::min)
            .ok_or(DataFrameError::EmptyDataFrame)
    }

    /// Returns the maximum non-null value.
    #[must_use]
    pub fn max(&self) -> DataFrameResult<f64> {
        self.valid_f64()?
            .into_iter()
            .reduce(f64::max)
            .ok_or(DataFrameError::EmptyDataFrame)
    }

    /// Returns the median of non-null values.
    #[must_use]
    pub fn median(&self) -> DataFrameResult<f64> {
        let mut vals = self.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        sort_f64(&mut vals);
        let n = vals.len();
        if n % 2 == 0 {
            Ok((vals[n / 2 - 1] + vals[n / 2]) / 2.0)
        } else {
            Ok(vals[n / 2])
        }
    }

    /// Returns the quantile at the given percentile (0.0 - 1.0) of non-null
    /// values using linear interpolation between closest ranks.
    #[must_use]
    pub fn quantile(&self, q: f64) -> DataFrameResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DataFrameError::InvalidOperation(
                "quantile must be between 0.0 and 1.0".to_string(),
            ));
        }
        let mut vals = self.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        sort_f64(&mut vals);
        let n = vals.len();
        let idx = q * (n - 1) as f64;
        let lo = math::floor(idx) as usize;
        let hi = math::ceil(idx) as usize;
        if lo == hi {
            Ok(vals[lo])
        } else {
            let frac = idx - lo as f64;
            Ok(vals[lo] * (1.0 - frac) + vals[hi] * frac)
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

    /// Returns the number of distinct non-null values (pandas `nunique`).
    #[must_use]
    pub fn nunique(&self) -> DataFrameResult<usize> {
        let mut vals = self.valid_f64()?;
        sort_f64(&mut vals);
        vals.dedup();
        Ok(vals.len())
    }

    /// Returns the most frequent non-null value and its count (pandas `mode`
    /// top). Ties are broken by smallest value.
    #[must_use]
    pub fn mode(&self) -> DataFrameResult<(f64, usize)> {
        let vals = self.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        let mut counts: alloc::collections::BTreeMap<u64, usize> = alloc::collections::BTreeMap::new();
        for v in &vals {
            *counts.entry(v.to_bits()).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(bits, c)| (f64::from_bits(bits), c))
            .ok_or(DataFrameError::EmptyDataFrame)
    }

    /// Returns (value, count) pairs for the `value_counts` histogram
    /// (descending by count, ties by ascending value).
    #[must_use]
    pub fn value_counts(&self) -> DataFrameResult<Vec<(f64, usize)>> {
        let vals = self.valid_f64()?;
        let mut counts: alloc::collections::BTreeMap<u64, usize> = alloc::collections::BTreeMap::new();
        for v in &vals {
            *counts.entry(v.to_bits()).or_default() += 1;
        }
        let mut out: Vec<(f64, usize)> = counts
            .into_iter()
            .map(|(bits, c)| (f64::from_bits(bits), c))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)));
        Ok(out)
    }

    /// Returns the first moment (mean) and second central moment (sample
    /// variance, ddof=1) in a single numerically stable Welford pass.
    #[must_use]
    pub fn moments(&self) -> DataFrameResult<(f64, f64)> {
        let vals = self.valid_f64()?;
        if vals.len() < 2 {
            return Err(DataFrameError::InvalidOperation(
                "moments require at least 2 non-null values".to_string(),
            ));
        }
        let (mean, m2) = welford(&vals);
        Ok((mean, m2 / (vals.len() - 1) as f64))
    }

    /// Returns a new series containing the cumulative sum. Null inputs
    /// produce `NaN` outputs while the running total continues from the last
    /// non-null value (pandas semantics).
    #[must_use]
    pub fn cumsum(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let mut running = 0.0;
        let data: Vec<f64> = (0..s.len())
            .map(|i| {
                if s.is_null(i) {
                    f64::NAN
                } else {
                    running += s.data()[i];
                    running
                }
            })
            .collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Returns a new series containing the cumulative product (pandas
    /// semantics for nulls, as in `cumsum`).
    #[must_use]
    pub fn cumprod(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let mut running = 1.0;
        let data: Vec<f64> = (0..s.len())
            .map(|i| {
                if s.is_null(i) {
                    f64::NAN
                } else {
                    running *= s.data()[i];
                    running
                }
            })
            .collect();
        Ok(AnyColumn::Float64(Series::new(s.name(), data)))
    }

    /// Returns a new series containing the first differences
    /// (`diff[i] = x[i] - x[i-1]`); the first element is `NaN`.
    #[must_use]
    pub fn diff(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        if data.is_empty() {
            return Ok(AnyColumn::Float64(Series::new(s.name(), vec![])));
        }
        let mut result = Vec::with_capacity(data.len());
        result.push(f64::NAN);
        for i in 1..data.len() {
            if s.is_null(i) || s.is_null(i - 1) {
                result.push(f64::NAN);
            } else {
                result.push(data[i] - data[i - 1]);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Returns a new series containing the percentage change; the first
    /// element and any null-involved pairs are `NaN`.
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
            if s.is_null(i) || s.is_null(i - 1) || data[i - 1] == 0.0 {
                result.push(f64::NAN);
            } else {
                result.push((data[i] - data[i - 1]) / data[i - 1]);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the mean.
    /// Windows containing any null yield `NaN` (pandas `min_periods=window`).
    #[must_use]
    pub fn rolling_mean(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        check_window(window, data.len())?;
        let mut result = Vec::with_capacity(data.len());
        let mut wsum = 0.0;
        for i in 0..data.len() {
            wsum += data[i];
            if i >= window {
                wsum -= data[i - window];
            }
            if i + 1 < window || (i + 1 - window..=i).any(|k| s.is_null(k)) {
                result.push(f64::NAN);
            } else {
                result.push(wsum / window as f64);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the sum.
    #[must_use]
    pub fn rolling_sum(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        check_window(window, data.len())?;
        let mut result = Vec::with_capacity(data.len());
        let mut wsum = 0.0;
        for i in 0..data.len() {
            wsum += data[i];
            if i >= window {
                wsum -= data[i - window];
            }
            if i + 1 < window || (i + 1 - window..=i).any(|k| s.is_null(k)) {
                result.push(f64::NAN);
            } else {
                result.push(wsum);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the min.
    #[must_use]
    pub fn rolling_min(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        check_window(window, data.len())?;
        let mut result = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if i + 1 < window || (i + 1 - window..=i).any(|k| s.is_null(k)) {
                result.push(f64::NAN);
            } else {
                result.push(
                    data[i + 1 - window..=i]
                        .iter()
                        .copied()
                        .reduce(f64::min)
                        .unwrap_or(f64::NAN),
                );
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Applies a rolling window of size `window` and returns the max.
    #[must_use]
    pub fn rolling_max(&self, window: usize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        check_window(window, data.len())?;
        let mut result = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if i + 1 < window || (i + 1 - window..=i).any(|k| s.is_null(k)) {
                result.push(f64::NAN);
            } else {
                result.push(
                    data[i + 1 - window..=i]
                        .iter()
                        .copied()
                        .reduce(f64::max)
                        .unwrap_or(f64::NAN),
                );
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }
}

impl DataFrame {
    /// Computes a pandas-style summary of the numeric columns and returns it
    /// as a new DataFrame whose first column (`statistic`) names each row.
    ///
    /// The rows are `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`.
    /// Non-numeric columns are skipped (pandas `describe` default).
    ///
    /// # Errors
    ///
    /// Returns an error if a numeric column cannot be summarized (e.g. a
    /// column with fewer than two non-null values, for which `std` is
    /// undefined).
    #[must_use]
    pub fn describe(&self) -> DataFrameResult<Self> {
        let mut result = DataFrame::new();
        let stats: [&str; 8] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        result.add_column("statistic", stats.iter().map(|s| String::from(*s)).collect::<Vec<_>>())?;

        for col in self.columns() {
            if matches!(
                col.dtype(),
                crate::dtype::DType::Float64
                    | crate::dtype::DType::Float32
                    | crate::dtype::DType::Int64
                    | crate::dtype::DType::Int32
            ) {
                let values = vec![
                    col.count() as f64,
                    col.mean()?,
                    col.std()?,
                    col.min()?,
                    col.quantile(0.25)?,
                    col.quantile(0.50)?,
                    col.quantile(0.75)?,
                    col.max()?,
                ];
                result.add_column(col.name(), values)?;
            }
        }
        Ok(result)
    }
}

/// Single-pass Welford algorithm: returns (mean, sum-of-squared-deviations).
fn welford(vals: &[f64]) -> (f64, f64) {
    let mut n = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for &x in vals {
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        m2 += delta * (x - mean);
    }
    (mean, m2)
}

/// NaN-aware ascending sort (NaNs sort last, matching pandas).
fn sort_f64(vals: &mut [f64]) {
    vals.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal),
    });
}

/// Validates a rolling window size against the data length.
fn check_window(window: usize, len: usize) -> DataFrameResult<()> {
    if window == 0 {
        return Err(DataFrameError::InvalidOperation(
            "window size must be at least 1".to_string(),
        ));
    }
    if window > len {
        return Err(DataFrameError::InvalidOperation(format!(
            "window size {window} is larger than series length {len}"
        )));
    }
    Ok(())
}
