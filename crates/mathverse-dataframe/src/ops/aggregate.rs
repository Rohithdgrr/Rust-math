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
            Self::Date(s) => s.non_null_count(),
            Self::DateTime(s) => s.non_null_count(),
            Self::Duration(s) => s.non_null_count(),
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

    /// Returns a boolean vector indicating whether each row is a duplicate of
    /// a previous row (based on non-null values only). The first occurrence
    /// of any unique value combination is considered not a duplicate; subsequent
    /// rows with the same non-null values are marked as duplicates.
    ///
    /// This is pandas-compatible: `df.duplicated()` returns a boolean Series
    /// where `True` means the row is a duplicate of a row that came before it.
    #[must_use]
    pub fn duplicated(&self) -> DataFrameResult<Series<bool>> {
        let vals = self.valid_f64()?;
        let n = vals.len();
        let mut seen: alloc::collections::BTreeSet<u64> = alloc::collections::BTreeSet::new();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let bits = vals[i].to_bits();
            if seen.contains(&bits) {
                result.push(true);
            } else {
                seen.insert(bits);
                result.push(false);
            }
        }
        let series = Series::new(self.name().to_string(), result);
        Ok(series)
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

    /// Returns a new series containing the values shifted by `n` positions.
    /// Positive `n` shifts values downward (the first `n` values become `NaN`);
    /// negative `n` shifts values upward.
    /// Null positions are preserved in their relative positions.
    ///
    /// # Examples
    ///
    /// Basic shift:
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    ///
    /// let col = AnyColumn::from(Series::new("x", vec![1.0, 2.0, 3.0, 4.0]));
    /// let shifted = col.shift(1).unwrap();
    /// // shifted == [NaN, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// Negative shift:
    ///
    /// ```
    /// use mathverse_dataframe::{AnyColumn, Series};
    ///
    /// let col = AnyColumn::from(Series::new("x", vec![1.0, 2.0, 3.0, 4.0]));
    /// let shifted = col.shift(-1).unwrap();
    /// // shifted == [2.0, 3.0, 4.0, NaN]
    /// ```
    #[must_use]
    pub fn shift(&self, n: isize) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        let len = data.len();
        let null_mask: Vec<bool> = (0..len).map(|i| s.is_null(i)).collect();
        let mut result_data = Vec::with_capacity(len);
        let mut result_nulls = Vec::with_capacity(len);

        if n > 0 {
            // Positive shift: move values down, first n become NaN
            for i in 0..len {
                if i < n as usize || null_mask[i] {
                    result_data.push(f64::NAN);
                    result_nulls.push(true);
                } else {
                    result_data.push(data[i - n as usize]);
                    result_nulls.push(false);
                }
            }
        } else if n < 0 {
            // Negative shift: move values upward, last |n| become NaN
            let shift = (-n) as usize;
            for i in 0..len {
                if i + shift >= len || null_mask[i] {
                    result_data.push(f64::NAN);
                    result_nulls.push(true);
                } else {
                    result_data.push(data[i + shift]);
                    result_nulls.push(false);
                }
            }
        } else {
            // n == 0: no shift
            for i in 0..len {
                result_data.push(data[i]);
                result_nulls.push(null_mask[i]);
            }
        }
        let mut series = Series::new(s.name(), result_data);
        // Set null positions
        for (i, &is_null) in result_nulls.iter().enumerate() {
            if is_null {
                series.set_null(i);
            }
        }
        Ok(AnyColumn::Float64(series))
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

    /// Returns the expanding (cumulative) mean. The first element is the
    /// value itself; subsequent elements are the running mean up to that
    /// position. Null inputs produce `NaN` outputs; the running mean
    /// continues from the last non-null value.
    #[must_use]
    pub fn expanding_mean(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        let mut count = 0.0;
        for i in 0..data.len() {
            if s.is_null(i) {
                result.push(f64::NAN);
            } else {
                sum += data[i];
                count += 1.0;
                result.push(sum / count);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Returns the expanding (cumulative) sum. Null inputs produce `NaN`
    /// outputs; the running sum continues from the last non-null value.
    #[must_use]
    pub fn expanding_sum(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        for i in 0..data.len() {
            if s.is_null(i) {
                result.push(f64::NAN);
            } else {
                sum += data[i];
                result.push(sum);
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Returns the expanding (cumulative) sample variance (ddof=1). The
    /// first two non-null values determine the variance; earlier outputs
    /// and null inputs are `NaN`.
    #[must_use]
    pub fn expanding_var(&self) -> DataFrameResult<AnyColumn> {
        let s = self.to_f64()?;
        let data = s.data();
        let mut result = Vec::with_capacity(data.len());
        let mut n = 0.0;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for i in 0..data.len() {
            if s.is_null(i) {
                result.push(f64::NAN);
            } else {
                n += 1.0;
                let delta = data[i] - mean;
                mean += delta / n;
                m2 += delta * (data[i] - mean);
                if n < 2.0 {
                    result.push(f64::NAN);
                } else {
                    result.push(m2 / (n - 1.0));
                }
            }
        }
        Ok(AnyColumn::Float64(Series::new(s.name(), result)))
    }

    /// Returns the exponentially weighted moving average with the given
    /// span. Equivalent to pandas `ewm(span=n).mean()`.
    #[must_use]
    pub fn ewm_mean(&self, span: usize) -> DataFrameResult<AnyColumn> {
        if span == 0 {
            return Err(DataFrameError::InvalidOperation(
                "ewm span must be at least 1".to_string(),
            ));
        }
        let s = self.to_f64()?;
        let data = s.data();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Vec::with_capacity(data.len());
        let mut prev: Option<f64> = None;
        for i in 0..data.len() {
            if s.is_null(i) {
                result.push(f64::NAN);
            } else {
                match prev {
                    None => {
                        result.push(data[i]);
                        prev = Some(data[i]);
                    }
                    Some(p) => {
                        let v = alpha * data[i] + (1.0 - alpha) * p;
                        result.push(v);
                        prev = Some(v);
                    }
                }
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
    /// Computes a pandas-style summary of the numeric columns and returns it
    /// as a new DataFrame whose first column (`statistic`) names each row.
    ///
    /// The rows are `count`, `mean`, `std`, `min`, `10%`, `25%`, `50%`, `75%`,
    /// `90%`, `95%`, `max`. Non-numeric columns are skipped (pandas `describe`
    /// default).
    ///
    /// # Errors
    ///
    /// Returns an error if a numeric column cannot be summarized (e.g. a
    /// column with fewer than two non-null values, for which `std` is
    /// undefined).
    #[must_use]
    pub fn describe(&self) -> DataFrameResult<Self> {
        let mut result = DataFrame::new();
        let stats: [&str; 11] = [
            "count", "mean", "std", "min", "10%", "25%", "50%", "75%", "90%", "95%", "max",
        ];
        result.add_column(
            "statistic",
            stats.iter().map(|s| String::from(*s)).collect::<Vec<_>>(),
        )?;

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
                    col.quantile(0.10)?,
                    col.quantile(0.25)?,
                    col.quantile(0.50)?,
                    col.quantile(0.75)?,
                    col.quantile(0.90)?,
                    col.quantile(0.95)?,
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
