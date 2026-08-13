use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::column::AnyColumn;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::index::Index;
use crate::schema::Schema;
use crate::series::Series;

/// A two-dimensional, column-oriented data structure with named, typed columns.
///
/// A `DataFrame` is the primary data structure for working with tabular data,
/// analogous to a pandas DataFrame or an R data frame.
///
/// # Examples
///
/// ```
/// use mathverse_dataframe::DataFrame;
///
/// let mut df = DataFrame::new();
/// df.add_column("x", vec![1.0, 2.0, 3.0]).unwrap();
/// df.add_column("y", vec![4.0, 5.0, 6.0]).unwrap();
///
/// assert_eq!(df.nrows(), 3);
/// assert_eq!(df.ncols(), 2);
/// assert!(df.has_column("x"));
/// ```
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<AnyColumn>,
    schema: Schema,
    index: Index,
}

impl DataFrame {
    /// Creates an empty DataFrame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            schema: Schema::empty(),
            index: Index::default_range(0),
        }
    }

    /// Creates a DataFrame from a list of `AnyColumn`s.
    ///
    /// # Errors
    ///
    /// Returns an error if columns have mismatched lengths or duplicate names.
    pub fn from_columns(columns: Vec<AnyColumn>) -> DataFrameResult<Self> {
        if columns.is_empty() {
            return Ok(Self::new());
        }

        let len = columns[0].len();
        for col in &columns[1..] {
            if col.len() != len {
                return Err(DataFrameError::DimensionMismatch {
                    message: format!(
                        "column `{}` has length {} but expected {}",
                        col.name(),
                        col.len(),
                        len
                    ),
                });
            }
        }

        let mut schema = Schema::empty();
        let mut names = Vec::with_capacity(columns.len());
        for col in &columns {
            let name = col.name().to_string();
            if schema.contains(&name) {
                return Err(DataFrameError::DuplicateColumn(name));
            }
            schema.add_field(crate::schema::Field::new(&name, col.dtype()))?;
            names.push(name);
        }

        Ok(Self {
            columns,
            schema,
            index: Index::default_range(len),
        })
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Returns the shape as (rows, columns).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Returns `true` if the DataFrame has no rows or columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nrows() == 0 || self.ncols() == 0
    }

    /// Returns the schema.
    #[must_use]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Returns the index.
    #[must_use]
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Returns the column names.
    #[must_use]
    pub fn column_names(&self) -> Vec<&str> {
        self.schema.names().collect()
    }

    /// Returns `true` if the DataFrame has a column with the given name.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.schema.contains(name)
    }

    /// Returns the index of a column by name.
    #[must_use]
    pub fn column_index(&self, name: &str) -> DataFrameResult<usize> {
        self.schema.index_of(name)
    }

    /// Returns a reference to the column at the given index.
    #[must_use]
    pub fn column_by_index(&self, index: usize) -> DataFrameResult<&AnyColumn> {
        self.columns
            .get(index)
            .ok_or_else(|| DataFrameError::IndexOutOfBounds {
                index,
                length: self.columns.len(),
            })
    }

    /// Returns a reference to a column by name.
    #[must_use]
    pub fn column(&self, name: &str) -> DataFrameResult<&AnyColumn> {
        let idx = self.schema.index_of(name)?;
        Ok(&self.columns[idx])
    }

    /// Returns a typed reference to a column's series.
    #[must_use]
    pub fn column_as<T: 'static>(&self, name: &str) -> DataFrameResult<&Series<T>> {
        let col = self.column(name)?;
        col.as_any().downcast_ref::<Series<T>>().ok_or_else(|| {
            DataFrameError::TypeMismatch {
                expected: "the requested type",
                actual: col.dtype().name(),
            }
        })
    }

    /// Adds a new column from a `Vec<T>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the column name already exists or lengths don't match.
    pub fn add_column<T: Into<AnyColumn>>(&mut self, name: &str, data: T) -> DataFrameResult<()> {
        let mut col = data.into();
        col.rename_mut(name);

        if self.schema.contains(name) {
            return Err(DataFrameError::DuplicateColumn(name.to_string()));
        }

        if !self.columns.is_empty() && col.len() != self.nrows() {
            return Err(DataFrameError::DimensionMismatch {
                message: format!(
                    "new column `{}` has length {} but DataFrame has {} rows",
                    name,
                    col.len(),
                    self.nrows()
                ),
            });
        }

        self.schema.add_field(crate::schema::Field::new(name, col.dtype()))?;
        let col_len = col.len();
        let is_first = self.columns.is_empty();
        self.columns.push(col);

        if is_first {
            self.index = Index::default_range(col_len);
        }

        Ok(())
    }

    /// Adds a new column from an `AnyColumn`.
    ///
    /// # Errors
    ///
    /// Returns an error if the column name already exists or lengths don't match.
    pub fn add_any_column(&mut self, col: AnyColumn) -> DataFrameResult<()> {
        let name = col.name().to_string();
        self.add_column(&name, col)
    }

    /// Removes a column by name, returning it.
    pub fn drop_column(&mut self, name: &str) -> DataFrameResult<AnyColumn> {
        let idx = self.schema.index_of(name)?;
        self.schema.remove_field(name)?;
        Ok(self.columns.remove(idx))
    }

    /// Renames a column.
    pub fn rename_column(&mut self, old_name: &str, new_name: &str) -> DataFrameResult<()> {
        if self.schema.contains(new_name) {
            return Err(DataFrameError::DuplicateColumn(new_name.to_string()));
        }
        let idx = self.schema.index_of(old_name)?;
        self.schema.rename(old_name, new_name)?;
        self.columns[idx].rename_mut(new_name);
        Ok(())
    }

    /// Returns the first `n` rows as a new DataFrame.
    #[must_use]
    pub fn head(&self, n: usize) -> Self {
        let n = n.min(self.nrows());
        self.select_rows(&(0..n).collect::<Vec<_>>())
            .expect("head positions are always in bounds")
    }

    /// Returns the last `n` rows as a new DataFrame.
    #[must_use]
    pub fn tail(&self, n: usize) -> Self {
        let n = n.min(self.nrows());
        let start = self.nrows() - n;
        self.select_rows(&(start..self.nrows()).collect::<Vec<_>>())
            .expect("tail positions are always in bounds")
    }

    /// Returns a new DataFrame with only the specified columns.
    #[must_use]
    pub fn select_columns(&self, names: &[&str]) -> DataFrameResult<Self> {
        let mut result = Self::new();
        for &name in names {
            let col = self.column(name)?.clone();
            result.add_any_column(col)?;
        }
        Ok(result)
    }

    /// Returns a new DataFrame with only the specified row positions.
    ///
    /// # Errors
    ///
    /// Returns an error if any position is out of bounds.
    pub fn select_rows(&self, positions: &[usize]) -> DataFrameResult<Self> {
        let nrows = self.nrows();
        if let Some(&p) = positions.iter().find(|&&p| p >= nrows) {
            return Err(DataFrameError::IndexOutOfBounds {
                index: p,
                length: nrows,
            });
        }
        let columns: Vec<AnyColumn> = self
            .columns
            .iter()
            .map(|col| {
                col.select_rows(positions)
                    .expect("positions validated against DataFrame length")
            })
            .collect();
        let index = self
            .index
            .select(positions)
            .expect("positions validated against DataFrame length");
        Ok(Self {
            columns,
            schema: self.schema.clone(),
            index,
        })
    }

    /// Returns a new DataFrame with rows where the boolean column is `true`.
    #[must_use]
    pub fn filter(&self, mask: &Series<bool>) -> DataFrameResult<Self> {
        if mask.len() != self.nrows() {
            return Err(DataFrameError::DimensionMismatch {
                message: format!(
                    "mask length {} doesn't match DataFrame rows {}",
                    mask.len(),
                    self.nrows()
                ),
            });
        }

        let positions: Vec<usize> = mask
            .data()
            .iter()
            .enumerate()
            .filter(|(_, &keep)| keep)
            .map(|(i, _)| i)
            .collect();

        self.select_rows(&positions)
    }

    /// Returns a transposed version of this DataFrame.
    ///
    /// All columns are cast to `f64` for the transpose.
    #[must_use]
    pub fn transpose(&self) -> DataFrameResult<Self> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let col_names: Vec<String> = (0..nrows).map(|i| format!("{i}")).collect();

        let mut result = DataFrame::new();
        for (ci, col_name) in col_names.iter().enumerate() {
            let mut row_data = Vec::with_capacity(ncols);
            for ri in 0..ncols {
                let val = match &self.columns[ri] {
                    AnyColumn::Float64(s) => s.data()[ci],
                    AnyColumn::Float32(s) => f64::from(s.data()[ci]),
                    AnyColumn::Int64(s) => s.data()[ci] as f64,
                    AnyColumn::Int32(s) => f64::from(s.data()[ci]),
                    AnyColumn::Bool(s) => if s.data()[ci] { 1.0 } else { 0.0 },
                    AnyColumn::Date(s) | AnyColumn::DateTime(s) | AnyColumn::Duration(s) => {
                        s.data()[ci] as f64
                    }
                    AnyColumn::Utf8(_) => {
                        return Err(DataFrameError::InvalidOperation(
                            "cannot transpose a DataFrame with string columns".to_string(),
                        ))
                    }
                };
                row_data.push(val);
            }
            result.add_column(col_name, row_data)?;
        }

        // Set row names from original column names.
        let row_names: Vec<String> = self.column_names().into_iter().map(String::from).collect();
        result.index = Index::labels(row_names);

        Ok(result)
    }

    /// Sorts the DataFrame by a single column.
    #[must_use]
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> DataFrameResult<Self> {
        let col = self.column(column_name)?;

        let mut positions: Vec<usize> = (0..self.nrows()).collect();

        match col {
            AnyColumn::Float64(s) => {
                let data = s.data();
                if ascending {
                    positions.sort_by(|&a, &b| {
                        data[a]
                            .partial_cmp(&data[b])
                            .unwrap_or(core::cmp::Ordering::Equal)
                    });
                } else {
                    positions.sort_by(|&a, &b| {
                        data[b]
                            .partial_cmp(&data[a])
                            .unwrap_or(core::cmp::Ordering::Equal)
                    });
                }
            }
            AnyColumn::Int64(s) => {
                let data = s.data();
                if ascending {
                    positions.sort_by(|&a, &b| data[a].cmp(&data[b]));
                } else {
                    positions.sort_by(|&a, &b| data[b].cmp(&data[a]));
                }
            }
            AnyColumn::Utf8(s) => {
                let data = s.data();
                if ascending {
                    positions.sort_by(|&a, &b| data[a].cmp(&data[b]));
                } else {
                    positions.sort_by(|&a, &b| data[b].cmp(&data[a]));
                }
            }
            _ => {
                return Err(DataFrameError::InvalidOperation(format!(
                    "sorting by column of type {} is not yet supported",
                    col.dtype()
                )));
            }
        }

        self.select_rows(&positions)
    }

    /// Returns a new DataFrame with duplicate rows removed.
    #[must_use]
    pub fn drop_duplicates(&self, column_names: &[&str]) -> DataFrameResult<Self> {
        let mut seen = alloc::collections::BTreeSet::new();
        let mut positions = Vec::new();

        for row_idx in 0..self.nrows() {
            let mut key = Vec::new();
            for &name in column_names {
                let col = self.column(name)?;
                match col {
                    AnyColumn::Float64(s) => {
                        if let Some(v) = s.data().get(row_idx) {
                            key.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                    }
                    AnyColumn::Int64(s) => {
                        if let Some(v) = s.data().get(row_idx) {
                            key.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    AnyColumn::Utf8(s) => {
                        if let Some(v) = s.data().get(row_idx) {
                            key.extend_from_slice(v.as_bytes());
                            key.push(0); // null terminator
                        }
                    }
                    _ => {
                        return Err(DataFrameError::InvalidOperation(format!(
                            "drop_duplicates not supported for column type {}",
                            col.dtype()
                        )));
                    }
                }
            }

            if seen.insert(key) {
                positions.push(row_idx);
            }
        }

        self.select_rows(&positions)
    }

    /// Returns a new DataFrame with null values in the specified column dropped.
    #[must_use]
    pub fn drop_nulls(&self, column_name: &str) -> DataFrameResult<Self> {
        let col = self.column(column_name)?;
        let positions: Vec<usize> = (0..col.len())
            .filter(|&i| !col.is_null(i))
            .collect();
        self.select_rows(&positions)
    }

    /// Fills null values in the specified column with a constant value (as f64) — in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not found or not f64.
    pub fn fill_nulls(&mut self, column_name: &str, fill_value: f64) -> DataFrameResult<()> {
        let col = self.columns.iter_mut().find(|c| c.name() == column_name)
            .ok_or_else(|| DataFrameError::ColumnNotFound(column_name.to_string()))?;

        match col {
            AnyColumn::Float64(s) => {
                for i in 0..s.len() {
                    if s.is_null(i) {
                        s.data_mut()[i] = fill_value;
                        s.set_valid(i);
                    }
                }
            }
            other => {
                return Err(DataFrameError::TypeMismatch {
                    expected: "f64",
                    actual: other.dtype().name(),
                });
            }
        }

        Ok(())
    }

    /// Forward-fills null values in the specified column — in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not found or the type is unsupported.
    pub fn forward_fill(&mut self, column_name: &str) -> DataFrameResult<()> {
        let col = self.columns.iter_mut().find(|c| c.name() == column_name)
            .ok_or_else(|| DataFrameError::ColumnNotFound(column_name.to_string()))?;

        match col {
            AnyColumn::Float64(s) => {
                let mut last_valid: Option<f64> = None;
                for i in 0..s.len() {
                    if s.is_null(i) {
                        if let Some(v) = last_valid {
                            s.data_mut()[i] = v;
                            s.set_valid(i);
                        }
                    } else {
                        last_valid = Some(s.data()[i]);
                    }
                }
            }
            AnyColumn::Int64(s) => {
                let mut last_valid: Option<i64> = None;
                for i in 0..s.len() {
                    if s.is_null(i) {
                        if let Some(v) = last_valid {
                            s.data_mut()[i] = v;
                            s.set_valid(i);
                        }
                    } else {
                        last_valid = Some(s.data()[i]);
                    }
                }
            }
            other => {
                return Err(DataFrameError::InvalidOperation(format!(
                    "forward_fill not supported for column type {}",
                    other.dtype()
                )));
            }
        }

        Ok(())
    }

    /// Collects all columns into a new vector, consuming `self`.
    #[must_use]
    pub fn into_columns(self) -> Vec<AnyColumn> {
        self.columns
    }

    /// Returns the underlying columns by reference.
    #[must_use]
    pub fn columns(&self) -> &[AnyColumn] {
        &self.columns
    }

    /// Sets one or more columns as a hierarchical `MultiIndex`. The
    /// specified columns are removed from the data and placed into the
    /// index (outermost-first in the order given). Returns a new
    /// `DataFrame`.
    ///
    /// # Errors
    ///
    /// Returns an error if any column name is not found.
    pub fn set_index(&self, col_names: &[&str]) -> DataFrameResult<Self> {
        let mut levels: Vec<Vec<String>> = Vec::with_capacity(col_names.len());
        for &name in col_names {
            let col = self.column(name)?;
            let level: Vec<String> = (0..self.nrows())
                .map(|r| col.get_str(r))
                .collect::<DataFrameResult<Vec<_>>>()?;
            levels.push(level);
        }
        // Build new DataFrame without the index columns.
        let remaining: Vec<&str> = self
            .column_names()
            .into_iter()
            .filter(|n| !col_names.contains(n))
            .collect();
        let mut result = self.select_columns(&remaining)?;
        result.index = crate::index::Index::MultiIndex(levels);
        Ok(result)
    }

    /// Moves the current hierarchical index back into regular columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is not a `MultiIndex`.
    pub fn reset_index(&self) -> DataFrameResult<Self> {
        match &self.index {
            crate::index::Index::MultiIndex(levels) => {
                let mut result = DataFrame::new();
                let nrows = self.nrows();
                // Add index columns first.
                for (i, level) in levels.iter().enumerate() {
                    let name = alloc::format!("level_{i}");
                    let col = AnyColumn::Utf8(Series::new(name.clone(), level.clone()));
                    result.add_any_column(col)?;
                }
                // Add original data columns.
                for name in self.column_names() {
                    let col = self.column(name)?;
                    let gathered = col.select_rows(&(0..nrows).collect::<Vec<_>>())?;
                    result.add_any_column(gathered)?;
                }
                result.index = crate::index::Index::default_range(nrows);
                Ok(result)
            }
            _ => Err(DataFrameError::InvalidOperation(
                "reset_index requires a MultiIndex".to_string(),
            )),
        }
    }

    /// Swaps two levels of a `MultiIndex`. Level 0 is outermost.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is not a `MultiIndex` or if
    /// `level_a`/`level_b` are out of range.
    /// Swaps two MultiIndex levels — in-place.
    pub fn swaplevel(&mut self, level_a: usize, level_b: usize) -> DataFrameResult<()> {
        match &mut self.index {
            crate::index::Index::MultiIndex(levels) => {
                if level_a >= levels.len() || level_b >= levels.len() {
                    return Err(DataFrameError::IndexOutOfBounds {
                        index: level_a.max(level_b),
                        length: levels.len(),
                    });
                }
                levels.swap(level_a, level_b);
                Ok(())
            }
            _ => Err(DataFrameError::InvalidOperation(
                "swaplevel requires a MultiIndex".to_string(),
            )),
        }
    }

    /// Sets a labels index — in-place.
    pub fn set_labels_index(&mut self, labels: Vec<String>) -> DataFrameResult<()> {
        if labels.len() != self.nrows() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "index length {} does not match row count {}",
                    labels.len(),
                    self.nrows()
                ),
            });
        }
        self.index = crate::index::Index::Labels(labels);
        Ok(())
    }

    /// Resamples a `Date` or `DateTime` column into time buckets and
    /// aggregates numeric columns within each bucket.
    ///
    /// `date_col` names the column containing dates (stored as i64
    /// days-since-epoch for `Date`, or microseconds-since-epoch for
    /// `DateTime`). `bucket_days` specifies the bucket width in days.
    /// All numeric columns are summed within each bucket; non-numeric
    /// columns are dropped.
    ///
    /// Returns a new `DataFrame` with one row per non-empty bucket.
    ///
    /// # Errors
    ///
    /// Returns an error if `date_col` is not found or is non-numeric.
    pub fn resample(&self, date_col: &str, bucket_days: i64) -> DataFrameResult<Self> {
        if bucket_days <= 0 {
            return Err(DataFrameError::InvalidOperation(
                "bucket_days must be positive".to_string(),
            ));
        }
        let col = self.column(date_col)?;
        let timestamps: Vec<i64> = match col {
            AnyColumn::Date(s) => s.data().to_vec(),
            AnyColumn::DateTime(s) => {
                // Convert microseconds to days (integer division).
                s.data().iter().map(|&us| us / 86_400_000_000).collect()
            }
            AnyColumn::Int64(s) => s.data().to_vec(),
            _ => {
                return Err(DataFrameError::InvalidOperation(alloc::format!(
                    "resample column '{}' must be Date, DateTime, or Int64",
                    date_col
                )));
            }
        };
        if timestamps.is_empty() {
            return Ok(DataFrame::new());
        }
        // Compute bucket id for each row: floor(ts / bucket_days).
        let bucket_ids: Vec<i64> = timestamps
            .iter()
            .map(|&ts| {
                if ts >= 0 {
                    ts / bucket_days
                } else {
                    (ts - bucket_days + 1) / bucket_days
                }
            })
            .collect();
        // Collect unique sorted bucket ids.
        let mut unique_buckets: Vec<i64> = bucket_ids.clone();
        unique_buckets.sort_unstable();
        unique_buckets.dedup();
        // Build result: one row per bucket.
        let mut result = DataFrame::new();
        let bucket_col = AnyColumn::Int64(Series::new("bucket", unique_buckets.clone()));
        result.add_any_column(bucket_col)?;
        // For each numeric column, aggregate by bucket.
        for name in self.column_names() {
            if name == date_col {
                continue;
            }
            let src = self.column(name)?;
            match src {
                AnyColumn::Float64(s) => {
                    let mut sums: alloc::collections::BTreeMap<i64, f64> =
                        alloc::collections::BTreeMap::new();
                    for (i, &bid) in bucket_ids.iter().enumerate() {
                        if !src.is_null(i) {
                            *sums.entry(bid).or_insert(0.0) += s.data()[i];
                        }
                    }
                    let data: Vec<f64> = unique_buckets
                        .iter()
                        .map(|&bid| sums.get(&bid).copied().unwrap_or(0.0))
                        .collect();
                    result.add_any_column(AnyColumn::Float64(Series::new(name, data)))?;
                }
                AnyColumn::Int64(s) => {
                    let mut sums: alloc::collections::BTreeMap<i64, i64> =
                        alloc::collections::BTreeMap::new();
                    for (i, &bid) in bucket_ids.iter().enumerate() {
                        if !src.is_null(i) {
                            *sums.entry(bid).or_insert(0) += s.data()[i];
                        }
                    }
                    let data: Vec<i64> = unique_buckets
                        .iter()
                        .map(|&bid| sums.get(&bid).copied().unwrap_or(0))
                        .collect();
                    result.add_any_column(AnyColumn::Int64(Series::new(name, data)))?;
                }
                _ => {} // skip non-numeric
            }
        }
        Ok(result)
    }

    // ── ML / AI / Data Science methods ────────────────────────────

    /// Returns a new DataFrame with `n` randomly sampled rows (without replacement).
    /// If `n` exceeds row count, returns all rows shuffled.
    /// Uses a simple LCG PRNG seeded by `seed` for reproducibility.
    #[must_use]
    pub fn sample(&self, n: usize, seed: u64) -> DataFrameResult<Self> {
        let nrows = self.nrows();
        if nrows == 0 {
            return Ok(DataFrame::new());
        }
        let n = n.min(nrows);
        // Fisher-Yates shuffle on indices via LCG.
        let mut indices: Vec<usize> = (0..nrows).collect();
        let mut state = seed;
        for i in (1..nrows).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (state as usize) % (i + 1);
            indices.swap(i, j);
        }
        indices.truncate(n);
        indices.sort_unstable();
        let mut result = DataFrame::new();
        for col in &self.columns {
            let gathered = crate::ops::join::gather(col, &indices.iter().map(|&i| (Some(i), Some(0))).collect::<Vec<_>>(), true)?;
            result.add_any_column(gathered)?;
        }
        result.index = Index::labels(
            indices.iter().map(|&i| alloc::format!("sample_{i}")).collect(),
        );
        Ok(result)
    }

    /// Returns a new DataFrame with all rows shuffled.
    /// Uses a simple LCG PRNG seeded by `seed` for reproducibility.
    #[must_use]
    pub fn shuffle(&self, seed: u64) -> DataFrameResult<Self> {
        self.sample(self.nrows(), seed)
    }

    /// Returns the quantile value at `q` (0.0–1.0) for the given column.
    /// Uses linear interpolation.
    pub fn percentile(&self, column_name: &str, q: f64) -> DataFrameResult<f64> {
        let col = self.column(column_name)?;
        let mut vals = col.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = vals.len();
        let rank = q * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            Ok(vals[lo])
        } else {
            let frac = rank - lo as f64;
            Ok(vals[lo] * (1.0 - frac) + vals[hi] * frac)
        }
    }

    /// Returns the weighted mean of the given column using `weights`.
    pub fn weighted_mean(&self, column_name: &str, weights: &[f64]) -> DataFrameResult<f64> {
        let col = self.column(column_name)?;
        let vals = col.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        if weights.len() != self.nrows() {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "weights length {} does not match row count {}",
                    weights.len(),
                    self.nrows()
                ),
            });
        }
        let mut w_sum = 0.0f64;
        let mut vw_sum = 0.0f64;
        for i in 0..self.nrows() {
            if !col.is_null(i) {
                let v = vals.get(i).copied().unwrap_or(0.0);
                vw_sum += v * weights[i];
                w_sum += weights[i];
            }
        }
        if w_sum == 0.0 {
            return Err(DataFrameError::InvalidOperation(
                "sum of weights is zero".to_string(),
            ));
        }
        Ok(vw_sum / w_sum)
    }

    /// Returns the Pearson correlation matrix for all numeric columns.
    /// Result is a square DataFrame with column names as both row and column labels.
    pub fn corr(&self) -> DataFrameResult<Self> {
        let names: Vec<&str> = self.column_names();
        let numeric: Vec<&str> = names.iter()
            .filter(|&&n| self.column(n).map(|c| c.dtype().is_numeric()).unwrap_or(false))
            .copied()
            .collect();
        let n = numeric.len();
        let mut matrix = vec![vec![0.0f64; n]; n];

        for i in 0..n {
                let _vi = self.column(numeric[i])?.valid_f64()?;
                for j in i..n {
                    let _vj = self.column(numeric[j])?.valid_f64()?;
                // Align on non-null positions.
                let col_i = self.column(numeric[i])?;
                let col_j = self.column(numeric[j])?;
                let mut xi = Vec::new();
                let mut xj = Vec::new();
                for row in 0..self.nrows() {
                    if !col_i.is_null(row) && !col_j.is_null(row) {
                        xi.push(col_i.to_f64()?.data()[row]);
                        xj.push(col_j.to_f64()?.data()[row]);
                    }
                }
                let len = xi.len() as f64;
                if len < 2.0 {
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 0.0;
                    continue;
                }
                let mean_i = xi.iter().sum::<f64>() / len;
                let mean_j = xj.iter().sum::<f64>() / len;
                let mut cov = 0.0f64;
                let mut var_i = 0.0f64;
                let mut var_j = 0.0f64;
                for k in 0..xi.len() {
                    let di = xi[k] - mean_i;
                    let dj = xj[k] - mean_j;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }
                let corr = if var_i == 0.0 || var_j == 0.0 {
                    0.0
                } else {
                    cov / (var_i * var_j).sqrt()
                };
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        // Build output DataFrame.
        let mut result = DataFrame::new();
        for (i, &name) in numeric.iter().enumerate() {
            let col: Vec<f64> = matrix[i].clone();
            result.add_column(name, col)?;
        }
        result.index = Index::labels(numeric.into_iter().map(String::from).collect());
        Ok(result)
    }

    /// Returns the covariance matrix for all numeric columns.
    pub fn cov(&self) -> DataFrameResult<Self> {
        let names: Vec<&str> = self.column_names();
        let numeric: Vec<&str> = names.iter()
            .filter(|&&n| self.column(n).map(|c| c.dtype().is_numeric()).unwrap_or(false))
            .copied()
            .collect();
        let n = numeric.len();
        let mut matrix = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            for j in i..n {
                let col_i = self.column(numeric[i])?;
                let col_j = self.column(numeric[j])?;
                let mut xi = Vec::new();
                let mut xj = Vec::new();
                for row in 0..self.nrows() {
                    if !col_i.is_null(row) && !col_j.is_null(row) {
                        xi.push(col_i.to_f64()?.data()[row]);
                        xj.push(col_j.to_f64()?.data()[row]);
                    }
                }
                let len = xi.len() as f64;
                if len < 2.0 {
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 0.0;
                    continue;
                }
                let mean_i = xi.iter().sum::<f64>() / len;
                let mean_j = xj.iter().sum::<f64>() / len;
                let cov: f64 = xi.iter().zip(xj.iter())
                    .map(|(&a, &b)| (a - mean_i) * (b - mean_j))
                    .sum::<f64>() / (len - 1.0);
                matrix[i][j] = cov;
                matrix[j][i] = cov;
            }
        }

        let mut result = DataFrame::new();
        for (i, &name) in numeric.iter().enumerate() {
            let col: Vec<f64> = matrix[i].clone();
            result.add_column(name, col)?;
        }
        result.index = Index::labels(numeric.into_iter().map(String::from).collect());
        Ok(result)
    }

    /// One-hot encodes the given categorical column. Returns a new DataFrame
    /// with the original columns (minus the source column) plus one Bool column
    /// per unique value named `{column}_{value}`.
    pub fn one_hot_encode(&self, column_name: &str) -> DataFrameResult<Self> {
        let col = self.column(column_name)?;
        // Collect unique non-null values in order of first appearance.
        let mut seen = alloc::collections::BTreeMap::<String, usize>::new();
        let mut order = Vec::new();
        for i in 0..self.nrows() {
            if !col.is_null(i) {
                let val = col.get_str(i).unwrap_or_default().to_string();
                if !seen.contains_key(&val) {
                    seen.insert(val.clone(), order.len());
                    order.push(val);
                }
            }
        }

        let mut result = DataFrame::new();
        // Copy non-source columns.
        for name in self.column_names() {
            if name != column_name {
                result.add_any_column(self.column(name)?.clone())?;
            }
        }
        // Add one Bool column per unique value.
        for val in &order {
            let flags: Vec<bool> = (0..self.nrows()).map(|i| {
                if col.is_null(i) { false }
                else { col.get_str(i).unwrap_or_default() == val.as_str() }
            }).collect();
            let cname = alloc::format!("{column_name}_{val}");
            result.add_column(&cname, flags)?;
        }
        Ok(result)
    }

    /// Fixed-width binning. Returns a new DataFrame with the specified column replaced by bin indices (0-based).
    /// `bins` specifies the number of equal-width bins. Values outside the range
    /// are assigned to the first/last bin.
    pub fn cut(&self, column_name: &str, bins: usize) -> DataFrameResult<AnyColumn> {
        if bins == 0 {
            return Err(DataFrameError::InvalidOperation(
                "bins must be at least 1".to_string(),
            ));
        }
        let col = self.column(column_name)?;
        let vals = col.valid_f64()?;
        if vals.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let width = (max - min) / bins as f64;
        let s = col.to_f64()?;
        let bin_indices: Vec<i64> = s.data().iter().enumerate().map(|(i, &v)| {
            if s.is_null(i) || width == 0.0 {
                0
            } else {
                let idx = ((v - min) / width).floor() as i64;
                idx.min((bins as i64) - 1).max(0)
            }
        }).collect();
        Ok(AnyColumn::Int64(Series::new(column_name, bin_indices)))
    }

    /// Quantile binning. Returns a new DataFrame with the specified column replaced by quantile indices (0-based).
    /// `q` specifies the number of equal-frequency bins.
    pub fn qcut(&self, column_name: &str, q: usize) -> DataFrameResult<AnyColumn> {
        if q == 0 {
            return Err(DataFrameError::InvalidOperation(
                "q must be at least 1".to_string(),
            ));
        }
        let col = self.column(column_name)?;
        let mut vals_with_idx: Vec<(f64, usize)> = Vec::new();
        for i in 0..self.nrows() {
            if !col.is_null(i) {
                let v = col.to_f64()?.data()[i];
                vals_with_idx.push((v, i));
            }
        }
        if vals_with_idx.is_empty() {
            return Err(DataFrameError::EmptyDataFrame);
        }
        vals_with_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        let n = vals_with_idx.len();
        let mut quantile_indices = vec![0i64; self.nrows()];
        for (rank, &(_, orig_idx)) in vals_with_idx.iter().enumerate() {
            let bin = ((rank * q) / n).min(q - 1);
            quantile_indices[orig_idx] = bin as i64;
        }
        Ok(AnyColumn::Int64(Series::new(column_name, quantile_indices)))
    }

    /// Interpolates null values in the specified column.
    /// `method` is `"linear"`, `"forward"`, or `"backward"`.
    pub fn interpolate(&mut self, column_name: &str, method: &str) -> DataFrameResult<()> {
        match method {
            "forward" => self.forward_fill(column_name)?,
            "backward" => {
                let col = self.columns.iter_mut().find(|c| c.name() == column_name)
                    .ok_or_else(|| DataFrameError::ColumnNotFound(column_name.to_string()))?;
                match col {
                    AnyColumn::Float64(s) => {
                        let mut last_valid: Option<f64> = None;
                        for i in (0..s.len()).rev() {
                            if s.is_null(i) {
                                if let Some(v) = last_valid {
                                    s.data_mut()[i] = v;
                                    s.set_valid(i);
                                }
                            } else {
                                last_valid = Some(s.data()[i]);
                            }
                        }
                    }
                    AnyColumn::Int64(s) => {
                        let mut last_valid: Option<i64> = None;
                        for i in (0..s.len()).rev() {
                            if s.is_null(i) {
                                if let Some(v) = last_valid {
                                    s.data_mut()[i] = v;
                                    s.set_valid(i);
                                }
                            } else {
                                last_valid = Some(s.data()[i]);
                            }
                        }
                    }
                    other => {
                        return Err(DataFrameError::InvalidOperation(alloc::format!(
                            "interpolate backward not supported for column type {}",
                            other.dtype()
                        )));
                    }
                }
            }
            "linear" => {
                let col = self.columns.iter_mut().find(|c| c.name() == column_name)
                    .ok_or_else(|| DataFrameError::ColumnNotFound(column_name.to_string()))?;
                match col {
                    AnyColumn::Float64(s) => {
                        let len = s.len();
                        let mut last_valid_idx: Option<usize> = None;
                        let mut last_valid_val: Option<f64> = None;
                        for i in 0..len {
                            if s.is_null(i) {
                                // Find next valid.
                                let mut next_idx = None;
                                let mut next_val = None;
                                for j in (i + 1)..len {
                                    if !s.is_null(j) {
                                        next_idx = Some(j);
                                        next_val = Some(s.data()[j]);
                                        break;
                                    }
                                }
                                if let (Some(lv), Some(lf), Some(nv), Some(ni)) =
                                    (last_valid_val, last_valid_idx, next_val, next_idx)
                                {
                                    let t = (i - lf) as f64 / (ni - lf) as f64;
                                    s.data_mut()[i] = lv + t * (nv - lv);
                                    s.set_valid(i);
                                } else if let Some(lv) = last_valid_val {
                                    // Forward fill from last valid.
                                    s.data_mut()[i] = lv;
                                    s.set_valid(i);
                                } else if let Some(nv) = next_val {
                                    // Backward fill from next valid.
                                    s.data_mut()[i] = nv;
                                    s.set_valid(i);
                                }
                            } else {
                                last_valid_idx = Some(i);
                                last_valid_val = Some(s.data()[i]);
                            }
                        }
                    }
                    other => {
                        return Err(DataFrameError::InvalidOperation(alloc::format!(
                            "interpolate linear not supported for column type {}",
                            other.dtype()
                        )));
                    }
                }
            }
            other => {
                return Err(DataFrameError::InvalidOperation(alloc::format!(
                    "unknown interpolate method '{other}', expected linear|forward|backward"
                )));
            }
        }
        Ok(())
    }

    /// Extended describe with custom quantile percentiles.
    /// Returns a DataFrame with rows for each stat: count, mean, std, min,
    /// then each custom quantile, then max.
    pub fn describe_with_quantiles(&self, quantiles: &[f64]) -> DataFrameResult<Self> {
        let numeric_cols: Vec<&str> = self.column_names().iter()
            .filter(|&&n| self.column(n).map(|c| c.dtype().is_numeric()).unwrap_or(false))
            .copied()
            .collect();
        let mut stat_names: Vec<String> = Vec::new();
        stat_names.push("count".into());
        stat_names.push("mean".into());
        stat_names.push("std".into());
        stat_names.push("min".into());
        for &q in quantiles {
            stat_names.push(alloc::format!("q{:.0}%", q * 100.0));
        }
        stat_names.push("max".into());

        let mut result = DataFrame::new();
        for &col_name in &numeric_cols {
            let vals = self.column(col_name)?.valid_f64()?;
            let n = vals.len();
            let count = n as f64;
            let mean = if n > 0 { vals.iter().sum::<f64>() / count } else { 0.0 };
            let std = if n > 1 {
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                var.sqrt()
            } else { 0.0 };
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let min = sorted.first().copied().unwrap_or(0.0);
            let max = sorted.last().copied().unwrap_or(0.0);
            let mut stats = vec![count, mean, std, min];
            for &q in quantiles {
                let rank = q * (n as f64 - 1.0);
                let lo = rank.floor() as usize;
                let hi = rank.ceil() as usize;
                let val = if lo == hi || n == 0 {
                    sorted.get(lo).copied().unwrap_or(0.0)
                } else {
                    let frac = rank - lo as f64;
                    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
                };
                stats.push(val);
            }
            stats.push(max);
            result.add_column(col_name, stats)?;
        }
        result.index = Index::labels(stat_names);
        Ok(result)
    }

    /// Extracts date components from Date/DateTime/Duration columns.
    /// `parts` is a list of: `"year"`, `"month"`, `"day"`, `"hour"`, `"dow"`.
    /// Returns a new DataFrame with one Int64 column per part per date column.
    pub fn dt_extract(&self, parts: &[&str]) -> DataFrameResult<Self> {
        let mut result = DataFrame::new();
        // Copy non-date columns.
        for name in self.column_names() {
            let col = self.column(name)?;
            match col {
                AnyColumn::Date(_) | AnyColumn::DateTime(_) => {}
                _ => { result.add_any_column(col.clone())?; }
            }
        }
        for name in self.column_names() {
            let col = self.column(name)?;
            match col {
                AnyColumn::Date(s) => {
                    let ts: Vec<i64> = s.data().to_vec();
                    for &part in parts {
                        let extracted: Vec<i64> = ts.iter().map(|&days| {
                            match part {
                                "year" => {
                                    // Rough epoch-based year.
                                    1970 + (days / 365)
                                }
                                "month" => {
                                    // Rough month from day-of-year.
                                    ((days % 365) / 30 + 1).min(12)
                                }
                                "day" => {
                                    days % 30 + 1
                                }
                                "dow" => {
                                    // Day of week (0=Mon).
                                    ((days % 7) + 7) % 7
                                }
                                _ => 0,
                            }
                        }).collect();
                        let cname = alloc::format!("{name}_{part}");
                        result.add_column(&cname, extracted)?;
                    }
                }
                AnyColumn::DateTime(s) => {
                    let ts: Vec<i64> = s.data().to_vec(); // microseconds
                    for &part in parts {
                        let extracted: Vec<i64> = ts.iter().map(|&us| {
                            let secs = us / 1_000_000;
                            match part {
                                "year" => 1970 + (secs / 31_557_600),
                                "month" => ((secs % 31_557_600) / 2_592_000 + 1).min(12),
                                "day" => (secs % 2_592_000) / 86400 + 1,
                                "hour" => (secs % 86400) / 3600,
                                "dow" => (secs / 86400 % 7 + 7) % 7,
                                _ => 0,
                            }
                        }).collect();
                        let cname = alloc::format!("{name}_{part}");
                        result.add_column(&cname, extracted)?;
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    /// Matrix dot product: `self` × `other`. Both must be all-numeric.
    /// `self` has shape (m × n), `other` has shape (n × p).
    /// Returns DataFrame of shape (m × p) with columns named `col_0`, `col_1`, etc.
    pub fn dot(&self, other: &DataFrame) -> DataFrameResult<Self> {
        let m = self.nrows();
        let n = self.ncols();
        if other.nrows() != n {
            return Err(DataFrameError::DimensionMismatch {
                message: alloc::format!(
                    "dot: left cols ({}) != right rows ({})",
                    n, other.nrows()
                ),
            });
        }
        let p = other.ncols();
        // Convert self to row-major f64 matrix.
        let mut left = vec![vec![0.0f64; n]; m];
        for (ci, name) in self.column_names().iter().enumerate() {
            let col = self.column(name)?;
            let f64col = col.to_f64()?;
            for ri in 0..m {
                left[ri][ci] = f64col.data()[ri];
            }
        }
        // Convert other to row-major f64 matrix.
        let mut right = vec![vec![0.0f64; p]; n];
        for (ci, name) in other.column_names().iter().enumerate() {
            let col = other.column(name)?;
            let f64col = col.to_f64()?;
            for ri in 0..n {
                right[ri][ci] = f64col.data()[ri];
            }
        }
        // Compute result.
        let mut result = DataFrame::new();
        for col_j in 0..p {
            let mut out_col = vec![0.0f64; m];
            for row_i in 0..m {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += left[row_i][k] * right[k][col_j];
                }
                out_col[row_i] = sum;
            }
            result.add_column(&alloc::format!("col_{col_j}"), out_col)?;
        }
        Ok(result)
    }

    /// Column profiling: returns a DataFrame with per-column statistics.
    /// For each column: dtype, count, null_count, unique_count, min, max, mean, std, skew.
    pub fn profile(&self) -> DataFrameResult<Self> {
        let mut col_names = Vec::new();
        let mut dtypes = Vec::new();
        let mut counts = Vec::new();
        let mut null_counts = Vec::new();
        let mut unique_counts = Vec::new();
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut skews = Vec::new();

        for name in self.column_names() {
            let col = self.column(name)?;
            col_names.push(name.to_string());
            dtypes.push(alloc::format!("{}", col.dtype()));
            let n = col.len();
            let nc = col.null_count();
            counts.push(n as f64);
            null_counts.push(nc as f64);

            if col.dtype().is_numeric() {
                let vals = col.valid_f64()?;
                let uniq: alloc::collections::BTreeSet<u64> = vals.iter()
                    .map(|v| v.to_bits())
                    .collect();
                unique_counts.push(uniq.len() as f64);
                if !vals.is_empty() {
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / (vals.len() as f64 - 1.0).max(1.0);
                    let std = variance.sqrt();
                    let skew = if std > 0.0 && vals.len() > 2 {
                        vals.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>()
                            / vals.len() as f64
                    } else { 0.0 };
                    mins.push(min);
                    maxs.push(max);
                    means.push(mean);
                    stds.push(std);
                    skews.push(skew);
                } else {
                    mins.push(0.0);
                    maxs.push(0.0);
                    means.push(0.0);
                    stds.push(0.0);
                    skews.push(0.0);
                }
            } else {
                unique_counts.push(0.0);
                mins.push(0.0);
                maxs.push(0.0);
                means.push(0.0);
                stds.push(0.0);
                skews.push(0.0);
            }
        }

        let mut result = DataFrame::new();
        result.add_column("column", col_names)?;
        result.add_column("dtype", dtypes)?;
        result.add_column("count", counts)?;
        result.add_column("null_count", null_counts)?;
        result.add_column("unique_count", unique_counts)?;
        result.add_column("min", mins)?;
        result.add_column("max", maxs)?;
        result.add_column("mean", means)?;
        result.add_column("std", stds)?;
        result.add_column("skew", skews)?;
        Ok(result)
    }

    /// Random train/test split. Returns `(train, test)`.
    /// `test_size` is the fraction of rows for the test set (0.0–1.0).
    /// Uses Fisher-Yates shuffle via LCG seeded by `seed`.
    pub fn train_test_split(&self, test_size: f64, seed: u64) -> DataFrameResult<(Self, Self)> {
        if !(0.0..1.0).contains(&test_size) {
            return Err(DataFrameError::InvalidOperation(
                "test_size must be between 0.0 and 1.0".to_string(),
            ));
        }
        let nrows = self.nrows();
        let n_test = (nrows as f64 * test_size).round() as usize;
        let n_train = nrows - n_test;

        // Shuffle indices.
        let mut indices: Vec<usize> = (0..nrows).collect();
        let mut state = seed;
        for i in (1..nrows).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (state as usize) % (i + 1);
            indices.swap(i, j);
        }

        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        let mut train = DataFrame::new();
        let mut test = DataFrame::new();
        for col in &self.columns {
            let train_gathered = crate::ops::join::gather(
                col,
                &train_indices.iter().map(|&i| (Some(i), Some(0))).collect::<Vec<_>>(),
                true,
            )?;
            train.add_any_column(train_gathered)?;
            let test_gathered = crate::ops::join::gather(
                col,
                &test_indices.iter().map(|&i| (Some(i), Some(0))).collect::<Vec<_>>(),
                true,
            )?;
            test.add_any_column(test_gathered)?;
        }
        Ok((train, test))
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_rows = 10;
        let nrows = self.nrows();

        // Column widths
        let mut widths: Vec<usize> = self.column_names().iter().map(|n| n.len()).collect();

        // Compute widths for each column based on displayed values
        for (ci, col) in self.columns.iter().enumerate() {
            for ri in 0..nrows.min(max_rows) {
                let cell = if col.is_null(ri) {
                    "null".to_string()
                } else {
                    match col {
                        AnyColumn::Float64(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Float32(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Int64(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Int32(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Bool(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Utf8(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Date(s) => format!("Date({})", s.data()[ri]),
                        AnyColumn::DateTime(s) => format!("DateTime({})", s.data()[ri]),
                        AnyColumn::Duration(s) => format!("Duration({})", s.data()[ri]),
                    }
                };
                widths[ci] = widths[ci].max(cell.len());
            }
        }

        // Header
        for (ci, name) in self.column_names().iter().enumerate() {
            if ci > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{:>width$}", name, width = widths[ci])?;
        }
        writeln!(f)?;

        // Separator
        for (ci, &w) in widths.iter().enumerate() {
            if ci > 0 {
                write!(f, "-+-")?;
            }
            for _ in 0..w {
                write!(f, "-")?;
            }
        }
        writeln!(f)?;

        // Data rows
        let display_rows = nrows.min(max_rows);
        for ri in 0..display_rows {
            for (ci, col) in self.columns.iter().enumerate() {
                if ci > 0 {
                    write!(f, " | ")?;
                }
                let cell = if col.is_null(ri) {
                    "null".to_string()
                } else {
                    match col {
                        AnyColumn::Float64(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Float32(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Int64(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Int32(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Bool(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Utf8(s) => format!("{}", s.data()[ri]),
                        AnyColumn::Date(s) => format!("Date({})", s.data()[ri]),
                        AnyColumn::DateTime(s) => format!("DateTime({})", s.data()[ri]),
                        AnyColumn::Duration(s) => format!("Duration({})", s.data()[ri]),
                    }
                };
                write!(f, "{:>width$}", cell, width = widths[ci])?;
            }
            writeln!(f)?;
        }

        if nrows > max_rows {
            writeln!(f, "... ({nrows} rows total)")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn empty_dataframe() {
        let df = DataFrame::new();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
        assert!(df.is_empty());
    }

    #[test]
    fn add_columns() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0, 3.0]).unwrap();
        df.add_column("y", vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
        assert!(df.has_column("x"));
        assert!(df.has_column("y"));
    }

    #[test]
    fn column_mismatch_error() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0]).unwrap();
        let result = df.add_column("y", vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_column_error() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0]).unwrap();
        let result = df.add_column("x", vec![2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn select_columns() {
        let mut df = DataFrame::new();
        df.add_column("a", vec![1, 2, 3]).unwrap();
        df.add_column("b", vec![4, 5, 6]).unwrap();
        df.add_column("c", vec![7, 8, 9]).unwrap();

        let sub = df.select_columns(&["a", "c"]).unwrap();
        assert_eq!(sub.ncols(), 2);
        assert!(sub.has_column("a"));
        assert!(sub.has_column("c"));
        assert!(!sub.has_column("b"));
    }

    #[test]
    fn head_tail() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1, 2, 3, 4, 5]).unwrap();

        let head = df.head(3);
        assert_eq!(head.nrows(), 3);

        let tail = df.tail(2);
        assert_eq!(tail.nrows(), 2);
    }

    #[test]
    fn sort_by() {
        let mut df = DataFrame::new();
        df.add_column("name", vec![String::from("c"), String::from("a"), String::from("b")])
            .unwrap();
        df.add_column("val", vec![3, 1, 2]).unwrap();

        let sorted = df.sort_by("val", true).unwrap();
        let col = sorted.column("val").unwrap().as_i64().unwrap();
        assert_eq!(col.data(), &[1, 2, 3]);
    }
}
