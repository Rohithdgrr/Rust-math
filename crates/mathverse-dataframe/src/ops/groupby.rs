//! Group-by aggregation, pandas-style.
//!
//! [`DataFrame::group_by`] groups rows by one or more key columns and exposes
//! aggregation operations (`sum`, `mean`, `count`, `min`, `max`, `median`,
//! `std`, `nunique`, `first`, `last`) over the remaining numeric columns.
//!
//! Null key values are excluded from grouping (pandas `dropna=True`
//! behaviour). Groups are emitted in ascending key order (pandas
//! `sort=True` behaviour). Aggregations skip null values within each group.
//!
//! # Examples
//!
//! ```
//! use mathverse_dataframe::DataFrame;
//!
//! let mut df = DataFrame::new();
//! df.add_column("city", vec![String::from("NYC"), String::from("LA"), String::from("NYC")]).unwrap();
//! df.add_column("temp", vec![20.0, 25.0, 22.0]).unwrap();
//!
//! let grouped = df.group_by(&["city"]).unwrap().mean().unwrap();
//! assert_eq!(grouped.nrows(), 2); // NYC, LA
//! let temps = grouped.column("temp_mean").unwrap().as_f64().unwrap();
//! let mut t: Vec<f64> = temps.data().to_vec();
//! t.sort_by(|a, b| a.partial_cmp(b).unwrap());
//! assert!((t[0] - 21.0).abs() < 1e-12); // NYC mean
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::dataframe::DataFrame;
use crate::errors::{DataFrameError, DataFrameResult};

/// A grouped view over a [`DataFrame`], produced by [`DataFrame::group_by`].
#[derive(Debug, Clone)]
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    keys: Vec<String>,
    ascending: bool,
}

/// Aggregation operations available on grouped data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    /// Sum of non-null values.
    Sum,
    /// Arithmetic mean of non-null values.
    Mean,
    /// Count of non-null values.
    Count,
    /// Minimum non-null value.
    Min,
    /// Maximum non-null value.
    Max,
    /// Median of non-null values.
    Median,
    /// Sample standard deviation (ddof=1) of non-null values.
    Std,
    /// Number of distinct non-null values.
    NUnique,
    /// First non-null value in the group.
    First,
    /// Last non-null value in the group.
    Last,
}

impl AggOp {
    /// Returns the suffix used when naming aggregated columns.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Count => "count",
            Self::Min => "min",
            Self::Max => "max",
            Self::Median => "median",
            Self::Std => "std",
            Self::NUnique => "nunique",
            Self::First => "first",
            Self::Last => "last",
        }
    }
}

impl<'a> GroupBy<'a> {
    /// Returns the key column names used for grouping.
    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Sets the sort order for grouped output. Default is ascending.
    pub fn sort_order(&mut self, ascending: bool) {
        self.ascending = ascending;
    }

    /// Aggregates each numeric column not used as a key with the given
    /// operation. Output columns are named `"{column}_{op}"`.
    ///
    /// # Errors
    ///
    /// Returns an error if any aggregated column is non-numeric, or if an
    /// aggregation fails (e.g. `std` on a group with fewer than two values).
    pub fn agg(&self, op: AggOp) -> DataFrameResult<DataFrame> {
        let groups = self.build_groups()?;
        let mut result = DataFrame::new();
        self.add_key_columns(&mut result, &groups)?;

        for name in self.df.column_names() {
            if self.keys.iter().any(|k| k == name) {
                continue;
            }
            let col = self.df.column(name)?;
            if !is_numeric(col) {
                return Err(DataFrameError::InvalidOperation(format!(
                    "groupby {op:?} requires numeric columns; `{name}` is {}",
                    col.dtype()
                )));
            }
            let mut values = Vec::with_capacity(groups.len());
            for (_, rows) in &groups {
                let sub = col.select_rows(rows)?;
                values.push(apply_op(&sub, op)?);
            }
            let out_name = format!("{}_{}", name, op.suffix());
            result.add_column(&out_name, values)?;
        }
        Ok(result)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Sum`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn sum(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Sum)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Mean`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn mean(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Mean)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Count`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn count(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Count)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Min`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn min(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Min)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Max`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn max(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Max)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Median`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn median(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Median)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::Std`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric or a group has
    /// fewer than two values.
    pub fn std(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::Std)
    }

    /// Aggregates all non-key numeric columns with [`AggOp::NUnique`].
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric.
    pub fn nunique(&self) -> DataFrameResult<DataFrame> {
        self.agg(AggOp::NUnique)
    }

    /// Transforms each non-key numeric column by broadcasting the group
    /// aggregation back to the original DataFrame's shape. Returns a new
    /// DataFrame with the same number of rows as the original, where each
    /// row's value is the aggregated value for its group.
    ///
    /// # Errors
    ///
    /// Returns an error if a non-key column is non-numeric or the
    /// aggregation fails.
    pub fn transform(&self, op: AggOp) -> DataFrameResult<DataFrame> {
        let groups = self.build_groups()?;
        let mut result = DataFrame::new();

        // Add key columns unchanged.
        for name in &self.keys {
            result.add_any_column(self.df.column(name)?.clone())?;
        }

        // For each non-key numeric column, compute per-group aggregated
        // values and broadcast back to original positions.
        for name in self.df.column_names() {
            if self.keys.iter().any(|k| k == name) {
                continue;
            }
            let col = self.df.column(name)?;
            if !is_numeric(col) {
                return Err(DataFrameError::InvalidOperation(format!(
                    "transform {op:?} requires numeric columns; `{name}` is {}",
                    col.dtype()
                )));
            }

            // Build a map from row index -> group value.
            let mut row_values: alloc::collections::BTreeMap<usize, f64> =
                alloc::collections::BTreeMap::new();
            for (_, rows) in &groups {
                let sub = col.select_rows(rows)?;
                let val = apply_op(&sub, op)?;
                for &row in rows {
                    row_values.insert(row, val);
                }
            }

            // Build the output column.
            let data: Vec<f64> = (0..self.df.nrows())
                .map(|i| row_values.get(&i).copied().unwrap_or(f64::NAN))
                .collect();
            let out_name = format!("{}_{}", name, op.suffix());
            result.add_column(&out_name, data)?;
        }
        Ok(result)
    }

    /// Groups rows into `(encoded_key, row_indices)` pairs, sorted by key.
    /// Rows with any null key are dropped.
    fn build_groups(&self) -> DataFrameResult<Vec<(Vec<u8>, Vec<usize>)>> {
        let mut groups: alloc::collections::BTreeMap<Vec<u8>, Vec<usize>> =
            alloc::collections::BTreeMap::new();
        for row in 0..self.df.nrows() {
            let mut key = Vec::new();
            let mut key_valid = true;
            for name in &self.keys {
                let col = self.df.column(name)?;
                if col.is_null(row) {
                    key_valid = false;
                    break;
                }
                encode_key_value(col, row, &mut key);
            }
            if key_valid {
                groups.entry(key).or_default().push(row);
            }
        }
        let mut result: Vec<_> = groups.into_iter().collect();
        if !self.ascending {
            result.reverse();
        }
        Ok(result)
    }

    /// Adds the key columns to the result, one row per group.
    fn add_key_columns(&self, result: &mut DataFrame, groups: &[(Vec<u8>, Vec<usize>)]) -> DataFrameResult<()> {
        for name in &self.keys {
            let col = self.df.column(name)?;
            let picks: Vec<usize> = groups.iter().map(|(_, rows)| rows[0]).collect();
            let selected = col.select_rows(&picks)?;
            result.add_any_column(selected)?;
        }
        Ok(())
    }
}

impl DataFrame {
    /// Groups the DataFrame by the given key columns.
    ///
    /// # Errors
    ///
    /// Returns an error if any key column does not exist.
    pub fn group_by(&self, by: &[&str]) -> DataFrameResult<GroupBy<'_>> {
        for name in by {
            if !self.has_column(name) {
                return Err(DataFrameError::ColumnNotFound(name.to_string()));
            }
        }
        Ok(GroupBy {
            df: self,
            keys: by.iter().map(|s| String::from(*s)).collect(),
            ascending: true,
        })
    }

    /// Creates a pivot table by aggregating `values_col` grouped by
    /// `index_col` (rows) and `columns_col` (columns).
    ///
    /// Returns a new DataFrame where:
    /// - The first column contains unique `index_col` values (named `index_col`)
    /// - Each subsequent column is named `{columns_col}={value}` and contains
    ///   the aggregated value for that (index, column) combination
    ///
    /// # Errors
    ///
    /// Returns an error if columns don't exist or `values_col` is non-numeric.
    pub fn pivot_table(
        &self,
        values_col: &str,
        index_col: &str,
        columns_col: &str,
        op: AggOp,
    ) -> DataFrameResult<Self> {
        if !self.has_column(values_col) {
            return Err(DataFrameError::ColumnNotFound(values_col.to_string()));
        }
        if !self.has_column(index_col) {
            return Err(DataFrameError::ColumnNotFound(index_col.to_string()));
        }
        if !self.has_column(columns_col) {
            return Err(DataFrameError::ColumnNotFound(columns_col.to_string()));
        }

        let values = self.column(values_col)?;
        if !is_numeric(values) {
            return Err(DataFrameError::InvalidOperation(format!(
                "pivot_table values column must be numeric; `{values_col}` is {}",
                values.dtype()
            )));
        }

        // Collect unique index values (using encoded keys for ordering).
        let index_col_data = self.column(index_col)?;
        let columns_col_data = self.column(columns_col)?;

        // Build (index_key, column_key) -> Vec<row> mapping.
        let mut cell_map: alloc::collections::BTreeMap<Vec<u8>, alloc::collections::BTreeMap<Vec<u8>, Vec<usize>>> =
            alloc::collections::BTreeMap::new();
        // Track unique column keys in order.
        let mut col_keys_seen: alloc::collections::BTreeMap<Vec<u8>, String> = alloc::collections::BTreeMap::new();
        // Track unique index keys in order.
        let mut idx_keys_seen: alloc::collections::BTreeMap<Vec<u8>, String> = alloc::collections::BTreeMap::new();

        for row in 0..self.nrows() {
            if index_col_data.is_null(row) || columns_col_data.is_null(row) || values.is_null(row) {
                continue;
            }
            let mut idx_key = Vec::new();
            encode_key_value(index_col_data, row, &mut idx_key);
            let mut col_key = Vec::new();
            encode_key_value(columns_col_data, row, &mut col_key);

            // Get display names.
            let idx_name = cell_display_name(index_col_data, row);
            let col_name = cell_display_name(columns_col_data, row);

            idx_keys_seen.entry(idx_key.clone()).or_insert(idx_name);
            col_keys_seen.entry(col_key.clone()).or_insert(col_name);
            cell_map.entry(idx_key).or_default().entry(col_key).or_default().push(row);
        }

        // Build result DataFrame.
        let mut result = DataFrame::new();

        // Index column.
        let idx_values: Vec<String> = idx_keys_seen.values().cloned().collect();
        result.add_column(index_col, idx_values)?;

        // One column per unique column-group value.
        let col_keys: Vec<Vec<u8>> = col_keys_seen.keys().cloned().collect();
        for col_key in &col_keys {
            let col_name = &col_keys_seen[col_key];
            let out_col_name = format!("{columns_col}={col_name}");
            let mut values_out = Vec::with_capacity(idx_keys_seen.len());
            for idx_key in idx_keys_seen.keys() {
                match cell_map.get(idx_key).and_then(|m| m.get(col_key)) {
                    Some(rows) => {
                        let sub = values.select_rows(rows)?;
                        values_out.push(apply_op(&sub, op)?);
                    }
                    None => {
                        values_out.push(f64::NAN);
                    }
                }
            }
            result.add_column(&out_col_name, values_out)?;
        }

        Ok(result)
    }
}

/// Appends an order-preserving byte encoding of a cell to `out`.
fn encode_key_value(col: &AnyColumn, row: usize, out: &mut Vec<u8>) {
    match col {
        AnyColumn::Float64(s) => {
            let bits = s.data()[row].to_bits();
            out.extend_from_slice(&order_preserving_f64(bits).to_be_bytes());
        }
        AnyColumn::Float32(s) => {
            let bits = f64::from(s.data()[row]).to_bits();
            out.extend_from_slice(&order_preserving_f64(bits).to_be_bytes());
        }
        AnyColumn::Int64(s) => {
            let v = s.data()[row];
            out.extend_from_slice(&(v ^ i64::MIN).to_be_bytes());
        }
        AnyColumn::Int32(s) => {
            let v = i64::from(s.data()[row]);
            out.extend_from_slice(&(v ^ i64::MIN).to_be_bytes());
        }
        AnyColumn::Bool(s) => out.push(u8::from(s.data()[row])),
        AnyColumn::Utf8(s) => {
            out.extend_from_slice(s.data()[row].as_bytes());
            out.push(0); // separator so ("ab","c") != ("a","bc")
        }
        AnyColumn::Date(s) | AnyColumn::DateTime(s) | AnyColumn::Duration(s) => {
            let v = s.data()[row];
            out.extend_from_slice(&(v ^ i64::MIN).to_be_bytes());
        }
    }
}

/// Maps an f64 bit pattern to a monotonically increasing u64 so that
/// big-endian byte order matches numeric order (including negatives).
fn order_preserving_f64(bits: u64) -> u64 {
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

/// Returns `true` if the column holds a numeric type.
fn is_numeric(col: &AnyColumn) -> bool {
    matches!(
        col,
        AnyColumn::Float64(_)
            | AnyColumn::Float32(_)
            | AnyColumn::Int64(_)
            | AnyColumn::Int32(_)
    )
}

/// Returns a display string for a cell value.
fn cell_display_name(col: &AnyColumn, row: usize) -> String {
    match col {
        AnyColumn::Float64(s) => alloc::format!("{}", s.data()[row]),
        AnyColumn::Float32(s) => alloc::format!("{}", s.data()[row]),
        AnyColumn::Int64(s) => alloc::format!("{}", s.data()[row]),
        AnyColumn::Int32(s) => alloc::format!("{}", s.data()[row]),
        AnyColumn::Bool(s) => alloc::format!("{}", s.data()[row]),
        AnyColumn::Utf8(s) => s.data()[row].clone(),
        AnyColumn::Date(s) => alloc::format!("Date({})", s.data()[row]),
        AnyColumn::DateTime(s) => alloc::format!("DateTime({})", s.data()[row]),
        AnyColumn::Duration(s) => alloc::format!("Duration({})", s.data()[row]),
    }
}

/// Applies an aggregation operation to a (possibly null-containing) column.
fn apply_op(col: &AnyColumn, op: AggOp) -> DataFrameResult<f64> {
    match op {
        AggOp::Sum => col.sum(),
        AggOp::Mean => col.mean(),
        AggOp::Count => Ok(col.count() as f64),
        AggOp::Min => col.min(),
        AggOp::Max => col.max(),
        AggOp::Median => col.median(),
        AggOp::Std => col.std(),
        AggOp::NUnique => Ok(col.nunique()? as f64),
        AggOp::First => col
            .valid_f64()?
            .first()
            .copied()
            .ok_or(DataFrameError::EmptyDataFrame),
        AggOp::Last => col
            .valid_f64()?
            .last()
            .copied()
            .ok_or(DataFrameError::EmptyDataFrame),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn sample() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("city", vec![String::from("NYC"), String::from("LA"), String::from("NYC"), String::from("LA")])
            .unwrap();
        df.add_column("temp", vec![20.0, 25.0, 22.0, 30.0]).unwrap();
        df.add_column("rain", vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        df
    }

    #[test]
    fn group_mean() {
        let df = sample();
        let g = df.group_by(&["city"]).unwrap().mean().unwrap();
        assert_eq!(g.nrows(), 2);
        assert_eq!(g.column_names(), vec!["city", "temp_mean", "rain_mean"]);
        let temps = g.column("temp_mean").unwrap().as_f64().unwrap();
        let sorted: Vec<f64> = {
            let mut v = temps.data().to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        assert!((sorted[0] - 21.0).abs() < 1e-12); // NYC
        assert!((sorted[1] - 27.5).abs() < 1e-12); // LA
    }

    #[test]
    fn group_count_and_nunique() {
        let df = sample();
        let g = df.group_by(&["city"]).unwrap().count().unwrap();
        assert_eq!(g.nrows(), 2);
        let counts = g.column("rain_count").unwrap().as_f64().unwrap();
        assert!(counts.data().iter().all(|&c| c == 2.0));
    }

    #[test]
    fn group_nulls_skipped() {
        let mut key = crate::series::Series::new("k", vec![1_i64, 1, 2]);
        key.set_null(2);
        let mut key_col = AnyColumn::Int64(key);
        key_col.rename_mut("k");
        let mut vals = crate::series::Series::new("v", vec![1.0, 2.0, 3.0]);
        vals.rename_mut("v");
        let df3 = DataFrame::from_columns(vec![key_col, AnyColumn::Float64(vals)]).unwrap();
        let g = df3.group_by(&["k"]).unwrap().sum().unwrap();
        assert_eq!(g.nrows(), 1); // null-key row dropped
        let v = g.column("v_sum").unwrap().as_f64().unwrap();
        assert!((v.data()[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn group_multi_key() {
        let mut df = DataFrame::new();
        df.add_column("a", vec![String::from("x"), String::from("x"), String::from("y")])
            .unwrap();
        df.add_column("b", vec![1_i64, 1, 2]).unwrap();
        df.add_column("v", vec![10.0, 20.0, 30.0]).unwrap();
        let g = df.group_by(&["a", "b"]).unwrap().sum().unwrap();
        assert_eq!(g.nrows(), 2); // (x,1) and (y,2)
    }

    #[test]
    fn group_missing_key_errors() {
        let df = sample();
        assert!(df.group_by(&["nope"]).is_err());
    }

    #[test]
    fn group_agg_named() {
        let df = sample();
        let g = df.group_by(&["city"]).unwrap().agg(AggOp::Min).unwrap();
        let mins = g.column("temp_min").unwrap().as_f64().unwrap();
        let sorted: Vec<f64> = {
            let mut v = mins.data().to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        assert!((sorted[0] - 20.0).abs() < 1e-12);
        assert!((sorted[1] - 25.0).abs() < 1e-12);
    }

    #[test]
    fn group_transform_mean() {
        let df = sample();
        let t = df.group_by(&["city"]).unwrap().transform(AggOp::Mean).unwrap();
        assert_eq!(t.nrows(), df.nrows()); // same shape
        assert_eq!(t.ncols(), df.ncols());
        let temps = t.column("temp_mean").unwrap().as_f64().unwrap();
        // NYC mean = (20 + 22) / 2 = 21, LA mean = (25 + 30) / 2 = 27.5
        let mut vals: Vec<f64> = temps.data().iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 21.0).abs() < 1e-12);
        assert!((vals[3] - 27.5).abs() < 1e-12);
    }

    #[test]
    fn pivot_table_mean() {
        let mut df = DataFrame::new();
        df.add_column("city", vec![
            String::from("NYC"), String::from("NYC"),
            String::from("LA"), String::from("LA"),
        ]).unwrap();
        df.add_column("season", vec![
            String::from("summer"), String::from("winter"),
            String::from("summer"), String::from("winter"),
        ]).unwrap();
        df.add_column("temp", vec![30.0, 0.0, 25.0, 10.0]).unwrap();

        let pivot = df.pivot_table("temp", "city", "season", AggOp::Mean).unwrap();
        assert_eq!(pivot.nrows(), 2); // NYC, LA
        assert!(pivot.has_column("city"));
        assert!(pivot.has_column("season=summer"));
        assert!(pivot.has_column("season=winter"));
    }
}
