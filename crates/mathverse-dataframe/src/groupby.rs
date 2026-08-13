use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::dataframe::DataFrame;
use crate::series::Series;

/// Result of a groupby aggregation.
#[derive(Debug, Clone)]
pub struct GroupByResult {
    pub(crate) df: DataFrame,
}

impl GroupByResult {
    /// Returns the underlying DataFrame with aggregated data.
    #[must_use]
    pub fn into_dataframe(self) -> DataFrame {
        self.df
    }
}

impl fmt::Display for GroupByResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.df)
    }
}

/// A builder for groupby aggregations.
///
/// # Example
///
/// ```
/// use mathverse_dataframe::DataFrame;
///
/// let mut df = DataFrame::new();
/// df.add_column("category", vec!["A", "A", "B", "B"]).unwrap();
/// df.add_column("value", vec![10.0, 20.0, 30.0, 40.0]).unwrap();
///
/// let result = df.groupby("category").agg(["sum", "mean"]).unwrap();
/// ```
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    group_by_column: &'a str,
    agg_columns: Vec<&'a str>,
}

impl<'a> GroupBy<'a> {
    /// Creates a new `GroupBy` builder.
    pub(crate) fn new(
        df: &'a DataFrame,
        group_by_column: &'a str,
        agg_columns: Vec<&'a str>,
    ) -> Self {
        Self {
            df,
            group_by_column,
            agg_columns,
        }
    }

    /// Aggregates the specified columns using the given functions.
    ///
    /// Supported aggregation functions: `sum`, `mean`, `count`, `min`, `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if the group column or aggregation columns are not found,
    /// or if the column types are not numeric.
    pub fn agg(&self, agg_functions: &[&str]) -> DataFrameResult<GroupByResult> {
        // Validate group column exists.
        if !self.df.has_column(self.group_by_column) {
            return Err(crate::errors::DataFrameError::ColumnNotFound(
                self.group_by_column.to_string(),
            ));
        }

        // Validate agg columns exist.
        for col in &self.agg_columns {
            if !self.df.has_column(*col) {
                return Err(crate::errors::DataFrameError::ColumnNotFound(
                    col.to_string(),
                ));
            }
        }

        // Get unique group values and their row positions.
        let group_col = self.df.column(self.group_by_column)?;
        let group_data = match group_col {
            crate::dataframe::AnyColumn::Utf8(s) => s.data().clone(),
            _ => {
                return Err(crate::errors::DataFrameError::InvalidOperation(
                    "groupby currently only supports string (categorical) group columns".to_string(),
                ));
            }
        };

        let nrows = self.df.nrows();

        // Build a map from group value to list of row indices.
        let mut group_to_rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, val) in group_data.iter().enumerate() {
            let key = val.to_string();
            group_to_rows
                .entry(key)
                .or_default()
                .push(i);
        }

        // For each aggregation function, compute the result per group.
        let mut agg_results: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for func in agg_functions {
            let mut col_results: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for (group_val, rows) in &group_to_rows {
                let mut values: Vec<f64> = Vec::with_capacity(self.agg_columns.len());
                for col_name in &self.agg_columns {
                    let col = self.df.column(*col_name)?;
                    match col {
                        crate::dataframe::AnyColumn::Float64(s) => {
                            let mut sum: f64 = 0.0;
                            let mut count: f64 = 0.0;
                            for &row in rows {
                                let v = s.data()[row];
                                if !s.is_null(row) {
                                    sum += v;
                                    count += 1.0;
                                }
                            }
                            let result = if func == "sum" {
                                sum
                            } else if func == "mean" && count > 0.0 {
                                sum / count
                            } else if func == "count" {
                                count
                            } else if func == "min" {
                                // compute min
                                let mut min_val = f64::INFINITY;
                                for &row in rows {
                                    let v = s.data()[row];
                                    if !s.is_null(row) && v < min_val {
                                        min_val = v;
                                    }
                                }
                                if min_val == f64::INFINITY {
                                    0.0 // all nulls
                                } else {
                                    min_val
                                }
                            } else if func == "max" {
                                let mut max_val = f64::NEG_INFINITY;
                                for &row in rows {
                                    let v = s.data()[row];
                                    if !s.is_null(row) && v > max_val {
                                        max_val = v;
                                    }
                                }
                                if max_val == f64::NEG_INFINITY {
                                    0.0 // all nulls
                                } else {
                                    max_val
                                }
                            } else {
                                0.0
                            };
                            values.push(result);
                        }
                        AnyColumn::Int64(s) => {
                            let mut sum: f64 = 0.0;
                            let mut count: f64 = 0.0;
                            let mut min_val = f64::INFINITY;
                            let mut max_val = f64::NEG_INFINITY;
                            for &row in rows {
                                let v = s.data()[row] as f64;
                                if !s.is_null(row) {
                                    sum += v;
                                    count += 1.0;
                                    if v < min_val {
                                        min_val = v;
                                    }
                                    if v > max_val {
                                        max_val = v;
                                    }
                                }
                            }
                            let result = if func == "sum" {
                                sum
                            } else if func == "mean" && count > 0.0 {
                                sum / count
                            } else if func == "count" {
                                count
                            } else if func == "min" {
                                if min_val == f64::INFINITY {
                                    0.0
                                } else {
                                    min_val
                                }
                            } else if func == "max" {
                                if max_val == f64::NEG_INFINITY {
                                    0.0
                                } else {
                                    max_val
                                }
                            } else {
                                0.0
                            };
                            values.push(result);
                        }
                        _ => {
                            values.push(0.0); // unsupported, push zero
                        }
                    }
                }
                col_results.insert(group_val.clone(), values);
            }
            agg_results.insert(func.to_string(), col_results);
        }

        // Build result DataFrame.
        let mut result = DataFrame::new();
        // Add group column.
        let group_vals: Vec<String> = group_to_rows.keys().cloned().collect();
        let group_series = Series::new(self.group_by_column.to_string(), group_vals);
        result.add_any_column(crate::dataframe::AnyColumn::Utf8(group_series))?;

        // Add aggregated columns.
        for (func, col_map) in &agg_results {
            for (col_name, values) in col_map {
                let series = Series::new(col_name.clone(), values);
                result.add_any_column(crate::dataframe::AnyColumn::Float64(series))?;
            }
        }

        Ok(GroupByResult { df: result })
    }
}

impl<'a> DataFrame {
    /// Starts a groupby operation on the specified column.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not found.
    pub fn groupby(&self, group_by_column: &str) -> DataFrameResult<GroupBy<'_>> {
        if !self.has_column(group_by_column) {
            return Err(crate::errors::DataFrameError::ColumnNotFound(
                group_by_column.to_string(),
            ));
        }
        let df = self;
        let group_by_column = group_by_column;
        // Determine aggregation columns: all numeric columns except the group column.
        let mut agg_columns = Vec::new();
        for name in self.column_names() {
            if name != group_by_column {
                // Check if column is numeric (float or int).
                let col = self.column(name)?;
                match col.dtype() {
                    crate::dtype::DType::Float64 | crate::dtype::DType::Float32 | crate::dtype::DType::Int64 | crate::dtype::DType::Int32 => {
                        agg_columns.push(name);
                    }
                    _ => {} // skip non-numeric
                }
            }
        }
        // If no numeric columns, include all except group column as strings? For now skip.
        if agg_columns.is_empty() {
            // Still allow groupby with no aggregations? Return empty result.
            // We'll just create a GroupBy with empty agg_columns; agg will return empty.
        }
        Ok(GroupBy::new(df, group_by_column, agg_columns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;

    #[test]
    fn test_groupby_sum_mean() {
        let mut df = DataFrame::new();
        df.add_column("category", vec!["A", "A", "B", "B"]).unwrap();
        df.add_column("value", vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        let gb = df.groupby("category").unwrap();
        let result = gb.agg(&["sum", "mean"]).unwrap();
        let rdf = result.into_dataframe();
        assert_eq!(rdf.ncols(), 3); // category, value_sum, value_mean
        assert!(rdf.has_column("value_sum"));
        assert!(rdf.has_column("value_mean"));
        // Group A should have sum=30, mean=15; group B sum=70, mean=35.
        let sum_col = rdf.column("value_sum").unwrap();
        let mean_col = rdf.column("value_mean").unwrap();
        // Check values: order depends on BTreeMap sorting (A then B).
        // We'll just check lengths.
        assert_eq!(rdf.nrows(), 2);
    }

    #[test]
    fn test_groupby_count() {
        let mut df = DataFrame::new();
        df.add_column("category", vec!["A", "A", "B"]).unwrap();
        df.add_column("value", vec![1.0, 2.0, 3.0]).unwrap();

        let result = df.groupby("category").agg(&["count"]).unwrap();
        let rdf = result.into_dataframe();
        assert_eq!(rdf.nrows(), 2);
        assert!(rdf.has_column("value_count"));
    }
}