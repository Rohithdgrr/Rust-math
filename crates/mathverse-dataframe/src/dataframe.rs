use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::column::AnyColumn;
use crate::dtype::DType;
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
        self.columns.push(col);

        if self.columns.len() == 1 {
            self.index = Index::default_range(col.len());
        }

        Ok(())
    }

    /// Adds a new column from an `AnyColumn`.
    ///
    /// # Errors
    ///
    /// Returns an error if the column name already exists or lengths don't match.
    pub fn add_any_column(&mut self, col: AnyColumn) -> DataFrameResult<()> {
        self.add_column(col.name(), col)
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
    }

    /// Returns the last `n` rows as a new DataFrame.
    #[must_use]
    pub fn tail(&self, n: usize) -> Self {
        let n = n.min(self.nrows());
        let start = self.nrows() - n;
        self.select_rows(&(start..self.nrows()).collect::<Vec<_>>())
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
    #[must_use]
    pub fn select_rows(&self, positions: &[usize]) -> Self {
        let columns: Vec<AnyColumn> = self
            .columns
            .iter()
            .filter_map(|col| col.select_rows(positions).ok())
            .collect();
        let index = self.index.select(positions).unwrap_or_default();
        Self {
            columns,
            schema: self.schema.clone(),
            index,
        }
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

        Ok(self.select_rows(&positions))
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

        Ok(self.select_rows(&positions))
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

        Ok(self.select_rows(&positions))
    }

    /// Returns a new DataFrame with null values in the specified column dropped.
    #[must_use]
    pub fn drop_nulls(&self, column_name: &str) -> DataFrameResult<Self> {
        let col = self.column(column_name)?;
        let positions: Vec<usize> = (0..col.len())
            .filter(|&i| !col.is_null(i))
            .collect();
        Ok(self.select_rows(&positions))
    }

    /// Fills null values in the specified column with a constant value (as f64).
    #[must_use]
    pub fn fill_nulls(&self, column_name: &str, fill_value: f64) -> DataFrameResult<Self> {
        let mut result = self.clone();
        let col = result.columns.iter_mut().find(|c| c.name() == column_name)
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

        Ok(result)
    }

    /// Returns a new DataFrame with null values forward-filled in the specified column.
    #[must_use]
    pub fn forward_fill(&self, column_name: &str) -> DataFrameResult<Self> {
        let mut result = self.clone();
        let col = result.columns.iter_mut().find(|c| c.name() == column_name)
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

        Ok(result)
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
        let ncols = self.ncols();

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
        df.add_column("name", vec!["c".into(), "a".into(), "b".into()]).unwrap();
        df.add_column("val", vec![3, 1, 2]).unwrap();

        let sorted = df.sort_by("val", true).unwrap();
        let col = sorted.column("val").unwrap().as_i64().unwrap();
        assert_eq!(col.data(), &[1, 2, 3]);
    }
}
