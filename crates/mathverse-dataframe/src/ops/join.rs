//! Relational joins and merges, pandas-style.
//!
//! [`DataFrame::join`] and [`DataFrame::merge`] combine two frames on a key
//! column. Four join types are supported — inner, left, right and outer —
//! matching the semantics of `pandas.merge`.
//!
//! Null key values never match (pandas behaviour). In left/outer joins, rows
//! without a match are emitted with null-padded values from the other side.
//!
//! # Examples
//!
//! ```
//! use mathverse_dataframe::{DataFrame, JoinType};
//!
//! let mut left = DataFrame::new();
//! left.add_column("id", vec![1_i64, 2, 3]).unwrap();
//! left.add_column("name", vec![String::from("a"), String::from("b"), String::from("c")]).unwrap();
//!
//! let mut right = DataFrame::new();
//! right.add_column("id", vec![2_i64, 3, 4]).unwrap();
//! right.add_column("score", vec![90.0, 80.0, 70.0]).unwrap();
//!
//! let merged = left.merge(&right, "id", JoinType::Inner).unwrap();
//! assert_eq!(merged.nrows(), 2); // ids 2 and 3
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::dataframe::DataFrame;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::null::NullBitmap;
use crate::series::Series;

/// The type of relational join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Keep only rows with matching keys on both sides.
    Inner,
    /// Keep all rows from the left; null-pad unmatched right columns.
    Left,
    /// Keep all rows from the right; null-pad unmatched left columns.
    Right,
    /// Keep all rows from both sides; null-pad whichever side is missing.
    Outer,
}

impl DataFrame {
    /// Joins this DataFrame with `other` on a same-named key column.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column is missing on either side.
    pub fn merge(&self, other: &DataFrame, on: &str, how: JoinType) -> DataFrameResult<Self> {
        self.join(other, on, on, how)
    }

    /// Joins this DataFrame with `other` on possibly differently-named key
    /// columns. Both key columns are kept in the output.
    ///
    /// # Errors
    ///
    /// Returns an error if either key column is missing.
    pub fn join(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        how: JoinType,
    ) -> DataFrameResult<Self> {
        if !self.has_column(left_on) {
            return Err(DataFrameError::ColumnNotFound(left_on.to_string()));
        }
        if !other.has_column(right_on) {
            return Err(DataFrameError::ColumnNotFound(right_on.to_string()));
        }

        // Build a lookup from encoded right key -> right row indices.
        let right_key = other.column(right_on)?;
        let mut lookup: alloc::collections::BTreeMap<Vec<u8>, Vec<usize>> =
            alloc::collections::BTreeMap::new();
        for row in 0..other.nrows() {
            if !right_key.is_null(row) {
                let mut key = Vec::new();
                encode_key_value(right_key, row, &mut key);
                lookup.entry(key).or_default().push(row);
            }
        }

        // Compute (left_row, right_row) pairs per the join type.
        let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
        let left_key = self.column(left_on)?;
        match how {
            JoinType::Inner | JoinType::Left => {
                for row in 0..self.nrows() {
                    let matches = if left_key.is_null(row) {
                        Vec::new()
                    } else {
                        let mut key = Vec::new();
                        encode_key_value(left_key, row, &mut key);
                        lookup.get(&key).cloned().unwrap_or_default()
                    };
                    if matches.is_empty() {
                        if how == JoinType::Left {
                            pairs.push((Some(row), None));
                        }
                    } else {
                        for r in matches {
                            pairs.push((Some(row), Some(r)));
                        }
                    }
                }
            }
            JoinType::Right | JoinType::Outer => {
                // First pass: all left rows (like Left join) so right rows
                // already matched are not duplicated; track which right rows
                // were consumed.
                let mut used: Vec<bool> = vec![false; other.nrows()];
                for row in 0..self.nrows() {
                    let matches = if left_key.is_null(row) {
                        Vec::new()
                    } else {
                        let mut key = Vec::new();
                        encode_key_value(left_key, row, &mut key);
                        lookup.get(&key).cloned().unwrap_or_default()
                    };
                    if matches.is_empty() {
                        if how == JoinType::Outer {
                            pairs.push((Some(row), None));
                        }
                    } else {
                        for r in &matches {
                            used[*r] = true;
                            pairs.push((Some(row), Some(*r)));
                        }
                    }
                }
                // Second pass: right rows never matched, for Right/Outer.
                if how == JoinType::Right || how == JoinType::Outer {
                    for (row, is_used) in used.iter().enumerate() {
                        if !is_used {
                            pairs.push((None, Some(row)));
                        }
                    }
                }
            }
        }

        // Assemble output columns.
        let mut result = DataFrame::new();
        // Key column first. For inner/left joins the left key always has a
        // value; for right/outer joins, unmatched right rows show the right
        // side's key value (pandas parity), so pick left value where present
        // and fall back to the right value.
        let left_key_col = self.column(left_on)?;
        let right_key_col = other.column(right_on)?;
        result.add_any_column(gather_key(left_key_col, right_key_col, &pairs)?)?;
        // Remaining left columns.
        for name in self.column_names() {
            if name == left_on {
                continue;
            }
            result.add_any_column(gather(self.column(name)?, &pairs, true)?)?;
        }
        // Right key column when names differ (SQL/merge semantics keep both).
        if right_on != left_on {
            result.add_any_column(gather(right_key_col, &pairs, false)?)?;
        }
        // Remaining right columns.
        for name in other.column_names() {
            if name == right_on {
                continue;
            }
            let col = other.column(name)?;
            // Avoid duplicate names in the output.
            let mut out_name = name.to_string();
            if result.has_column(&out_name) {
                out_name = format!("{name}_right");
            }
            result.add_any_column(gather(col, &pairs, false)?.with_name(out_name))?;
        }
        Ok(result)
    }
}

/// Gathers the key column value for each pair: uses the left key where the
/// left row exists, otherwise the right key (for unmatched right rows in
/// right/outer joins). Falls back to the right key when the left key is
/// itself null so valid right keys are not lost.
fn gather_key(left: &AnyColumn, right: &AnyColumn, pairs: &[(Option<usize>, Option<usize>)]) -> DataFrameResult<AnyColumn> {
    let len = pairs.len();
    let mut nulls: Vec<bool> = Vec::with_capacity(len);
    let mut picks_left: Vec<Option<usize>> = Vec::with_capacity(len);
    let mut picks_right: Vec<Option<usize>> = Vec::with_capacity(len);
    for (l, r) in pairs {
        let use_left = match (l, r) {
            (Some(lr), Some(rr)) => {
                // Both present: keys are equal, either works. Prefer the
                // non-null side.
                let left_null = left.is_null(*lr);
                let right_null = right.is_null(*rr);
                match (left_null, right_null) {
                    (false, _) => true,
                    (true, false) => false,
                    (true, true) => true,
                }
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => true,
        };
        if use_left {
            picks_left.push(*l);
            picks_right.push(None);
            nulls.push(l.is_none());
        } else {
            picks_left.push(None);
            picks_right.push(*r);
            nulls.push(r.is_none());
        }
    }
    let validity = NullBitmap::from_bools(&nulls);
    match left {
        AnyColumn::Float64(s) => {
            let data: Vec<f64> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p])
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Float64(r) => r.data()[p],
                            _ => 0.0,
                        }))
                        .unwrap_or(0.0)
                })
                .collect();
            Ok(AnyColumn::Float64(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Float32(s) => {
            let data: Vec<f32> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p])
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Float32(r) => r.data()[p],
                            _ => 0.0,
                        }))
                        .unwrap_or(0.0)
                })
                .collect();
            Ok(AnyColumn::Float32(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Int64(s) => {
            let data: Vec<i64> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p])
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Int64(r) => r.data()[p],
                            _ => 0,
                        }))
                        .unwrap_or(0)
                })
                .collect();
            Ok(AnyColumn::Int64(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Int32(s) => {
            let data: Vec<i32> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p])
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Int32(r) => r.data()[p],
                            _ => 0,
                        }))
                        .unwrap_or(0)
                })
                .collect();
            Ok(AnyColumn::Int32(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Bool(s) => {
            let data: Vec<bool> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p])
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Bool(r) => r.data()[p],
                            _ => false,
                        }))
                        .unwrap_or(false)
                })
                .collect();
            Ok(AnyColumn::Bool(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Utf8(s) => {
            let data: Vec<String> = (0..len)
                .map(|i| {
                    picks_left[i]
                        .map(|p| s.data()[p].clone())
                        .or_else(|| picks_right[i].map(|p| match right {
                            AnyColumn::Utf8(r) => r.data()[p].clone(),
                            _ => String::new(),
                        }))
                        .unwrap_or_default()
                })
                .collect();
            Ok(AnyColumn::Utf8(Series::with_validity(s.name(), data, validity)))
        }
    }
}

/// Gathers values from `col` according to the join pairs. When `from_left`
/// is `true`, the `left` side of each pair selects the value; otherwise the
/// `right` side does. Pairs whose selected side is `None` yield nulls.
fn gather(col: &AnyColumn, pairs: &[(Option<usize>, Option<usize>)], from_left: bool) -> DataFrameResult<AnyColumn> {
    let len = pairs.len();
    let mut nulls: Vec<bool> = Vec::with_capacity(len);
    let pick: Vec<Option<usize>> = pairs
        .iter()
        .map(|(l, r)| if from_left { *l } else { *r })
        .collect();
    for p in &pick {
        nulls.push(p.is_none());
    }
    let validity = NullBitmap::from_bools(&nulls);

    match col {
        AnyColumn::Float64(s) => {
            let data: Vec<f64> = pick.iter().map(|&p| p.map_or(0.0, |i| s.data()[i])).collect();
            Ok(AnyColumn::Float64(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Float32(s) => {
            let data: Vec<f32> = pick.iter().map(|&p| p.map_or(0.0, |i| s.data()[i])).collect();
            Ok(AnyColumn::Float32(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Int64(s) => {
            let data: Vec<i64> = pick.iter().map(|&p| p.map_or(0, |i| s.data()[i])).collect();
            Ok(AnyColumn::Int64(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Int32(s) => {
            let data: Vec<i32> = pick.iter().map(|&p| p.map_or(0, |i| s.data()[i])).collect();
            Ok(AnyColumn::Int32(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Bool(s) => {
            let data: Vec<bool> = pick.iter().map(|&p| p.map_or(false, |i| s.data()[i])).collect();
            Ok(AnyColumn::Bool(Series::with_validity(s.name(), data, validity)))
        }
        AnyColumn::Utf8(s) => {
            let data: Vec<String> = pick
                .iter()
                .map(|&p| p.map_or_else(String::new, |i| s.data()[i].clone()))
                .collect();
            Ok(AnyColumn::Utf8(Series::with_validity(s.name(), data, validity)))
        }
    }
}

/// Appends an equality-preserving byte encoding of a cell to `out`.
fn encode_key_value(col: &AnyColumn, row: usize, out: &mut Vec<u8>) {
    match col {
        AnyColumn::Float64(s) => out.extend_from_slice(&s.data()[row].to_bits().to_le_bytes()),
        AnyColumn::Float32(s) => out.extend_from_slice(&s.data()[row].to_bits().to_le_bytes()),
        AnyColumn::Int64(s) => out.extend_from_slice(&s.data()[row].to_le_bytes()),
        AnyColumn::Int32(s) => out.extend_from_slice(&s.data()[row].to_le_bytes()),
        AnyColumn::Bool(s) => out.push(u8::from(s.data()[row])),
        AnyColumn::Utf8(s) => {
            out.extend_from_slice(s.data()[row].as_bytes());
            out.push(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn left() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("id", vec![1_i64, 2, 3]).unwrap();
        df.add_column("name", vec![String::from("a"), String::from("b"), String::from("c")]).unwrap();
        df
    }

    fn right() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("id", vec![2_i64, 3, 4]).unwrap();
        df.add_column("score", vec![90.0, 80.0, 70.0]).unwrap();
        df
    }

    #[test]
    fn inner_join() {
        let merged = left().merge(&right(), "id", JoinType::Inner).unwrap();
        assert_eq!(merged.nrows(), 2);
        assert_eq!(merged.ncols(), 3);
        let ids = merged.column("id").unwrap().as_i64().unwrap();
        let mut sorted = ids.data().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![2, 3]);
        let scores = merged.column("score").unwrap().as_f64().unwrap();
        assert!(scores.data().iter().all(|s| s.is_finite()));
    }

    #[test]
    fn left_join_null_pads() {
        let merged = left().merge(&right(), "id", JoinType::Left).unwrap();
        assert_eq!(merged.nrows(), 3);
        let scores = merged.column("score").unwrap().as_f64().unwrap();
        assert!(scores.is_null(0)); // id=1 has no match
        assert!(!scores.is_null(1));
        assert!(!scores.is_null(2));
    }

    #[test]
    fn right_join() {
        let merged = left().merge(&right(), "id", JoinType::Right).unwrap();
        assert_eq!(merged.nrows(), 3);
        let names = merged.column("name").unwrap().as_utf8().unwrap();
        assert!(names.is_null(2)); // id=4 has no left match
        // Key parity: unmatched right row (id=4) still shows its key value.
        let ids = merged.column("id").unwrap().as_i64().unwrap();
        let mut sorted = ids.data().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![2, 3, 4]);
    }

    #[test]
    fn outer_join_key_parity() {
        // Unmatched rows on both sides must keep their key values.
        let merged = left().merge(&right(), "id", JoinType::Outer).unwrap();
        let ids = merged.column("id").unwrap().as_i64().unwrap();
        assert!(!ids.is_null(0)); // id=1 (left-only) present
        let mut vals: Vec<i64> = ids.data().iter().copied().collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn outer_join() {
        let merged = left().merge(&right(), "id", JoinType::Outer).unwrap();
        assert_eq!(merged.nrows(), 4);
    }

    #[test]
    fn join_on_different_names() {
        let mut r = DataFrame::new();
        r.add_column("key", vec![1_i64, 2]).unwrap();
        r.add_column("v", vec![10.0, 20.0]).unwrap();
        let joined = left().join(&r, "id", "key", JoinType::Inner).unwrap();
        assert_eq!(joined.nrows(), 2);
        assert!(joined.has_column("id"));
        assert!(joined.has_column("key"));
        assert!(joined.has_column("v"));
    }

    #[test]
    fn missing_key_errors() {
        assert!(left().merge(&right(), "nope", JoinType::Inner).is_err());
    }

    #[test]
    fn null_keys_never_match() {
        let mut l = DataFrame::new();
        l.add_column("k", vec![1_i64, 2]).unwrap();
        let mut rk = crate::series::Series::new("k", vec![1_i64, 2]);
        rk.set_null(0);
        let mut rk_col = AnyColumn::Int64(rk);
        rk_col.rename_mut("k");
        let mut rv = crate::series::Series::new("v", vec![100.0, 200.0]);
        rv.rename_mut("v");
        let r = DataFrame::from_columns(vec![rk_col, AnyColumn::Float64(rv)]).unwrap();
        let merged = l.merge(&r, "k", JoinType::Inner).unwrap();
        assert_eq!(merged.nrows(), 1); // only k=2 matches
    }

    #[test]
    fn duplicate_column_suffix() {
        // Right frame shares the `name` column name; it must be suffixed.
        let mut r = DataFrame::new();
        r.add_column("id", vec![1_i64, 2]).unwrap();
        r.add_column("name", vec![String::from("z"), String::from("y")]).unwrap();
        let merged = left().merge(&r, "id", JoinType::Left).unwrap();
        assert!(merged.has_column("name"));
        assert!(merged.has_column("name_right"));
    }
}
