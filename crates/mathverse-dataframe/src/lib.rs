//! MathVerse DataFrame: pandas-like tabular data structures.
//!
//! This crate provides a `DataFrame` — a two-dimensional, column-oriented
//! data structure with named, typed columns. It is the primary way to work
//! with tabular data in the MathVerse ecosystem.
//!
//! # Quick Start
//!
//! ```
//! use mathverse_dataframe::DataFrame;
//!
//! let mut df = DataFrame::new();
//! df.add_column("name", vec![String::from("Alice"), String::from("Bob"), String::from("Charlie")]).unwrap();
//! df.add_column("age", vec![25.0, 30.0, 35.0]).unwrap();
//! df.add_column("score", vec![88.5, 92.3, 76.1]).unwrap();
//!
//! println!("{df}");
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::approx_constant)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::unreadable_literal)]

extern crate alloc;

#[cfg(feature = "csv")]
pub mod io;
#[cfg(feature = "json")]
pub mod json;

mod column;
mod dataframe;
mod dtype;
mod errors;
mod index;
mod math;
mod null;
mod ops;
mod schema;
mod series;

pub use column::AnyColumn;
pub use dataframe::DataFrame;
pub use dtype::DType;
pub use errors::{DataFrameError, DataFrameResult};
pub use index::Index;
pub use ops::groupby::{AggOp, GroupBy};
pub use ops::join::JoinType;
pub use ops::join::MergeConfig;
pub use schema::{Field, Schema};
pub use series::Series;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    use crate::null::NullBitmap;

    #[test]
    fn smoke_test() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0, 3.0]).unwrap();
        df.add_column("y", vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }

    // ── AnyColumn basic ops ─────────────────────────────────────────

    #[test]
    fn column_name_and_len() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0]));
        assert_eq!(c.name(), "a");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn column_dtype_roundtrip() {
        let cols: Vec<(AnyColumn, &str)> = vec![
            (AnyColumn::Float64(Series::new("a", vec![])), "f64"),
            (AnyColumn::Float32(Series::new("a", vec![])), "f32"),
            (AnyColumn::Int64(Series::new("a", vec![])), "i64"),
            (AnyColumn::Int32(Series::new("a", vec![])), "i32"),
            (AnyColumn::Bool(Series::new("a", vec![])), "bool"),
            (AnyColumn::Utf8(Series::new("a", vec![])), "str"),
            (AnyColumn::Date(Series::new("a", vec![])), "date"),
            (AnyColumn::DateTime(Series::new("a", vec![])), "datetime"),
            (AnyColumn::Duration(Series::new("a", vec![])), "duration"),
        ];
        for (col, expected) in cols {
            assert_eq!(col.dtype().name(), expected);
        }
    }

    #[test]
    fn column_null_tracking() {
        let mut c = AnyColumn::Int64(Series::new("n", vec![1, 2, 3]));
        assert_eq!(c.null_count(), 0);
        assert!(!c.is_null(0));
        c.set_null(1);
        assert!(c.is_null(1));
        assert_eq!(c.null_count(), 1);
    }

    #[test]
    fn column_rename() {
        let mut c = AnyColumn::Float64(Series::new("old", vec![1.0]));
        c.rename_mut("new");
        assert_eq!(c.name(), "new");
        assert_eq!(c.with_name("x").name(), "x");
    }

    #[test]
    fn column_to_f64() {
        let c = AnyColumn::Int64(Series::new("a", vec![10, 20]));
        let f = c.to_f64().unwrap();
        assert_eq!(f.data()[0], 10.0);
        let c = AnyColumn::Utf8(Series::new("a", vec!["x".into()]));
        assert!(c.to_f64().is_err());
    }

    #[test]
    fn column_as_accessors() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0]));
        assert!(c.as_f64().is_ok());
        assert!(c.as_i64().is_err());
        let c = AnyColumn::Int64(Series::new("a", vec![1]));
        assert!(c.as_i64().is_ok());
        let c = AnyColumn::Bool(Series::new("a", vec![true]));
        assert!(c.as_bool().is_ok());
        let c = AnyColumn::Utf8(Series::new("a", vec!["x".into()]));
        assert!(c.as_utf8().is_ok());
    }

    #[test]
    fn column_get_str() {
        let c = AnyColumn::Float64(Series::new("a", vec![3.14]));
        assert_eq!(c.get_str(0).unwrap(), "3.14");
        let c = AnyColumn::Int64(Series::new("a", vec![42]));
        assert_eq!(c.get_str(0).unwrap(), "42");
        let c = AnyColumn::Bool(Series::new("a", vec![true]));
        assert_eq!(c.get_str(0).unwrap(), "true");
        let c = AnyColumn::Utf8(Series::new("a", vec!["hello".into()]));
        assert_eq!(c.get_str(0).unwrap(), "hello");
        let c = AnyColumn::Date(Series::new("a", vec![19000]));
        assert_eq!(c.get_str(0).unwrap(), "19000");
        assert!(c.get_str(1).is_err());
    }

    // ── String ops ──────────────────────────────────────────────────

    fn utf8_col(data: Vec<&str>) -> AnyColumn {
        let strings: Vec<String> = data.into_iter().map(String::from).collect();
        AnyColumn::Utf8(Series::new("s", strings))
    }

    #[test]
    fn str_contains_basic() {
        let c = utf8_col(vec!["abc", "def", "ab"]);
        let r = c.str_contains("ab").unwrap();
        assert_eq!(r.as_bool().unwrap().data(), &[true, false, true]);
    }

    #[test]
    fn str_startswith_basic() {
        let c = utf8_col(vec!["abc", "def", "ab"]);
        let r = c.str_startswith("ab").unwrap();
        assert_eq!(r.as_bool().unwrap().data(), &[true, false, true]);
    }

    #[test]
    fn str_endswith_basic() {
        let c = utf8_col(vec!["abc", "def", "bc"]);
        let r = c.str_endswith("bc").unwrap();
        assert_eq!(r.as_bool().unwrap().data(), &[true, false, true]);
    }

    #[test]
    fn str_to_lowercase() {
        let c = utf8_col(vec!["ABC", "Def"]);
        let r = c.str_to_lowercase().unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["abc", "def"]);
    }

    #[test]
    fn str_to_uppercase() {
        let c = utf8_col(vec!["abc", "Def"]);
        let r = c.str_to_uppercase().unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["ABC", "DEF"]);
    }

    #[test]
    fn str_strip() {
        let c = utf8_col(vec!["  a  ", "b"]);
        let r = c.str_strip().unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["a", "b"]);
    }

    #[test]
    fn str_replace_all() {
        let c = utf8_col(vec!["aab", "aba"]);
        let r = c.str_replace("a", "x", None).unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["xxb", "xbx"]);
    }

    #[test]
    fn str_replace_limited() {
        let c = utf8_col(vec!["aaa"]);
        let r = c.str_replace("a", "x", Some(2)).unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["xxa"]);
    }

    #[test]
    fn str_split_basic() {
        let c = utf8_col(vec!["a,b,c", "x,y"]);
        let r = c.str_split(",", 1).unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["b", "y"]);
    }

    #[test]
    fn str_split_out_of_range() {
        let c = utf8_col(vec!["a"]);
        let r = c.str_split(",", 5).unwrap();
        // Index out of range -> null (validity true)
        assert!(r.is_null(0));
    }

    #[test]
    fn str_matches_alias() {
        let c = utf8_col(vec!["abc", "def"]);
        let r = c.str_matches("bc").unwrap();
        assert_eq!(r.as_bool().unwrap().data(), &[true, false]);
    }

    #[test]
    fn str_len_basic() {
        let c = utf8_col(vec!["hi", "hello"]);
        let r = c.str_len().unwrap();
        assert_eq!(r.as_i64().unwrap().data(), &[2, 5]);
    }

    #[test]
    fn str_repeat_basic() {
        let c = utf8_col(vec!["ab", "c"]);
        let r = c.str_repeat(3).unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["ababab", "ccc"]);
    }

    #[test]
    fn str_pad_left() {
        let c = utf8_col(vec!["a", "bc"]);
        let r = c.str_pad_left(3, '0').unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["00a", "0bc"]);
    }

    #[test]
    fn str_ops_on_non_utf8_errors() {
        let c = AnyColumn::Int64(Series::new("a", vec![1]));
        assert!(c.str_contains("x").is_err());
        assert!(c.str_startswith("x").is_err());
        assert!(c.str_endswith("x").is_err());
        assert!(c.str_to_lowercase().is_err());
        assert!(c.str_to_uppercase().is_err());
        assert!(c.str_strip().is_err());
        assert!(c.str_replace("a", "b", None).is_err());
        assert!(c.str_split(",", 0).is_err());
        assert!(c.str_matches("x").is_err());
        assert!(c.str_len().is_err());
        assert!(c.str_repeat(2).is_err());
        assert!(c.str_pad_left(5, ' ').is_err());
    }

    // ── Aggregate ops ───────────────────────────────────────────────

    #[test]
    fn aggregate_count() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        assert_eq!(c.count(), 3);
    }

    #[test]
    fn aggregate_duplicated() {
        let c = AnyColumn::Int64(Series::new("a", vec![1, 2, 1, 3]));
        let r = c.duplicated().unwrap();
        assert_eq!(r.data(), &[false, false, true, false]);
    }

    #[test]
    fn aggregate_shift_positive() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        let r = c.shift(1).unwrap().as_f64().unwrap().clone();
        assert!(r.is_null(0));
        assert_eq!(r.data()[1], 1.0);
        assert_eq!(r.data()[2], 2.0);
    }

    #[test]
    fn aggregate_shift_negative() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        let r = c.shift(-1).unwrap().as_f64().unwrap().clone();
        assert_eq!(r.data()[0], 2.0);
        assert_eq!(r.data()[1], 3.0);
        assert!(r.is_null(2));
    }

    #[test]
    fn aggregate_cumsum() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        let r = c.cumsum().unwrap().as_f64().unwrap().clone();
        assert_eq!(r.data(), &[1.0, 3.0, 6.0]);
    }

    #[test]
    fn aggregate_diff() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 3.0, 6.0]));
        let r = c.diff().unwrap().as_f64().unwrap().clone();
        assert!(r.data()[0].is_nan());
        assert_eq!(r.data()[1], 2.0);
        assert_eq!(r.data()[2], 3.0);
    }

    #[test]
    fn aggregate_rolling_mean() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0, 4.0]));
        let r = c.rolling_mean(2).unwrap().as_f64().unwrap().clone();
        assert!(r.data()[0].is_nan());
        assert_eq!(r.data()[1], 1.5);
        assert_eq!(r.data()[2], 2.5);
        assert_eq!(r.data()[3], 3.5);
    }

    #[test]
    fn aggregate_expanding_mean() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        let r = c.expanding_mean().unwrap().as_f64().unwrap().clone();
        assert_eq!(r.data()[0], 1.0);
        assert_eq!(r.data()[1], 1.5);
        assert_eq!(r.data()[2], 2.0);
    }

    #[test]
    fn aggregate_ewm_mean() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0, 3.0]));
        let r = c.ewm_mean(2).unwrap().as_f64().unwrap().clone();
        assert_eq!(r.data()[0], 1.0);
        // ewm with span=2: alpha=2/(2+1)=2/3
        // second: alpha*2 + (1-alpha)*1 = 2/3*2 + 1/3*1 = 5/3
        let expected_second = 5.0 / 3.0;
        assert!((r.data()[1] - expected_second).abs() < 1e-10);
        // third: alpha*3 + (1-alpha)*5/3 = 2/3*3 + 1/3*5/3 = 2 + 5/9 = 23/9
        let expected_third = 23.0 / 9.0;
        assert!((r.data()[2] - expected_third).abs() < 1e-10);
    }

    // ── Index ops ───────────────────────────────────────────────────

    #[test]
    fn index_range_basic() {
        let idx = Index::default_range(5);
        assert_eq!(idx.len(), 5);
        assert_eq!(idx.get_int(0).unwrap(), 0);
        assert_eq!(idx.get_int(4).unwrap(), 4);
        assert!(idx.is_unique());
    }

    #[test]
    fn index_int64_basic() {
        let idx = Index::int64(vec![10, 20, 30]);
        assert_eq!(idx.get_int(1).unwrap(), 20);
        assert_eq!(idx.get_str(2).unwrap(), "30");
    }

    #[test]
    fn index_labels_basic() {
        let idx = Index::labels(vec!["a".into(), "b".into()]);
        assert_eq!(idx.get_str(0).unwrap(), "a");
        assert!(idx.is_unique());
    }

    #[test]
    fn index_categorical_from_labels() {
        let idx = Index::categorical_from_labels(vec!["b".into(), "a".into(), "b".into()]);
        // categories sorted: ["a","b"]; codes: [1,0,1]
        assert_eq!(idx.get_str(0).unwrap(), "b");
        assert_eq!(idx.get_str(1).unwrap(), "a");
        assert!(!idx.is_unique());
    }

    #[test]
    fn index_multi_basic() {
        let idx = Index::MultiIndex(vec![
            vec!["A".into(), "B".into()],
            vec!["x".into(), "y".into()],
        ]);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.get_str(0).unwrap(), "A, x");
        assert_eq!(idx.get_str(1).unwrap(), "B, y");
        assert!(idx.is_unique());
    }

    #[test]
    fn index_select_variants() {
        let idx = Index::default_range(4);
        let s = idx.select(&[1, 3]).unwrap();
        assert_eq!(s.get_int(0).unwrap(), 1);
        assert_eq!(s.get_int(1).unwrap(), 3);

        let idx = Index::labels(vec!["a".into(), "b".into(), "c".into()]);
        let s = idx.select(&[2]).unwrap();
        assert_eq!(s.get_str(0).unwrap(), "c");
    }

    // ── DataFrame core ──────────────────────────────────────────────

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("a", vec![1.0, 2.0, 3.0]).unwrap();
        df.add_column("b", vec![4.0, 5.0, 6.0]).unwrap();
        df
    }

    #[test]
    fn df_ncols_nrows() {
        let df = sample_df();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn df_column_names() {
        let df = sample_df();
        let names = df.column_names();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn df_select_columns() {
        let df = sample_df();
        let r = df.select_columns(&["b"]).unwrap();
        assert_eq!(r.ncols(), 1);
        assert_eq!(r.column("b").unwrap().to_f64().unwrap().data()[0], 4.0);
    }

    #[test]
    fn df_add_columns() {
        let mut df = sample_df();
        df.add_column("c", vec![7.0, 8.0, 9.0]).unwrap();
        assert_eq!(df.ncols(), 3);
    }

    #[test]
    fn df_head_tail() {
        let df = sample_df();
        assert_eq!(df.head(2).nrows(), 2);
        assert_eq!(df.tail(1).nrows(), 1);
    }

    #[test]
    fn df_sort_by() {
        let df = sample_df();
        let r = df.sort_by("a", false).unwrap();
        assert_eq!(r.column("a").unwrap().to_f64().unwrap().data()[0], 3.0);
    }

    #[test]
    fn df_drop_duplicates() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 1.0, 2.0]).unwrap();
        let r = df.drop_duplicates(&["x"]).unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn df_set_index_and_reset() {
        let mut df = DataFrame::new();
        df.add_column("group", vec![String::from("A"), String::from("A"), String::from("B")])
            .unwrap();
        df.add_column("val", vec![1.0, 2.0, 3.0]).unwrap();
        let indexed = df.set_index(&["group"]).unwrap();
        assert_eq!(indexed.ncols(), 1);
        let reset = indexed.reset_index().unwrap();
        assert_eq!(reset.ncols(), 2);
    }

    #[test]
    fn df_swaplevel() {
        let mut df = DataFrame::new();
        df.add_column("l1", vec![String::from("A"), String::from("B")]).unwrap();
        df.add_column("l2", vec![String::from("x"), String::from("y")]).unwrap();
        df.add_column("v", vec![1.0, 2.0]).unwrap();
        let mut indexed = df.set_index(&["l1", "l2"]).unwrap();
        indexed.swaplevel(0, 1).unwrap();
        let idx = indexed.index();
        assert_eq!(idx.get_str(0).unwrap(), "x, A");
    }

    #[test]
    fn df_set_labels_index() {
        let mut df = sample_df();
        df.set_labels_index(vec!["r0".into(), "r1".into(), "r2".into()]).unwrap();
        assert_eq!(df.index().get_str(0).unwrap(), "r0");
    }

    #[test]
    fn df_resample_basic() {
        let mut df = DataFrame::new();
        // Dates as days: 0, 1, 5, 6, 10
        df.add_column("date", vec![0i64, 1, 5, 6, 10]).unwrap();
        df.add_column("val", vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let r = df.resample("date", 5).unwrap();
        // Buckets: 0->{0,1}, 1->{5,6}, 2->{10}
        assert_eq!(r.nrows(), 3);
    }

    #[test]
    fn df_resample_rejects_nonpositive() {
        let df = sample_df();
        assert!(df.resample("a", 0).is_err());
    }

    // ── Join ops ────────────────────────────────────────────────────

    fn left_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("key", vec![1.0, 2.0, 3.0]).unwrap();
        df.add_column("val_l", vec![10.0, 20.0, 30.0]).unwrap();
        df
    }

    fn right_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("key", vec![2.0, 3.0, 4.0]).unwrap();
        df.add_column("val_r", vec![200.0, 300.0, 400.0]).unwrap();
        df
    }

    #[test]
    fn join_inner() {
        let r = left_df().merge(&right_df(), "key", JoinType::Inner).unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn join_left() {
        let r = left_df().merge(&right_df(), "key", JoinType::Left).unwrap();
        assert_eq!(r.nrows(), 3);
    }

    #[test]
    fn join_right() {
        let r = left_df().merge(&right_df(), "key", JoinType::Right).unwrap();
        assert_eq!(r.nrows(), 3);
    }

    #[test]
    fn join_outer() {
        let r = left_df().merge(&right_df(), "key", JoinType::Outer).unwrap();
        assert_eq!(r.nrows(), 4);
    }

    #[test]
    fn join_semi() {
        let r = left_df().semi_join(&right_df(), "key", "key").unwrap();
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 2);
    }

    #[test]
    fn join_anti() {
        let r = left_df().anti_join(&right_df(), "key", "key").unwrap();
        assert_eq!(r.nrows(), 1);
    }

    #[test]
    fn merge_with_custom_suffixes() {
        let mut left = DataFrame::new();
        left.add_column("key", vec![1.0, 2.0]).unwrap();
        left.add_column("val", vec![10.0, 20.0]).unwrap();
        let mut right = DataFrame::new();
        right.add_column("key", vec![2.0, 3.0]).unwrap();
        right.add_column("val", vec![200.0, 300.0]).unwrap();
        let r = left
            .merge_with(
                &right,
                &MergeConfig::on("key", JoinType::Inner)
                    .with_suffixes("_L", "_R"),
            )
            .unwrap();
        let names = r.column_names();
        // Right suffix applied on collision; left keeps original name
        assert!(names.contains(&"val"));
        assert!(names.contains(&"val_R"));
    }

    #[test]
    fn merge_with_indicator() {
        let r = left_df()
            .merge_with(&right_df(), &MergeConfig::on("key", JoinType::Inner).with_indicator())
            .unwrap();
        assert!(r.column_names().contains(&"_merge"));
    }

    // ── GroupBy ops ─────────────────────────────────────────────────

    fn group_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("g", vec![String::from("A"), String::from("A"), String::from("B")])
            .unwrap();
        df.add_column("v", vec![10.0, 20.0, 30.0]).unwrap();
        df
    }

    #[test]
    fn groupby_mean() {
        let r = group_df().group_by(&["g"]).unwrap().mean().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_sum() {
        let r = group_df().group_by(&["g"]).unwrap().sum().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_count() {
        let r = group_df().group_by(&["g"]).unwrap().count().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_min() {
        let r = group_df().group_by(&["g"]).unwrap().min().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_max() {
        let r = group_df().group_by(&["g"]).unwrap().max().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_std() {
        let mut df = DataFrame::new();
        df.add_column("g", vec![String::from("A"), String::from("A"), String::from("B"), String::from("B")])
            .unwrap();
        df.add_column("v", vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let r = df.group_by(&["g"]).unwrap().std().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_nunique() {
        let r = group_df().group_by(&["g"]).unwrap().nunique().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    #[test]
    fn groupby_transform_mean() {
        let r = group_df().group_by(&["g"]).unwrap().transform(crate::ops::groupby::AggOp::Mean).unwrap();
        assert_eq!(r.nrows(), 3);
    }

    #[test]
    fn groupby_pivot_table() {
        let mut df = DataFrame::new();
        df.add_column("row", vec![String::from("A"), String::from("A"), String::from("B")])
            .unwrap();
        df.add_column("col", vec![String::from("X"), String::from("Y"), String::from("X")])
            .unwrap();
        df.add_column("val", vec![1.0, 2.0, 3.0]).unwrap();
        let r = df.pivot_table("val", "row", "col", crate::ops::groupby::AggOp::Sum).unwrap();
        assert!(r.nrows() >= 1);
    }

    #[test]
    fn groupby_sort_order_descending() {
        let df = group_df();
        let mut g = df.group_by(&["g"]).unwrap();
        g.sort_order(false);
        let r = g.mean().unwrap();
        assert_eq!(r.nrows(), 2);
    }

    // ── I/O CSV ─────────────────────────────────────────────────────

    #[cfg(feature = "std")]
    mod io_tests {
        use super::*;

        #[test]
        fn csv_roundtrip() {
            let mut df = DataFrame::new();
            df.add_column("x", vec![1.0, 2.0]).unwrap();
            df.add_column("y", vec![String::from("a"), String::from("b")]).unwrap();
            let csv = io::to_csv_string(&df);
            let df2 = io::read_csv_from_str(&csv).unwrap();
            assert_eq!(df2.nrows(), 2);
            assert_eq!(df2.ncols(), 2);
        }
    }

    // ── JSON ────────────────────────────────────────────────────────

    #[test]
    fn json_roundtrip() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0]).unwrap();
        df.add_column("y", vec![String::from("a"), String::from("b")]).unwrap();
        let json = json::to_json_string(&df);
        let df2 = json::from_json_str(&json).unwrap();
        assert_eq!(df2.nrows(), 2);
    }

    // ── Series ──────────────────────────────────────────────────────

    #[test]
    fn series_basic() {
        let s = Series::new("x", vec![1, 2, 3]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.data()[0], 1);
    }

    #[test]
    fn series_with_validity() {
        use crate::null::NullBitmap;
        let v = NullBitmap::from_bools(&[false, true, false]);
        let s = Series::with_validity("x", vec![1, 2, 3], v);
        assert!(!s.is_null(0));
        assert!(s.is_null(1));
        assert!(!s.is_null(2));
    }

    #[test]
    fn series_head_tail() {
        let s = Series::new("x", vec![1, 2, 3, 4, 5]);
        assert_eq!(s.head(3).data(), &[1, 2, 3]);
        assert_eq!(s.tail(2).data(), &[4, 5]);
    }

    #[test]
    fn series_map() {
        let s = Series::new("x", vec![1, 2, 3]);
        let r = s.map(|v| v * 2);
        assert_eq!(r.data(), &[2, 4, 6]);
    }

    #[test]
    fn series_drop_nulls() {
        use crate::null::NullBitmap;
        let v = NullBitmap::from_bools(&[false, true, false]);
        let s = Series::with_validity("x", vec![1, 2, 3], v);
        let r = s.drop_nulls();
        assert_eq!(r.data(), &[1, 3]);
    }

    // ── NullBitmap ──────────────────────────────────────────────────

    #[test]
    fn null_bitmap_from_bools() {
        let b = NullBitmap::from_bools(&[true, false, true]);
        assert!(b.is_null(0));
        assert!(!b.is_null(1));
        assert!(b.is_null(2));
        assert_eq!(b.null_count(), 2);
    }

    #[test]
    fn null_bitmap_set() {
        let mut b = NullBitmap::from_bools(&[false, false]);
        b.set_null(0);
        assert!(b.is_null(0));
        assert!(!b.is_null(1));
    }

    // ── DType ───────────────────────────────────────────────────────

    #[test]
    fn dtype_properties() {
        assert!(DType::Float64.is_numeric());
        assert!(DType::Int32.is_integer());
        assert!(DType::Float32.is_float());
        assert!(DType::Date.is_temporal());
        assert!(DType::DateTime.is_temporal());
        assert!(DType::Duration.is_temporal());
        assert!(!DType::Bool.is_numeric());
        assert!(!DType::Utf8.is_temporal());
    }

    // ── Schema ──────────────────────────────────────────────────────

    #[test]
    fn schema_basic() {
        let mut s = Schema::empty();
        s.add_field(Field::new("x", DType::Float64)).unwrap();
        s.add_field(Field::new("y", DType::Utf8)).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.dtype_of("x").unwrap(), DType::Float64);
    }

    #[test]
    fn schema_remove() {
        let mut s = Schema::empty();
        s.add_field(Field::new("x", DType::Float64)).unwrap();
        s.add_field(Field::new("y", DType::Utf8)).unwrap();
        s.remove_field("x").unwrap();
        assert_eq!(s.len(), 1);
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn arithmetic_add() {
        let a = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![3.0, 4.0]));
        let r = a.add(&b).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[4.0, 6.0]);
    }

    #[test]
    fn arithmetic_sub() {
        let a = AnyColumn::Float64(Series::new("a", vec![5.0, 7.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![1.0, 2.0]));
        let r = a.sub(&b).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[4.0, 5.0]);
    }

    #[test]
    fn arithmetic_mul() {
        let a = AnyColumn::Float64(Series::new("a", vec![2.0, 3.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![4.0, 5.0]));
        let r = a.mul(&b).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[8.0, 15.0]);
    }

    #[test]
    fn arithmetic_div() {
        let a = AnyColumn::Float64(Series::new("a", vec![10.0, 15.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![2.0, 3.0]));
        let r = a.div(&b).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[5.0, 5.0]);
    }

    #[test]
    fn arithmetic_scalar() {
        let a = AnyColumn::Float64(Series::new("a", vec![1.0, 2.0]));
        let r = a.add_scalar(10.0).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[11.0, 12.0]);
    }

    #[test]
    fn arithmetic_comparison() {
        let a = AnyColumn::Float64(Series::new("a", vec![1.0, 3.0, 5.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![2.0, 2.0, 2.0]));
        assert_eq!(a.gt(&b).unwrap().data(), &[false, true, true]);
        assert_eq!(a.gte(&b).unwrap().data(), &[false, true, true]);
        assert_eq!(a.lt(&b).unwrap().data(), &[true, false, false]);
        assert_eq!(a.lte(&b).unwrap().data(), &[true, false, false]);
    }

    #[test]
    fn arithmetic_dimension_mismatch() {
        let a = AnyColumn::Float64(Series::new("a", vec![1.0]));
        let b = AnyColumn::Float64(Series::new("b", vec![1.0, 2.0]));
        assert!(a.add(&b).is_err());
    }

    // ── Cast ────────────────────────────────────────────────────────

    #[test]
    fn cast_i64_to_f64() {
        let c = AnyColumn::Int64(Series::new("a", vec![1, 2]));
        let r = c.cast(DType::Float64).unwrap();
        assert_eq!(r.dtype(), DType::Float64);
        assert_eq!(r.as_f64().unwrap().data(), &[1.0, 2.0]);
    }

    #[test]
    fn cast_bool_to_f64() {
        let c = AnyColumn::Bool(Series::new("a", vec![true, false]));
        let r = c.cast(DType::Float64).unwrap();
        assert_eq!(r.as_f64().unwrap().data(), &[1.0, 0.0]);
    }

    #[test]
    fn cast_to_utf8() {
        let c = AnyColumn::Int64(Series::new("a", vec![42]));
        let r = c.cast(DType::Utf8).unwrap();
        assert_eq!(r.as_utf8().unwrap().data(), &["42"]);
    }

    #[test]
    fn cast_utf8_to_bool() {
        let c = AnyColumn::Utf8(Series::new("a", vec!["".into(), "x".into()]));
        let r = c.cast(DType::Bool).unwrap();
        assert_eq!(r.as_bool().unwrap().data(), &[false, true]);
    }

    #[test]
    fn cast_f64_to_i64() {
        let c = AnyColumn::Float64(Series::new("a", vec![1.7, 2.3]));
        let r = c.cast(DType::Int64).unwrap();
        assert_eq!(r.as_i64().unwrap().data(), &[1, 2]);
    }

    #[test]
    fn cast_unsupported_errors() {
        let c = AnyColumn::Utf8(Series::new("a", vec!["x".into()]));
        assert!(c.cast(DType::Float64).is_err());
    }

    // ── Display formatting ──────────────────────────────────────────

    #[test]
    fn dataframe_display() {
        let df = sample_df();
        let s = format!("{df}");
        assert!(s.contains("a"));
        assert!(s.contains("b"));
    }

    #[test]
    fn column_display() {
        let c = AnyColumn::Float64(Series::new("x", vec![1.0, 2.0]));
        let s = format!("{c}");
        assert!(s.contains("x"));
    }

    #[test]
    fn index_display() {
        let idx = Index::labels(vec!["a".into(), "b".into()]);
        let s = format!("{idx}");
        assert!(s.contains("a"));
    }

    // ── Security tests ──────────────────────────────────────────────

    #[test]
    fn json_depth_limit_prevents_stack_overflow() {
        // Build a deeply nested JSON array that would stack-overflow without a depth limit
        let mut input = String::new();
        for _ in 0..200 {
            input.push('[');
        }
        input.push_str("1");
        for _ in 0..200 {
            input.push(']');
        }
        let result = crate::json::from_json_str(&input);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nesting depth"));
    }

    #[test]
    fn select_rows_out_of_bounds_returns_error() {
        let df = sample_df();
        let result = df.select_rows(&[0, 999]);
        assert!(result.is_err());
    }

    #[test]
    fn merge_config_without_key_returns_error() {
        use crate::ops::join::{JoinType, MergeConfig};
        let mut left = DataFrame::new();
        left.add_column("a", vec![1.0]).unwrap();
        let mut right = DataFrame::new();
        right.add_column("a", vec![1.0]).unwrap();
        // MergeConfig with no on/left_on/right_on — should error, not panic
        let config = MergeConfig {
            how: JoinType::Inner,
            on: None,
            left_on: None,
            right_on: None,
            suffixes: ("_x".into(), "_y".into()),
            indicator: false,
        };
        let result = left.merge_with(&right, &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("must have"));
    }
}
