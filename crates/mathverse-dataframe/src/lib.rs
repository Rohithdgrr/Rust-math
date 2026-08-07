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
pub use schema::{Field, Schema};
pub use series::Series;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn smoke_test() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0, 3.0]).unwrap();
        df.add_column("y", vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }
}
