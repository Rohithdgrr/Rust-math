# MathVerse DataFrame

[![Crates.io](https://img.shields.io/crates/v/mathverse-dataframe.svg)](https://crates.io/crates/mathverse-dataframe)
[![docs.rs](https://docs.rs/mathverse-dataframe/badge.svg)](https://docs.rs/mathverse-dataframe)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/Rohithdgrr/Rust-math)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Pandas-like tabular data structures for the MathVerse ecosystem: a two-dimensional, column-oriented `DataFrame` with named, typed, nullable columns. Zero external dependencies.

---

## Features

- **`DataFrame`** — column-oriented tabular structure with schema, row index, and rich display
- **`Series<T>`** — named, typed, nullable column primitive (f64, f32, i64, i32, bool, String)
- **Null handling** — compact `NullBitmap` validity tracking, `drop_nulls`, `fill_nulls`, `forward_fill`
- **Selection** — `select_columns`, `select_rows`, `head`, `tail`, boolean `filter`
- **Transforms** — `transpose`, `sort_by`, `drop_duplicates`, `rename_column`
- **Arithmetic** — element-wise column + column and column + scalar (`add`, `sub`, `mul`, `div`, `neg`, `abs`)
- **Comparisons** — element-wise `gt` / `gte` / `lt` / `lte` / `eq` / `neq`, plus scalar variants
- **Aggregations** — `sum`, `mean`, `var`, `std`, `min`, `max`, `median`, `quantile`, `count`, `describe_numeric`
- **Window / cumulative ops** — `cumsum`, `cumprod`, `diff`, `pct_change`, `rolling_mean` / `rolling_sum` / `rolling_min` / `rolling_max`
- **Type casting** — `cast` between numeric / bool / string dtypes, `to_f64` for numeric ops
- **Schema** — `Field` + `Schema` metadata: add, remove, rename, lookup by name or index
- **Index** — `Range`, `Int64`, and `Labels` row indices with `select` and uniqueness checks
- **`no_std` ready** — `#![no_std]` with `alloc`, `#![forbid(unsafe_code)]`, fully documented public API

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `dataframe` | The `DataFrame` type — construction, selection, transforms, null ops |
| `series` | The `Series<T>` primitive — data, validity, slicing, iteration, mapping |
| `column` | `AnyColumn` type-erased column — downcasting, casting, row selection |
| `dtype` | The `DType` enum — `Float64`, `Float32`, `Int64`, `Int32`, `Bool`, `Utf8`, temporal types |
| `schema` | `Field` + `Schema` — ordered column names and their dtypes |
| `index` | `Index` row labels — `Range`, `Int64`, `Labels` |
| `null` | `NullBitmap` — compact 1-bit-per-element validity tracking |
| `errors` | `DataFrameError` enum and `DataFrameResult<T>` alias |
| `ops` | Arithmetic, comparison, and aggregation operator impls on `AnyColumn` |

---

## Installation

```toml
[dependencies]
mathverse-dataframe = "0.1"
```

---

## Quick Start

```rust
use mathverse_dataframe::DataFrame;

fn main() {
    let mut df = DataFrame::new();
    df.add_column("name", vec!["Alice".into(), "Bob".into(), "Charlie".into()]).unwrap();
    df.add_column("age", vec![25.0, 30.0, 35.0]).unwrap();
    df.add_column("score", vec![88.5, 92.3, 76.1]).unwrap();

    println!("{df}");
    //    name | age | score
    //   ------+-----+-------
    //   Alice |  25 |  88.5
    //     Bob |  30 |  92.3
    // Charlie |  35 |  76.1
}
```

---

## Guided Tour

### Building a DataFrame

```rust
use mathverse_dataframe::DataFrame;

let mut df = DataFrame::new();
df.add_column("x", vec![1.0, 2.0, 3.0]).unwrap();
df.add_column("y", vec![4.0, 5.0, 6.0]).unwrap();
df.add_column("label", vec!["a".into(), "b".into(), "c".into()]).unwrap();

assert_eq!(df.shape(), (3, 3));
assert!(df.has_column("x"));
assert_eq!(df.column_names(), vec!["x", "y", "label"]);
```

### Working with nulls

```rust
use mathverse_dataframe::{DataFrame, Series};

let mut s = Series::new("prices", vec![10.0, 20.0, 30.0]);
s.set_null(1);
assert!(s.is_null(1));
assert_eq!(s.null_count(), 1);

let cleaned = s.drop_nulls();
assert_eq!(cleaned.data(), &[10.0, 30.0]);
```

DataFrame-level null handling:

```rust
let df = df.fill_nulls("x", 0.0)?;       // fill with constant
let df = df.forward_fill("x")?;          // carry last valid value forward
let df = df.drop_nulls("x")?;            // drop rows where `x` is null
```

### Selection

```rust
let sub = df.select_columns(&["x", "label"]).unwrap();
let head = df.head(2);
let tail = df.tail(2);

// Boolean filter
let mask: Series<bool> = df.column("x").unwrap().gt_scalar(1.5).unwrap();
let filtered = df.filter(&mask).unwrap();
```

### Arithmetic

```rust
use mathverse_dataframe::{AnyColumn, Series};

let x = AnyColumn::from(Series::new("x", vec![1.0, 2.0, 3.0]));
let y = AnyColumn::from(Series::new("y", vec![4.0, 5.0, 6.0]));

let sum = x.add(&y).unwrap();             // 5, 7, 9
let scaled = x.mul_scalar(2.0).unwrap();  // 2, 4, 6
let flag = x.gt_scalar(1.5).unwrap();     // false, true, true
```

### Aggregation

```rust
let col = df.column("score").unwrap();

let mean   = col.mean().unwrap();
let std    = col.std().unwrap();
let median = col.median().unwrap();
let q25    = col.quantile(0.25).unwrap();
let (min, max, mu, sd, n) = col.describe_numeric().unwrap();
```

### Rolling and cumulative windows

```rust
use mathverse_dataframe::{AnyColumn, Series};

let s = AnyColumn::from(Series::new("t", vec![1.0, 2.0, 3.0, 4.0, 5.0]));

let cum = s.cumsum().unwrap();            // 1, 3, 6, 10, 15
let dif = s.diff().unwrap();              // NaN, 1, 1, 1, 1
let pct = s.pct_change().unwrap();        // NaN, 1.0, 0.5, 0.333, 0.25
let mov = s.rolling_mean(3).unwrap();     // NaN, NaN, 2, 3, 4
```

### Sorting and de-duplication

```rust
let sorted = df.sort_by("age", true).unwrap();
let unique = df.drop_duplicates(&["name"]).unwrap();
```

### Transpose

```rust
let t = df.transpose().unwrap();   // numeric-only; column names become the new index labels
```

### Type casting

```rust
let as_f64 = df.column("x").unwrap().to_f64().unwrap();
let as_str = df.column("x").unwrap().cast(mathverse_dataframe::DType::Utf8).unwrap();
let back   = df.column("x").unwrap().cast(mathverse_dataframe::DType::Int64).unwrap();
```

---

## Error Handling

All fallible operations return `DataFrameResult<T>` (a `Result<T, DataFrameError>`), so errors are never silently swallowed:

```rust
use mathverse_dataframe::{DataFrame, DataFrameError};

// Column lookup failure
assert!(matches!(
    df.column("does_not_exist"),
    Err(DataFrameError::ColumnNotFound(_))
));

// Length mismatch on add_column
df.add_column("bad", vec![1.0]).unwrap_err();

// Type mismatch on typed access
assert!(df.column_as::<i64>("x").is_err());
```

The error enum is `#[non_exhaustive]` and includes `ColumnNotFound`, `IndexOutOfBounds`, `TypeMismatch`, `DimensionMismatch`, `DuplicateColumn`, `JoinKeyNotFound`, `EmptyDataFrame`, `ParseError`, `Io`, and `InvalidOperation`.

---

## DType Reference

| `DType` | Rust type | `is_numeric` | `is_float` | `is_integer` | native size |
|---------|-----------|:---:|:---:|:---:|:---:|
| `Float64` | `f64` | ✓ | ✓ | — | 8 |
| `Float32` | `f32` | ✓ | ✓ | — | 4 |
| `Int64` | `i64` | ✓ | — | ✓ | 8 |
| `Int32` | `i32` | ✓ | — | ✓ | 4 |
| `Bool` | `bool` | — | — | — | 4 |
| `Utf8` | `String` | — | — | — | 0 |
| `Date` | — | — | — | — | 8 |
| `DateTime` | — | — | — | — | 8 |
| `Duration` | — | — | — | — | 8 |

Numeric aggregate/arithmetic operations cast to `f64` internally; string columns are not convertible to numeric types.

---

## Future Scope

- CSV / JSON I/O behind the existing `csv` / `json` feature flags
- Group-by / split-apply-combine operations
- Joins and merges (keys already surfaced via `JoinKeyNotFound`)
- Multi-column sort and stable ordering
- Date / DateTime / Duration column support
- Memory-efficient `Int64`-style bitmap storage and SIMD kernels

## License

Licensed under either of Apache License, Version 2.0 or MIT license — see the workspace root for the full license texts.
