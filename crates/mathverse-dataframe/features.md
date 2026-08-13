# MathVerse DataFrame — Feature Inventory

This document catalogs the features implemented in `mathverse-dataframe` v0.1.x, organized by functional domain. It serves as both a user reference and a roadmap for future development.

## Legend
- ✅ **Implemented** — fully functional, tested, and documented
- ⚠️ **Partial** — basic implementation exists, additional functionality planned
- ❌ **Not yet implemented** — planned but not started

---

## 📐 Core Data Structure

| Feature | Status | Notes |
|---------|--------|-------|
| `DataFrame` — column-oriented two-dimensional data structure | ✅ | Primary data container; `#![no_std]` compatible |
| `Series<T>` — typed, nullable column (f64, f32, i64, i32, bool, String) | ✅ | Compile-time type enforcement via `AnyColumn` enum |
| `DType` enum — `Float64`, `Float32`, `Int64`, `Int32`, `Bool`, `Utf8`, `Date`, `DateTime`, `Duration` | ✅ | Supports `is_numeric()`, `is_float()`, `is_integer()`, `native_size()` |
| `NullBitmap` — compact 1-bit-per-element validity tracking | ✅ | Enables zero-cost null handling; `all_valid()`, `is_null(pos)`, `null_count()` |
| `Schema` / `Field` — ordered column names with dtypes; add/remove/rename/lookup | ✅ | Supports dynamic schema evolution |
| `Index` types — `Range`, `Int64`, `Labels`, `Categorical`, `MultiIndex` row labels with `select` and uniqueness checks | ✅ | `Categorical` supports codes, categories, inference; `MultiIndex` supports hierarchical indexing |
| `duplicated()` — boolean mask of duplicate rows (pandas-compatible) | ✅ | **New in v0.2.0** — first occurrence not duplicate, subsequent duplicates marked |
| `nunique()` — count of distinct non-null values (pandas `nunique`) | ✅ | Returns `DataFrameResult<usize>` via `valid_f64()` |
| `mode()` — most frequent value and its count (pandas `mode` top) | ✅ | Ties broken by smallest value |
| `value_counts()` — (value, count) pairs sorted descending | ✅ | Histogram of value frequencies |
| `shift(n)` — shift values by n positions; first/last |n| become NaN | ✅ | **New in v0.2.0** — positive n shifts down, negative n shifts up |

---

## 🔤 Data Input / Output

| Feature | Status | Notes |
|---------|--------|-------|
| `json::to_json_string(df)` — serialize DataFrame to JSON array-of-objects | ✅ | Roundtrip with null and missing key handling |
| `json::from_json_str(input)` — parse JSON into DataFrame | ✅ | Schema inferred from key union; type inference planned |
| CSV read/write behind `csv` feature flag | ✅ | Read with type inference, write with header/options |
| `describe()` — compute summary statistics (count, mean, std, min, 25%, 50%, 75%, max) | ✅ | Numeric columns only; extends with additional percentiles planned |
| `valid_f64()` — extract non-null f64 values from any numeric column | ✅ | Handles int→f64 casting internally |
| `get_str(pos)` — string representation of any column value | ✅ | Handles Date/DateTime/Duration formatting |

---

## 🧮 Arithmetic & Comparison

| Feature | Status | Notes |
|---------|--------|-------|
| `x.add(&y)` / `x.sub(&y)` / `x.mul_scalar(c)` / `x.div_scalar(c)` — element-wise operations | ✅ | Between `AnyColumn` instances; also `column.op_scalar()` variants |
| `x.gt_scalar(c)` / `x.eq_scalar(c)` / `x.lt_scalar(c)` / `x.gte_scalar(c)` / `x.lte_scalar(c)` / `x.neq_scalar(c)` — comparisons | ✅ | Return `Series<bool>`; support chained filtering |
| `abs()` — element-wise absolute value | ✅ | On `AnyColumn` |
| `neg()` — element-wise negation | ✅ | On `AnyColumn` |
| `cumsum()` — cumulative sum | ✅ | NaN for null positions; running total continues from last non-null |
| `cumprod()` — cumulative product | ✅ | Same null semantics as `cumsum()` |
| `diff()` — first differences (Δ[i] = x[i] - x[i-1]); first element NaN | ✅ | On `AnyColumn` |
| `pct_change()` — percentage change; first element and null-involved pairs NaN | ✅ | On `AnyColumn` |
| `rolling_mean(n)` / `rolling_sum(n)` / `rolling_min(n)` / `rolling_max(n)` — fixed-width window | ✅ | Basic rolling; expanding windows and `min_periods` planned |
| `shift(n)` — shift values by n positions; first/last |n| become NaN | ✅ | **New in v0.2.0** — positive n shifts down, negative n shifts up |
| `expanding_mean()` / `expanding_sum()` / `expanding_var()` — cumulative window | ✅ | **New in v0.3.0** — running aggregation from first to current position |
| `ewm_mean(span)` — exponentially weighted moving average | ✅ | **New in v0.3.0** — span-based EWM matching pandas `ewm(span=n).mean()` |
| `rank()` — rank values (competition, dense, min, max methods) | ❌ | Not yet implemented |
| `scale()` / `standardize()` — z-score or robust scaling | ❌ | Not yet implemented |

---

## 📊 Aggregations & Statistics

| Feature | Status | Notes |
|---------|--------|-------|
| `mean()` — mean of non-null values | ✅ | `AnyColumn` method; numerically stable Welford variant available |
| `std()` — sample standard deviation (ddof=1) | ✅ | Via Welford algorithm |
| `var()` — sample variance (ddof=1) | ✅ | Available via `valid_f64()` + Welford |
| `median()` — median of non-null values | ✅ | Interpolation-free for odd; average-of-two for even |
| `quantile(q)` — quantile at percentile q (0.0–1.0) via linear interpolation | ✅ | Supports edge cases (NaN handling planned) |
| `min()` / `max()` — min/max of non-null values | ✅ | Available on `AnyColumn` |
| `count()` — count of non-null values | ✅ | Distinct from `len()`; returns usize |
| `nunique()` — count of distinct non-null values (pandas `nunique`) | ✅ | Returns distinct count |
| `mode()` — most frequent value and its count (pandas `mode` top) | ✅ | Ties broken by smallest value |
| `value_counts()` — (value, count) pairs sorted descending | ✅ | Histogram of value frequencies |
| `moments()` — (mean, variance) single-pass Welford | ✅ | |
| `describe_numeric()` — tuple (min, max, mean, std, count) | ✅ | Core summary; expanding to full `describe()` parity planned |

---

## 🪟 Window / Cumulative Operations

| Feature | Status | Notes |
|---------|--------|-------|
| `cumsum()` — cumulative sum | ✅ | NaN for null positions; running total continues from last non-null |
| `cumprod()` — cumulative product | ✅ | Same null semantics as `cumsum()` |
| `diff()` — first differences | ✅ | First element NaN; null-aware differencing |
| `pct_change()` — percentage change | ✅ | First element and null-involved pairs NaN |
| `rolling_mean(n)` / `rolling_sum(n)` / `rolling_min(n)` / `rolling_max(n)` | ✅ | Basic rolling; expanding windows and `min_periods` planned |
| `expanding_mean()` / `expanding_sum()` / `expanding_var()` | ✅ | **New in v0.3.0** — cumulative aggregation |
| `ewm_mean(span)` — exponential weighted moving average | ✅ | **New in v0.3.0** — span-based EWM |

---

## 🔗 Joins & Merges

| Feature | Status | Notes |
|---------|--------|-------|
| `inner_join(&df, other, "key")` — inner join on column key | ✅ | Basic key-column join; expanding to merge-on-index planned |
| `left_join(&df, other, "key")` — left join | ✅ | Pads nulls on right side for unmatched keys |
| `right_join(&df, other, "key")` — right join | ✅ | Pads nulls on left side for unmatched keys |
| `outer_join(&df, other, "key")` — outer join | ✅ | All keys from both sides; null-padded mismatches |
| `duplicate_column_suffix` handling during joins | ✅ | Pandas-style `_x`/`_y` suffixes via `MergeConfig::with_suffixes()` |
| `merge()` with `how`, `on`, `left_on`, `right_on`, `suffixes`, `indicator` | ✅ | **New in v0.3.0** — builder-style `MergeConfig` with `_merge` indicator column |

---

## 🔀 GroupBy

| Feature | Status | Notes |
|---------|--------|-------|
| `group_mean(&df, "value", &["key"])` — group-by mean | ✅ | Basic group-by; returns map of key→statistic |
| `group_count(&df, "key")` — group-by count | ✅ | Via `GroupBy::count()` |
| `group_sum(&df, "value", &["key"])` — group-by sum | ✅ | Via `GroupBy::sum()` |
| `group_std(&df, "value", &["key"])` — group-by std | ✅ | Via `GroupBy::std()` |
| `group_agg(&df, "value", &["key"], AggOp::Sum)` — flexible aggregation | ✅ | Via `GroupBy::agg(op)` |
| `group_transform()` — transform result back to original shape | ✅ | **New in v0.3.0** — broadcasts per-group aggregation to original row positions |
| `pivot_table()` — pivot with value column, index, columns, and aggregation | ✅ | **New in v0.3.0** — pivots index×column with aggregated values |
| `sort` parameter in group-by | ✅ | `GroupBy::sort_order(ascending)` — ascending/descending sort |

---

## 🧩 Type Conversion & Casting

| Feature | Status | Notes |
|---------|--------|-------|
| `cast(DType)` — cast column to target dtype | ✅ | Supports Float64↔Int64, Bool, Utf8; cross-type errors handled |
| `to_f64()` — convert column to `Series<f64>` (integers cast, bools→0.0/1.0) | ✅ | Utf8 → error; null positions preserved |
| `as_f64()` / `as_i64()` / `as_bool()` / `as_utf8()` — typed accessor | ✅ | Returns `DataFrameResult<&Series<T>>`; type mismatch errors |
| `into_data(self) -> Vec<T>` — consume Series and get data vector | ✅ | Series method |

---

## 🛠️ Utility & Inspection

| Feature | Status | Notes |
|---------|--------|-------|
| `shape()` — (nrows, ncols) | ✅ | |
| `nrows()` / `ncols()` — row/column count | ✅ | |
| `column_names()` — Vec<&str> of column names | ✅ | |
| `has_column(name)` — check column existence | ✅ | |
| `columns()` — iterator over column `AnyColumn`s | ✅ | |
| `column("name")` — get `AnyColumn` by name | ✅ | |
| `column_by_index(usize)` — get column by position | ✅ | |
| `iter()` — iterator over `(usize, Option<&T>)` pairs per series | ✅ | Series method |
| `valid_mask()` — boolean vector of non-null positions | ✅ | Series method |
| `non_null_count()` / `non_null_indices()` | ✅ | Series methods |
| `set_index(col_names)` — set one or more columns as the index | ✅ | **New in v1.0.0** |
| `reset_index()` — restore default RangeIndex | ✅ | **New in v1.0.0** |
| `swaplevel(a, b)` — swap two MultiIndex levels | ✅ | **New in v1.0.0** |
| `set_labels_index(labels)` — set a Labels index | ✅ | **New in v1.0.0** |
| `resample(date_col, bucket_days)` — downsample by day buckets | ✅ | **New in v1.0.0** |

---

## 📦 I/O & Serialization

| Feature | Status | Notes |
|---------|--------|-------|
| `json::to_json_string(df)` — serialize to JSON array-of-objects | ✅ | Roundtrip with null/missing key handling |
| `json::from_json_str(input)` — parse JSON into DataFrame | ✅ | Schema inferred from key union |
| CSV read/write behind `csv` feature flag | ✅ | Read with type inference, write with header/options |
| Excel/Parquet I/O | ❌ | Post-v1.0 |

---

## 🛣️ Feature Development Roadmap

### v0.2.0 — "Feature Complete Core" (complete)
- [x] `duplicated()` — boolean mask of duplicate rows
- [x] `shift(n)` — shift values by n positions
- [x] `value_counts()` — value frequency histogram
- [x] `nunique()` — distinct count (already existed, verified)
- [x] `mode()` — most frequent value (already existed, verified)
- [x] Enhanced `describe()` with 0.10/0.90/0.95 percentiles
- [x] `group_count()`, `group_sum()`, `group_std()` (already existed via `GroupBy`)
- [x] `semi_join` / `anti_join` variants
- [x] `group` parameter support (ascending/descending sort order)
- [x] `cast()` expansion: Int32↔Float32, Int64→Float32, Int32→Utf8, Float32→Utf8, Bool→Utf8, Utf8→Bool

### v0.3.0 — "Ecosystem Parity" (complete)
- [x] Full `merge()` with `on`, `left_on`, `right_on`, `suffixes`, `indicator` (via `MergeConfig`)
- [x] `groupby` transform (result back to original shape)
- [x] `pivot_table()` with value/index/columns
- [x] Rolling window: `expanding_mean()`, `expanding_sum()`, `expanding_var()`
- [x] `ewm()` exponential weighted moving average
- [x] CSV I/O behind `csv` feature flag (already existed)
- [x] Index: `CategoricalIndex` basic support

### v1.0.0 — "Production Release"
- [x] Time column types: `Date`, `DateTime`, `Duration` with full ops (column variants, match arms updated across codebase)
- [x] `MultiIndex`-style hierarchical indexing (basic)
- [x] String operations: `str_contains`, `str_startswith`, `str_endswith`, `str_to_lowercase`, `str_to_uppercase`, `str_strip`, `str_replace`, `str_split`, `str_matches`, `str_len`, `str_repeat`, `str_pad_left`
- [ ] Full pandas `merge()` parity
- [ ] No-std verified with `defconfig` CI
- [ ] 1,000+ unit tests across all features
- [ ] MSRV bump to 1.70 (if no‑std feature gated) or keep 1.87

### v1.1.0 — "ML / AI / Data Science Features"
- [ ] `sample(n)` / `shuffle()` — random sampling and shuffling
- [ ] `clip(min, max)` — clamp values to range
- [ ] `rank(method)` — rank column (dense, min, max, average methods)
- [ ] `percentile(q)` — single quantile value
- [ ] `outliers_iqr(k)` — IQR-based outlier detection mask
- [ ] `zscore()` / `normalize(method)` — standardize / min-max scale
- [ ] `weighted_mean(weights)` — weighted average
- [ ] `corr()` — Pearson correlation matrix
- [ ] `cov()` — covariance matrix
- [ ] `one_hot_encode(col)` — categorical → binary columns
- [ ] `cut(bins)` / `qcut(q)` — fixed-width / quantile binning
- [ ] `interpolate(method)` — linear / forward / backward gap fill
- [ ] `describe_with_quantiles(quantiles)` — full describe with custom percentiles
- [ ] `dt_extract(parts)` — pull year/month/day/hour/dow from date columns
- [ ] `pivot(index, columns, values)` — row-level reshape (inverse of melt)
- [ ] `dot(other)` — matrix multiply
- [ ] `profile()` — column-level summary (nulls, unique, skew, top values)
- [ ] `train_test_split(test_size, seed)` — stratified random split

### 📌 Known Gaps (Compared to pandas)

| Category | Gap | Priority |
|----------|-----|----------|
| Merge/Join | Full `merge()` with all parameters — now implemented | 🟢 Resolved |
| GroupBy grammar | `group_agg`, `group_transform`, `pivot_table` — now implemented | 🟢 Resolved |
| Rolling windows | `min_periods` parameter, `center` option | 🟠 Medium |
| String ops | `str_contains`, `str_startswith`, `str_endswith`, `str_to_lowercase`, `str_to_uppercase`, `str_strip`, `str_replace`, `str_split` implemented; no `.str` accessor or regex support | 🟠 Medium |
| Time types | `Date`/`DateTime`/`Duration` column variants implemented; no timezone ops, no resample, no `DatetimeIndex` | 🔴 High |
| Categorical | CategoricalIndex now implemented; CategoricalDtype for columns pending | 🟠 Medium |
| I/O | CSV/Excel/Parquet beyond basic JSON | 🔴 Low (out of zero-deps scope) |

### 💡 Contributing New Features

Feature additions follow this process:

1. **Check the roadmap** — ensure the feature isn't already planned or duplicate
2. **Implement with `#![no_std]` compatibility** — all new code must compile without `std`
3. **Add `#[must_use]`** where appropriate (core design principle)
4. **Write unit tests** — 100% test coverage for new code
5. **Update `features.md`** — document the new feature and its status
6. **Open a PR** — preferably with examples in `examples/` directory

Preferred first contributions (high impact, low complexity):
- Enhanced `describe()` percentiles
- `group_count()`, `group_sum()`
- Rolling `min_periods` parameter

--- 

## 🔒 Security Hardening (v1.0.0)

| Fix | Severity | Description |
|-----|----------|-------------|
| JSON depth limit | **High** | Recursive-descent JSON parser capped at 128 nesting levels to prevent stack overflow on adversarial input |
| `select_rows` → `Result` | **Medium** | Out-of-bounds positions now return `DataFrameResult` instead of panicking |
| `MergeConfig` key validation | **Medium** | Missing `on`/`left_on` returns error instead of panicking via `expect()` |
| Defensive `column_by_index` | **Medium** | `to_json_string` and CSV writer use `match` instead of `expect()` on column access |
| `ResourceExhausted` error | **Low** | New error variant for future resource-limit enforcement |

**Known accepted risks** (documented, not yet fixed):
- Join cross-product can be O(left × right) — accepted for in-memory use case
- `read_to_string` for CSV/JSON files — no streaming parser; documented OOM risk for multi-GB files
- `Series` methods (`is_null`, `get`, `set`) use `assert!` — public API panics on invalid index

---

## 🔬 ML / AI / Data Science Features

| Feature | Status | Notes |
|---------|--------|-------|
| `sample(n)` / `shuffle(seed)` — random sampling and row shuffling | ❌ | Deterministic via LCG seeded PRNG |
| `clip(min, max)` — clamp values to [min, max] | ❌ | Element-wise; supports null passthrough |
| `rank(method)` — rank values (dense, min, max, average) | ❌ | On `AnyColumn`; ties broken per method |
| `percentile(q)` — single quantile at q (0.0–1.0) | ❌ | Linear interpolation; reuse existing `quantile` |
| `outliers_iqr(k)` — IQR-based outlier boolean mask | ❌ | Returns `Series<bool>`; k × IQR beyond Q1/Q3 |
| `zscore()` — z-score standardization (μ=0, σ=1) | ❌ | Returns new column; null-safe |
| `normalize(method)` — min-max or max-abs scaling | ❌ | `method="minmax"` or `"maxabs"` |
| `weighted_mean(weights)` — weighted average | ❌ | Per-column weighted mean |
| `corr()` — Pearson correlation matrix | ❌ | DataFrame-wide; returns square matrix |
| `cov()` — covariance matrix | ❌ | DataFrame-wide; returns square matrix |
| `one_hot_encode(col)` — categorical → binary columns | ❌ | One new Bool column per unique value |
| `cut(bins)` — fixed-width binning | ❌ | Returns Int64 column of bin indices |
| `qcut(q)` — quantile binning | ❌ | Returns Int64 column of quantile indices |
| `interpolate(method)` — gap fill | ❌ | `method="linear"`, `"forward"`, `"backward"` |
| `describe_with_quantiles(quantiles)` — describe with custom percentiles | ❌ | Extends existing `describe()` |
| `dt_extract(parts)` — extract date components | ❌ | `parts=["year","month","day","hour","dow"]` |
| `pivot(index, columns, values)` — row-level reshape | ❌ | Inverse of melt; distinct from `pivot_table` |
| `dot(other)` — matrix multiply | ❌ | DataFrame × DataFrame dot product |
| `profile()` — column-level profiling summary | ❌ | nulls, unique, skew, top values per column |
| `train_test_split(test_size, seed)` — random stratified split | ❌ | Returns (train, test) tuple |

---

*Last updated: 2026-08-13 | MathVerse v0.1.x — "Foundations" release | 183 tests (164 unit + 19 doc-tests)*