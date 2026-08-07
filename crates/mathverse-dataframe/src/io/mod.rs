//! CSV and JSON I/O for [`crate::DataFrame`].
//!
//! The `csv` feature provides RFC 4180-compliant reading and writing with
//! per-column type inference. The `json` feature provides JSON serialization
//! and deserialization.
//!
//! # Examples
//!
//! ```
//! use mathverse_dataframe::{DataFrame, io::read_csv_from_str};
//!
//! let csv = "name,age,score\nAlice,25,88.5\nBob,30,92.3\n";
//! let df = read_csv_from_str(csv).unwrap();
//! assert_eq!(df.nrows(), 2);
//! assert_eq!(df.ncols(), 3);
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::dataframe::DataFrame;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

/// Options controlling CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    /// Field delimiter (default `','`).
    pub delimiter: u8,
    /// Whether the first record is a header row (default `true`).
    pub has_header: bool,
    /// String representing a missing value (default `""` — empty fields).
    pub null_value: String,
    /// If `true`, columns are kept as strings instead of type-inferred.
    pub keep_strings: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            null_value: String::new(),
            keep_strings: false,
        }
    }
}

/// Options controlling CSV writing.
#[derive(Debug, Clone)]
pub struct CsvWriteOptions {
    /// Field delimiter (default `','`).
    pub delimiter: u8,
    /// Whether to include the header row (default `true`).
    pub include_header: bool,
    /// String written for null values (default empty).
    pub null_value: String,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            include_header: true,
            null_value: String::new(),
        }
    }
}

/// Parses a CSV `&str` into a [`DataFrame`] with default options
/// (comma-delimited, header row, per-column type inference).
///
/// # Errors
///
/// Returns an error if the CSV is malformed (e.g. unbalanced quotes) or if
/// rows have inconsistent column counts.
pub fn read_csv_from_str(input: &str) -> DataFrameResult<DataFrame> {
    read_csv_with_options(input, &CsvReadOptions::default())
}

/// Parses a CSV `&str` into a [`DataFrame`] with custom options.
///
/// # Errors
///
/// Returns an error if the CSV is malformed, rows have inconsistent column
/// counts, or a header row contains duplicate names.
pub fn read_csv_with_options(input: &str, options: &CsvReadOptions) -> DataFrameResult<DataFrame> {
    let records = parse_records(input, options.delimiter)?;

    let (header, body) = if options.has_header {
        let Some(first) = records.first() else {
            return Ok(DataFrame::new());
        };
        (first.clone(), &records[1..])
    } else {
        let ncols = records.first().map_or(0, Vec::len);
        let header: Vec<String> = (0..ncols).map(|i| format!("column_{i}")).collect();
        (header, &records[..])
    };

    if header.is_empty() {
        return Ok(DataFrame::new());
    }

    let ncols = header.len();
    for (row_idx, record) in body.iter().enumerate() {
        if record.len() != ncols {
            return Err(DataFrameError::ParseError {
                value: format!("row {} has {} fields, expected {ncols}", row_idx + 1, record.len()),
                target: "csv",
            });
        }
    }

    let mut df = DataFrame::new();
    for (ci, name) in header.iter().enumerate() {
        let raw: Vec<&str> = body.iter().map(|r| r[ci].as_str()).collect();
        if options.keep_strings {
            let data: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
            df.add_any_column(AnyColumn::Utf8(Series::new(name, data)))?;
            continue;
        }
        df.add_any_column(infer_column(name, &raw, &options.null_value)?)?;
    }
    Ok(df)
}

/// Serializes a [`DataFrame`] to a CSV `String` with default options.
#[must_use]
pub fn to_csv_string(df: &DataFrame) -> String {
    write_csv_with_options(df, &CsvWriteOptions::default())
}

/// Serializes a [`DataFrame`] to a CSV `String` with custom options.
#[must_use]
pub fn write_csv_with_options(df: &DataFrame, options: &CsvWriteOptions) -> String {
    let delim = options.delimiter as char;
    let mut out = String::new();
    let ncols = df.ncols();
    if ncols == 0 {
        return out;
    }

    if options.include_header {
        for (i, name) in df.column_names().iter().enumerate() {
            if i > 0 {
                out.push(delim);
            }
            out.push_str(&escape_field(name, options.delimiter));
        }
        out.push('\n');
    }

    for row in 0..df.nrows() {
        for ci in 0..ncols {
            if ci > 0 {
                out.push(delim);
            }
            let col = df.column_by_index(ci).expect("column index within range");
            if col.is_null(row) {
                out.push_str(&options.null_value);
            } else {
                let cell = match col {
                    AnyColumn::Float64(s) => s.data()[row].to_string(),
                    AnyColumn::Float32(s) => s.data()[row].to_string(),
                    AnyColumn::Int64(s) => s.data()[row].to_string(),
                    AnyColumn::Int32(s) => s.data()[row].to_string(),
                    AnyColumn::Bool(s) => s.data()[row].to_string(),
                    AnyColumn::Utf8(s) => s.data()[row].clone(),
                };
                out.push_str(&escape_field(&cell, options.delimiter));
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(feature = "std")]
mod file_io {
    use super::*;

    /// Reads a CSV file into a [`DataFrame`] (default options).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the CSV is malformed.
    pub fn read_csv(path: &str) -> DataFrameResult<DataFrame> {
        let contents = std::fs::read_to_string(path).map_err(DataFrameError::from)?;
        read_csv_from_str(&contents)
    }

    /// Writes a [`DataFrame`] to a CSV file (default options).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_csv(df: &DataFrame, path: &str) -> DataFrameResult<()> {
        let csv = to_csv_string(df);
        std::fs::write(path, csv).map_err(DataFrameError::from)
    }
}

#[cfg(feature = "std")]
pub use file_io::{read_csv, write_csv};

/// Splits `input` into records and fields, honoring quoted fields per RFC 4180.
fn parse_records(input: &str, delimiter: u8) -> DataFrameResult<Vec<Vec<String>>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut field = String::new();
    let mut record: Vec<String> = Vec::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                // An escaped quote is `""` inside a quoted field.
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c as u8 == delimiter {
            record.push(core::mem::take(&mut field));
        } else if c == '\n' {
            record.push(core::mem::take(&mut field));
            // Tolerate bare CR and CRLF line endings.
            if record.len() == 1 && record[0].is_empty() {
                record.clear();
            } else {
                records.push(core::mem::take(&mut record));
            }
        } else if c == '\r' {
            // Peek for CRLF; otherwise treat as a record terminator.
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            record.push(core::mem::take(&mut field));
            if record.len() == 1 && record[0].is_empty() {
                record.clear();
            } else {
                records.push(core::mem::take(&mut record));
            }
        } else {
            field.push(c);
        }
    }

    if in_quotes {
        return Err(DataFrameError::ParseError {
            value: "unbalanced quote in CSV input".to_string(),
            target: "csv",
        });
    }
    // Flush a trailing record without a newline.
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

/// Infers the best-fit column dtype and builds the corresponding [`AnyColumn`].
fn infer_column(name: &str, raw: &[&str], null_value: &str) -> DataFrameResult<AnyColumn> {
    let is_null = |s: &str| s == null_value;
    if raw.iter().all(|s| is_null(s)) {
        // All-missing column: store as Float64 with all-null validity.
        let mut series = Series::new(name, vec![0.0; raw.len()]);
        for i in 0..raw.len() {
            series.set_null(i);
        }
        return Ok(AnyColumn::Float64(series));
    }

    // Integer inference.
    if raw
        .iter()
        .all(|s| is_null(s) || s.parse::<i64>().is_ok())
    {
        let data: Vec<i64> = raw
            .iter()
            .map(|s| if is_null(s) { 0 } else { s.parse().unwrap_or(0) })
            .collect();
        let mut series = Series::new(name, data);
        for (i, s) in raw.iter().enumerate() {
            if is_null(s) {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Int64(series));
    }

    // Float inference (must contain a decimal point or exponent marker to
    // avoid misreading integer-looking strings).
    if raw.iter().all(|s| {
        is_null(s)
            || ((s.contains('.') || s.contains('e') || s.contains('E') || s.contains('+'))
                && s.parse::<f64>().is_ok())
    }) && raw.iter().any(|s| !is_null(s) && (s.contains('.') || s.contains('e') || s.contains('E')))
    {
        let data: Vec<f64> = raw
            .iter()
            .map(|s| if is_null(s) { 0.0 } else { s.parse().unwrap_or(f64::NAN) })
            .collect();
        let mut series = Series::new(name, data);
        for (i, s) in raw.iter().enumerate() {
            if is_null(s) {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Float64(series));
    }

    // Bool inference.
    let bool_map = |s: &str| -> Option<bool> {
        match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    };
    if raw.iter().all(|s| is_null(s) || bool_map(s).is_some()) {
        let data: Vec<bool> = raw
            .iter()
            .map(|s| if is_null(s) { false } else { bool_map(s).unwrap_or(false) })
            .collect();
        let mut series = Series::new(name, data);
        for (i, s) in raw.iter().enumerate() {
            if is_null(s) {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Bool(series));
    }

    // Fallback: strings.
    let data: Vec<String> = raw
        .iter()
        .map(|s| if is_null(s) { String::new() } else { s.to_string() })
        .collect();
    let mut series = Series::new(name, data);
    for (i, s) in raw.iter().enumerate() {
        if is_null(s) {
            series.set_null(i);
        }
    }
    Ok(AnyColumn::Utf8(series))
}

/// Escapes a field if it contains the delimiter, quotes, or newlines.
fn escape_field(field: &str, delimiter: u8) -> String {
    let needs_quote = field
        .bytes()
        .any(|b| b == delimiter || b == b'"' || b == b'\n' || b == b'\r');
    if !needs_quote {
        return field.to_string();
    }
    let mut out = String::with_capacity(field.len() + 2);
    out.push('"');
    for c in field.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_simple_csv() {
        let csv = "name,age,score\nAlice,25,88.5\nBob,30,92.3\n";
        let df = read_csv_from_str(csv).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 3);
        assert!(df.has_column("name"));
        let age = df.column("age").unwrap().as_i64().unwrap();
        assert_eq!(age.data(), &[25, 30]);
        let score = df.column("score").unwrap().as_f64().unwrap();
        assert!((score.data()[0] - 88.5).abs() < 1e-12);
    }

    #[test]
    fn read_quoted_fields() {
        let csv = "name,note\n\"Doe, John\",\"said \"\"hi\"\"\"\n";
        let df = read_csv_from_str(csv).unwrap();
        let names = df.column("name").unwrap().as_utf8().unwrap();
        assert_eq!(names.data()[0], "Doe, John");
        let notes = df.column("note").unwrap().as_utf8().unwrap();
        assert_eq!(notes.data()[0], "said \"hi\"");
    }

    #[test]
    fn read_without_header() {
        let options = CsvReadOptions {
            has_header: false,
            ..Default::default()
        };
        let df = read_csv_with_options("1,2\n3,4\n", &options).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    }

    #[test]
    fn read_malformed_quotes_errors() {
        let result = read_csv_from_str("a,b\n\"unterminated,2\n");
        assert!(result.is_err());
    }

    #[test]
    fn read_ragged_rows_errors() {
        let result = read_csv_from_str("a,b\n1,2\n3\n");
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_with_special_chars() {
        let mut df = DataFrame::new();
        df.add_column("a", vec![1, 2]).unwrap();
        df.add_column("b", vec![String::from("x,y"), String::from("he said \"hi\"")])
            .unwrap();
        let csv = to_csv_string(&df);
        let back = read_csv_from_str(&csv).unwrap();
        assert_eq!(back.nrows(), 2);
        let b = back.column("b").unwrap().as_utf8().unwrap();
        assert_eq!(b.data()[0], "x,y");
        assert_eq!(b.data()[1], "he said \"hi\"");
    }

    #[test]
    fn write_options() {
        let mut df = DataFrame::new();
        df.add_column("x", vec![1.0, 2.0]).unwrap();
        let opts = CsvWriteOptions {
            include_header: false,
            null_value: "NA".to_string(),
            ..Default::default()
        };
        let csv = write_csv_with_options(&df, &opts);
        assert_eq!(csv, "1\n2\n");
    }

    #[test]
    fn infer_bool_column() {
        let csv = "flag\ntrue\nfalse\ntrue\n";
        let df = read_csv_from_str(csv).unwrap();
        let flags = df.column("flag").unwrap().as_bool().unwrap();
        assert_eq!(flags.data(), &[true, false, true]);
    }

    #[test]
    fn infer_null_cells() {
        let csv = "a,b\n1,\n,2\n";
        let df = read_csv_from_str(csv).unwrap();
        let a = df.column("a").unwrap();
        assert!(a.is_null(1));
        let b = df.column("b").unwrap();
        assert!(b.is_null(0));
    }

    #[test]
    fn crlf_line_endings() {
        let csv = "a,b\r\n1,2\r\n3,4\r\n";
        let df = read_csv_from_str(csv).unwrap();
        assert_eq!(df.nrows(), 2);
    }

    #[test]
    fn empty_and_single_field() {
        let df = read_csv_from_str("").unwrap();
        assert_eq!(df.nrows(), 0);
        let df = read_csv_from_str("only_one_column\nvalue\n").unwrap();
        assert_eq!(df.ncols(), 1);
        assert_eq!(df.nrows(), 1);
    }
}
