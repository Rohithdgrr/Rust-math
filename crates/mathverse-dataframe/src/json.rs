//! JSON serialization and deserialization for [`crate::DataFrame`].
//!
//! The wire format is a JSON array of row objects:
//!
//! ```json
//! [{"name": "Alice", "age": 25, "score": 88.5}, {"name": "Bob", "age": 30}]
//! ```
//!
//! Nulls are encoded as JSON `null`; missing keys on read are treated as
//! nulls. Numbers are parsed with the column's inferred dtype where possible
//! (integer columns stay integers), and strings are kept as `Utf8`.
//!
//! # Examples
//!
//! ```
//! use mathverse_dataframe::{DataFrame, json::{to_json_string, from_json_str}};
//!
//! let mut df = DataFrame::new();
//! df.add_column("name", vec![String::from("Alice"), String::from("Bob")]).unwrap();
//! df.add_column("age", vec![25, 30]).unwrap();
//! let json = to_json_string(&df);
//! let back = from_json_str(&json).unwrap();
//! assert_eq!(back.nrows(), 2);
//! assert_eq!(back.ncols(), 2);
//! ```

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::column::AnyColumn;
use crate::dataframe::DataFrame;
use crate::errors::{DataFrameError, DataFrameResult};
use crate::series::Series;

/// Serializes a [`DataFrame`] to a JSON array-of-objects `String`.
#[must_use]
pub fn to_json_string(df: &DataFrame) -> String {
    let mut out = String::from("[");
    let names: Vec<&str> = df.column_names();
    for row in 0..df.nrows() {
        if row > 0 {
            out.push(',');
        }
        out.push('{');
        for (ci, name) in names.iter().enumerate() {
            if ci > 0 {
                out.push(',');
            }
            out.push_str(&json_escape(name));
            out.push(':');
            let col = match df.column_by_index(ci) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if col.is_null(row) {
                out.push_str("null");
                continue;
            }
            match col {
                AnyColumn::Float64(s) => out.push_str(&float_to_json(s.data()[row])),
                AnyColumn::Float32(s) => out.push_str(&float_to_json(f64::from(s.data()[row]))),
                AnyColumn::Int64(s) => out.push_str(&s.data()[row].to_string()),
                AnyColumn::Int32(s) => out.push_str(&s.data()[row].to_string()),
                AnyColumn::Bool(s) => out.push_str(if s.data()[row] { "true" } else { "false" }),
                AnyColumn::Utf8(s) => out.push_str(&json_escape(&s.data()[row])),
                AnyColumn::Date(s) | AnyColumn::DateTime(s) | AnyColumn::Duration(s) => {
                    out.push_str(&s.data()[row].to_string());
                }
            }
        }
        out.push('}');
    }
    out.push(']');
    out
}

/// Parses a JSON array-of-objects `String` into a [`DataFrame`].
///
/// The schema is inferred from the union of keys across all objects, ordered
/// by first appearance. Columns are typed per-column: if every non-null value
/// in a column parses as `i64`, it becomes `Int64`; otherwise `f64` or `Utf8`.
///
/// # Errors
///
/// Returns an error if the input is not valid JSON or not an array of objects.
pub fn from_json_str(input: &str) -> DataFrameResult<DataFrame> {
    let value = parse_json(input)?;
    match value {
        Value::Array(rows) => {
            for row in &rows {
                if !matches!(row, Value::Object(_)) {
                    return Err(DataFrameError::ParseError {
                        value: "expected every element to be a JSON object".to_string(),
                        target: "json",
                    });
                }
            }
            let mut col_names: Vec<String> = Vec::new();
            for row in &rows {
                if let Value::Object(fields) = row {
                    for (k, _) in fields {
                        if !col_names.contains(k) {
                            col_names.push(k.clone());
                        }
                    }
                }
            }
            let mut df = DataFrame::new();
            for name in col_names {
                let mut raw: Vec<Option<JsonScalar>> = Vec::with_capacity(rows.len());
                for row in &rows {
                    match row {
                        Value::Object(fields) => {
                            let cell = fields
                                .iter()
                                .find(|(k, _)| *k == name)
                                .map(|(_, v)| json_scalar_from_value(v))
                                .flatten();
                            raw.push(cell);
                        }
                        _ => raw.push(None),
                    }
                }
                df.add_any_column(build_column(&name, &raw)?)?;
            }
            Ok(df)
        }
        _ => Err(DataFrameError::ParseError {
            value: "expected a JSON array of objects".to_string(),
            target: "json",
        }),
    }
}

#[cfg(feature = "std")]
mod file_io {
    use super::*;

    /// Reads a JSON file into a [`DataFrame`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn read_json(path: &str) -> DataFrameResult<DataFrame> {
        let contents = std::fs::read_to_string(path).map_err(DataFrameError::from)?;
        from_json_str(&contents)
    }

    /// Writes a [`DataFrame`] to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_json(df: &DataFrame, path: &str) -> DataFrameResult<()> {
        let json = to_json_string(df);
        std::fs::write(path, json).map_err(DataFrameError::from)
    }
}

#[cfg(feature = "std")]
pub use file_io::{read_json, write_json};

/// A minimal, dependency-free JSON value tree (subset sufficient for
/// array-of-objects data frames).
enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// A scalar cell extracted from a JSON value (numbers/bools/strings).
enum JsonScalar {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

fn json_scalar_from_value(v: &Value) -> Option<JsonScalar> {
    match v {
        Value::Null => None,
        Value::Bool(b) => Some(JsonScalar::Bool(*b)),
        Value::Number(f) => {
            if f.fract() == 0.0 && f.is_finite() && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                Some(JsonScalar::Int(*f as i64))
            } else {
                Some(JsonScalar::Float(*f))
            }
        }
        Value::String(s) => Some(JsonScalar::Str(s.clone())),
        _ => None,
    }
}

/// Builds a typed column from raw scalar cells.
fn build_column(name: &str, raw: &[Option<JsonScalar>]) -> DataFrameResult<AnyColumn> {
    let has_non_null = raw.iter().any(Option::is_some);

    if !has_non_null {
        let mut series = Series::new(name, vec![0.0; raw.len()]);
        for i in 0..raw.len() {
            series.set_null(i);
        }
        return Ok(AnyColumn::Float64(series));
    }

    // All integers → Int64.
    if raw.iter().all(|c| match c {
        None | Some(JsonScalar::Int(_)) => true,
        _ => false,
    }) {
        let data: Vec<i64> = raw
            .iter()
            .map(|c| match c {
                Some(JsonScalar::Int(i)) => *i,
                _ => 0,
            })
            .collect();
        let mut series = Series::new(name, data);
        for (i, c) in raw.iter().enumerate() {
            if c.is_none() {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Int64(series));
    }

    // All numeric → Float64.
    if raw.iter().all(|c| match c {
        None | Some(JsonScalar::Int(_)) | Some(JsonScalar::Float(_)) => true,
        _ => false,
    }) {
        let data: Vec<f64> = raw
            .iter()
            .map(|c| match c {
                Some(JsonScalar::Int(i)) => *i as f64,
                Some(JsonScalar::Float(f)) => *f,
                _ => 0.0,
            })
            .collect();
        let mut series = Series::new(name, data);
        for (i, c) in raw.iter().enumerate() {
            if c.is_none() {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Float64(series));
    }

    // All bool → Bool.
    if raw.iter().all(|c| match c {
        None | Some(JsonScalar::Bool(_)) => true,
        _ => false,
    }) {
        let data: Vec<bool> = raw
            .iter()
            .map(|c| matches!(c, Some(JsonScalar::Bool(true))))
            .collect();
        let mut series = Series::new(name, data);
        for (i, c) in raw.iter().enumerate() {
            if c.is_none() {
                series.set_null(i);
            }
        }
        return Ok(AnyColumn::Bool(series));
    }

    // Mixed → Utf8.
    let data: Vec<String> = raw
        .iter()
        .map(|c| match c {
            Some(JsonScalar::Int(i)) => i.to_string(),
            Some(JsonScalar::Float(f)) => float_to_json(*f),
            Some(JsonScalar::Bool(b)) => b.to_string(),
            Some(JsonScalar::Str(s)) => s.clone(),
            None => String::new(),
        })
        .collect();
    let mut series = Series::new(name, data);
    for (i, c) in raw.iter().enumerate() {
        if c.is_none() {
            series.set_null(i);
        }
    }
    Ok(AnyColumn::Utf8(series))
}

/// Recursive-descent JSON parser (RFC 8259 subset). Returns the parsed tree.
fn parse_json(input: &str) -> DataFrameResult<Value> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.err("trailing characters after JSON value"));
    }
    Ok(value)
}

/// Maximum nesting depth for the JSON parser to prevent stack overflow.
const MAX_JSON_DEPTH: usize = 128;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> DataFrameError {
        DataFrameError::ParseError {
            value: format!("JSON error at byte {}: {msg}", self.pos),
            target: "json",
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn expect(&mut self, b: u8) -> DataFrameResult<()> {
        if self.next() == Some(b) {
            Ok(())
        } else {
            Err(self.err("unexpected character"))
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> DataFrameResult<Value> {
        match self.peek() {
            Some(b'n') => {
                self.expect_literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.expect_literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.expect_literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => self.parse_string().map(Value::String),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> DataFrameResult<()> {
        for &b in lit {
            if self.next() != Some(b) {
                return Err(self.err("invalid literal"));
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> DataFrameResult<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.next() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => return Ok(out),
                Some(b'\\') => {
                    match self.next() {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{0008}'),
                        Some(b'f') => out.push('\u{000C}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            let mut code = 0u32;
                            for _ in 0..4 {
                                match self.next().and_then(|b| (b as char).to_digit(16)) {
                                    Some(d) => code = code * 16 + d,
                                    None => return Err(self.err("invalid \\u escape")),
                                }
                            }
                            if let Some(c) = char::from_u32(code) {
                                out.push(c);
                            } else {
                                return Err(self.err("invalid unicode escape"));
                            }
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                Some(b) => {
                    // Reject raw control characters (RFC 8259 §8.2).
                    if b < 0x20 {
                        return Err(self.err("unescaped control character in string"));
                    }
                    let len = utf8_len(b);
                    let end = self.pos + len - 1;
                    if end > self.bytes.len() {
                        return Err(self.err("truncated UTF-8 sequence"));
                    }
                    let slice = &self.bytes[self.pos - 1..end];
                    match core::str::from_utf8(slice) {
                        Ok(s) => {
                            out.push_str(s);
                            self.pos = end;
                        }
                        Err(_) => return Err(self.err("invalid UTF-8")),
                    }
                }
            }
        }
    }

    fn parse_number(&mut self) -> DataFrameResult<Value> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')) {
            self.pos += 1;
        }
        let text = core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        text.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| self.err("invalid number"))
    }

    fn parse_array(&mut self) -> DataFrameResult<Value> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(self.err("maximum nesting depth exceeded"));
        }
        self.depth += 1;
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => {
                    self.depth -= 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self) -> DataFrameResult<Value> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(self.err("maximum nesting depth exceeded"));
        }
        self.depth += 1;
        self.expect(b'{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Object(fields));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let value = self.parse_value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => {
                    self.depth -= 1;
                    return Ok(Value::Object(fields));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }
}

/// Length (in bytes) of a UTF-8 code point from its leading byte.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

/// Escapes a string for inclusion in JSON.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Formats a float for JSON, ensuring NaN/Inf are not emitted (encoded as null).
fn float_to_json(x: f64) -> String {
    if x.is_finite() {
        x.to_string()
    } else {
        "null".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mixed_types() {
        let mut df = DataFrame::new();
        df.add_column("name", vec![String::from("Alice"), String::from("Bob")]).unwrap();
        df.add_column("age", vec![25, 30]).unwrap();
        df.add_column("score", vec![88.5, 92.3]).unwrap();
        df.add_column("active", vec![true, false]).unwrap();
        let json = to_json_string(&df);
        let back = from_json_str(&json).unwrap();
        assert_eq!(back.nrows(), 2);
        assert_eq!(back.ncols(), 4);
        let age = back.column("age").unwrap().as_i64().unwrap();
        assert_eq!(age.data(), &[25, 30]);
        let score = back.column("score").unwrap().as_f64().unwrap();
        assert!((score.data()[0] - 88.5).abs() < 1e-12);
        let active = back.column("active").unwrap().as_bool().unwrap();
        assert_eq!(active.data(), &[true, false]);
    }

    #[test]
    fn null_and_missing_keys() {
        let json = r#"[{"a": 1, "b": "x"}, {"a": null}, {"b": "y"}]"#;
        let df = from_json_str(json).unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
        assert!(df.column("a").unwrap().is_null(1));
        assert!(df.column("b").unwrap().is_null(1));
    }

    #[test]
    fn malformed_json_errors() {
        assert!(from_json_str("[1,2,3]").is_err());
        assert!(from_json_str("{\"a\": 1}").is_err());
        assert!(from_json_str("[{\"a\": }]").is_err());
        assert!(from_json_str("not json").is_err());
        assert!(from_json_str("[{\"a\": \"unterminated}]").is_err());
    }

    #[test]
    fn nested_arrays_are_null_cells() {
        let json = r#"[{"a": [1,2]}]"#;
        let df = from_json_str(json).unwrap();
        assert!(df.column("a").unwrap().is_null(0));
    }

    #[test]
    fn unicode_and_escapes() {
        let json = r#"[{"s": "caf\u00e9\n\"quoted\""}]"#;
        let df = from_json_str(json).unwrap();
        let s = df.column("s").unwrap().as_utf8().unwrap();
        assert_eq!(s.data()[0], "café\n\"quoted\"");
    }

    #[test]
    fn floats_and_integers() {
        let json = r#"[{"x": 1}, {"x": 2.5}]"#;
        let df = from_json_str(json).unwrap();
        let x = df.column("x").unwrap().as_f64().unwrap();
        assert!((x.data()[0] - 1.0).abs() < 1e-12);
        assert!((x.data()[1] - 2.5).abs() < 1e-12);
    }
}
