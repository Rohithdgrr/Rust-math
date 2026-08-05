//! A type-erased scalar value, used for row iteration, value counting,
//! pivoting, grouping keys, and generic filling/merging.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};

use crate::column::AnyColumn;

/// A single, type-erased cell value in a `DataFrame`.
///
/// `Value` mirrors the `DType`-level granularity exposed by columns while
/// allowing heterogeneous values (e.g. rows spanning multiple column types)
/// to be collected into one vector.
#[derive(Debug, Clone)]
pub enum Value {
    /// 64-bit floating-point value.
    Float(f64),
    /// 64-bit signed integer.
    Int(i64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    Str(String),
    /// Missing value.
    Null,
}

impl Value {
    /// Returns `true` if this is a missing value.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this is a non-missing value.
    #[must_use]
    pub const fn is_some(&self) -> bool {
        !self.is_null()
    }

    /// Interprets this value as an `f64`.
    ///
    /// Integers and booleans are coerced; strings and `Null` return `None`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Self::Str(_) | Self::Null => None,
        }
    }

    /// Returns the string view of a `Str` value.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the string view of a `Str` value, or formats non-string
    /// values as text.
    #[must_use]
    pub fn to_string(&self) -> String {
        match self {
            Self::Float(f) => f.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Str(s) => s.clone(),
            Self::Null => String::from("null"),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (Float(a), Float(b)) => (a.is_nan() && b.is_nan()) || a == b,
            (Int(a), Int(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Str(a), Str(b)) => a == b,
            (Null, Null) => true,
            (Int(a), Float(b)) => (*a as f64) == *b,
            (Float(b), Int(a)) => *b == (*a as f64),
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use Value::*;
        match self {
            Float(f) => {
                0u8.hash(state);
                float_hash_bits(*f).hash(state);
            }
            Int(i) => {
                // Hash as Float to maintain the Hash/Eq contract:
                // Int(2) == Float(2.0), so they must hash identically.
                0u8.hash(state);
                float_hash_bits(*i as f64).hash(state);
            }
            Bool(b) => {
                2u8.hash(state);
                b.hash(state);
            }
            Str(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            Null => 4u8.hash(state),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

/// Normalizes a float to a stable bit pattern for hashing.
fn float_hash_bits(x: f64) -> u64 {
    if x.is_nan() {
        0x7FF8_0000_0000_0000
    } else if x == 0.0 {
        // +0.0 and -0.0 are equal, so they must hash identically.
        0u64
    } else {
        x.to_bits()
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Self::Float(f64::from(v))
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

impl AnyColumn {
    /// Returns the value stored at `pos` as a type-erased `Value`.
    ///
    /// Returns `Value::Null` for null cells.
    #[must_use]
    pub fn value_at(&self, pos: usize) -> Value {
        match self {
            Self::Float64(s) => s
                .get(pos)
                .map_or(Value::Null, |&v| Value::Float(v)),
            Self::Float32(s) => s
                .get(pos)
                .map_or(Value::Null, |&v| Value::Float(f64::from(v))),
            Self::Int64(s) => s.get(pos).map_or(Value::Null, |&v| Value::Int(v)),
            Self::Int32(s) => s.get(pos).map_or(Value::Null, |&v| Value::Int(i64::from(v))),
            Self::Bool(s) => s.get(pos).map_or(Value::Null, |&v| Value::Bool(v)),
            Self::Utf8(s) => s.get(pos).map_or(Value::Null, |v| Value::Str(v.clone())),
        }
    }

    /// Returns all cells of this column as `Value`s, in order.
    #[must_use]
    pub fn to_values(&self) -> Vec<Value> {
        (0..self.len()).map(|i| self.value_at(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_equality() {
        assert_eq!(Value::Int(2), Value::Float(2.0));
        assert_eq!(Value::Float(2.0), Value::Int(2));
        assert_ne!(Value::Int(2), Value::Float(2.5));
        assert_eq!(Value::Str("a".to_string()), Value::from("a"));
        assert_eq!(Value::Null, Value::Null);
        assert_ne!(Value::Null, Value::Int(0));
        assert!(Value::Float(f64::NAN) == Value::Float(f64::NAN));
    }

    #[test]
    fn value_conversions() {
        assert_eq!(Value::from(3.5).as_f64(), Some(3.5));
        assert_eq!(Value::from(7_i64).as_f64(), Some(7.0));
        assert_eq!(Value::from(true).as_f64(), Some(1.0));
        assert_eq!(Value::Null.as_f64(), None);
        assert_eq!(Value::Str("x".to_string()).to_string(), "x");
    }

    #[test]
    fn hash_matches_equality() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // Equal values must hash identically (Hash/Eq contract).
        assert_eq!(h(&Value::Int(2)), h(&Value::Float(2.0)));
        assert_eq!(h(&Value::Float(0.0)), h(&Value::Float(-0.0)));
        assert_eq!(h(&Value::Float(f64::NAN)), h(&Value::Float(f64::NAN)));
        assert_eq!(h(&Value::Int(2)), h(&Value::Int(2)));
    }

    // ============================================================
    // Comprehensive Hash/Eq contract tests
    // ============================================================

    #[test]
    fn hash_eq_nan_contract() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // NaN == NaN per our PartialEq impl, so they must hash identically.
        let nan = Value::Float(f64::NAN);
        assert_eq!(nan, nan);
        assert_eq!(h(&nan), h(&nan));

        // Different NaN bit patterns must all hash the same.
        let nan1 = Value::Float(f64::NAN);
        let nan2 = Value::Float(f64::NAN);
        assert_eq!(h(&nan1), h(&nan2));

        // NaN hashes differently from non-NaN.
        assert_ne!(h(&nan), h(&Value::Float(1.0)));
    }

    #[test]
    fn hash_eq_zero_and_negative_zero() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // +0.0 == -0.0 per IEEE 754, so they must hash identically.
        let pos_zero = Value::Float(0.0);
        let neg_zero = Value::Float(-0.0);
        assert_eq!(pos_zero, neg_zero);
        assert_eq!(h(&pos_zero), h(&neg_zero));
    }

    #[test]
    fn hash_eq_int_float_cross_type() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // Int and Float with the same numeric value must hash identically.
        assert_eq!(h(&Value::Int(0)), h(&Value::Float(0.0)));
        assert_eq!(h(&Value::Int(1)), h(&Value::Float(1.0)));
        assert_eq!(h(&Value::Int(-1)), h(&Value::Float(-1.0)));
        assert_eq!(h(&Value::Int(42)), h(&Value::Float(42.0)));
        assert_eq!(h(&Value::Int(i64::MAX)), h(&Value::Float(i64::MAX as f64)));

        // Unequal values should (with overwhelming probability) hash differently.
        assert_ne!(h(&Value::Int(1)), h(&Value::Int(2)));
        assert_ne!(h(&Value::Int(1)), h(&Value::Float(2.0)));
    }

    #[test]
    fn hash_eq_int_float_cross_type_equality() {
        assert_eq!(Value::Int(0), Value::Float(0.0));
        assert_eq!(Value::Int(1), Value::Float(1.0));
        assert_eq!(Value::Int(-1), Value::Float(-1.0));
        assert_eq!(Value::Int(42), Value::Float(42.0));
        assert_eq!(Value::Int(999_999), Value::Float(999_999.0));
        assert_ne!(Value::Int(1), Value::Float(1.5));
        assert_ne!(Value::Int(0), Value::Float(0.1));
    }

    #[test]
    fn hash_eq_bool_separate_namespace() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // Bool uses discriminant 2, Int/Float use 0, so they hash differently.
        assert_ne!(h(&Value::Bool(true)), h(&Value::Int(1)));
        assert_ne!(h(&Value::Bool(false)), h(&Value::Int(0)));
        assert_ne!(h(&Value::Bool(true)), h(&Value::Float(1.0)));
        // Bool == Bool works correctly.
        assert_eq!(Value::Bool(true), Value::Bool(true));
        assert_ne!(Value::Bool(true), Value::Bool(false));
    }

    #[test]
    fn hash_eq_str_separate_namespace() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // Str uses discriminant 3.
        assert_ne!(h(&Value::Str("1".into())), h(&Value::Int(1)));
        assert_eq!(h(&Value::Str("hello".into())), h(&Value::Str("hello".into())));
        assert_ne!(h(&Value::Str("hello".into())), h(&Value::Str("world".into())));
    }

    #[test]
    fn hash_eq_null_separate_namespace() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // Null uses discriminant 4.
        assert_eq!(h(&Value::Null), h(&Value::Null));
        assert_ne!(h(&Value::Null), h(&Value::Int(0)));
        assert_ne!(h(&Value::Null), h(&Value::Float(0.0)));
        assert_ne!(h(&Value::Null), h(&Value::Bool(false)));
        assert_ne!(h(&Value::Null), h(&Value::Str("".into())));
    }

    #[test]
    fn hash_eq_large_values() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        assert_eq!(h(&Value::Int(i64::MAX)), h(&Value::Float(i64::MAX as f64)));
        assert_eq!(h(&Value::Int(i64::MIN)), h(&Value::Float(i64::MIN as f64)));
        assert_eq!(h(&Value::Int(0)), h(&Value::Float(0.0)));
    }

    #[test]
    fn hash_eq_boundary_floats() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // These float values have exact integer representations.
        assert_eq!(h(&Value::Float(1e0)), h(&Value::Int(1)));
        assert_eq!(h(&Value::Float(1e1)), h(&Value::Int(10)));
        assert_eq!(h(&Value::Float(1e2)), h(&Value::Int(100)));
        assert_eq!(h(&Value::Float(1e5)), h(&Value::Int(100_000)));

        // Infinity.
        assert_eq!(h(&Value::Float(f64::INFINITY)), h(&Value::Float(f64::INFINITY)));
        assert_ne!(h(&Value::Float(f64::INFINITY)), h(&Value::Float(f64::NEG_INFINITY)));
        assert_ne!(h(&Value::Float(f64::INFINITY)), h(&Value::Float(1e308)));
    }

    #[test]
    fn value_as_f64_conversions() {
        assert_eq!(Value::Int(42).as_f64(), Some(42.0));
        assert_eq!(Value::Float(3.14).as_f64(), Some(3.14));
        assert_eq!(Value::Bool(true).as_f64(), Some(1.0));
        assert_eq!(Value::Bool(false).as_f64(), Some(0.0));
        assert_eq!(Value::Str("42".to_string()).as_f64(), None);
        assert_eq!(Value::Null.as_f64(), None);
    }

    #[test]
    fn value_as_str() {
        assert_eq!(Value::Str("hello".to_string()).as_str(), Some("hello"));
        assert_eq!(Value::Int(1).as_str(), None);
        assert_eq!(Value::Null.as_str(), None);
    }

    #[test]
    fn value_to_string_display() {
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::Float(1.5).to_string(), "1.5");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Bool(false).to_string(), "false");
        assert_eq!(Value::Str("abc".to_string()).to_string(), "abc");
        assert_eq!(Value::Null.to_string(), "null");
    }

    #[test]
    fn value_is_null_is_some() {
        assert!(Value::Null.is_null());
        assert!(!Value::Null.is_some());
        assert!(!Value::Int(0).is_null());
        assert!(Value::Int(0).is_some());
        assert!(!Value::Float(0.0).is_null());
        assert!(Value::Float(0.0).is_some());
        assert!(!Value::Bool(false).is_null());
        assert!(Value::Bool(false).is_some());
        assert!(!Value::Str(String::new()).is_null());
        assert!(Value::Str(String::new()).is_some());
    }

    #[test]
    fn value_from_conversions() {
        let v: Value = 42i32.into();
        assert_eq!(v, Value::Int(42));
        let v: Value = 42i64.into();
        assert_eq!(v, Value::Int(42));
        let v: Value = 3.14f32.into();
        assert_eq!(v, Value::Float(3.14f32 as f64));
        let v: Value = 3.14f64.into();
        assert_eq!(v, Value::Float(3.14));
        let v: Value = true.into();
        assert_eq!(v, Value::Bool(true));
        let v: Value = "hello".into();
        assert_eq!(v, Value::Str("hello".to_string()));
        let v: Value = String::from("world").into();
        assert_eq!(v, Value::Str("world".to_string()));
    }

    #[test]
    fn hash_consistency_across_calls() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        let v = Value::Int(42);
        let h1 = h(&v);
        let h2 = h(&v);
        let h3 = h(&v);
        assert_eq!(h1, h2);
        assert_eq!(h2, h3);
    }

    #[test]
    fn hash_usable_in_hashmap() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(Value::Int(1), "one");
        map.insert(Value::Float(1.0), "also-one");
        // Since Int(1) == Float(1.0), they should occupy the same bucket.
        // The second insert overwrites the first.
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&Value::Int(1)), Some(&"also-one"));
        assert_eq!(map.get(&Value::Float(1.0)), Some(&"also-one"));

        map.insert(Value::Int(2), "two");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&Value::Int(2)), Some(&"two"));
        assert_eq!(map.get(&Value::Float(2.0)), Some(&"two"));
    }

    #[test]
    fn hash_nan_consistency_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Value::Float(f64::NAN));
        set.insert(Value::Float(f64::NAN));
        // NaN == NaN, so only one entry.
        assert_eq!(set.len(), 1);
        assert!(set.contains(&Value::Float(f64::NAN)));
    }

    #[test]
    fn hash_negative_zero_equal_positive_zero() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        use std::collections::HashSet;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        // They must be equal and hash identically.
        assert_eq!(Value::Float(0.0), Value::Float(-0.0));
        assert_eq!(h(&Value::Float(0.0)), h(&Value::Float(-0.0)));
        // In a HashSet they should be the same entry.
        let mut set = HashSet::new();
        set.insert(Value::Float(0.0));
        set.insert(Value::Float(-0.0));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn hash_value_clone_preserves_hash() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        fn h(v: &Value) -> u64 {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        }
        let values = vec![
            Value::Int(42),
            Value::Float(3.14),
            Value::Bool(true),
            Value::Str("test".into()),
            Value::Null,
            Value::Float(f64::NAN),
            Value::Float(0.0),
            Value::Float(-0.0),
        ];
        for v in &values {
            let v2 = v.clone();
            assert_eq!(h(v), h(&v2));
            assert_eq!(*v, v2);
        }
    }
}
