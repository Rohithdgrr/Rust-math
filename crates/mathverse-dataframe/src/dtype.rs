use core::fmt;

/// Supported data types for DataFrame columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 64-bit floating point.
    Float64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit signed integer.
    Int32,
    /// Boolean.
    Bool,
    /// Variable-length UTF-8 string.
    Utf8,
    /// Date as days since Unix epoch.
    Date,
    /// DateTime as microseconds since Unix epoch.
    DateTime,
    /// Signed duration in microseconds.
    Duration,
}

impl DType {
    /// Returns `true` if this is a numeric type (float or integer).
    #[must_use]
    pub const fn is_numeric(self) -> bool {
        matches!(
            self,
            Self::Float64 | Self::Float32 | Self::Int64 | Self::Int32
        )
    }

    /// Returns `true` if this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float64 | Self::Float32)
    }

    /// Returns `true` if this is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Int64 | Self::Int32)
    }

    /// Returns `true` if this is a temporal type (date, datetime, or duration).
    #[must_use]
    pub const fn is_temporal(self) -> bool {
        matches!(self, Self::Date | Self::DateTime | Self::Duration)
    }

    /// Returns the byte size of the native type, or 0 for variable-size types.
    #[must_use]
    pub const fn native_size(self) -> usize {
        match self {
            Self::Float64 | Self::Int64 | Self::DateTime | Self::Duration | Self::Date => 8,
            Self::Float32 | Self::Int32 | Self::Bool => 4,
            Self::Utf8 => 0,
        }
    }

    /// Returns a human-readable name for the dtype.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float64 => "f64",
            Self::Float32 => "f32",
            Self::Int64 => "i64",
            Self::Int32 => "i32",
            Self::Bool => "bool",
            Self::Utf8 => "str",
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::Duration => "duration",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
