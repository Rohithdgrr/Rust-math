use core::fmt;

/// Errors that can occur in DataFrame operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataFrameError {
    /// Column not found by name.
    ColumnNotFound(String),
    /// Index out of bounds.
    IndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Length of the collection.
        length: usize,
    },
    /// Type mismatch between expected and actual dtype.
    TypeMismatch {
        /// Expected dtype.
        expected: &'static str,
        /// Actual dtype.
        actual: &'static str,
    },
    /// Dimension mismatch in operations requiring matching shapes.
    DimensionMismatch {
        /// Description of the mismatch.
        message: String,
    },
    /// Duplicate column name.
    DuplicateColumn(String),
    /// Join key not found.
    JoinKeyNotFound(String),
    /// Empty DataFrame where a non-empty one is required.
    EmptyDataFrame,
    /// Parse error for a value.
    ParseError {
        /// The value that failed to parse.
        value: String,
        /// Target type.
        target: &'static str,
    },
    /// I/O error.
    Io(String),
    /// Invalid operation.
    InvalidOperation(String),
}

impl fmt::Display for DataFrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnNotFound(name) => write!(f, "column not found: `{name}`"),
            Self::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "index {index} out of bounds for length {length}"
                )
            }
            Self::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected}, got {actual}")
            }
            Self::DimensionMismatch { message } => {
                write!(f, "dimension mismatch: {message}")
            }
            Self::DuplicateColumn(name) => {
                write!(f, "duplicate column name: `{name}`")
            }
            Self::JoinKeyNotFound(name) => {
                write!(f, "join key not found: `{name}`")
            }
            Self::EmptyDataFrame => {
                write!(f, "operation requires a non-empty DataFrame")
            }
            Self::ParseError { value, target } => {
                write!(
                    f,
                    "failed to parse `{value}` as {target}"
                )
            }
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
        }
    }
}

impl From<std::io::Error> for DataFrameError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<DataFrameError> for String {
    fn from(e: DataFrameError) -> String {
        e.to_string()
    }
}

/// Convenience alias for DataFrame operations.
pub type DataFrameResult<T> = Result<T, DataFrameError>;
