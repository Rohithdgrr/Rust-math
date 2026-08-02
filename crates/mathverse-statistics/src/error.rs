//! Error types for the statistics crate.

/// Every failure mode this crate recognizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MathError {
    /// Input outside the function's domain.
    Domain,
    /// Division or modulus by zero.
    DivisionByZero,
    /// A parameter was invalid; carries a human-readable reason.
    InvalidArgument(&'static str),
    /// An iterative method failed to converge.
    NotConverged(&'static str),
    /// Matrix/vector shapes don't line up.
    DimensionMismatch,
    /// Result exceeded the representable range.
    Overflow,
    /// Result fell below the representable range.
    Underflow,
    /// Matrix is not invertible.
    Singular,
    /// Input must be non-negative.
    NegativeInput,
}

/// Convenience alias used everywhere.
pub type MathResult<T> = Result<T, MathError>;

impl core::fmt::Display for MathError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Domain => f.write_str("input is outside the function domain"),
            Self::DivisionByZero => f.write_str("division by zero"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::NotConverged(method) => write!(f, "{method} failed to converge"),
            Self::DimensionMismatch => f.write_str("dimension mismatch"),
            Self::Overflow => f.write_str("numeric overflow"),
            Self::Underflow => f.write_str("numeric underflow"),
            Self::Singular => f.write_str("singular matrix"),
            Self::NegativeInput => f.write_str("input must be non-negative"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MathError {}
