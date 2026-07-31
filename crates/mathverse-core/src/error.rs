//! Error taxonomy shared by the whole ecosystem.

/// Every failure mode MathVerse recognizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathError {
    /// Input outside the function's domain (e.g. `ln(-1)`).
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
    /// Computation exceeded a time or iteration limit.
    Timeout,
    /// I/O operation failed.
    Io,
    /// Parsing of a string or expression failed.
    Parse,
    /// Feature or operation not yet implemented.
    NotImplemented,
    /// Value outside a valid range.
    OutOfRange,
    /// No solution exists for the given inputs.
    NoSolution,
    /// Infinitely many solutions exist.
    InfiniteSolutions,
    /// Base of a number system is invalid.
    InvalidBase,
    /// Input must be non-negative.
    NegativeInput,
}

/// Convenience alias used everywhere.
pub type MathResult<T> = Result<T, MathError>;

impl core::fmt::Display for MathError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MathError::Domain => write!(f, "input is outside the function domain"),
            MathError::DivisionByZero => write!(f, "division by zero"),
            MathError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            MathError::NotConverged(method) => write!(f, "{method} failed to converge"),
            MathError::DimensionMismatch => write!(f, "dimension mismatch"),
            MathError::Overflow => write!(f, "numeric overflow"),
            MathError::Underflow => write!(f, "numeric underflow"),
            MathError::Singular => write!(f, "singular matrix"),
            MathError::Timeout => write!(f, "computation timed out"),
            MathError::Io => write!(f, "I/O error"),
            MathError::Parse => write!(f, "parse error"),
            MathError::NotImplemented => write!(f, "not implemented"),
            MathError::OutOfRange => write!(f, "value out of range"),
            MathError::NoSolution => write!(f, "no solution exists"),
            MathError::InfiniteSolutions => write!(f, "infinitely many solutions"),
            MathError::InvalidBase => write!(f, "invalid base"),
            MathError::NegativeInput => write!(f, "input must be non-negative"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MathError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_with_context() {
        assert_eq!(MathError::DivisionByZero.to_string(), "division by zero");
        assert_eq!(
            MathError::InvalidArgument("negative radius").to_string(),
            "invalid argument: negative radius"
        );
        assert_eq!(MathError::Timeout.to_string(), "computation timed out");
        assert_eq!(MathError::NoSolution.to_string(), "no solution exists");
        assert_eq!(MathError::Parse.to_string(), "parse error");
    }

    #[test]
    fn result_flow() {
        fn invert(x: f64) -> MathResult<f64> {
            if x == 0.0 {
                Err(MathError::DivisionByZero)
            } else {
                Ok(1.0 / x)
            }
        }
        assert_eq!(invert(4.0), Ok(0.25));
        assert_eq!(invert(0.0), Err(MathError::DivisionByZero));
    }
}
