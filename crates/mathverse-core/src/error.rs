//! Error taxonomy shared by the whole ecosystem.
//!
//! Every failure mode `MathVerse` recognizes is represented as a variant of
//! [`MathError`]. The [`MathResult<T>`] type alias simplifies return types.

use alloc::string::{String, ToString};

/// Every failure mode `MathVerse` recognizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
    /// Numerical failure: NaN/Inf detected during computation.
    NumericalFailure(&'static str),
}

/// Convenience alias used everywhere.
pub type MathResult<T> = Result<T, MathError>;

impl MathError {
    /// Returns a human-readable description of the error.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathverse_core::error::MathError;
    ///
    /// assert_eq!(MathError::Domain.msg(), "input is outside the function domain");
    /// assert_eq!(MathError::DivisionByZero.msg(), "division by zero");
    /// assert_eq!(MathError::Overflow.msg(), "numeric overflow");
    /// ```
    #[must_use]
    pub const fn msg(&self) -> &'static str {
        match self {
            Self::Domain => "input is outside the function domain",
            Self::DivisionByZero => "division by zero",
            Self::InvalidArgument(m) | Self::NotConverged(m) => m,
            Self::DimensionMismatch => "dimension mismatch",
            Self::Overflow => "numeric overflow",
            Self::Underflow => "numeric underflow",
            Self::Singular => "singular matrix",
            Self::Timeout => "computation timed out",
            Self::Io => "I/O error",
            Self::Parse => "parse error",
            Self::NotImplemented => "not implemented",
            Self::OutOfRange => "value out of range",
            Self::NoSolution => "no solution exists",
            Self::InfiniteSolutions => "infinitely many solutions",
            Self::InvalidBase => "invalid base",
            Self::NegativeInput => "input must be non-negative",
            Self::NumericalFailure(m) => m,
        }
    }

    /// Returns `true` if this error variant carries a custom message.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathverse_core::error::MathError;
    ///
    /// assert!(MathError::InvalidArgument("bad").has_context());
    /// assert!(!MathError::Domain.has_context());
    /// ```
    #[must_use]
    pub const fn has_context(&self) -> bool {
        matches!(self, Self::InvalidArgument(_) | Self::NotConverged(_))
    }
}

impl core::fmt::Display for MathError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::NotConverged(method) => write!(f, "{method} failed to converge"),
            other => f.write_str(other.msg()),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MathError {}

#[allow(clippy::use_self)]
impl From<MathError> for String {
    fn from(e: MathError) -> String {
        e.to_string()
    }
}

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
    fn msg_method() {
        assert_eq!(MathError::Domain.msg(), "input is outside the function domain");
        assert_eq!(
            MathError::InvalidArgument("test").msg(),
            "test"
        );
        assert_eq!(MathError::Overflow.msg(), "numeric overflow");
    }

    #[test]
    fn has_context_method() {
        assert!(MathError::InvalidArgument("x").has_context());
        assert!(MathError::NotConverged("newton").has_context());
        assert!(!MathError::Domain.has_context());
        assert!(!MathError::Overflow.has_context());
    }

    #[test]
    fn into_string() {
        let e: String = MathError::DivisionByZero.into();
        assert_eq!(e, "division by zero");
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
