//! Rational numbers: exact arithmetic with fractions.

use mathverse_core::error::{MathError, MathResult};

/// Rational number representation as a fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub numerator: i64,
    pub denominator: i64,
}

impl Rational {
    /// Create new rational number.
    pub fn new(numerator: i64, denominator: i64) -> MathResult<Self> {
        if denominator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let mut r = Rational {
            numerator,
            denominator,
        };
        r.reduce();
        Ok(r)
    }

    /// Create from integer.
    pub fn from_integer(value: i64) -> Self {
        Rational {
            numerator: value,
            denominator: 1,
        }
    }

    /// Create from floating point (approximate).
    pub fn from_f64(value: f64, tolerance: f64) -> MathResult<Self> {
        if value.is_nan() || value.is_infinite() {
            return Err(MathError::InvalidArgument("cannot convert NaN or infinity to rational"));
        }
        
        // Use continued fraction approximation
        let mut cf = Vec::new();
        let mut x = value;
        
        for _ in 0..20 {
            let a = x.floor() as i64;
            cf.push(a);
            
            let frac = x - a as f64;
            if frac.abs() < tolerance {
                break;
            }
            
            x = 1.0 / frac;
        }
        
        Self::from_continued_fraction(&cf)
    }

    /// Create from continued fraction.
    pub fn from_continued_fraction(cf: &[i64]) -> MathResult<Self> {
        if cf.is_empty() {
            return Err(MathError::InvalidArgument("empty continued fraction"));
        }
        
        let mut result = Rational::from_integer(cf[cf.len() - 1]);
        
        for i in (0..cf.len() - 1).rev() {
            let a = Rational::from_integer(cf[i]);
            result = a.add(&Rational::new(1, result.denominator)?)?;
        }
        
        Ok(result)
    }

    /// Reduce fraction to lowest terms.
    pub fn reduce(&mut self) {
        if self.denominator < 0 {
            self.numerator *= -1;
            self.denominator *= -1;
        }
        
        let gcd = Self::gcd(self.numerator.abs(), self.denominator.abs());
        self.numerator /= gcd;
        self.denominator /= gcd;
    }

    /// GCD using Euclidean algorithm.
    fn gcd(a: i64, b: i64) -> i64 {
        let mut a = a;
        let mut b = b;
        
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        
        a.abs()
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Get integer part.
    pub fn floor(&self) -> i64 {
        if self.numerator >= 0 {
            self.numerator / self.denominator
        } else {
            (self.numerator - self.denominator + 1) / self.denominator
        }
    }

    /// Get ceiling.
    pub fn ceil(&self) -> i64 {
        if self.numerator >= 0 {
            (self.numerator + self.denominator - 1) / self.denominator
        } else {
            self.numerator / self.denominator
        }
    }

    /// Get absolute value.
    pub fn abs(&self) -> Self {
        Rational {
            numerator: self.numerator.abs(),
            denominator: self.denominator,
        }
    }

    /// Check if positive.
    pub fn is_positive(&self) -> bool {
        self.numerator > 0
    }

    /// Check if negative.
    pub fn is_negative(&self) -> bool {
        self.numerator < 0
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.numerator == 0
    }

    /// Check if integer.
    pub fn is_integer(&self) -> bool {
        self.denominator == 1 || self.numerator % self.denominator == 0
    }

    /// Add two rational numbers.
    pub fn add(&self, other: &Rational) -> Self {
        let num = self.numerator * other.denominator + other.numerator * self.denominator;
        let den = self.denominator * other.denominator;
        
        Rational::new(num, den).unwrap()
    }

    /// Subtract two rational numbers.
    pub fn sub(&self, other: &Rational) -> Self {
        let num = self.numerator * other.denominator - other.numerator * self.denominator;
        let den = self.denominator * other.denominator;
        
        Rational::new(num, den).unwrap()
    }

    /// Multiply two rational numbers.
    pub fn mul(&self, other: &Rational) -> Self {
        let num = self.numerator * other.numerator;
        let den = self.denominator * other.denominator;
        
        Rational::new(num, den).unwrap()
    }

    /// Divide two rational numbers.
    pub fn div(&self, other: &Rational) -> MathResult<Self> {
        if other.numerator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let num = self.numerator * other.denominator;
        let den = self.denominator * other.numerator;
        
        Rational::new(num, den)
    }

    /// Negate rational number.
    pub fn neg(&self) -> Self {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }

    /// Raise to integer power.
    pub fn pow(&self, exp: i32) -> MathResult<Self> {
        if exp < 0 {
            return Err(MathError::InvalidArgument("negative exponent not implemented"));
        }
        
        let mut result = Rational::from_integer(1);
        let mut base = *self;
        let mut exp = exp as u32;
        
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        
        Ok(result)
    }

    /// Reciprocal.
    pub fn reciprocal(&self) -> MathResult<Self> {
        if self.numerator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        Rational::new(self.denominator, self.numerator)
    }

    /// Compare to another rational.
    pub fn cmp(&self, other: &Rational) -> std::cmp::Ordering {
        let left = self.numerator * other.denominator;
        let right = other.numerator * self.denominator;
        
        left.cmp(&right)
    }

    /// Get continued fraction representation.
    pub fn to_continued_fraction(&self, max_terms: usize) -> Vec<i64> {
        let mut cf = Vec::new();
        let mut a = *self;
        
        for _ in 0..max_terms {
            if a.denominator == 0 {
                break;
            }
            
            let integer = a.numerator / a.denominator;
            cf.push(integer);
            
            let remainder = a.numerator % a.denominator;
            if remainder == 0 {
                break;
            }
            
            a = Rational::new(a.denominator, remainder).unwrap();
        }
        
        cf
    }

    /// Get mixed number representation (whole, numerator, denominator).
    pub fn to_mixed_number(&self) -> (i64, i64, i64) {
        let whole = self.floor();
        let remainder = self.sub(&Rational::from_integer(whole));
        
        (whole, remainder.numerator, remainder.denominator)
    }
}

/// Rational number operations.
pub struct RationalOps;

impl RationalOps {
    /// Least Common Multiple of denominators.
    pub fn lcm_denominator(a: &Rational, b: &Rational) -> i64 {
        let gcd = Rational::gcd(a.denominator, b.denominator);
        (a.denominator / gcd) * b.denominator
    }

    /// Find common denominator for list of rationals.
    pub fn common_denominator(rationals: &[Rational]) -> i64 {
        if rationals.is_empty() {
            return 1;
        }
        
        rationals.iter().skip(1).fold(rationals[0].denominator, |acc, r| {
            let gcd = Rational::gcd(acc, r.denominator);
            (acc / gcd) * r.denominator
        })
    }

    /// Compare with tolerance.
    pub fn almost_equal(a: &Rational, b: &Rational, tolerance: f64) -> bool {
        (a.to_f64() - b.to_f64()).abs() < tolerance
    }

    /// Find rational approximation of f64 using continued fractions.
    pub fn approximate(value: f64, max_denominator: i64) -> MathResult<Rational> {
        let mut cf = Vec::new();
        let mut x = value;
        
        for _ in 0..20 {
            let a = x.floor() as i64;
            cf.push(a);
            
            let frac = x - a as f64;
            if frac.abs() < 1e-15 {
                break;
            }
            
            x = 1.0 / frac;
        }
        
        // Find best approximation with denominator <= max_denominator
        let mut best = Rational::from_integer(value.round() as i64);
        let mut best_error = (value - best.to_f64()).abs();
        
        for i in 1..=cf.len() {
            let r = Rational::from_continued_fraction(&cf[..i])?;
            if r.denominator <= max_denominator {
                let error = (value - r.to_f64()).abs();
                if error < best_error {
                    best = r;
                    best_error = error;
                }
            }
        }
        
        Ok(best)
    }

    /// Mediant of two rationals: (a/b) ⊕ (c/d) = (a+c)/(b+d).
    pub fn mediant(a: &Rational, b: &Rational) -> Rational {
        Rational::new(
            a.numerator + b.numerator,
            a.denominator + b.denominator,
        ).unwrap()
    }

    /// Farey sequence of order n: all rationals in [0,1] with denominator ≤ n.
    pub fn farey_sequence(n: i64) -> Vec<Rational> {
        let mut sequence = vec![Rational::new(0, 1).unwrap(), Rational::new(1, 1).unwrap()];
        
        for d in 2..=n {
            let mut new_terms = Vec::new();
            
            for i in 0..sequence.len() - 1 {
                let a = &sequence[i];
                let b = &sequence[i + 1];
                
                let mediant = Self::mediant(a, b);
                if mediant.denominator == d {
                    new_terms.push(mediant);
                }
            }
            
            // Insert new terms in correct positions
            for term in new_terms {
                let pos = sequence.binary_search_by(|r| r.cmp(&term)).unwrap_or_else(|e| e);
                if pos >= sequence.len() || sequence[pos] != term {
                    sequence.insert(pos, term);
                }
            }
        }
        
        sequence
    }

    /// Stern-Brocot tree traversal.
    pub fn stern_brocot(depth: usize) -> Vec<Rational> {
        let mut tree = vec![Rational::new(0, 1).unwrap(), Rational::new(1, 1).unwrap()];
        
        for _ in 0..depth {
            let mut new_level = Vec::new();
            
            for i in 0..tree.len() - 1 {
                new_level.push(tree[i].clone());
                new_level.push(Self::mediant(&tree[i], &tree[i + 1]));
            }
            
            new_level.push(tree.last().unwrap().clone());
            tree = new_level;
        }
        
        tree
    }
}

/// Fraction arithmetic utilities.
pub struct FractionUtils;

impl FractionUtils {
    /// Convert decimal string to fraction.
    pub fn from_decimal(s: &str) -> MathResult<Rational> {
        let parts: Vec<&str> = s.split('.').collect();
        
        match parts.len() {
            1 => {
                // Integer
                let numerator: i64 = parts[0].parse()
                    .map_err(|_| MathError::InvalidArgument("invalid integer"))?;
                Ok(Rational::from_integer(numerator))
            }
            2 => {
                // Decimal
                let integer: i64 = parts[0].parse()
                    .unwrap_or(0);
                let decimal = parts[1];
                let denominator = 10_i64.pow(decimal.len() as u32);
                let decimal_num: i64 = decimal.parse()
                    .map_err(|_| MathError::InvalidArgument("invalid decimal"))?;
                
                let numerator = integer * denominator + if integer >= 0 { decimal_num } else { -decimal_num };
                Rational::new(numerator, denominator)
            }
            _ => Err(MathError::InvalidArgument("invalid decimal format")),
        }
    }

    /// Convert percentage to fraction.
    pub fn from_percentage(percent: f64) -> MathResult<Rational> {
        Rational::from_f64(percent / 100.0, 1e-10)
    }

    /// Convert mixed number to improper fraction.
    pub fn from_mixed_number(whole: i64, numerator: i64, denominator: i64) -> MathResult<Rational> {
        if denominator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        let num = whole * denominator + if whole >= 0 { numerator } else { -numerator };
        Rational::new(num, denominator)
    }

    /// Simplify fraction by dividing by common factors.
    pub fn simplify(numerator: i64, denominator: i64) -> MathResult<(i64, i64)> {
        let r = Rational::new(numerator, denominator)?;
        Ok((r.numerator, r.denominator))
    }

    /// Check if two fractions are equivalent.
    pub fn equivalent(n1: i64, d1: i64, n2: i64, d2: i64) -> bool {
        let r1 = Rational::new(n1, d1).unwrap();
        let r2 = Rational::new(n2, d2).unwrap();
        r1 == r2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_creation() {
        let r = Rational::new(3, 4).unwrap();
        assert_eq!(r.numerator, 3);
        assert_eq!(r.denominator, 4);
        
        let r_neg = Rational::new(3, -4).unwrap();
        assert_eq!(r_neg.numerator, -3);
        assert_eq!(r_neg.denominator, 4);
    }

    #[test]
    fn test_rational_reduction() {
        let r = Rational::new(4, 8).unwrap();
        assert_eq!(r.numerator, 1);
        assert_eq!(r.denominator, 2);
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();
        
        let sum = a.add(&b);
        assert_eq!(sum.numerator, 5);
        assert_eq!(sum.denominator, 6);
        
        let product = a.mul(&b);
        assert_eq!(product.numerator, 1);
        assert_eq!(product.denominator, 6);
    }

    #[test]
    fn test_rational_division() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 4).unwrap();
        
        let quotient = a.div(&b).unwrap();
        assert_eq!(quotient.numerator, 2);
        assert_eq!(quotient.denominator, 1);
    }

    #[test]
    fn test_rational_power() {
        let r = Rational::new(2, 3).unwrap();
        let squared = r.pow(2).unwrap();
        
        assert_eq!(squared.numerator, 4);
        assert_eq!(squared.denominator, 9);
    }

    #[test]
    fn test_rational_comparison() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();
        
        assert!(a.cmp(&b) == std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_from_f64() {
        let r = Rational::from_f64(0.75, 1e-10).unwrap();
        assert_eq!(r.numerator, 3);
        assert_eq!(r.denominator, 4);
    }

    #[test]
    fn test_continued_fraction() {
        let r = Rational::new(22, 7).unwrap();
        let cf = r.to_continued_fraction(10);
        
        assert_eq!(cf, vec![3, 7]);
    }

    #[test]
    fn test_mixed_number() {
        let r = Rational::new(7, 4).unwrap();
        let (whole, num, den) = r.to_mixed_number();
        
        assert_eq!(whole, 1);
        assert_eq!(num, 3);
        assert_eq!(den, 4);
    }

    #[test]
    fn test_farey_sequence() {
        let farey = RationalOps::farey_sequence(5);
        
        assert!(farey.contains(&Rational::new(0, 1).unwrap()));
        assert!(farey.contains(&Rational::new(1, 5).unwrap()));
        assert!(farey.contains(&Rational::new(1, 1).unwrap()));
    }

    #[test]
    fn test_from_decimal() {
        let r = FractionUtils::from_decimal("0.75").unwrap();
        assert_eq!(r.numerator, 3);
        assert_eq!(r.denominator, 4);
    }

    #[test]
    fn test_from_mixed_number() {
        let r = FractionUtils::from_mixed_number(1, 3, 4).unwrap();
        assert_eq!(r.numerator, 7);
        assert_eq!(r.denominator, 4);
    }
}
