/// Dense `f64` vector backed by `Vec<f64>`.
///
/// Provides basic element-wise arithmetic and dot product operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The underlying data storage.
    pub data: Vec<f64>,
}

use mathverse_core::error::{MathError, MathResult};

impl Vector {
    /// Creates a new `Vector` from the given data.
    pub fn new(data: Vec<f64>) -> Self { Self { data } }

    /// Creates a zero vector of length `n`.
    pub fn zeros(n: usize) -> Self { Self { data: vec![0.0; n] } }

    /// Returns the number of elements.
    pub fn len(&self) -> usize { self.data.len() }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Returns the element at index `i`.
    pub fn get(&self, i: usize) -> f64 { self.data[i] }

    /// Sets the element at index `i` to `v`.
    pub fn set(&mut self, i: usize, v: f64) { self.data[i] = v; }

    /// Element-wise addition of two vectors.
    pub fn add(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect())
    }

    /// Element-wise subtraction of two vectors.
    pub fn sub(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect())
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Vector {
        Vector::new(self.data.iter().map(|x| x * scalar).collect())
    }

    /// Dot product of two vectors.
    pub fn dot(&self, other: &Vector) -> f64 {
        crate::operations::dot(&self.data, &other.data)
    }

    /// Unit vector in the same direction.
    ///
    /// Returns [`MathError::DivisionByZero`] if the vector has zero length, so
    /// callers can react instead of silently producing a zero "unit" vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathverse_vector::Vector;
    ///
    /// let v = Vector::new(vec![3.0, 4.0]);
    /// let u = v.normalized().unwrap();
    /// assert!((u.get(0) - 0.6).abs() < 1e-12);
    /// assert!(Vector::zeros(2).normalized().is_err());
    /// ```
    pub fn normalized(&self) -> MathResult<Vector> {
        let m = crate::operations::magnitude(&self.data);
        if m == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(Vector::new(self.data.iter().map(|x| x / m).collect()))
    }

    /// 3D cross product `self × other`.
    ///
    /// Returns [`MathError::DimensionMismatch`] unless both vectors have
    /// exactly three elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathverse_vector::Vector;
    ///
    /// let i = Vector::new(vec![1.0, 0.0, 0.0]);
    /// let j = Vector::new(vec![0.0, 1.0, 0.0]);
    /// let k = i.cross3(&j).unwrap();
    /// assert!((k.get(2) - 1.0).abs() < 1e-12);
    /// ```
    pub fn cross3(&self, other: &Vector) -> MathResult<Vector> {
        if self.len() != 3 || other.len() != 3 {
            return Err(MathError::DimensionMismatch);
        }
        Ok(Vector::new(crate::operations::cross(&self.data, &other.data)))
    }
}
