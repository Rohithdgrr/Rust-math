/// Dense `f64` vector backed by `Vec<f64>`.
///
/// Provides basic element-wise arithmetic and dot product operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The underlying data storage.
    pub data: Vec<f64>,
}

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
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
}
