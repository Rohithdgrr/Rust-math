//! Fused operation types for zero-allocation intermediate results.

use alloc::vec::Vec;

/// Fused addition of two slices into an output buffer.
pub struct FusedAdd<'a> {
    a: &'a [f64],
    b: &'a [f64],
}

impl<'a> FusedAdd<'a> {
    /// Create a new fused addition.
    pub fn new(a: &'a [f64], b: &'a [f64]) -> Self {
        Self { a, b }
    }

    /// Execute the fused operation, writing into `out`.
    pub fn eval(self, out: &mut [f64]) {
        for (o, (a, b)) in out.iter_mut().zip(self.a.iter().zip(self.b)) {
            *o = a + b;
        }
    }

    /// Execute and return a new vector.
    pub fn eval_to_vec(self) -> Vec<f64> {
        self.a.iter().zip(self.b).map(|(a, b)| a + b).collect()
    }
}

/// Fused multiplication of two slices.
pub struct FusedMul<'a> {
    a: &'a [f64],
    b: &'a [f64],
}

impl<'a> FusedMul<'a> {
    /// Create a new fused multiplication.
    pub fn new(a: &'a [f64], b: &'a [f64]) -> Self {
        Self { a, b }
    }

    /// Execute the fused operation, writing into `out`.
    pub fn eval(self, out: &mut [f64]) {
        for (o, (a, b)) in out.iter_mut().zip(self.a.iter().zip(self.b)) {
            *o = a * b;
        }
    }

    /// Execute and return a new vector.
    pub fn eval_to_vec(self) -> Vec<f64> {
        self.a.iter().zip(self.b).map(|(a, b)| a * b).collect()
    }
}

/// Fused scale (scalar multiply) of a slice.
pub struct FusedScale<'a> {
    data: &'a [f64],
    scalar: f64,
}

impl<'a> FusedScale<'a> {
    /// Create a new fused scale.
    pub fn new(data: &'a [f64], scalar: f64) -> Self {
        Self { data, scalar }
    }

    /// Execute the fused operation, writing into `out`.
    pub fn eval(self, out: &mut [f64]) {
        for (o, &v) in out.iter_mut().zip(self.data) {
            *o = v * self.scalar;
        }
    }

    /// Execute and return a new vector.
    pub fn eval_to_vec(self) -> Vec<f64> {
        self.data.iter().map(|&v| v * self.scalar).collect()
    }
}

/// Fused multiply-add: `a * b + c` in a single pass.
pub struct FusedMulAdd<'a> {
    a: &'a [f64],
    b: &'a [f64],
    c: &'a [f64],
}

impl<'a> FusedMulAdd<'a> {
    /// Create a new fused multiply-add.
    pub fn new(a: &'a [f64], b: &'a [f64], c: &'a [f64]) -> Self {
        Self { a, b, c }
    }

    /// Execute the fused operation, writing into `out`.
    pub fn eval(self, out: &mut [f64]) {
        for (o, ((&a, &b), &c)) in out
            .iter_mut()
            .zip(self.a.iter().zip(self.b).zip(self.c))
        {
            *o = a * b + c;
        }
    }

    /// Execute and return a new vector.
    pub fn eval_to_vec(self) -> Vec<f64> {
        self.a
            .iter()
            .zip(self.b)
            .zip(self.c)
            .map(|((&a, &b), &c)| a * b + c)
            .collect()
    }
}

/// Fused negate-scale: `-a * scalar`.
pub struct FusedNegScale<'a> {
    data: &'a [f64],
    scalar: f64,
}

impl<'a> FusedNegScale<'a> {
    /// Create a new fused negate-scale.
    pub fn new(data: &'a [f64], scalar: f64) -> Self {
        Self { data, scalar }
    }

    /// Execute and return a new vector.
    pub fn eval_to_vec(self) -> Vec<f64> {
        self.data.iter().map(|&v| -v * self.scalar).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = FusedAdd::new(&a, &b).eval_to_vec();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn fused_mul() {
        let a = [2.0, 3.0];
        let b = [4.0, 5.0];
        let result = FusedMul::new(&a, &b).eval_to_vec();
        assert_eq!(result, vec![8.0, 15.0]);
    }

    #[test]
    fn fused_scale() {
        let a = [1.0, 2.0, 3.0];
        let result = FusedScale::new(&a, 3.0).eval_to_vec();
        assert_eq!(result, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn fused_mul_add() {
        let a = [2.0, 3.0];
        let b = [4.0, 5.0];
        let c = [1.0, 1.0];
        let result = FusedMulAdd::new(&a, &b, &c).eval_to_vec();
        assert_eq!(result, vec![9.0, 16.0]);
    }

    #[test]
    fn fused_into_buffer() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let mut out = [0.0; 2];
        FusedAdd::new(&a, &b).eval(&mut out);
        assert_eq!(out, [4.0, 6.0]);
    }
}
