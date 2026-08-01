/// Dense f64 vector backed by Vec<f64>.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self { Self { data } }
    pub fn zeros(n: usize) -> Self { Self { data: vec![0.0; n] } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn get(&self, i: usize) -> f64 { self.data[i] }
    pub fn set(&mut self, i: usize, v: f64) { self.data[i] = v; }

    pub fn add(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect())
    }

    pub fn sub(&self, other: &Vector) -> Vector {
        Vector::new(self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect())
    }

    pub fn scale(&self, scalar: f64) -> Vector {
        Vector::new(self.data.iter().map(|x| x * scalar).collect())
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
}
