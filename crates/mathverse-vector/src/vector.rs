/// Dense f64 vector backed by `Vec<f64>`.
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
}
