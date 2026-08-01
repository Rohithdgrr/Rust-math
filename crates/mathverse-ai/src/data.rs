//! Data loading utilities: DataLoader with batching and shuffling.

use crate::tensor::Tensor;

/// A mini-batch of inputs and targets.
pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

/// DataLoader: iterates over data in shuffled mini-batches.
pub struct DataLoader {
    pub x: Tensor,
    pub y: Tensor,
    pub batch_size: usize,
    pub shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl DataLoader {
    pub fn new(x: Tensor, y: Tensor, batch_size: usize, shuffle: bool) -> Self {
        let n = x.shape[0];
        let indices: Vec<usize> = (0..n).collect();
        Self { x, y, batch_size, shuffle, indices, pos: 0 }
    }

    /// Reset iterator.
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            use std::cell::Cell;
            thread_local! { static S: Cell<u64> = const { Cell::new(0xABCD) }; }
            // Fisher-Yates shuffle with xorshift
            for i in (1..self.indices.len()).rev() {
                S.with(|s| {
                    let mut x = s.get();
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    s.set(x);
                    let j = (x as usize) % (i + 1);
                    self.indices.swap(i, j);
                });
            }
        }
    }

    /// Number of batches.
    pub fn num_batches(&self) -> usize {
        self.x.shape[0].div_ceil(self.batch_size)
    }

    /// Get next batch.
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.pos >= self.x.shape[0] { return None; }
        let end = (self.pos + self.batch_size).min(self.x.shape[0]);
        let batch_len = end - self.pos;
        let dim: usize = self.x.shape[1..].iter().product();
        let y_dim: usize = if self.y.shape.len() > 1 { self.y.shape[1..].iter().product() } else { 1 };

        let mut x_data = Vec::with_capacity(batch_len * dim);
        let mut y_data = Vec::with_capacity(batch_len * y_dim);
        for &idx in &self.indices[self.pos..end] {
            x_data.extend_from_slice(&self.x.data[idx * dim..(idx + 1) * dim]);
            y_data.extend_from_slice(&self.y.data[idx * y_dim..(idx + 1) * y_dim]);
        }

        let mut x_shape = self.x.shape.clone();
        x_shape[0] = batch_len;
        let mut y_shape = self.y.shape.clone();
        y_shape[0] = batch_len;

        self.pos = end;
        Some(Batch {
            x: Tensor { shape: x_shape, data: x_data },
            y: Tensor { shape: y_shape, data: y_data },
        })
    }
}

/// Iterator adapter for DataLoader.
impl Iterator for DataLoader {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Train/test split: returns (x_train, x_test, y_train, y_test).
pub fn train_test_split(x: &Tensor, y: &Tensor, test_ratio: f64, seed: u64) -> (Tensor, Tensor, Tensor, Tensor) {
    let n = x.shape[0];
    let test_size = (n as f64 * test_ratio) as usize;
    let train_size = n - test_size;

    let mut indices: Vec<usize> = (0..n).collect();
    // Deterministic shuffle
    let mut state = seed;
    for i in (1..n).rev() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }

    let x_dim: usize = x.shape[1..].iter().product();
    let y_dim: usize = if y.shape.len() > 1 { y.shape[1..].iter().product() } else { 1 };

    let mut x_train_data = Vec::with_capacity(train_size * x_dim);
    let mut x_test_data = Vec::with_capacity(test_size * x_dim);
    let mut y_train_data = Vec::with_capacity(train_size * y_dim);
    let mut y_test_data = Vec::with_capacity(test_size * y_dim);

    for (i, &idx) in indices.iter().enumerate() {
        if i < train_size {
            x_train_data.extend_from_slice(&x.data[idx * x_dim..(idx + 1) * x_dim]);
            y_train_data.extend_from_slice(&y.data[idx * y_dim..(idx + 1) * y_dim]);
        } else {
            x_test_data.extend_from_slice(&x.data[idx * x_dim..(idx + 1) * x_dim]);
            y_test_data.extend_from_slice(&y.data[idx * y_dim..(idx + 1) * y_dim]);
        }
    }

    let mut x_shape = x.shape.clone();
    let mut y_shape = y.shape.clone();
    x_shape[0] = train_size;
    y_shape[0] = train_size;
    let x_train = Tensor { shape: x_shape, data: x_train_data };
    let y_train = Tensor { shape: y_shape, data: y_train_data };

    let mut x_shape = x.shape.clone();
    let mut y_shape = y.shape.clone();
    x_shape[0] = test_size;
    y_shape[0] = test_size;
    let x_test = Tensor { shape: x_shape, data: x_test_data };
    let y_test = Tensor { shape: y_shape, data: y_test_data };

    (x_train, x_test, y_train, y_test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataloader_test() {
        let x = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]).unwrap();
        let y = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]).unwrap();
        let mut loader = DataLoader::new(x, y, 3, false);
        assert_eq!(loader.num_batches(), 4);
        let batch = loader.next().unwrap();
        assert_eq!(batch.x.shape, vec![3, 1]);
    }

    #[test]
    fn train_test_split_test() {
        let x = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]).unwrap();
        let y = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]).unwrap();
        let (_x_tr, _x_te, _y_tr, _y_te) = train_test_split(&x, &y, 0.2, 42);
        assert_eq!(x_tr.shape[0], 8);
        assert_eq!(x_te.shape[0], 2);
    }
}


