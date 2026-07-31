//! Isolation Forest for anomaly detection.

use mathverse_core::error::MathResult;

/// Isolation Forest model.
pub struct IsolationForest {
    pub n_trees: usize,
    pub subsample_size: usize,
    pub trees: Vec<IsolationTree>,
    pub threshold: f64,
}

struct IsolationTree {
    feature: usize,
    threshold: f64,
    left: Option<Box<IsolationTreeData>>,
    right: Option<Box<IsolationTreeData>>,
    depth: usize,
    size: usize,
}

enum IsolationTreeData {
    Branch { tree: IsolationTree },
    Leaf { size: usize },
}

fn build_tree(data: &[Vec<f64>], max_depth: usize, current_depth: usize, rng_state: &mut u64) -> IsolationTree {
    let n = data.len();
    let p = data[0].len();

    if n <= 1 || current_depth >= max_depth {
        return IsolationTree { feature: 0, threshold: 0.0, left: None, right: None, depth: current_depth, size: n };
    }

    // Pick random feature
    *rng_state ^= *rng_state << 13; *rng_state ^= *rng_state >> 7; *rng_state ^= *rng_state << 17;
    let feature = (*rng_state as usize) % p;

    // Pick random threshold between min and max of feature
    let min_val = data.iter().map(|xi| xi[feature]).fold(f64::INFINITY, f64::min);
    let max_val = data.iter().map(|xi| xi[feature]).fold(f64::NEG_INFINITY, f64::max);
    if (max_val - min_val).abs() < 1e-15 {
        return IsolationTree { feature: 0, threshold: 0.0, left: None, right: None, depth: current_depth, size: n };
    }
    *rng_state ^= *rng_state << 13; *rng_state ^= *rng_state >> 7; *rng_state ^= *rng_state << 17;
    let threshold = min_val + ((*rng_state as f64) / (u64::MAX as f64)) * (max_val - min_val);

    let left_data: Vec<Vec<f64>> = data.iter().filter(|xi| xi[feature] < threshold).cloned().collect();
    let right_data: Vec<Vec<f64>> = data.iter().filter(|xi| xi[feature] >= threshold).cloned().collect();

    let left = if left_data.is_empty() {
        None
    } else {
        Some(Box::new(IsolationTreeData::Branch { tree: build_tree(&left_data, max_depth, current_depth + 1, rng_state) }))
    };
    let right = if right_data.is_empty() {
        None
    } else {
        Some(Box::new(IsolationTreeData::Branch { tree: build_tree(&right_data, max_depth, current_depth + 1, rng_state) }))
    };

    IsolationTree { feature, threshold, left, right, depth: current_depth, size: n }
}

impl IsolationTree {
    fn path_length(&self, x: &[f64]) -> f64 {
        match (&self.left, &self.right) {
            (None, None) => {
                // Leaf
                self.depth as f64 + c_factor(self.size)
            }
            (Some(left), Some(right)) => {
                if x[self.feature] < self.threshold {
                    match left.as_ref() {
                        IsolationTreeData::Branch { tree } => tree.path_length(x),
                        IsolationTreeData::Leaf { size } => self.depth as f64 + c_factor(*size),
                    }
                } else {
                    match right.as_ref() {
                        IsolationTreeData::Branch { tree } => tree.path_length(x),
                        IsolationTreeData::Leaf { size } => self.depth as f64 + c_factor(*size),
                    }
                }
            }
            (Some(left), None) | (None, Some(left)) => {
                if x[self.feature] < self.threshold {
                    match left.as_ref() {
                        IsolationTreeData::Branch { tree } => tree.path_length(x),
                        IsolationTreeData::Leaf { size } => self.depth as f64 + c_factor(*size),
                    }
                } else {
                    self.depth as f64 + c_factor(self.size)
                }
            }
        }
    }
}

/// Average path length of unsuccessful search in BST.
fn c_factor(n: usize) -> f64 {
    if n <= 1 { return 0.0; }
    2.0 * ((n as f64 - 1.0).ln()) - 2.0 * (n as f64 - 1.0) / n as f64 + 1.0
}

impl IsolationForest {
    pub fn new(n_trees: usize, subsample_size: usize) -> Self {
        Self { n_trees, subsample_size, trees: Vec::new(), threshold: -0.5 }
    }

    pub fn fit(&mut self, x: &[Vec<f64>]) {
        let n = x.len();
        let mut rng = 0xABCD_1234_u64;
        let max_depth = (self.subsample_size as f64).ceil() as usize;

        self.trees = (0..self.n_trees).map(|_| {
            // Subsample
            let mut indices: Vec<usize> = (0..n).collect();
            let sub_n = self.subsample_size.min(n);
            for i in (1..n).rev() {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let j = (rng as usize) % (i + 1);
                indices.swap(i, j);
            }
            let subsample: Vec<Vec<f64>> = indices.iter().take(sub_n).map(|&i| x[i].clone()).collect();
            build_tree(&subsample, max_depth, 0, &mut rng)
        }).collect();
    }

    /// Compute anomaly scores: higher = more anomalous.
    pub fn score_samples(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len() as f64;
        let avg_c = c_factor(self.subsample_size);
        x.iter().map(|xi| {
            let avg_path: f64 = self.trees.iter().map(|t| t.path_length(xi)).sum::<f64>() / self.trees.len() as f64;
            // Anomaly score: 2^(-avg_path / c_factor(n))
            (-avg_path / avg_c * 2.0_f64.ln()).exp()
        }).collect()
    }

    /// Predict: -1 for anomaly, 1 for normal.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i32> {
        self.score_samples(x).iter()
            .map(|&s| if s > 0.5 { -1 } else { 1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_forest_test() {
        let mut normal: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let anomaly = vec![vec![1000.0, 2000.0]];
        let mut x = normal.clone();
        x.extend(anomaly.clone());

        let mut forest = IsolationForest::new(100, 256);
        forest.fit(&x);
        let scores = forest.score_samples(&x);
        // Last point (anomaly) should have highest score
        assert!(*scores.last().unwrap() > scores[0]);
    }

    #[test]
    fn predict_test() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![100.0, 200.0]];
        let mut forest = IsolationForest::new(10, 3);
        forest.fit(&x);
        let pred = forest.predict(&x);
        assert_eq!(pred.len(), 3);
    }
}
