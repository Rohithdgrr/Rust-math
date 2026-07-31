//! Random forest: bagged decision trees with feature subsampling.

use crate::tree::DecisionTree;

/// Random forest classifier.
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    feature_indices: Vec<Vec<usize>>,
    n_trees: usize,
    max_depth: usize,
    max_features: usize,
}

impl RandomForest {
    pub fn new(n_trees: usize, max_depth: usize, max_features: usize) -> Self {
        Self { trees: Vec::new(), feature_indices: Vec::new(), n_trees, max_depth, max_features }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let _n = x.len();
        let p = x[0].len();
        let mf = self.max_features.min(p);
        self.trees.clear();
        self.feature_indices.clear();

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let (bx, by) = bootstrap_sample(x, y);
            // Random feature subset
            let mut feats: Vec<usize> = (0..p).collect();
            use_xorshift_shuffle(&mut feats);
            feats.truncate(mf);
            feats.sort();
            self.feature_indices.push(feats.clone());
            // Subset features
            let bx_sub: Vec<Vec<f64>> = bx.iter().map(|row| feats.iter().map(|&j| row[j]).collect()).collect();
            let mut tree = DecisionTree::new(self.max_depth, 2);
            tree.fit(&bx_sub, &by);
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let all_preds: Vec<Vec<f64>> = self.trees.iter().zip(&self.feature_indices)
            .map(|(tree, feats)| {
                let x_sub: Vec<Vec<f64>> = x.iter().map(|row| feats.iter().map(|&j| row[j]).collect()).collect();
                tree.predict(&x_sub)
            }).collect();
        let n = x.len();
        (0..n).map(|i| {
            let mut counts = std::collections::HashMap::new();
            for preds in &all_preds {
                *counts.entry(preds[i].to_bits()).or_insert(0usize) += 1;
            }
            f64::from_bits(*counts.iter().max_by_key(|(_, c)| *c).unwrap().0)
        }).collect()
    }

    pub fn predict_proba(&self, x: &[Vec<f64>], classes: &[f64]) -> Vec<Vec<f64>> {
        let all_preds: Vec<Vec<f64>> = self.trees.iter().zip(&self.feature_indices)
            .map(|(tree, feats)| {
                let x_sub: Vec<Vec<f64>> = x.iter().map(|row| feats.iter().map(|&j| row[j]).collect()).collect();
                tree.predict(&x_sub)
            }).collect();
        let n = x.len();
        let nc = classes.len();
        let total = self.n_trees as f64;
        (0..n).map(|i| {
            (0..nc).map(|c| {
                all_preds.iter().filter(|preds| (preds[i] - classes[c]).abs() < 1e-10).count() as f64 / total
            }).collect()
        }).collect()
    }
}

fn bootstrap_sample<'a>(x: &'a [Vec<f64>], y: &'a [f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let mut bx = Vec::with_capacity(n);
    let mut by = Vec::with_capacity(n);
    let mut state: u32 = 0x1234_5678;
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let idx = (state as usize) % n;
        bx.push(x[idx].clone());
        by.push(y[idx]);
    }
    (bx, by)
}

fn use_xorshift_shuffle(v: &mut [usize]) {
    let n = v.len();
    let mut state: u32 = 0xDEAD_BEEF;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict() {
        let x = vec![
            vec![1.0, 2.0], vec![1.5, 1.8], vec![1.2, 2.2],
            vec![5.0, 6.0], vec![5.5, 5.8], vec![4.8, 6.2],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut rf = RandomForest::new(10, 5, 2);
        rf.fit(&x, &y);
        let preds = rf.predict(&x);
        let correct = preds.iter().zip(&y).filter(|(&p, &t)| (p - t).abs() < 0.5).count();
        assert!(correct >= 5);
    }

    #[test]
    fn predict_proba_sums() {
        let x = vec![vec![1.0], vec![5.0]];
        let y = vec![0.0, 1.0];
        let mut rf = RandomForest::new(5, 3, 1);
        rf.fit(&x, &y);
        let probs = rf.predict_proba(&x, &[0.0, 1.0]);
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
