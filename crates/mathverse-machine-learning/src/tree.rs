//! Decision tree (CART) for classification.

#[derive(Debug, Clone)]
enum Node {
    Leaf { prediction: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// Fitted decision tree.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: Node,
    max_depth: usize,
    min_samples_split: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self { root: Node::Leaf { prediction: 0.0 }, max_depth, min_samples_split }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.root = build_tree(x, y, 0, self.max_depth, self.min_samples_split);
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| predict_one(&self.root, row)).collect()
    }

    pub fn predict_proba(&self, x: &[Vec<f64>], classes: &[f64]) -> Vec<Vec<f64>> {
        x.iter().map(|row| {
            let leaf_vals = collect_leaf_values(&self.root, row);
            let total = leaf_vals.len() as f64;
            classes.iter().map(|&c| {
                leaf_vals.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64 / total
            }).collect()
        }).collect()
    }
}

fn build_tree(x: &[Vec<f64>], y: &[f64], depth: usize, max_depth: usize, min_samples: usize) -> Node {
    if y.is_empty() { return Node::Leaf { prediction: 0.0 }; }
    // All same class or depth limit or too few samples
    let all_same = y.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
    if all_same || depth >= max_depth || y.len() < min_samples {
        return Node::Leaf { prediction: majority_class(y) };
    }
    // Find best split
    if let Some((feat, thresh, _gain)) = find_best_split(x, y) {
        let (left_x, left_y, right_x, right_y) = split_data(x, y, feat, thresh);
        let left = build_tree(&left_x, &left_y, depth + 1, max_depth, min_samples);
        let right = build_tree(&right_x, &right_y, depth + 1, max_depth, min_samples);
        Node::Split { feature: feat, threshold: thresh, left: Box::new(left), right: Box::new(right) }
    } else {
        Node::Leaf { prediction: majority_class(y) }
    }
}

fn find_best_split(x: &[Vec<f64>], y: &[f64]) -> Option<(usize, f64, f64)> {
    let n = y.len();
    let p = x[0].len();
    let parent_impurity = gini(y);
    let mut best = None;
    let mut best_gain = -1e-10;
    for j in 0..p {
        let mut vals: Vec<(f64, f64)> = x.iter().zip(y).map(|(row, &yi)| (row[j], yi)).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..n - 1 {
            if (vals[i].0 - vals[i + 1].0).abs() < 1e-10 { continue; }
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            let left_y: Vec<f64> = vals.iter().take(i + 1).map(|(_, y)| *y).collect();
            let right_y: Vec<f64> = vals.iter().skip(i + 1).map(|(_, y)| *y).collect();
            if left_y.is_empty() || right_y.is_empty() { continue; }
            let li = left_y.len() as f64 / n as f64;
            let ri = right_y.len() as f64 / n as f64;
            let gain = parent_impurity - li * gini(&left_y) - ri * gini(&right_y);
            if gain > best_gain {
                best_gain = gain;
                best = Some((j, thresh, gain));
            }
        }
    }
    best
}

fn gini(y: &[f64]) -> f64 {
    if y.is_empty() { return 0.0; }
    let n = y.len() as f64;
    let mut counts = std::collections::HashMap::new();
    for &yi in y { *counts.entry(yi.to_bits()).or_insert(0usize) += 1; }
    1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

fn majority_class(y: &[f64]) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for &yi in y { *counts.entry(yi.to_bits()).or_insert(0usize) += 1; }
    counts.iter().max_by_key(|(_, c)| *c).map(|(&v, _)| f64::from_bits(v)).unwrap_or(0.0)
}

fn split_data(x: &[Vec<f64>], y: &[f64], feature: usize, threshold: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut lx = Vec::new(); let mut ly = Vec::new();
    let mut rx = Vec::new(); let mut ry = Vec::new();
    for (row, &yi) in x.iter().zip(y) {
        if row[feature] <= threshold { lx.push(row.clone()); ly.push(yi); }
        else { rx.push(row.clone()); ry.push(yi); }
    }
    (lx, ly, rx, ry)
}

fn predict_one(node: &Node, x: &[f64]) -> f64 {
    match node {
        Node::Leaf { prediction } => *prediction,
        Node::Split { feature, threshold, left, right } => {
            if x[*feature] <= *threshold { predict_one(left, x) }
            else { predict_one(right, x) }
        }
    }
}

fn collect_leaf_values(node: &Node, x: &[f64]) -> Vec<f64> {
    match node {
        Node::Leaf { prediction } => vec![*prediction],
        Node::Split { feature, threshold, left, right } => {
            if x[*feature] <= *threshold { collect_leaf_values(left, x) }
            else { collect_leaf_values(right, x) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict_xor_like() {
        let x = vec![
            vec![0.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 0.0], vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let mut tree = DecisionTree::new(10, 2);
        tree.fit(&x, &y);
        let preds = tree.predict(&x);
        let correct = preds.iter().zip(&y).filter(|(&p, &t)| (p - t).abs() < 0.5).count();
        assert_eq!(correct, 4);
    }

    #[test]
    fn depth_one() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut tree = DecisionTree::new(1, 2);
        tree.fit(&x, &y);
        let preds = tree.predict(&x);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[3], 1.0);
    }

    #[test]
    fn single_class() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, 1.0];
        let mut tree = DecisionTree::new(5, 2);
        tree.fit(&x, &y);
        assert_eq!(tree.predict(&x), vec![1.0, 1.0]);
    }
}
