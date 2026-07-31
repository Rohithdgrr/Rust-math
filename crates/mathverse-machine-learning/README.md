# mathverse-machine-learning

Classical machine learning algorithms in pure Rust: supervised learning, unsupervised clustering, ensemble methods, model evaluation, and feature preprocessing — zero dependencies.

## Features

- **Supervised**: Linear Regression (OLS, Ridge, Lasso), Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting
- **Unsupervised**: K-Means, DBSCAN, Agglomerative Hierarchical Clustering, Gaussian Mixture Model
- **Model Selection**: Train/test split, K-fold cross-validation, Accuracy, Confusion Matrix, ROC Curve, AUC
- **Feature Preprocessing**: Standardization, Min-Max Normalization, One-Hot Encoding, Polynomial Features

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `linear` | Linear regression, Ridge, Lasso | `fit`, `fit_ridge`, `fit_lasso`, `predict` |
| `logistic` | Logistic regression via GD | `fit`, `predict`, `predict_proba`, `cross_entropy` |
| `knn` | k-Nearest Neighbors | `classify`, `regress` |
| `naive_bayes` | Gaussian Naive Bayes | `fit`, `predict`, `predict_proba` |
| `tree` | CART Decision Tree | `DecisionTree::fit`, `predict`, `predict_proba` |
| `forest` | Random Forest | `RandomForest::new`, `fit`, `predict` |
| `boosting` | Gradient Boosting | `GradientBoostingRegressor::new`, `fit`, `predict` |
| `kmeans` | K-Means clustering | `kmeans` |
| `dbscan` | DBSCAN clustering | `dbscan` |
| `hierarchical` | Agglomerative clustering | `agglomerative` |
| `gmm` | Gaussian Mixture Model | `fit_gmm`, `predict` |
| `model_selection` | Evaluation & CV | `train_test_split`, `k_fold_cv`, `roc_curve`, `auc` |
| `feature` | Feature preprocessing | `standardize`, `min_max`, `one_hot`, `polynomial_features` |

## Installation

```bash
cargo add mathverse-machine-learning
```

Or add to `Cargo.toml`:

```toml
[dependencies]
mathverse-machine-learning = { path = "../mathverse-machine-learning" }
```

## Quick Start

```rust
use mathverse_machine_learning::knn;

fn main() {
    let x_train = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
    let y_train = vec![0.0, 0.0, 1.0, 1.0];
    let x_test = vec![vec![0.5], vec![10.5]];

    let preds = knn::classify(&x_train, &y_train, &x_test, 1).unwrap();
    println!("Predictions: {:?}", preds);
    // Predictions: [0.0, 1.0]
}
```

## Module Documentation

### Linear Regression

Ordinary Least Squares, Ridge (L2), and Lasso (L1) regression via Cholesky decomposition and coordinate descent.

```
Decision Boundary:

  y│        ·
   │      ·   OLS fit: y = 2x + 1
   │    ·
   │  ·
   │·
   └──────────── x
```

```rust
use mathverse_machine_learning::linear;

let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
let y: Vec<f64> = (0..5).map(|i| 2.0 * i as f64 + 1.0).collect();

let result = linear::fit(&x, &y).unwrap();
println!("Coefficients: {:?}", result.coefficients); // [2.0]
println!("Intercept: {:.1}", result.intercept);       // 1.0
println!("R²: {:.4}", result.r_squared);              // 1.0000

// Ridge regression (regularized)
let ridge = linear::fit_ridge(&x, &y, 0.1).unwrap();

// Lasso regression (feature selection)
let lasso = linear::fit_lasso(&x, &y, 0.1, 1000, 1e-6).unwrap();

// Predict
let preds = linear::predict(&vec![vec![5.0], vec![6.0]], &result.coefficients, result.intercept);
// [11.0, 13.0]
```

**Formula**: `y = β₀ + β₁x₁ + ... + βₚxₚ`

### Logistic Regression

Binary classification via gradient descent with sigmoid activation.

```rust
use mathverse_machine_learning::logistic;

let x: Vec<Vec<f64>> = (-10..10).map(|i| vec![i as f64]).collect();
let y: Vec<f64> = (-10..10).map(|i| if i > 0 { 1.0 } else { 0.0 }).collect();

let result = logistic::fit(&x, &y, 0.1, 1000, 1e-8).unwrap();
println!("Weight: {:.2}", result.coefficients[0]); // positive

let probs = logistic::predict_proba(&x, &result.coefficients, result.intercept);
let preds = logistic::predict(&x, &result.coefficients, result.intercept);
```

**Formula**: `P(y=1) = σ(wᵀx + b)`, where `σ(z) = 1/(1+e⁻ᶻ)`

### K-Nearest Neighbors

Instance-based learning for classification and regression.

```
KNN Decision Boundary (k=3):

  Class 0        │     Class 1
    ○ ○          │     ● ●
   ○ ○ ○         │   ● ● ●
  ○ ○ ○ ○   ?────┤  ● ● ● ●
   ○ ○ ○    ▲    │   ● ● ●
    ○ ○     │    │     ● ●
            │    │
  ◄───── nearest ─────►
         neighbors
```

```rust
use mathverse_machine_learning::knn;

let x_train = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
let y_train = vec![0.0, 0.0, 1.0, 1.0];

// Classification
let preds = knn::classify(&x_train, &y_train, &vec![vec![0.5], vec![10.5]], 1).unwrap();
// [0.0, 1.0]

// Regression
let preds = knn::regress(&x_train, &y_train, &vec![vec![1.5]], 2).unwrap();
// [15.0] — mean of 10.0 and 20.0 (nearest 2 neighbors)
```

### Decision Tree (CART)

Recursive binary splitting using Gini impurity.

```
Decision Tree:

              [feature_0 ≤ 2.5]
               /            \
           [Yes]           [No]
            /                \
    [Class: A]          [feature_1 ≤ 1.5]
                        /            \
                    [Yes]           [No]
                     /                \
             [Class: B]          [Class: C]
```

```rust
use mathverse_machine_learning::tree::DecisionTree;

let x = vec![
    vec![0.0, 0.0], vec![0.0, 1.0],
    vec![1.0, 0.0], vec![1.0, 1.0],
];
let y = vec![0.0, 1.0, 1.0, 0.0]; // XOR

let mut tree = DecisionTree::new(10, 2);
tree.fit(&x, &y);
let preds = tree.predict(&x);
// [0.0, 1.0, 1.0, 0.0] — perfect on training data
```

**Formula**: `Gini(S) = 1 - Σ(pᵢ²)` where `pᵢ` is fraction of class `i` in set `S`.

### Random Forest

Ensemble of bagged decision trees with random feature subsampling.

```
Random Forest:

  ┌─────────────┐
  │  Bootstrap   │
  │  Sample 1    │──► Tree 1 (features A,B)  ──► Vote
  ├─────────────┤                                │
  │  Bootstrap   │                                │
  │  Sample 2    │──► Tree 2 (features B,C)  ──► Vote
  ├─────────────┤                                │
  │  Bootstrap   │                                ▼
  │  Sample N    │──► Tree N (features A,C)  ──► Majority
  └─────────────┘                                ▼
                                             Prediction
```

```rust
use mathverse_machine_learning::forest::RandomForest;

let x = vec![
    vec![1.0, 2.0], vec![1.5, 1.8], vec![1.2, 2.2],
    vec![5.0, 6.0], vec![5.5, 5.8], vec![4.8, 6.2],
];
let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

let mut rf = RandomForest::new(10, 5, 2);
rf.fit(&x, &y);
let preds = rf.predict(&x);
// Majority vote from 10 trees
```

### Gradient Boosting

Sequential ensemble of weak learners (stumps) trained on residuals.

```rust
use mathverse_machine_learning::boosting::GradientBoostingRegressor;

let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
let y: Vec<f64> = (0..20).map(|i| i as f64 * 2.0 + 1.0).collect();

let mut gb = GradientBoostingRegressor::new(50, 0.1, 3);
gb.fit(&x, &y);
let preds = gb.predict(&x);
// Predictions closely follow y = 2x + 1
```

### K-Means Clustering

Partitional clustering with K-means++ initialization.

```
K-Means Iterations:

  Iter 0:          Iter 1:          Converged:
  · · · ·          · · · ·          · · · ·
  ·   · ·          ·   · ·          ●●●●●●●
  · · · ·          ●●●●●●●          ●●●●●●●
  ●●●●●●●          ●●●●●●●          · · · ·
  ●●●●●●●          · · · ·          · · · ·
  ●●●●●●●          · · · ·          · · · ·
```

```rust
use mathverse_machine_learning::kmeans;

let mut x: Vec<Vec<f64>> = Vec::new();
for i in 0..10 { x.push(vec![i as f64, 0.0]); }
for i in 0..10 { x.push(vec![i as f64 + 100.0, 0.0]); }

let result = kmeans::kmeans(&x, 2, 100, 1e-6).unwrap();
println!("Labels: {:?}", result.labels);
// [0, 0, 0, ..., 1, 1, 1, ...]
println!("Inertia: {:.2}", result.inertia);
```

### DBSCAN

Density-based spatial clustering that finds arbitrary-shaped clusters and noise.

```
DBSCAN:

  · · · · · · · · ·    · = noise (label -1)
  · ○ ○ ○ · · · · ·    ○ = cluster 0
  · ○ ○ ○ · · · · ·    ● = cluster 1
  · · · · · · · · ·
  · · · · ● ● ● · ·    eps = radius
  · · · ● ● ● ● · ·    min_pts = 3
  · · · ● ● ● · · ·
  · · · · · · · · ·
```

```rust
use mathverse_machine_learning::dbscan;

let mut x: Vec<Vec<f64>> = Vec::new();
for i in 0..10 { x.push(vec![i as f64, 0.0]); }
for i in 0..10 { x.push(vec![i as f64 + 100.0, 0.0]); }
x.push(vec![500.0, 500.0]); // noise

let result = dbscan::dbscan(&x, 1.0, 3);
println!("Clusters: {}", result.n_clusters); // 2
// labels[20] == -1 (noise)
```

### Gaussian Mixture Model

Soft clustering via Expectation-Maximization.

```rust
use mathverse_machine_learning::gmm;

let mut x = Vec::new();
for _ in 0..20 { x.push(vec![randn(), 0.0]); }
for _ in 0..20 { x.push(vec![randn() + 10.0, 0.0]); }

let result = gmm::fit_gmm(&x, 2, 50, 1e-6).unwrap();
println!("Log-likelihood: {:.2}", result.log_likelihood);
let labels = gmm::predict(&result, &x);
```

### Model Selection

```rust
use mathverse_machine_learning::model_selection;

let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
let y: Vec<f64> = (0..100).map(|i| i as f64).collect();

// Train/test split
let (x_train, x_test, y_train, y_test) =
    model_selection::train_test_split(&x, &y, 0.2, 42);

// K-fold cross-validation
let accs = model_selection::k_fold_cv(&x, &y, 5, 42, |xtr, ytr, xte| {
    xte.iter().map(|row| if row[0] < 50.0 { 0.0 } else { 1.0 }).collect()
});
let mean_acc: f64 = accs.iter().sum::<f64>() / 5.0;

// ROC / AUC
let scores = vec![0.1, 0.4, 0.35, 0.8, 0.9];
let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
let points = model_selection::roc_curve(&scores, &labels);
let auc = model_selection::auc(&points);
```

### Feature Preprocessing

```rust
use mathverse_machine_learning::feature;

// Standardize (zero mean, unit variance)
let mut x = vec![vec![1.0, 200.0], vec![3.0, 400.0], vec![5.0, 600.0]];
let (means, stds) = feature::standardize(&mut x);
// x now: [[-1.22, -1.22], [0.0, 0.0], [1.22, 1.22]]

// Min-Max normalization [0, 1]
let mut x = vec![vec![0.0, 10.0], vec![5.0, 20.0], vec![10.0, 30.0]];
feature::min_max(&mut x);
// x[0] = [0.0, 0.0], x[2] = [1.0, 1.0]

// One-hot encoding
let labels = vec![0.0, 1.0, 2.0];
let encoded = feature::one_hot(&labels, 3);
// [[1,0,0], [0,1,0], [0,0,1]]

// Polynomial features
let x = vec![vec![2.0, 3.0]];
let pf = feature::polynomial_features(&x, 2);
// [2, 3, 4, 6, 9] → x₁, x₂, x₁², x₁x₂, x₂²
```

## Future Scope

- [ ] Support Vector Machines (SVM)
- [ ] PCA / dimensionality reduction
- [ ] Feature selection methods
- [ ] XGBoost / LightGBM-style boosting
- [ ] Neural network MLP
- [ ] Parallel tree training via rayon
- [ ] Serde serialization for fitted models

## License

MIT OR Apache-2.0
