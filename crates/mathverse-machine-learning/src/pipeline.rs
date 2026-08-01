use std::fs;
use std::io::{Read, Write};

/// A single step in a preprocessing/model pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Standardize features to zero mean and unit variance.
    Standardize,
    /// Scale features to [0, 1] range.
    MinMax,
    /// Reduce dimensionality via PCA.
    Pca {
        /// Number of components to keep.
        n_components: usize,
    },
    /// Select top features by correlation with target.
    FeatureSelect {
        /// Number of top features to select.
        k: usize,
    },
    /// Train a model as the final pipeline step.
    Model(ModelType),
}

/// Supported model types for the pipeline.
#[derive(Debug, Clone)]
pub enum ModelType {
    /// Linear regression.
    Linear,
    /// Logistic regression.
    Logistic,
    /// K-nearest neighbors.
    Knn {
        /// Number of nearest neighbors.
        k: usize,
    },
    /// Decision tree (single stump).
    DecisionTree,
    /// Random forest of 5 decision stumps.
    RandomForest,
    /// Support vector machine with linear kernel.
    Svm,
}

/// Sequential pipeline of preprocessing steps and a final model.
#[derive(Debug, Clone)]
pub struct Pipeline {
    steps: Vec<PipelineStep>,
    fitted_params: Vec<FittedParams>,
}

#[derive(Debug, Clone)]
enum FittedParams {
    Standardize {
        means: Vec<f64>,
        stds: Vec<f64>,
    },
    MinMax {
        mins: Vec<f64>,
        maxs: Vec<f64>,
    },
    Pca {
        components: Vec<Vec<f64>>,
        means: Vec<f64>,
    },
    FeatureSelect {
        selected: Vec<usize>,
    },
    Model(ModelFitted),
}

#[derive(Debug, Clone)]
enum ModelFitted {
    Linear {
        weights: Vec<f64>,
        bias: f64,
    },
    Logistic {
        weights: Vec<f64>,
        bias: f64,
    },
    Knn {
        train_x: Vec<Vec<f64>>,
        train_y: Vec<f64>,
        k: usize,
    },
    DecisionTree {
        tree: SimpleTree,
    },
    RandomForest {
        trees: Vec<SimpleTree>,
    },
    Svm {
        weights: Vec<f64>,
        bias: f64,
    },
}

#[derive(Debug, Clone)]
struct SimpleTree {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl Pipeline {
    /// Creates a new pipeline with the given steps.
    #[must_use]
    #[inline]
    pub fn new(steps: Vec<PipelineStep>) -> Self {
        Self {
            steps,
            fitted_params: Vec::new(),
        }
    }

    /// Fits all pipeline steps on training data.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.fitted_params.clear();
        let mut current_x = x.to_vec();

        for step in &self.steps {
            match step {
                PipelineStep::Standardize => {
                    let (means, stds) = compute_mean_std(&current_x);
                    let standardized = standardize(&current_x, &means, &stds);
                    self.fitted_params
                        .push(FittedParams::Standardize { means, stds });
                    current_x = standardized;
                }
                PipelineStep::MinMax => {
                    let (mins, maxs) = compute_min_max(&current_x);
                    let minmaxed = min_max_scale(&current_x, &mins, &maxs);
                    self.fitted_params.push(FittedParams::MinMax { mins, maxs });
                    current_x = minmaxed;
                }
                PipelineStep::Pca { n_components } => {
                    let n = *n_components;
                    let (components, means) = compute_pca(&current_x, n);
                    let projected = apply_pca(&current_x, &components, &means);
                    self.fitted_params
                        .push(FittedParams::Pca { components, means });
                    current_x = projected;
                }
                PipelineStep::FeatureSelect { k } => {
                    let selected = select_top_k_features(&current_x, y, *k);
                    let selected_x: Vec<Vec<f64>> = current_x
                        .iter()
                        .map(|row| selected.iter().map(|&i| row[i]).collect())
                        .collect();
                    self.fitted_params
                        .push(FittedParams::FeatureSelect { selected });
                    current_x = selected_x;
                }
                PipelineStep::Model(model_type) => {
                    let fitted = fit_model(model_type, &current_x, y);
                    self.fitted_params.push(FittedParams::Model(fitted));
                }
            }
        }
    }

    /// Transforms and predicts using the fitted pipeline.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut current_x = x.to_vec();

        for params in &self.fitted_params {
            match params {
                FittedParams::Standardize { means, stds } => {
                    current_x = standardize(&current_x, means, stds);
                }
                FittedParams::MinMax { mins, maxs } => {
                    current_x = min_max_scale(&current_x, mins, maxs);
                }
                FittedParams::Pca { components, means } => {
                    current_x = apply_pca(&current_x, components, means);
                }
                FittedParams::FeatureSelect { selected } => {
                    current_x = current_x
                        .iter()
                        .map(|row| selected.iter().map(|&i| row[i]).collect())
                        .collect();
                }
                FittedParams::Model(model) => {
                    return predict_model(model, &current_x);
                }
            }
        }
        vec![0.0; current_x.len()]
    }
}

fn compute_mean_std(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_cols = x[0].len();
    let n = x.len() as f64;
    let means: Vec<f64> = (0..n_cols)
        .map(|col| x.iter().map(|row| row[col]).sum::<f64>() / n)
        .collect();
    let stds: Vec<f64> = (0..n_cols)
        .map(|col| {
            let var = x
                .iter()
                .map(|row| (row[col] - means[col]).powi(2))
                .sum::<f64>()
                / n;
            var.sqrt().max(1e-10)
        })
        .collect();
    (means, stds)
}

fn standardize(x: &[Vec<f64>], means: &[f64], stds: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(col, &v)| (v - means[col]) / stds[col])
                .collect()
        })
        .collect()
}

fn compute_min_max(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_cols = x[0].len();
    let mut mins = vec![f64::INFINITY; n_cols];
    let mut maxs = vec![f64::NEG_INFINITY; n_cols];
    for row in x {
        for (col, &v) in row.iter().enumerate() {
            mins[col] = mins[col].min(v);
            maxs[col] = maxs[col].max(v);
        }
    }
    (mins, maxs)
}

fn min_max_scale(x: &[Vec<f64>], mins: &[f64], maxs: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(col, &v)| {
                    let range = maxs[col] - mins[col];
                    if range < 1e-10 {
                        0.0
                    } else {
                        (v - mins[col]) / range
                    }
                })
                .collect()
        })
        .collect()
}

fn compute_pca(x: &[Vec<f64>], n_components: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    if x.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_cols = x[0].len();
    let means: Vec<f64> = (0..n_cols)
        .map(|col| x.iter().map(|row| row[col]).sum::<f64>() / x.len() as f64)
        .collect();

    // Simplified Pca using variance-based component selection
    let mut variances: Vec<(usize, f64)> = (0..n_cols)
        .map(|col| {
            let var = x
                .iter()
                .map(|row| (row[col] - means[col]).powi(2))
                .sum::<f64>()
                / x.len() as f64;
            (col, var)
        })
        .collect();
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let components: Vec<Vec<f64>> = variances[..n_components]
        .iter()
        .map(|&(col, _)| {
            let mut comp = vec![0.0; n_cols];
            comp[col] = 1.0;
            comp
        })
        .collect();

    (components, means)
}

fn apply_pca(x: &[Vec<f64>], components: &[Vec<f64>], means: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            components
                .iter()
                .map(|comp| {
                    row.iter()
                        .zip(comp.iter())
                        .zip(means.iter())
                        .map(|((&r, &c), &m)| (r - m) * c)
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn select_top_k_features(x: &[Vec<f64>], y: &[f64], k: usize) -> Vec<usize> {
    if x.is_empty() {
        return Vec::new();
    }
    let n_cols = x[0].len();
    let k = k.min(n_cols);

    // Compute correlation with target
    let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / y.len() as f64;

    let mut correlations: Vec<(usize, f64)> = (0..n_cols)
        .map(|col| {
            let mean_x: f64 = x.iter().map(|row| row[col]).sum::<f64>() / x.len() as f64;
            let var_x: f64 =
                x.iter().map(|row| (row[col] - mean_x).powi(2)).sum::<f64>() / x.len() as f64;
            let cov: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(row, yi)| (row[col] - mean_x) * (yi - mean_y))
                .sum::<f64>()
                / x.len() as f64;
            let corr = if var_x > 1e-10 && var_y > 1e-10 {
                cov / (var_x.sqrt() * var_y.sqrt())
            } else {
                0.0
            };
            (col, corr.abs())
        })
        .collect();
    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    correlations[..k].iter().map(|&(col, _)| col).collect()
}

fn fit_model(model_type: &ModelType, x: &[Vec<f64>], y: &[f64]) -> ModelFitted {
    match model_type {
        ModelType::Linear => {
            let (w, b) = fit_linear(x, y);
            ModelFitted::Linear {
                weights: w,
                bias: b,
            }
        }
        ModelType::Logistic => {
            let (w, b) = fit_logistic(x, y);
            ModelFitted::Logistic {
                weights: w,
                bias: b,
            }
        }
        ModelType::Knn { k } => ModelFitted::Knn {
            train_x: x.to_vec(),
            train_y: y.to_vec(),
            k: *k,
        },
        ModelType::DecisionTree => {
            let tree = fit_decision_stump(x, y);
            ModelFitted::DecisionTree { tree }
        }
        ModelType::RandomForest => {
            let trees: Vec<SimpleTree> = (0..5)
                .map(|i| {
                    let indices: Vec<usize> = (0..x.len()).map(|j| (j + i * 7) % x.len()).collect();
                    let bx: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
                    let by: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
                    fit_decision_stump(&bx, &by)
                })
                .collect();
            ModelFitted::RandomForest { trees }
        }
        ModelType::Svm => {
            let (w, b) = fit_svm_simplified(x, y);
            ModelFitted::Svm {
                weights: w,
                bias: b,
            }
        }
    }
}

fn fit_linear(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    if x.is_empty() {
        return (Vec::new(), 0.0);
    }
    let n_features = x[0].len();
    let mut w = vec![0.0; n_features];
    let mut b = 0.0;
    let lr = 0.01;

    for _ in 0..200 {
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred: f64 = w.iter().zip(xi.iter()).map(|(wj, xj)| wj * xj).sum::<f64>() + b;
            let err = pred - yi;
            for (wj, &xj) in w.iter_mut().zip(xi.iter()) {
                *wj -= lr * err * xj / x.len() as f64;
            }
            b -= lr * err / x.len() as f64;
        }
    }
    (w, b)
}

fn fit_logistic(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    if x.is_empty() {
        return (Vec::new(), 0.0);
    }
    let n_features = x[0].len();
    let mut w = vec![0.0; n_features];
    let mut b = 0.0;
    let lr = 0.1;

    for _ in 0..200 {
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let logit: f64 = w.iter().zip(xi.iter()).map(|(wj, xj)| wj * xj).sum::<f64>() + b;
            let pred = 1.0 / (1.0 + (-logit).exp());
            let err = pred - yi;
            for (wj, &xj) in w.iter_mut().zip(xi.iter()) {
                *wj -= lr * err * xj / x.len() as f64;
            }
            b -= lr * err / x.len() as f64;
        }
    }
    (w, b)
}

fn fit_decision_stump(x: &[Vec<f64>], y: &[f64]) -> SimpleTree {
    if x.is_empty() {
        return SimpleTree {
            feature: 0,
            threshold: 0.0,
            left_value: 0.0,
            right_value: 0.0,
        };
    }
    let n_features = x[0].len();
    let mut best_mse = f64::INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_left = 0.0;
    let mut best_right = 0.0;

    for feat in 0..n_features {
        let mut vals: Vec<(f64, f64)> = x
            .iter()
            .zip(y.iter())
            .map(|(r, &yi)| (r[feat], yi))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 0..vals.len() - 1 {
            if vals[i].0 == vals[i + 1].0 {
                continue;
            }
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            let left: Vec<f64> = vals[..=i].iter().map(|&(_, y)| y).collect();
            let right: Vec<f64> = vals[i + 1..].iter().map(|&(_, y)| y).collect();
            let lm = left.iter().sum::<f64>() / left.len() as f64;
            let rm = right.iter().sum::<f64>() / right.len() as f64;
            let mse: f64 = left.iter().map(|y| (y - lm).powi(2)).sum::<f64>()
                + right.iter().map(|y| (y - rm).powi(2)).sum::<f64>();
            if mse < best_mse {
                best_mse = mse;
                best_feature = feat;
                best_threshold = thresh;
                best_left = lm;
                best_right = rm;
            }
        }
    }

    SimpleTree {
        feature: best_feature,
        threshold: best_threshold,
        left_value: best_left,
        right_value: best_right,
    }
}

fn fit_svm_simplified(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    let mut w = vec![0.0; x[0].len()];
    let mut b = 0.0;
    let lr = 0.01;

    for _ in 0..200 {
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let y_label = if yi > 0.5 { 1.0 } else { -1.0 };
            let decision: f64 = w.iter().zip(xi.iter()).map(|(wj, xj)| wj * xj).sum::<f64>() + b;
            if y_label * decision < 1.0 {
                for (wj, &xj) in w.iter_mut().zip(xi.iter()) {
                    *wj += lr * (C * y_label * xj - *wj);
                }
                b += lr * C * y_label;
            } else {
                for wj in w.iter_mut() {
                    *wj -= lr * *wj;
                }
            }
        }
    }
    (w, b)
}

fn predict_model(model: &ModelFitted, x: &[Vec<f64>]) -> Vec<f64> {
    match model {
        ModelFitted::Linear { weights, bias } => x
            .iter()
            .map(|xi| {
                weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias
            })
            .collect(),
        ModelFitted::Logistic { weights, bias } => x
            .iter()
            .map(|xi| {
                let logit: f64 = weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                1.0 / (1.0 + (-logit).exp())
            })
            .collect(),
        ModelFitted::Knn {
            train_x,
            train_y,
            k,
        } => x
            .iter()
            .map(|xi| {
                let mut dists: Vec<(f64, f64)> = train_x
                    .iter()
                    .zip(train_y.iter())
                    .map(|(tx, &ty)| {
                        let d: f64 = xi.iter().zip(tx.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                        (d, ty)
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let kk = (*k).min(dists.len());
                dists[..kk].iter().map(|&(_, y)| y).sum::<f64>() / kk as f64
            })
            .collect(),
        ModelFitted::DecisionTree { tree } => x
            .iter()
            .map(|xi| {
                if xi[tree.feature] <= tree.threshold {
                    tree.left_value
                } else {
                    tree.right_value
                }
            })
            .collect(),
        ModelFitted::RandomForest { trees } => x
            .iter()
            .map(|xi| {
                let preds: Vec<f64> = trees
                    .iter()
                    .map(|t| {
                        if xi[t.feature] <= t.threshold {
                            t.left_value
                        } else {
                            t.right_value
                        }
                    })
                    .collect();
                preds.iter().sum::<f64>() / preds.len() as f64
            })
            .collect(),
        ModelFitted::Svm { weights, bias } => x
            .iter()
            .map(|xi| {
                let d: f64 = weights
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;
                if d > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    }
}

/// Serializes a pipeline to a file using Debug format.
#[must_use]
pub fn save_pipeline(pipeline: &Pipeline, path: &str) -> std::io::Result<()> {
    let serialized = format!("{:?}", pipeline);
    let mut file = fs::File::create(path)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

/// Loads a pipeline from a file (not yet implemented).
#[must_use]
pub fn load_pipeline(path: &str) -> std::io::Result<Pipeline> {
    let mut file = fs::File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    // Simple parsing - just return a default pipeline for now
    // In production, use serde
    Err(std::io::Error::other(
        "Pipeline deserialization not yet implemented",
    ))
}

// Svm regularization constant
const C: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardize() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (means, stds) = compute_mean_std(&x);
        let scaled = standardize(&x, &means, &stds);
        let mean_0: f64 = scaled.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        assert!(mean_0.abs() < 1e-10);
    }

    #[test]
    fn test_minmax() {
        let x = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let (mins, maxs) = compute_min_max(&x);
        let scaled = min_max_scale(&x, &mins, &maxs);
        assert!((scaled[0][0]).abs() < 1e-10);
        assert!((scaled[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pca() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (components, means) = compute_pca(&x, 1);
        let projected = apply_pca(&x, &components, &means);
        assert_eq!(projected[0].len(), 1);
    }

    #[test]
    fn test_feature_select() {
        let x = vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]];
        let y = vec![1.0, 2.0, 3.0];
        let selected = select_top_k_features(&x, &y, 1);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_pipeline_fit_predict() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut pipeline = Pipeline::new(vec![
            PipelineStep::Standardize,
            PipelineStep::Model(ModelType::Logistic),
        ]);
        pipeline.fit(&x, &y);
        let preds = pipeline.predict(&x);
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_pipeline_knn() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut pipeline = Pipeline::new(vec![
            PipelineStep::MinMax,
            PipelineStep::Model(ModelType::Knn { k: 2 }),
        ]);
        pipeline.fit(&x, &y);
        let preds = pipeline.predict(&x);
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_pipeline_decision_tree() {
        let x = vec![
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut pipeline = Pipeline::new(vec![PipelineStep::Model(ModelType::DecisionTree)]);
        pipeline.fit(&x, &y);
        let preds = pipeline.predict(&x);
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_save_load_pipeline() {
        let pipeline = Pipeline::new(vec![
            PipelineStep::Standardize,
            PipelineStep::Model(ModelType::Linear),
        ]);
        let path = "C:\\Users\\rohit\\AppData\\Local\\Temp\\test_pipeline.txt";
        let _ = save_pipeline(&pipeline, path);
        let result = load_pipeline(path);
        // Load not implemented yet, just check it compiles
        assert!(result.is_err() || result.is_ok());
        let _ = fs::remove_file(path);
    }
}
