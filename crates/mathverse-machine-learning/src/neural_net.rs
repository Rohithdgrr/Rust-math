/// A single layer in a neural network.
#[derive(Debug, Clone)]
pub enum Layer {
    /// Fully connected layer with weights and bias.
    Linear {
        /// Weight matrix (rows = output neurons, cols = input features).
        weights: Vec<Vec<f64>>,
        /// Bias vector.
        bias: Vec<f64>,
    },
    /// Rectified linear unit activation.
    ReLU,
    /// Sigmoid activation.
    Sigmoid,
    /// Softmax activation (normalizes to probabilities).
    Softmax,
}

/// A simple feedforward neural network.
#[derive(Debug)]
pub struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    /// Create a new neural network with the given layer sequence.
    #[must_use]
    #[inline]
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Compute the forward pass through all layers.
    #[must_use]
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut activations = x.to_vec();
        for layer in &self.layers {
            activations = match layer {
                Layer::Linear { weights, bias } => {
                    let n_out = weights.len();
                    let mut out = Vec::with_capacity(activations.len());
                    for row in &activations {
                        let mut result = vec![0.0; n_out];
                        for j in 0..n_out {
                            let mut sum = bias.get(j).copied().unwrap_or(0.0);
                            for (k, &val) in row.iter().enumerate() {
                                if k < weights[j].len() {
                                    sum += val * weights[j][k];
                                }
                            }
                            result[j] = sum;
                        }
                        out.push(result);
                    }
                    out
                }
                Layer::ReLU => activations
                    .iter()
                    .map(|row| row.iter().map(|&v| v.max(0.0)).collect())
                    .collect(),
                Layer::Sigmoid => activations
                    .iter()
                    .map(|row| row.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
                    .collect(),
                Layer::Softmax => activations
                    .iter()
                    .map(|row| {
                        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
                        let sum: f64 = exps.iter().sum();
                        exps.iter().map(|&e| e / sum).collect()
                    })
                    .collect(),
            };
        }
        activations
    }

    /// Train the network with gradient descent for the given number of epochs.
    /// This implements proper backpropagation with the chain rule.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
        for _ in 0..epochs {
            for (sample_idx, xi) in x.iter().enumerate() {
                // Forward pass with cache
                let mut activations = vec![xi.clone()];
                
                for layer in &self.layers {
                    let current = activations.last().unwrap();
                    match layer {
                        Layer::Linear { weights, bias } => {
                            let n_out = weights.len();
                            let mut z = vec![0.0; n_out];
                            for j in 0..n_out {
                                let mut sum = bias.get(j).copied().unwrap_or(0.0);
                                for (k, &val) in current.iter().enumerate() {
                                    if k < weights[j].len() {
                                        sum += val * weights[j][k];
                                    }
                                }
                            z[j] = sum;
                        }
                        activations.push(z);
                        }
                        Layer::ReLU => {
                            let activated: Vec<f64> = current.iter().map(|&v| v.max(0.0)).collect();
                            activations.push(activated);
                        }
                        Layer::Sigmoid => {
                            let activated: Vec<f64> = current.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                            activations.push(activated);
                        }
                        Layer::Softmax => {
                            let max_val = current.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let exps: Vec<f64> = current.iter().map(|&v| (v - max_val).exp()).collect();
                            let sum: f64 = exps.iter().sum();
                            let activated: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
                            activations.push(activated);
                        }
                    }
                }
                
                // Backward pass
                let output = activations.last().unwrap();
                let target = y[sample_idx];
                
                // Output error (MSE loss derivative: output - target)
                let mut delta: Vec<f64> = output.iter().map(|&o| o - target).collect();
                
                // Backpropagate through layers in reverse
                for layer_idx in (0..self.layers.len()).rev() {
                    let prev_activation = if layer_idx > 0 {
                        activations[layer_idx].clone()
                    } else {
                        xi.clone()
                    };
                    
                    // First: compute new delta using immutable borrow
                    let new_delta = match &self.layers[layer_idx] {
                        Layer::Linear { weights, .. } => {
                            if layer_idx > 0 {
                                let mut nd = vec![0.0; prev_activation.len()];
                                for k in 0..prev_activation.len() {
                                    let mut sum = 0.0;
                                    for (j, d) in delta.iter().enumerate() {
                                        if j < weights.len() && k < weights[j].len() {
                                            sum += d * weights[j][k];
                                        }
                                    }
                                    nd[k] = sum;
                                }
                                Some(nd)
                            } else {
                                None
                            }
                        }
                        Layer::ReLU => {
                            let z = &activations[layer_idx];
                            let nd: Vec<f64> = delta.iter().zip(z.iter()).map(|(d, &z)| {
                                if z > 0.0 { *d } else { 0.0 }
                            }).collect();
                            Some(nd)
                        }
                        Layer::Sigmoid => {
                            let activation = &activations[layer_idx + 1];
                            let nd: Vec<f64> = delta.iter().zip(activation.iter()).map(|(d, &a)| {
                                d * a * (1.0 - a)
                            }).collect();
                            Some(nd)
                        }
                        Layer::Softmax => {
                            let activation = &activations[layer_idx + 1];
                            let nd: Vec<f64> = delta.iter().zip(activation.iter()).map(|(d, &a)| {
                                d * a * (1.0 - a)
                            }).collect();
                            Some(nd)
                        }
                    };
                    
                    // Then: apply weight updates using mutable borrow
                    if let Layer::Linear { weights, bias } = &mut self.layers[layer_idx] {
                        for (j, w_row) in weights.iter_mut().enumerate() {
                            for (k, w) in w_row.iter_mut().enumerate() {
                                if k < prev_activation.len() && j < delta.len() {
                                    *w -= lr * delta[j] * prev_activation[k];
                                }
                            }
                        }
                        for (j, b) in bias.iter_mut().enumerate() {
                            if j < delta.len() {
                                *b -= lr * delta[j];
                            }
                        }
                    }
                    
                    // Apply new delta
                    if let Some(nd) = new_delta {
                        delta = nd;
                    }
                }
            }
        }
    }

    /// Predict output values for the given inputs.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let out = self.forward(x);
        // For classification, return argmax; for regression, return the single output
        out.iter()
            .map(|row| {
                if row.len() > 1 {
                    // Classification: return argmax
                    row.iter()
                        .enumerate()
                        .filter(|(_, v)| !v.is_nan())
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as f64)
                        .unwrap_or(0.0)
                } else {
                    // Regression: return the single output
                    row[0]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_linear() {
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![0.1, 0.2];
        let net = NeuralNet::new(vec![Layer::Linear { weights, bias }]);
        let out = net.forward(&[vec![1.0, 2.0]]);
        assert!((out[0][0] - 1.1).abs() < 1e-10);
        assert!((out[0][1] - 2.2).abs() < 1e-10);
    }

    #[test]
    fn forward_relu() {
        let net = NeuralNet::new(vec![Layer::ReLU]);
        let out = net.forward(&[vec![-1.0, 2.0, -0.5]]);
        assert_eq!(out[0], vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn forward_sigmoid() {
        let net = NeuralNet::new(vec![Layer::Sigmoid]);
        let out = net.forward(&[vec![0.0]]);
        assert!((out[0][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn forward_softmax() {
        let net = NeuralNet::new(vec![Layer::Softmax]);
        let out = net.forward(&[vec![1.0, 2.0, 3.0]]);
        let sum: f64 = out[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(out[0][2] > out[0][1]);
        assert!(out[0][1] > out[0][0]);
    }

    #[test]
    fn fit_xor() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0];
        // 2→2→1 with sigmoid. Asymmetric init breaks symmetry.
        let mut net = NeuralNet::new(vec![
            Layer::Linear {
                weights: vec![vec![2.0, 2.0], vec![-2.0, -2.0]],
                bias: vec![-1.0, 3.0],
            },
            Layer::Sigmoid,
            Layer::Linear {
                weights: vec![vec![2.0, 2.0]],
                bias: vec![-3.0],
            },
            Layer::Sigmoid,
        ]);
        net.fit(&x, &y, 0.5, 1000);
        let preds = net.predict(&x);
        assert!(preds[0] < 0.1, "preds[0]={}", preds[0]);
        assert!(preds[1] > 0.9, "preds[1]={}", preds[1]);
        assert!(preds[2] > 0.9, "preds[2]={}", preds[2]);
        assert!(preds[3] < 0.1, "preds[3]={}", preds[3]);
    }

    #[test]
    fn fit_simple_classification() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut net = NeuralNet::new(vec![
            Layer::Linear {
                weights: vec![vec![0.5]],
                bias: vec![0.0],
            },
            Layer::Sigmoid,
        ]);
        net.fit(&x, &y, 0.5, 1000);
        let preds = net.predict(&x);
        assert!(preds[0] < preds[2]);
        assert!(preds[1] < preds[3]);
    }

    #[test]
    fn test_backpropagation_improvement() {
        // Test that backpropagation actually improves the loss
        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![0.0, 1.0];
        let mut net = NeuralNet::new(vec![
            Layer::Linear {
                weights: vec![vec![0.5]],
                bias: vec![0.0],
            },
            Layer::Sigmoid,
        ]);
        
        // Get initial predictions
        let initial_preds = net.predict(&x);
        let initial_loss: f64 = initial_preds.iter().zip(&y).map(|(p, t)| (p - t).powi(2)).sum();
        
        // Train for a few epochs
        net.fit(&x, &y, 0.5, 100);
        
        // Get final predictions
        let final_preds = net.predict(&x);
        let final_loss: f64 = final_preds.iter().zip(&y).map(|(p, t)| (p - t).powi(2)).sum();
        
        // Loss should decrease
        assert!(final_loss < initial_loss, "Loss should decrease after training");
    }

    #[test]
    fn test_predict_argmax() {
        // Test that predict returns argmax for multi-class
        let x = vec![vec![1.0, 2.0]];
        let net = NeuralNet::new(vec![
            Layer::Linear {
                weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                bias: vec![0.0, 0.0],
            },
            Layer::Softmax,
        ]);
        
        let preds = net.predict(&x);
        // Should return the index of the maximum value (1.0 in this case since 2.0 > 1.0)
        assert_eq!(preds[0], 1.0);
    }
}
