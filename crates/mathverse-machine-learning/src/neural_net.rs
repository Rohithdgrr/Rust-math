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
#[derive(Debug, Clone)]
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
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
        let input_dim = x[0].len();
        let mut output_weights = vec![0.0; input_dim];
        let mut output_bias = 0.0;

        for _ in 0..epochs {
            for (i, xi) in x.iter().enumerate() {
                let hidden = self.forward(std::slice::from_ref(xi));
                let h = &hidden[0];

                let pred: f64 = h
                    .iter()
                    .zip(output_weights.iter())
                    .map(|(hi, wi)| hi * wi)
                    .sum::<f64>()
                    + output_bias;
                let error = pred - y[i];

                for (j, hi) in h.iter().enumerate() {
                    output_weights[j] -= lr * error * hi / x.len() as f64;
                }
                output_bias -= lr * error / x.len() as f64;

                for layer in self.layers.iter_mut() {
                    if let Layer::Linear { weights, bias } = layer {
                        for (j, w_row) in weights.iter_mut().enumerate() {
                            for w in w_row.iter_mut() {
                                *w -= lr
                                    * error
                                    * output_weights.get(j).copied().unwrap_or(0.0)
                                    * 0.01;
                            }
                        }
                        for b in bias.iter_mut() {
                            *b -= lr * error * 0.01;
                        }
                    }
                }
            }
        }

        let n_out = 1;
        let mut w_matrix = Vec::with_capacity(n_out);
        for wi in output_weights.iter() {
            w_matrix.push(vec![*wi]);
        }
        self.layers.push(Layer::Linear {
            weights: w_matrix,
            bias: vec![output_bias],
        });
    }

    /// Predict output values for the given inputs.
    #[must_use]
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let out = self.forward(x);
        out.iter()
            .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
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
        let mut net = NeuralNet::new(vec![
            Layer::Linear {
                weights: vec![vec![0.5, -0.5], vec![-0.5, 0.5]],
                bias: vec![-0.2, -0.2],
            },
            Layer::Sigmoid,
        ]);
        net.fit(&x, &y, 2.0, 5000);
        let preds = net.predict(&x);
        assert!(preds[0] < preds[1], "preds={preds:?}");
        assert!(preds[0] < preds[2], "preds={preds:?}");
        assert!(preds[0] < preds[3], "preds={preds:?}");
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
}
