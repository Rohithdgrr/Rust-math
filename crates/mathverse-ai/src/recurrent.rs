//! Recurrent layers: vanilla RNN, LSTM, and GRU forward passes.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

/// Vanilla RNN forward pass.
/// `x`: [batch, seq_len, input_size]
/// `w_ih`: [hidden_size, input_size]
/// `w_hh`: [hidden_size, hidden_size]
/// `h0`: [batch, hidden_size] optional (defaults to zeros)
/// Returns (output [batch, seq_len, hidden_size], final_hidden [batch, hidden_size]).
pub fn rnn_forward(
    x: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    h0: Option<&Tensor>,
    activation: fn(f64) -> f64,
) -> MathResult<(Tensor, Tensor)> {
    if x.shape.len() != 3 {
        return Err(MathError::InvalidArgument("x must be [batch, seq_len, input_size]"));
    }
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let input_size = x.shape[2];
    let hidden_size = w_ih.shape[0];
    if w_ih.shape[1] != input_size {
        return Err(MathError::DimensionMismatch);
    }
    if w_hh.shape != [hidden_size, hidden_size] {
        return Err(MathError::DimensionMismatch);
    }

    let mut h = match h0 {
        Some(h0) => {
            if h0.shape != [batch, hidden_size] {
                return Err(MathError::DimensionMismatch);
            }
            h0.data.clone()
        }
        None => vec![0.0; batch * hidden_size],
    };

    let mut output = vec![0.0; batch * seq_len * hidden_size];

    for t in 0..seq_len {
        for b in 0..batch {
            let mut new_h = vec![0.0; hidden_size];
#[allow(clippy::needless_range_loop)]            for i in 0..hidden_size {
                let mut val = 0.0;
                // W_ih @ x[t]
                for j in 0..input_size {
                    val += w_ih.data[i * input_size + j] * x.data[b * seq_len * input_size + t * input_size + j];
                }
                // W_hh @ h[t-1]
                for j in 0..hidden_size {
                    val += w_hh.data[i * hidden_size + j] * h[b * hidden_size + j];
                }
                new_h[i] = activation(val);
            }
            h[b * hidden_size..(b + 1) * hidden_size].copy_from_slice(&new_h);
            output[b * seq_len * hidden_size + t * hidden_size..b * seq_len * hidden_size + (t + 1) * hidden_size]
                .copy_from_slice(&new_h);
        }
    }

    let output = Tensor::from_vec(&[batch, seq_len, hidden_size], output)?;
    let final_hidden = Tensor::from_vec(&[batch, hidden_size], h)?;
    Ok((output, final_hidden))
}

/// LSTM forward pass.
/// `x`: [batch, seq_len, input_size]
/// `w_ih`: [4*hidden_size, input_size] (i, f, g, o gates)
/// `w_hh`: [4*hidden_size, hidden_size]
/// `h0`, `c0`: [batch, hidden_size] optional
/// Returns (output [batch, seq_len, hidden_size], final_h, final_c).
pub fn lstm_forward(
    x: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    h0: Option<&Tensor>,
    c0: Option<&Tensor>,
) -> MathResult<(Tensor, Tensor, Tensor)> {
    if x.shape.len() != 3 {
        return Err(MathError::InvalidArgument("x must be [batch, seq_len, input_size]"));
    }
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let input_size = x.shape[2];
    let hidden_size = w_ih.shape[0] / 4;
    if w_ih.shape[1] != input_size {
        return Err(MathError::DimensionMismatch);
    }
    if w_hh.shape != [4 * hidden_size, hidden_size] {
        return Err(MathError::DimensionMismatch);
    }

    let mut h = match h0 {
        Some(h0) => {
            if h0.shape != [batch, hidden_size] { return Err(MathError::DimensionMismatch); }
            h0.data.clone()
        }
        None => vec![0.0; batch * hidden_size],
    };

    let mut c = match c0 {
        Some(c0) => {
            if c0.shape != [batch, hidden_size] { return Err(MathError::DimensionMismatch); }
            c0.data.clone()
        }
        None => vec![0.0; batch * hidden_size],
    };

    let mut output = vec![0.0; batch * seq_len * hidden_size];

    for t in 0..seq_len {
        for b in 0..batch {
            let mut gates = vec![0.0; 4 * hidden_size];
#[allow(clippy::needless_range_loop)]            for i in 0..4 * hidden_size {
                let mut val = 0.0;
                for j in 0..input_size {
                    val += w_ih.data[i * input_size + j] * x.data[b * seq_len * input_size + t * input_size + j];
                }
                for j in 0..hidden_size {
                    val += w_hh.data[i * hidden_size + j] * h[b * hidden_size + j];
                }
                gates[i] = val;
            }

            let mut new_c = vec![0.0; hidden_size];
            let mut new_h = vec![0.0; hidden_size];
#[allow(clippy::needless_range_loop)]            for i in 0..hidden_size {
                let input_gate = sigmoid_val(gates[i]);
                let forget_gate = sigmoid_val(gates[hidden_size + i]);
                let cell_gate = gates[2 * hidden_size + i].tanh();
                let output_gate = sigmoid_val(gates[3 * hidden_size + i]);

                new_c[i] = forget_gate * c[b * hidden_size + i] + input_gate * cell_gate;
                new_h[i] = output_gate * new_c[i].tanh();
            }

            h[b * hidden_size..(b + 1) * hidden_size].copy_from_slice(&new_h);
            c[b * hidden_size..(b + 1) * hidden_size].copy_from_slice(&new_c);
            output[b * seq_len * hidden_size + t * hidden_size..b * seq_len * hidden_size + (t + 1) * hidden_size]
                .copy_from_slice(&new_h);
        }
    }

    let output = Tensor::from_vec(&[batch, seq_len, hidden_size], output)?;
    let final_h = Tensor::from_vec(&[batch, hidden_size], h)?;
    let final_c = Tensor::from_vec(&[batch, hidden_size], c)?;
    Ok((output, final_h, final_c))
}

/// GRU forward pass.
/// `x`: [batch, seq_len, input_size]
/// `w_ih`: [3*hidden_size, input_size] (z, r, n gates)
/// `w_hh`: [3*hidden_size, hidden_size]
/// `h0`: [batch, hidden_size] optional
/// Returns (output [batch, seq_len, hidden_size], final_hidden).
pub fn gru_forward(
    x: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    h0: Option<&Tensor>,
) -> MathResult<(Tensor, Tensor)> {
    if x.shape.len() != 3 {
        return Err(MathError::InvalidArgument("x must be [batch, seq_len, input_size]"));
    }
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let input_size = x.shape[2];
    let hidden_size = w_ih.shape[0] / 3;
    if w_ih.shape[1] != input_size {
        return Err(MathError::DimensionMismatch);
    }
    if w_hh.shape != [3 * hidden_size, hidden_size] {
        return Err(MathError::DimensionMismatch);
    }

    let mut h = match h0 {
        Some(h0) => {
            if h0.shape != [batch, hidden_size] { return Err(MathError::DimensionMismatch); }
            h0.data.clone()
        }
        None => vec![0.0; batch * hidden_size],
    };

    let mut output = vec![0.0; batch * seq_len * hidden_size];

    for t in 0..seq_len {
        for b in 0..batch {
            let mut gates_ih = vec![0.0; 3 * hidden_size];
            let mut gates_hh = vec![0.0; 3 * hidden_size];
            for i in 0..3 * hidden_size {
                for j in 0..input_size {
                    gates_ih[i] += w_ih.data[i * input_size + j] * x.data[b * seq_len * input_size + t * input_size + j];
                }
                for j in 0..hidden_size {
                    gates_hh[i] += w_hh.data[i * hidden_size + j] * h[b * hidden_size + j];
                }
            }

            let mut new_h = vec![0.0; hidden_size];
#[allow(clippy::needless_range_loop)]            for i in 0..hidden_size {
                let z = sigmoid_val(gates_ih[i] + gates_hh[i]);
                let r = sigmoid_val(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
                let n = (gates_ih[2 * hidden_size + i] + r * gates_hh[2 * hidden_size + i]).tanh();
                new_h[i] = (1.0 - z) * n + z * h[b * hidden_size + i];
            }

            h[b * hidden_size..(b + 1) * hidden_size].copy_from_slice(&new_h);
            output[b * seq_len * hidden_size + t * hidden_size..b * seq_len * hidden_size + (t + 1) * hidden_size]
                .copy_from_slice(&new_h);
        }
    }

    let output = Tensor::from_vec(&[batch, seq_len, hidden_size], output)?;
    let final_hidden = Tensor::from_vec(&[batch, hidden_size], h)?;
    Ok((output, final_hidden))
}

fn sigmoid_val(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-5;

#[test]
    fn rnn_forward_test() {
        let x = Tensor::new(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let w_ih = Tensor::new(&[2, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let w_hh = Tensor::new(&[2, 2], &[0.5, 0.1, 0.1, 0.5]).unwrap();
        let (out, h) = rnn_forward(&x, &w_ih, &w_hh, None, f64::tanh).unwrap();
        assert_eq!(out.shape, vec![1, 2, 2]);
        assert_eq!(h.shape, vec![1, 2]);
        // Hidden state should be non-zero after processing
        assert!(h.data.iter().any(|&v| v.abs() > E));
    }

#[test]
    fn rnn_with_initial_hidden() {
        let x = Tensor::new(&[1, 1, 2], &[1.0, 1.0]).unwrap();
        let w_ih = Tensor::new(&[2, 2], &[0.1, 0.2, 0.3, 0.4]).unwrap();
        let w_hh = Tensor::new(&[2, 2], &[0.5, 0.0, 0.0, 0.5]).unwrap();
        let h0 = Tensor::new(&[1, 2], &[0.5, 0.5]).unwrap();
        let (_, h) = rnn_forward(&x, &w_ih, &w_hh, Some(&h0), f64::tanh).unwrap();
        // h should differ from h0
        assert!((h.data[0] - 0.5).abs() > E || (h.data[1] - 0.5).abs() > E);
    }

    #[test]
    fn lstm_forward_test() {
        let x = Tensor::new(&[1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let w_ih = Tensor::randn(&[8, 4]);
        let w_hh = Tensor::randn(&[8, 2]);
        let (out, h, c) = lstm_forward(&x, &w_ih, &w_hh, None, None).unwrap();
        assert_eq!(out.shape, vec![1, 2, 2]);
        assert_eq!(h.shape, vec![1, 2]);
        assert_eq!(c.shape, vec![1, 2]);
        // LSTM cell state should be non-zero
        assert!(c.data.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn gru_forward_test() {
        let x = Tensor::new(&[1, 3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let w_ih = Tensor::randn(&[6, 2]);
        let w_hh = Tensor::randn(&[6, 2]);
        let (out, h) = gru_forward(&x, &w_ih, &w_hh, None).unwrap();
        assert_eq!(out.shape, vec![1, 3, 2]);
        assert_eq!(h.shape, vec![1, 2]);
        // Output at each timestep should be non-zero
        assert!(out.data.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn rnn_multibatch() {
        let x = Tensor::new(&[2, 1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let w_ih = Tensor::new(&[2, 2], &[0.1, 0.2, 0.3, 0.4]).unwrap();
        let w_hh = Tensor::new(&[2, 2], &[0.5, 0.1, 0.1, 0.5]).unwrap();
        let (out, h) = rnn_forward(&x, &w_ih, &w_hh, None, f64::tanh).unwrap();
        assert_eq!(out.shape, vec![2, 1, 2]);
        assert_eq!(h.shape, vec![2, 2]);
        // Different batches should produce different hidden states
        assert!((h.data[0] - h.data[2]).abs() > 1e-6 || (h.data[1] - h.data[3]).abs() > 1e-6);
    }
}








