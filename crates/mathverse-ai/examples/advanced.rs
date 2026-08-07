use mathverse_ai::{Tensor, softmax, mse};

fn main() {
    let logits = Tensor::new(&[1, 3], &[2.0, 1.0, 0.5]).unwrap();
    let probs = softmax(&logits, 1).unwrap();
    println!("Softmax: {:?}", probs.data);
    let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
    println!("MSE: {:.4}", mse(&pred, &target).unwrap());
}
