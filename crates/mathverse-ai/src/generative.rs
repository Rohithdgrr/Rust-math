//! Generative model components: VAE, GAN, and diffusion utilities.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

/// VAE reparameterization trick: z = mean + std * eps, eps ~ N(0,1).
pub fn vae_reparameterize(mean: &Tensor, logvar: &Tensor) -> MathResult<Tensor> {
    if mean.shape != logvar.shape {
        return Err(MathError::DimensionMismatch);
    }
    let std = logvar.mul_scalar(0.5).exp();
    let eps = Tensor::randn(&mean.shape);
    mean.add(&std.mul(&eps)?)
}

/// VAE loss = reconstruction loss (MSE) + KL divergence.
/// KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
pub fn vae_loss(
    recon: &Tensor,
    original: &Tensor,
    mean: &Tensor,
    logvar: &Tensor,
) -> MathResult<f64> {
    if recon.shape != original.shape || mean.shape != logvar.shape {
        return Err(MathError::DimensionMismatch);
    }
    // Reconstruction: MSE
    let diff = recon.sub(original)?;
    let recon_loss = diff.data.iter().map(|x| x * x).sum::<f64>() / diff.numel() as f64;

    // KL divergence
    let kl: f64 = mean.data.iter().zip(&logvar.data)
        .map(|(m, lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
        .sum();

    Ok(recon_loss + kl)
}

/// GAN discriminator loss: -E[log(D(real))] - E[log(1 - D(fake))]
pub fn gan_discriminator_loss(real_pred: &Tensor, fake_pred: &Tensor) -> MathResult<f64> {
    let eps = 1e-12;
    let real_loss: f64 = real_pred.data.iter()
        .map(|&x| -(x.max(eps).ln()))
        .sum::<f64>() / real_pred.numel() as f64;
    let fake_loss: f64 = fake_pred.data.iter()
        .map(|&x| -((1.0 - x).max(eps).ln()))
        .sum::<f64>() / fake_pred.numel() as f64;
    Ok(real_loss + fake_loss)
}

/// GAN generator loss: wants D(fake) → 1.
/// Returns mean(-log(D(fake))), which the generator minimizes.
pub fn gan_generator_loss(fake_pred: &Tensor) -> MathResult<f64> {
    let eps = 1e-12;
    let loss: f64 = fake_pred.data.iter()
        .map(|&x| -(x.max(eps).ln()))
        .sum::<f64>() / fake_pred.numel() as f64;
    Ok(loss)
}

/// Diffusion forward process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
/// Returns (noisy_x, noise).
pub fn diffusion_forward_process(
    x0: &Tensor,
    t: usize,
    betas: &[f64],
) -> MathResult<(Tensor, Tensor)> {
    if t >= betas.len() {
        return Err(MathError::InvalidArgument("t must be < betas.len()"));
    }
    // Compute alpha_bar_t = prod(1 - beta_i) for i in 0..=t
    let mut alpha_bar = 1.0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..=t {
        alpha_bar *= 1.0 - betas[i];
    }
    let sqrt_alpha = alpha_bar.sqrt();
    let sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt();

    let noise = Tensor::randn(&x0.shape);
    let noisy = x0.mul_scalar(sqrt_alpha).add(&noise.mul_scalar(sqrt_one_minus_alpha))?;
    Ok((noisy, noise))
}

/// Diffusion reverse process: denoise one step.
/// x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise) + sigma_t * z
pub fn diffusion_reverse_process(
    x_t: &Tensor,
    predicted_noise: &Tensor,
    t: usize,
    betas: &[f64],
) -> MathResult<Tensor> {
    if t >= betas.len() {
        return Err(MathError::InvalidArgument("t must be < betas.len()"));
    }
    let beta_t = betas[t];
    let alpha_t = 1.0 - beta_t;

    let mut alpha_bar = 1.0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..=t {
        alpha_bar *= 1.0 - betas[i];
    }

    let sqrt_alpha = alpha_t.sqrt();
    let coeff = beta_t / (1.0 - alpha_bar).sqrt();

    let mean = x_t.sub(&predicted_noise.mul_scalar(coeff))?.div_scalar(sqrt_alpha);
    let sigma = beta_t.sqrt();
    let z = Tensor::randn(&x_t.shape);
    mean.add(&z.mul_scalar(sigma))
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-4;

    #[test]
    fn vae_reparameterize_test() {
        let mean = Tensor::new(&[2, 3], &[0.0; 6]).unwrap();
        let logvar = Tensor::new(&[2, 3], &[0.0; 6]).unwrap();
        let z = vae_reparameterize(&mean, &logvar).unwrap();
        assert_eq!(z.shape, vec![2, 3]);
        // With logvar=0, std=1, so z ≈ eps ~ N(0,1)
        assert!(z.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn vae_loss_test() {
        let recon = Tensor::new(&[2], &[0.5, 0.5]).unwrap();
        let original = Tensor::new(&[2], &[1.0, 1.0]).unwrap();
        let mean = Tensor::new(&[2], &[0.0, 0.0]).unwrap();
        let logvar = Tensor::new(&[2], &[0.0, 0.0]).unwrap();
        let loss = vae_loss(&recon, &original, &mean, &logvar).unwrap();
        // MSE = (0.25 + 0.25)/2 = 0.25, KL = 0 → loss ≈ 0.25
        assert!((loss - 0.25).abs() < E);
    }

    #[test]
    fn gan_discriminator_loss_test() {
        let real_pred = Tensor::new(&[4], &[0.9, 0.8, 0.85, 0.95]).unwrap();
        let fake_pred = Tensor::new(&[4], &[0.1, 0.2, 0.15, 0.05]).unwrap();
        let loss = gan_discriminator_loss(&real_pred, &fake_pred).unwrap();
        // Should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn gan_generator_loss_test() {
        let fake_pred = Tensor::new(&[4], &[0.9, 0.8, 0.85, 0.95]).unwrap();
        let loss = gan_generator_loss(&fake_pred).unwrap();
        // Generator loss = mean(-log(D(fake))), should be positive and finite
        assert!(loss > 0.0);
        assert!(loss.is_finite());
        // High D(fake) → low loss
        let fake_bad = Tensor::new(&[4], &[0.1, 0.1, 0.1, 0.1]).unwrap();
        let loss_bad = gan_generator_loss(&fake_bad).unwrap();
        assert!(loss_bad > loss);
    }

    #[test]
    fn diffusion_forward_test() {
        let x0 = Tensor::new(&[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let betas: Vec<f64> = (0..10).map(|i| 0.001 + i as f64 * 0.0001).collect();
        let (noisy, noise) = diffusion_forward_process(&x0, 5, &betas).unwrap();
        assert_eq!(noisy.shape, vec![1, 4]);
        assert_eq!(noise.shape, vec![1, 4]);
        // noisy ≈ sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise, should differ from x0
        let diff: f64 = noisy.data.iter().zip(&x0.data).map(|(a, b)| (a - b).abs()).sum::<f64>() / 4.0;
        assert!(diff > 1e-6);
    }

    #[test]
    fn diffusion_reverse_test() {
        let x_t = Tensor::new(&[1, 4], &[0.5, 1.5, 2.5, 3.5]).unwrap();
        let noise = Tensor::new(&[1, 4], &[0.1, 0.1, 0.1, 0.1]).unwrap();
        let betas: Vec<f64> = (0..10).map(|i| 0.001 + i as f64 * 0.0001).collect();
        let x_prev = diffusion_reverse_process(&x_t, &noise, 5, &betas).unwrap();
        assert_eq!(x_prev.shape, vec![1, 4]);
        assert!(x_prev.data.iter().all(|&x| x.is_finite()));
    }
}









