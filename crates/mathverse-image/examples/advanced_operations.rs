//! Advanced image operations example.
//!
//! This example demonstrates thresholding, noise, arithmetic operations,
//! and other advanced features.

use mathverse_image::GrayImage;

fn main() {
    // Create a test image
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            let value = ((x + y) % 32) as f64 / 32.0;
            img.set(x, y, value);
        }
    }
    
    println!("Original image mean: {:.3}", img.mean());
    println!("Original image std dev: {:.3}", img.std_dev());
    
    // Apply simple thresholding
    let thresholded = img.threshold(0.5);
    let thresholded_count = thresholded.data.iter().filter(|v| **v > 0.5).count();
    println!("Thresholded (0.5) pixels above threshold: {}", thresholded_count);
    
    // Apply adaptive thresholding
    let adaptive = img.adaptive_threshold(8, 0.1);
    println!("Adaptive thresholded mean: {:.3}", adaptive.mean());
    
    // Add Gaussian noise
    let noisy = img.add_gaussian_noise(0.0, 0.1);
    println!("Noisy image mean: {:.3}", noisy.mean());
    println!("Noisy image std dev: {:.3}", noisy.std_dev());
    
    // Add salt-and-pepper noise
    let sp_noisy = img.add_salt_pepper_noise(0.05);
    println!("Salt-pepper noisy image mean: {:.3}", sp_noisy.mean());
    
    // Invert the image
    let inverted = img.invert();
    println!("Inverted image mean: {:.3}", inverted.mean());
    
    // Apply gamma correction
    let gamma = img.gamma_correction(2.0);
    println!("Gamma corrected (2.0) mean: {:.3}", gamma.mean());
    
    // Normalize the image
    let normalized = img.normalize();
    println!("Normalized image min: {:.3}", normalized.min_value());
    println!("Normalized image max: {:.3}", normalized.max_value());
    
    // Contrast stretching
    let stretched = img.contrast_stretch(0.25, 0.75);
    println!("Contrast stretched mean: {:.3}", stretched.mean());
    
    // Arithmetic operations
    let img2 = img.scale(0.5);
    let sum = img.add(&img2);
    let diff = img.subtract(&img2);
    let prod = img.multiply(&img2);
    
    println!("Sum mean: {:.3}", sum.mean());
    println!("Difference mean: {:.3}", diff.mean());
    println!("Product mean: {:.3}", prod.mean());
    
    // Scale and offset
    let scaled = img.scale(1.5);
    let offset = img.offset(0.25);
    
    println!("Scaled (1.5) mean: {:.3}", scaled.mean());
    println!("Offset (+0.25) mean: {:.3}", offset.mean());
}
