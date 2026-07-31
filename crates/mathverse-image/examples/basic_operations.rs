//! Basic image operations example.
//!
//! This example demonstrates basic image creation, manipulation, and processing.

use mathverse_image::{box_blur, sharpen, GrayImage};

fn main() {
    // Create a new blank image
    let mut img = GrayImage::new(64, 64);
    
    // Create a simple pattern
    for y in 0..64 {
        for x in 0..64 {
            let value = ((x + y) % 32) as f64 / 32.0;
            img.set(x, y, value);
        }
    }
    
    println!("Original image: {}x{}", img.w, img.h);
    println!("Mean value: {:.3}", img.mean());
    
    // Apply box blur
    let blurred = box_blur(&img);
    println!("Blurred image mean: {:.3}", blurred.mean());
    
    // Apply sharpening
    let sharpened = sharpen(&img);
    println!("Sharpened image mean: {:.3}", sharpened.mean());
    
    // Apply Gaussian blur
    let gaussian = img.gaussian_blur(3, 1.5);
    println!("Gaussian blurred image mean: {:.3}", gaussian.mean());
    
    // Resize the image
    let resized = img.resize(128, 128);
    println!("Resized image: {}x{}", resized.w, resized.h);
    
    // Flip horizontally
    let flipped = img.flip_h();
    println!("Flipped image: {}x{}", flipped.w, flipped.h);
    
    // Rotate 90 degrees
    let rotated = img.rotate90();
    println!("Rotated image: {}x{}", rotated.w, rotated.h);
    
    // Compute histogram
    let histogram = img.histogram();
    let max_bin = histogram.iter().cloned().fold(0usize, usize::max);
    println!("Histogram max bin count: {}", max_bin);
}
