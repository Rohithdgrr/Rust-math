//! Edge detection example.
//!
//! This example demonstrates Sobel gradients and Canny edge detection.

use mathverse_image::{canny::canny, GrayImage};

fn main() {
    // Create a test image with a step edge
    let mut img = GrayImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            img.set(x, y, if x < 32 { 0.0 } else { 1.0 });
        }
    }
    
    println!("Test image: {}x{}", img.w, img.h);
    
    // Compute Sobel gradients
    let (magnitude, direction) = img.sobel();
    println!("Sobel magnitude mean: {:.3}", magnitude.mean());
    println!("Sobel magnitude max: {:.3}", magnitude.max_value());
    
    // Apply Canny edge detection
    let edges = canny(&img, 1.5, 0.05, 0.15);
    println!("Canny edges mean: {:.3}", edges.mean());
    
    // Count edge pixels
    let edge_count = edges.data.iter().filter(|v| **v > 0.5).count();
    println!("Edge pixels: {}", edge_count);
}
