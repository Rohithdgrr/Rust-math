//! Morphological operations example.
//!
//! This example demonstrates binary morphological operations.

use mathverse_image::{GrayImage, morphology::{binarize, erode, dilate, open, close, sum}};

fn main() {
    // Create a test image with a square
    let mut img = GrayImage::new(64, 64).unwrap();
    for y in 16..48 {
        for x in 16..48 {
            img.set(x, y, 1.0);
        }
    }
    
    println!("Original image sum: {:.1}", sum(&img));
    
    // Binarize the image
    let binary = binarize(&img, 0.5);
    println!("Binarized image sum: {:.1}", sum(&binary));
    
    // Erode the image
    let eroded = erode(&binary);
    println!("Eroded image sum: {:.1}", sum(&eroded));
    
    // Dilate the image
    let dilated = dilate(&binary);
    println!("Dilated image sum: {:.1}", sum(&dilated));
    
    // Open the image (erosion followed by dilation)
    let opened = open(&binary);
    println!("Opened image sum: {:.1}", sum(&opened));
    
    // Close the image (dilation followed by erosion)
    let closed = close(&binary);
    println!("Closed image sum: {:.1}", sum(&closed));
}
