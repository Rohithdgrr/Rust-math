//! Image I/O example.
//!
//! This example demonstrates loading and saving images.

use mathverse_image::{io::{load, save}, GrayImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test image
    let mut img = GrayImage::new(64, 64).unwrap();
    for y in 0..64 {
        for x in 0..64 {
            let value = ((x * y) % 64) as f64 / 64.0;
            img.set(x, y, value);
        }
    }
    
    // Save the image
    save(&img, "test_output.png")?;
    println!("Saved image to test_output.png");
    
    // Load the image back
    let loaded = load("test_output.png")?;
    println!("Loaded image: {}x{}", loaded.w, loaded.h);
    
    // Verify the dimensions match
    assert_eq!(img.w, loaded.w);
    assert_eq!(img.h, loaded.h);
    println!("Image dimensions match!");
    
    Ok(())
}
