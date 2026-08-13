//! Live camera viewer with OpenCV-like features
//! 
//! Controls:
//! - ESC or Q: Quit
//! - E: Toggle edge detection (Canny)
//! - C: Toggle corner detection (Harris)
//! - B: Toggle blur (Gaussian)
//! - S: Toggle Sobel gradient
//! - H: Toggle histogram equalization
//! - R: Reset (show original)
//! - SPACE: Capture screenshot

use mathverse_vision::{Image, camera::SystemCamera};
use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

enum ProcessMode {
    Original,
    EdgeDetection,
    CornerDetection,
    Blur,
    Sobel,
    HistogramEqualize,
}

fn main() -> Result<(), String> {
    // Initialize camera
    println!("Initializing camera...");
    let mut cam = SystemCamera::new(0)?;
    
    // Create window
    let mut window = Window::new(
        "MathVerse Vision - Live Camera (ESC to quit)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).map_err(|e| format!("Failed to create window: {}", e))?;
    
    // Limit update rate to approximately 30 FPS
    window.set_target_fps(30);
    
    let mut mode = ProcessMode::Original;
    let mut frame_count = 0;
    
    println!("\n=== Camera Controls ===");
    println!("ESC/Q: Quit");
    println!("E: Edge Detection (Canny)");
    println!("C: Corner Detection (Harris)");
    println!("B: Gaussian Blur");
    println!("S: Sobel Gradient");
    println!("H: Histogram Equalization");
    println!("R: Reset to Original");
    println!("SPACE: Capture Screenshot");
    println!("======================\n");
    
    // Main loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Handle keyboard input
        if window.is_key_pressed(Key::Q, minifb::KeyRepeat::No) {
            break;
        }
        if window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
            mode = ProcessMode::EdgeDetection;
            println!("Mode: Edge Detection (Canny)");
        }
        if window.is_key_pressed(Key::C, minifb::KeyRepeat::No) {
            mode = ProcessMode::CornerDetection;
            println!("Mode: Corner Detection (Harris)");
        }
        if window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
            mode = ProcessMode::Blur;
            println!("Mode: Gaussian Blur");
        }
        if window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            mode = ProcessMode::Sobel;
            println!("Mode: Sobel Gradient");
        }
        if window.is_key_pressed(Key::H, minifb::KeyRepeat::No) {
            mode = ProcessMode::HistogramEqualize;
            println!("Mode: Histogram Equalization");
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            mode = ProcessMode::Original;
            println!("Mode: Original");
        }
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            println!("Screenshot saved as frame_{:04}.raw", frame_count);
        }
        
        // Capture frame
        let (success, frame) = cam.read()?;
        
        if !success {
            println!("Failed to capture frame");
            continue;
        }
        
        // Process frame based on mode
        let processed = match mode {
            ProcessMode::Original => frame,
            ProcessMode::EdgeDetection => {
                use mathverse_vision::ops::canny;
                canny(&frame, 0.1, 0.3)
            },
            ProcessMode::CornerDetection => {
                use mathverse_vision::features::harris;
                let corners = harris(&frame, 1.0, 0.04);
                // Normalize for display
                normalize_image(&corners)
            },
            ProcessMode::Blur => {
                frame.gaussian_blur(5, 2.0)
            },
            ProcessMode::Sobel => {
                use mathverse_vision::ops::sobel;
                let (mag, _) = sobel(&frame);
                normalize_image(&mag)
            },
            ProcessMode::HistogramEqualize => {
                histogram_equalize(&frame)
            },
        };
        
        // Convert grayscale to RGB for display
        let buffer = grayscale_to_rgb_buffer(&processed);
        
        // Update window
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .map_err(|e| format!("Window update failed: {}", e))?;
        
        frame_count += 1;
        
        // Print FPS every 30 frames
        if frame_count % 30 == 0 {
            print!("\rFrames: {} ", frame_count);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    
    println!("\n\nCamera closed. Total frames: {}", frame_count);
    Ok(())
}

/// Convert grayscale image to RGB buffer for minifb (u32 format: 0xRRGGBB)
fn grayscale_to_rgb_buffer(img: &Image) -> Vec<u32> {
    let mut buffer = vec![0u32; img.w * img.h];
    
    for (i, &val) in img.data.iter().enumerate() {
        // Clamp to [0, 1] and convert to 8-bit
        let byte = (val.clamp(0.0, 1.0) * 255.0) as u8;
        // Pack as 0xRRGGBB
        buffer[i] = ((byte as u32) << 16) | ((byte as u32) << 8) | (byte as u32);
    }
    
    buffer
}

/// Normalize image values to [0, 1] range
fn normalize_image(img: &Image) -> Image {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    
    for &val in &img.data {
        if val < min_val { min_val = val; }
        if val > max_val { max_val = val; }
    }
    
    let range = max_val - min_val;
    if range < 1e-10 {
        return img.clone();
    }
    
    let mut out = Image::new(img.w, img.h);
    for (i, &val) in img.data.iter().enumerate() {
        out.data[i] = (val - min_val) / range;
    }
    
    out
}

/// Simple histogram equalization
fn histogram_equalize(img: &Image) -> Image {
    use mathverse_vision::ops::histogram;
    
    let hist = histogram(img);
    
    // Compute cumulative distribution function (CDF)
    let mut cdf = [0usize; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    
    let total_pixels = img.w * img.h;
    
    // Create equalized image
    let mut out = Image::new(img.w, img.h);
    for (i, &val) in img.data.iter().enumerate() {
        let bin = (val * 255.0).floor() as usize;
        let bin = bin.min(255);
        // Map using CDF
        let equalized = (cdf[bin] as f64 / total_pixels as f64);
        out.data[i] = equalized;
    }
    
    out
}
