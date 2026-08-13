//! Simple camera window example - OpenCV style
//! 
//! This is a minimal example showing how to open a live camera feed
//! in a window, similar to OpenCV's cv2.imshow() and cv2.VideoCapture()

use mathverse_vision::{Image, camera::SystemCamera};
use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

fn main() -> Result<(), String> {
    // Create camera capture object (like cv2.VideoCapture(0))
    let mut cap = SystemCamera::new(0)?;
    
    // Create window for display (like cv2.namedWindow())
    let mut window = Window::new(
        "Camera Feed - Press ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).map_err(|e| format!("Window creation failed: {}", e))?;
    
    // Set framerate limit (~30 FPS)
    window.set_target_fps(30);
    
    println!("Camera opened successfully!");
    println!("Window opened - you should see animated patterns");
    println!("Press ESC to quit...\n");
    
    let mut frame_count = 0;
    let start_time = std::time::Instant::now();
    
    // Main loop (like while cap.isOpened())
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Read frame (like cap.read())
        let (ret, frame) = cap.read()?;
        
        if !ret {
            eprintln!("Failed to capture frame");
            continue;
        }
        
        // Convert grayscale image to RGB buffer for display
        let buffer = image_to_buffer(&frame);
        
        // Display frame (like cv2.imshow())
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .map_err(|e| format!("Display update failed: {}", e))?;
        
        frame_count += 1;
        
        // Print status every 30 frames with FPS
        if frame_count % 30 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            println!("Frames: {} | FPS: {:.1}", frame_count, fps);
        }
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / elapsed;
    
    println!("\n=== Session Summary ===");
    println!("Total frames captured: {}", frame_count);
    println!("Duration: {:.2}s", elapsed);
    println!("Average FPS: {:.1}", avg_fps);
    println!("Camera released successfully.");
    
    // Window and camera are automatically closed when dropped
    Ok(())
}

/// Convert Image to RGB buffer for minifb display
fn image_to_buffer(img: &Image) -> Vec<u32> {
    let mut buffer = vec![0u32; img.w * img.h];
    
    for (i, &val) in img.data.iter().enumerate() {
        // Convert f64 [0.0, 1.0] to u8 [0, 255]
        let gray = (val.clamp(0.0, 1.0) * 255.0) as u8;
        
        // Pack as RGB (0x00RRGGBB format for minifb)
        buffer[i] = ((gray as u32) << 16) | ((gray as u32) << 8) | (gray as u32);
    }
    
    buffer
}
