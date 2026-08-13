use mathverse_vision::camera::{SystemCamera, properties};

fn main() -> Result<(), String> {
    // Auto-detects platform: V4L2 on Linux/macOS, Win32 on Windows
    let mut cam = SystemCamera::new(0)?;  // index 0 = default/integrated camera
    
    // Read a frame (Python-opencv style)
    let (ret, frame) = cam.read()?;
    
    if ret {
        println!("Frame captured: {}x{}", frame.w, frame.h);
        println!("Frame data length: {}", frame.data.len());
        
        // Display basic info - this is the camera pipeline working
        println!("\nCamera Properties:");
        println!("  Width: {}", cam.get(properties::FRAME_WIDTH)?);
        println!("  Height: {}", cam.get(properties::FRAME_HEIGHT)?);
        println!("  FPS: {}", cam.get(properties::FPS)?);
    } else {
        println!("Failed to capture frame");
    }
    
    // Demonstrate property control
    cam.set(properties::FRAME_WIDTH, 640.0)?;
    cam.set(properties::FRAME_HEIGHT, 480.0)?;
    
    let width = cam.get(properties::FRAME_WIDTH)?;
    println!("Width after set: {}", width);
    
    Ok(())
}
