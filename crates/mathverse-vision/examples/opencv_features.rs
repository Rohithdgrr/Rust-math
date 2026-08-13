//! OpenCV-style camera viewer with advanced features
//! 
//! This example demonstrates a complete OpenCV-like interface with:
//! - Live camera feed
//! - Multiple processing modes
//! - Drawing overlays
//! - Real-time feature detection
//! - Mouse interaction (future)
//! 
//! Controls:
//! - ESC/Q: Quit
//! - 1: Original view
//! - 2: Edge detection (Canny)
//! - 3: Corner detection (Harris) with markers
//! - 4: Gaussian blur
//! - 5: Sobel gradient magnitude
//! - 6: Laplacian edge detection
//! - 7: Threshold (binary)
//! - 8: Adaptive threshold
//! - 9: Motion detection (frame difference)
//! - 0: Show histogram
//! - F: Toggle FPS display
//! - I: Toggle info overlay
//! - SPACE: Freeze frame / Resume
//! - S: Save screenshot

use mathverse_vision::{
    Image,
    camera::SystemCamera,
    ops::{canny, sobel, laplacian, histogram, bounding_box},
    features::harris,
    threshold::{binary, adaptive},
    drawing::{circle, rectangle},
};
use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    Original = 1,
    EdgeCanny = 2,
    CornerHarris = 3,
    GaussianBlur = 4,
    SobelGradient = 5,
    Laplacian = 6,
    ThresholdBinary = 7,
    ThresholdAdaptive = 8,
    MotionDetection = 9,
    Histogram = 0,
}

struct AppState {
    mode: ViewMode,
    show_fps: bool,
    show_info: bool,
    frozen: bool,
    frozen_frame: Option<Image>,
    previous_frame: Option<Image>,
    frame_times: Vec<std::time::Instant>,
}

impl AppState {
    fn new() -> Self {
        Self {
            mode: ViewMode::Original,
            show_fps: true,
            show_info: true,
            frozen: false,
            frozen_frame: None,
            previous_frame: None,
            frame_times: Vec::new(),
        }
    }
    
    fn update_fps(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push(now);
        
        // Keep only last 30 frames for FPS calculation
        if self.frame_times.len() > 30 {
            self.frame_times.remove(0);
        }
    }
    
    fn get_fps(&self) -> f64 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }
        
        let duration = self.frame_times.last().unwrap()
            .duration_since(*self.frame_times.first().unwrap());
        let seconds = duration.as_secs_f64();
        
        if seconds > 0.0 {
            (self.frame_times.len() - 1) as f64 / seconds
        } else {
            0.0
        }
    }
}

fn main() -> Result<(), String> {
    println!("=== MathVerse Vision - OpenCV Features Demo ===\n");
    println!("Initializing camera...");
    
    let mut cam = SystemCamera::new(0)?;
    
    let mut window = Window::new(
        "MathVerse Vision - OpenCV Features (Press H for help)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).map_err(|e| format!("Failed to create window: {}", e))?;
    
    window.set_target_fps(30);
    
    let mut state = AppState::new();
    
    print_help();
    
    // Main processing loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Handle keyboard input
        handle_input(&window, &mut state);
        
        if window.is_key_pressed(Key::Q, minifb::KeyRepeat::No) {
            break;
        }
        
        // Capture or use frozen frame
        let frame = if state.frozen && state.frozen_frame.is_some() {
            state.frozen_frame.as_ref().unwrap().clone()
        } else {
            let (success, frame) = cam.read()?;
            if !success {
                println!("Failed to capture frame");
                continue;
            }
            
            // Store frozen frame if just frozen
            if state.frozen && state.frozen_frame.is_none() {
                state.frozen_frame = Some(frame.clone());
            }
            
            // Store for motion detection
            if state.mode == ViewMode::MotionDetection {
                state.previous_frame = Some(frame.clone());
            }
            
            frame
        };
        
        // Process frame based on mode
        let mut processed = process_frame(&frame, &state);
        
        // Draw overlays
        draw_overlays(&mut processed, &state, &frame);
        
        // Convert to display buffer
        let buffer = grayscale_to_rgb_buffer(&processed);
        
        // Update window
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .map_err(|e| format!("Window update failed: {}", e))?;
        
        state.update_fps();
    }
    
    println!("\nCamera closed.");
    Ok(())
}

fn handle_input(window: &Window, state: &mut AppState) {
    // Mode selection
    if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
        state.mode = ViewMode::Original;
        println!("Mode: Original");
    }
    if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
        state.mode = ViewMode::EdgeCanny;
        println!("Mode: Edge Detection (Canny)");
    }
    if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
        state.mode = ViewMode::CornerHarris;
        println!("Mode: Corner Detection (Harris)");
    }
    if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
        state.mode = ViewMode::GaussianBlur;
        println!("Mode: Gaussian Blur");
    }
    if window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) {
        state.mode = ViewMode::SobelGradient;
        println!("Mode: Sobel Gradient");
    }
    if window.is_key_pressed(Key::Key6, minifb::KeyRepeat::No) {
        state.mode = ViewMode::Laplacian;
        println!("Mode: Laplacian");
    }
    if window.is_key_pressed(Key::Key7, minifb::KeyRepeat::No) {
        state.mode = ViewMode::ThresholdBinary;
        println!("Mode: Binary Threshold");
    }
    if window.is_key_pressed(Key::Key8, minifb::KeyRepeat::No) {
        state.mode = ViewMode::ThresholdAdaptive;
        println!("Mode: Adaptive Threshold");
    }
    if window.is_key_pressed(Key::Key9, minifb::KeyRepeat::No) {
        state.mode = ViewMode::MotionDetection;
        println!("Mode: Motion Detection");
    }
    if window.is_key_pressed(Key::Key0, minifb::KeyRepeat::No) {
        state.mode = ViewMode::Histogram;
        println!("Mode: Histogram View");
    }
    
    // Toggle options
    if window.is_key_pressed(Key::F, minifb::KeyRepeat::No) {
        state.show_fps = !state.show_fps;
        println!("FPS display: {}", if state.show_fps { "ON" } else { "OFF" });
    }
    if window.is_key_pressed(Key::I, minifb::KeyRepeat::No) {
        state.show_info = !state.show_info;
        println!("Info overlay: {}", if state.show_info { "ON" } else { "OFF" });
    }
    if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
        state.frozen = !state.frozen;
        if state.frozen {
            println!("Frame FROZEN - Press SPACE to resume");
        } else {
            println!("Frame RESUMED");
            state.frozen_frame = None;
        }
    }
    if window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
        println!("Screenshot feature - save to file (not implemented in this example)");
    }
    if window.is_key_pressed(Key::H, minifb::KeyRepeat::No) {
        print_help();
    }
}

fn process_frame(frame: &Image, state: &AppState) -> Image {
    match state.mode {
        ViewMode::Original => frame.clone(),
        
        ViewMode::EdgeCanny => {
            let edges = canny(frame, 0.1, 0.3);
            edges
        },
        
        ViewMode::CornerHarris => {
            let corners = harris(frame, 1.0, 0.04);
            
            // Find strong corners and draw markers
            let mut display = frame.clone();
            let threshold = 0.01; // Adjust for sensitivity
            
            for y in 10..frame.h - 10 {
                for x in 10..frame.w - 10 {
                    let idx = y * frame.w + x;
                    if corners.data[idx] > threshold {
                        // Draw a small circle at corner location
                        circle(&mut display, (x, y), 3, 1.0, 1);
                    }
                }
            }
            
            display
        },
        
        ViewMode::GaussianBlur => {
            frame.gaussian_blur(5, 2.0)
        },
        
        ViewMode::SobelGradient => {
            let (mag, _) = sobel(frame);
            normalize_image(&mag)
        },
        
        ViewMode::Laplacian => {
            let lap = laplacian(frame);
            normalize_image(&lap)
        },
        
        ViewMode::ThresholdBinary => {
            binary(frame, 0.5, 1.0)
        },
        
        ViewMode::ThresholdAdaptive => {
            adaptive(frame, 15, 0.05)
        },
        
        ViewMode::MotionDetection => {
            if let Some(ref prev) = state.previous_frame {
                // Compute frame difference
                let mut diff = Image::new(frame.w, frame.h);
                for i in 0..frame.data.len() {
                    diff.data[i] = (frame.data[i] - prev.data[i]).abs();
                }
                
                // Threshold to detect motion
                let motion = binary(&diff, 0.1, 1.0);
                
                // Find bounding box of motion
                let mut display = frame.clone();
                if let Some((x0, y0, x1, y1)) = bounding_box(&motion) {
                    rectangle(&mut display, (x0, y0), (x1, y1), 1.0, 2);
                }
                
                display
            } else {
                frame.clone()
            }
        },
        
        ViewMode::Histogram => {
            draw_histogram(frame)
        },
    }
}

fn draw_overlays(img: &mut Image, state: &AppState, _original: &Image) {
    // Draw FPS counter (top-left corner indicator)
    if state.show_fps {
        let fps = state.get_fps();
        // Draw a small rectangle to indicate FPS area
        // In a real implementation, you'd use text rendering
        let fps_indicator_size = (fps / 60.0 * 50.0).min(50.0) as usize;
        for y in 10..15 {
            for x in 10..(10 + fps_indicator_size) {
                if x < img.w && y < img.h {
                    img.set(x, y, 1.0);
                }
            }
        }
    }
    
    // Draw mode indicator (top-right corner)
    if state.show_info {
        let mode_num = state.mode as usize;
        // Draw small indicator dots
        for i in 0..10 {
            let x = img.w - 20 - i * 5;
            let y = 10;
            let brightness = if i == mode_num { 1.0 } else { 0.3 };
            if x < img.w && y < img.h {
                circle(img, (x, y), 2, brightness, 1);
            }
        }
    }
    
    // Draw frozen indicator
    if state.frozen {
        // Draw pause symbol (two vertical bars) in bottom-left
        for y in img.h - 30..img.h - 10 {
            for x in 10..15 {
                if x < img.w && y < img.h {
                    img.set(x, y, 1.0);
                }
            }
            for x in 20..25 {
                if x < img.w && y < img.h {
                    img.set(x, y, 1.0);
                }
            }
        }
    }
}

fn draw_histogram(img: &Image) -> Image {
    let hist = histogram(img);
    
    // Create histogram visualization
    let hist_width = WIDTH;
    let hist_height = HEIGHT;
    let mut hist_img = Image::new(hist_width, hist_height);
    
    // Find max bin count for normalization
    let max_count = *hist.iter().max().unwrap_or(&1);
    
    // Draw histogram bars
    let bin_width = hist_width / 256;
    for (i, &count) in hist.iter().enumerate() {
        let bar_height = (count as f64 / max_count as f64 * (hist_height as f64 * 0.8)) as usize;
        let x = i * bin_width;
        
        // Draw vertical bar
        for dy in 0..bar_height {
            let y = hist_height - 1 - dy;
            for dx in 0..bin_width {
                if x + dx < hist_width && y < hist_height {
                    hist_img.set(x + dx, y, 0.8);
                }
            }
        }
    }
    
    hist_img
}

fn grayscale_to_rgb_buffer(img: &Image) -> Vec<u32> {
    let mut buffer = vec![0u32; img.w * img.h];
    
    for (i, &val) in img.data.iter().enumerate() {
        let byte = (val.clamp(0.0, 1.0) * 255.0) as u8;
        buffer[i] = ((byte as u32) << 16) | ((byte as u32) << 8) | (byte as u32);
    }
    
    buffer
}

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

fn print_help() {
    println!("\n=== CONTROLS ===");
    println!("ESC/Q: Quit");
    println!("\nProcessing Modes:");
    println!("  1: Original view");
    println!("  2: Edge detection (Canny)");
    println!("  3: Corner detection (Harris)");
    println!("  4: Gaussian blur");
    println!("  5: Sobel gradient");
    println!("  6: Laplacian");
    println!("  7: Binary threshold");
    println!("  8: Adaptive threshold");
    println!("  9: Motion detection");
    println!("  0: Histogram");
    println!("\nDisplay Options:");
    println!("  F: Toggle FPS display");
    println!("  I: Toggle info overlay");
    println!("  SPACE: Freeze/Resume");
    println!("  S: Save screenshot");
    println!("  H: Show this help");
    println!("================\n");
}
