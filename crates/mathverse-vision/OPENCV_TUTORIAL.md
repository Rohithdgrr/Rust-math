# OpenCV-Style Live Camera Tutorial

This guide shows how to use MathVerse Vision as an OpenCV replacement for live camera applications.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Camera Capture](#basic-camera-capture)
3. [Image Processing](#image-processing)
4. [Drawing Overlays](#drawing-overlays)
5. [Real-time Feature Detection](#real-time-feature-detection)
6. [Complete Examples](#complete-examples)

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathverse-vision = { path = "crates/mathverse-vision" }
minifb = "0.27"
```

### Minimal Example

```rust
use mathverse_vision::camera::SystemCamera;
use minifb::{Key, Window, WindowOptions};

fn main() -> Result<(), String> {
    let mut cap = SystemCamera::new(0)?;
    let mut window = Window::new("Camera", 640, 480, WindowOptions::default())?;
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let (ret, frame) = cap.read()?;
        if ret {
            let buffer = to_rgb_buffer(&frame);
            window.update_with_buffer(&buffer, 640, 480)?;
        }
    }
    Ok(())
}

fn to_rgb_buffer(img: &mathverse_vision::Image) -> Vec<u32> {
    img.data.iter().map(|&v| {
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        ((byte as u32) << 16) | ((byte as u32) << 8) | (byte as u32)
    }).collect()
}
```

## Basic Camera Capture

### Opening a Camera

```rust
use mathverse_vision::camera::SystemCamera;

// Open default camera (like cv2.VideoCapture(0))
let mut cap = SystemCamera::new(0)?;

// On Windows: uses Win32 API
// On Linux: uses V4L2 (/dev/video0)
// Fallback: DummyCamera for testing
```

### Reading Frames

```rust
// Read a single frame (like cap.read() in OpenCV)
let (success, frame) = cap.read()?;

if success {
    println!("Frame size: {}x{}", frame.w, frame.h);
    println!("Pixel at (10, 20): {}", frame.get(10, 20));
}

// Grab without decoding (like cap.grab())
cap.grab()?;
```

### Camera Properties

```rust
use mathverse_vision::camera::properties;

// Get properties (like cap.get())
let width = cap.get(properties::FRAME_WIDTH)?;
let height = cap.get(properties::FRAME_HEIGHT)?;
let fps = cap.get(properties::FPS)?;

// Set properties (like cap.set())
cap.set(properties::FRAME_WIDTH, 1280.0)?;
cap.set(properties::FRAME_HEIGHT, 720.0)?;
```

## Image Processing

### Edge Detection

```rust
use mathverse_vision::ops::{canny, sobel, laplacian};

// Canny edge detection
let edges = canny(&frame, 0.1, 0.3);  // low and high thresholds

// Sobel gradient
let (magnitude, direction) = sobel(&frame);

// Laplacian
let lap = laplacian(&frame);
```

### Blurring and Filtering

```rust
// Gaussian blur (like cv2.GaussianBlur)
let blurred = frame.gaussian_blur(5, 2.0);  // radius, sigma

// Custom convolution
let kernel = [
    -1.0, -1.0, -1.0,
    -1.0,  8.0, -1.0,
    -1.0, -1.0, -1.0,
];
let filtered = frame.convolve3(&kernel);
```

### Thresholding

```rust
use mathverse_vision::threshold::{binary, adaptive};

// Binary threshold (like cv2.threshold)
let thresh = binary(&frame, 0.5, 1.0);

// Adaptive threshold (like cv2.adaptiveThreshold)
let adaptive_thresh = adaptive(&frame, 15, 0.05);
```

### Transformations

```rust
use mathverse_vision::transform::{resize, rotate, affine};

// Resize (like cv2.resize)
let resized = resize(&frame, 320, 240);

// Rotate
let rotated = rotate(&frame, 45.0);  // degrees

// Affine transform
let transformed = affine(&frame, 1.0, 0.0, 10.0, 0.0, 1.0, 20.0);
```

## Drawing Overlays

### Basic Shapes

```rust
use mathverse_vision::drawing::{line, rectangle, circle};

let mut display = frame.clone();

// Draw line (like cv2.line)
line(&mut display, (10, 10), (100, 100), 1.0, 2);

// Draw rectangle (like cv2.rectangle)
rectangle(&mut display, (50, 50), (200, 150), 1.0, 2);

// Draw circle (like cv2.circle)
circle(&mut display, (150, 150), 30, 1.0, 2);
```

### Drawing on Video Stream

```rust
while window.is_open() {
    let (_, frame) = cap.read()?;
    let mut display = frame.clone();
    
    // Add timestamp indicator
    circle(&mut display, (30, 30), 10, 1.0, 2);
    
    // Add frame counter
    let x = frame_count as usize % frame.w;
    line(&mut display, (x, 0), (x, 10), 1.0, 1);
    
    // Display
    window.update_with_buffer(&to_rgb_buffer(&display), 640, 480)?;
    frame_count += 1;
}
```

## Real-time Feature Detection

### Corner Detection

```rust
use mathverse_vision::features::harris;

let corners = harris(&frame, 1.0, 0.04);

// Find and mark strong corners
let mut display = frame.clone();
let threshold = 0.01;

for y in 10..frame.h - 10 {
    for x in 10..frame.w - 10 {
        if corners.get(x, y) > threshold {
            circle(&mut display, (x, y), 3, 1.0, 1);
        }
    }
}
```

### Motion Detection

```rust
use mathverse_vision::{threshold::binary, ops::bounding_box};

let mut prev_frame: Option<Image> = None;

while window.is_open() {
    let (_, frame) = cap.read()?;
    
    if let Some(prev) = &prev_frame {
        // Compute frame difference
        let mut diff = Image::new(frame.w, frame.h);
        for i in 0..frame.data.len() {
            diff.data[i] = (frame.data[i] - prev.data[i]).abs();
        }
        
        // Threshold to detect motion
        let motion = binary(&diff, 0.1, 1.0);
        
        // Draw bounding box around motion
        if let Some((x0, y0, x1, y1)) = bounding_box(&motion) {
            let mut display = frame.clone();
            rectangle(&mut display, (x0, y0), (x1, y1), 1.0, 2);
            // Display...
        }
    }
    
    prev_frame = Some(frame);
}
```

### Histogram Analysis

```rust
use mathverse_vision::ops::histogram;

let hist = histogram(&frame);  // 256-bin histogram

// Find brightest regions
let max_bin = hist.iter().enumerate()
    .max_by_key(|(_, &count)| count)
    .map(|(bin, _)| bin)
    .unwrap();

println!("Most common intensity: {}/255", max_bin);
```

## Complete Examples

### Interactive Camera App

```rust
use mathverse_vision::{
    Image, camera::SystemCamera,
    ops::{canny, sobel},
    features::harris,
    drawing::circle,
};
use minifb::{Key, Window, WindowOptions};

enum Mode { Original, Edges, Corners, Blur }

fn main() -> Result<(), String> {
    let mut cap = SystemCamera::new(0)?;
    let mut window = Window::new("Camera", 640, 480, WindowOptions::default())?;
    let mut mode = Mode::Original;
    
    println!("Controls: 1=Original, 2=Edges, 3=Corners, 4=Blur, ESC=Quit");
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Handle input
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            mode = Mode::Original;
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            mode = Mode::Edges;
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            mode = Mode::Corners;
        }
        if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
            mode = Mode::Blur;
        }
        
        // Capture and process
        let (_, frame) = cap.read()?;
        
        let processed = match mode {
            Mode::Original => frame,
            Mode::Edges => canny(&frame, 0.1, 0.3),
            Mode::Corners => {
                let corners = harris(&frame, 1.0, 0.04);
                let mut display = frame.clone();
                for y in 10..frame.h-10 {
                    for x in 10..frame.w-10 {
                        if corners.get(x, y) > 0.01 {
                            circle(&mut display, (x, y), 3, 1.0, 1);
                        }
                    }
                }
                display
            },
            Mode::Blur => frame.gaussian_blur(5, 2.0),
        };
        
        // Display
        let buffer = to_rgb_buffer(&processed);
        window.update_with_buffer(&buffer, 640, 480)?;
    }
    
    Ok(())
}

fn to_rgb_buffer(img: &Image) -> Vec<u32> {
    img.data.iter().map(|&v| {
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        ((byte as u32) << 16) | ((byte as u32) << 8) | (byte as u32)
    }).collect()
}
```

### Recording to Video

```rust
use mathverse_vision::{camera::SystemCamera, video::VideoWriter};

fn main() -> Result<(), String> {
    let mut cap = SystemCamera::new(0)?;
    let mut writer = VideoWriter::new("output.raw", 640, 480, 30);
    
    println!("Recording 300 frames...");
    
    for i in 0..300 {
        let (ret, frame) = cap.read()?;
        if ret {
            writer.write_frame(&frame)?;
            if i % 30 == 0 {
                println!("Recorded {} frames", i);
            }
        }
    }
    
    let total = writer.close();
    println!("Saved {} frames to output.raw", total);
    
    Ok(())
}
```

## Performance Tips

1. **Use frame skipping for expensive operations:**
```rust
if frame_count % 5 == 0 {
    // Only process every 5th frame
    let corners = harris(&frame, 1.0, 0.04);
}
```

2. **Pre-allocate buffers:**
```rust
let mut display_buffer = vec![0u32; 640 * 480];
// Reuse buffer each frame
```

3. **Adjust window update rate:**
```rust
window.limit_update_rate(Some(std::time::Duration::from_micros(33333))); // ~30 FPS
```

4. **Use smaller resolution for processing:**
```rust
let small = resize(&frame, 320, 240);
let edges = canny(&small, 0.1, 0.3);
let display = resize(&edges, 640, 480);
```

## Comparison with OpenCV

| OpenCV Python | MathVerse Vision Rust |
|--------------|----------------------|
| `cv2.VideoCapture(0)` | `SystemCamera::new(0)?` |
| `cap.read()` | `cap.read()?` |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | `cap.get(properties::FRAME_WIDTH)?` |
| `cv2.Canny(img, 50, 150)` | `canny(&img, 0.2, 0.6)` |
| `cv2.GaussianBlur(img, (5,5), 2)` | `img.gaussian_blur(5, 2.0)` |
| `cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)` | `binary(&img, 0.5, 1.0)` |
| `cv2.Sobel(img, -1, 1, 0)` | `sobel(&img)` |
| `cv2.line(img, (0,0), (100,100), 255, 2)` | `line(&mut img, (0,0), (100,100), 1.0, 2)` |
| `cv2.imshow("window", img)` | `window.update_with_buffer(&buf, w, h)?` |

## Next Steps

- Run the examples: `cargo run --example opencv_features`
- Explore advanced features: homography, optical flow, epipolar geometry
- Build your own computer vision app!
