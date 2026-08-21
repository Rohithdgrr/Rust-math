//! Camera module for mathverse-vision.

use crate::Image;

/// Trait abstracting over camera backends.
/// Implementations can be zero-dependency (DummyCamera) or FFI-based
/// (V4L2, Win32, AVFoundation) when dependencies are available.
pub trait CameraTrait {
    /// Capture a single frame, returning f64 grayscale Image with values in [0.0, 1.0].
    fn capture_frame(&mut self) -> Result<Image, String>;

    /// Get camera width in pixels.
    fn width(&self) -> usize;

    /// Get camera height in pixels.
    fn height(&self) -> usize;

    /// Get camera FPS.
    fn fps(&self) -> f64;
}

/// Pinhole camera model with focal lengths `fx`, `fy` and principal point `(cx, cy)`.
/// Used for projection/unprojection and camera matrix operations.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Focal length in x direction (pixels)
    pub fx: f64,
    /// Focal length in y direction (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
}

impl Camera {
    /// Creates a new camera model with given focal length `(fx, fy)` and principal point `(cx, cy)`.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Camera { fx, fy, cx, cy }
    }

    /// Projects 3D camera coordinates `(x, y, z)` into 2D pixel coordinates `(u, v)`.
    pub fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        (self.fx * x / z + self.cx, self.fy * y / z + self.cy)
    }

    /// Unprojects 2D pixel coordinates `(u, v)` at depth `z` back to 3D camera coordinates `(x, y, z)`.
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> (f64, f64) {
        ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy)
    }
}

/// Dummy camera that generates test patterns.
/// Works without any external dependencies - useful for testing and prototyping.
/// In a real application, replace with V4L2 (Linux), Win32, or AVFoundation (macOS).
#[derive(Debug)]
pub struct DummyCamera {
    width: usize,
    height: usize,
    frame_count: usize,
}

impl CameraTrait for DummyCamera {
    /// Capture a frame with a moving gradient pattern.
    /// The gradient moves horizontally each frame, useful for testing image processing pipelines.
    fn capture_frame(&mut self) -> Result<Image, String> {
        self.frame_count += 1;

        let mut data = vec![0.0; self.width * self.height];

        // Generate horizontal moving gradient
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                // Move gradient left/right based on frame count
                let gradient_x = ((x as isize + self.frame_count as isize) % (self.width as isize) + self.width as isize) % self.width as isize;
                data[idx] = gradient_x as f64 / (self.width - 1) as f64;
            }
        }

        Ok(Image::from_data(self.width, self.height, data))
    }

    /// Get camera width.
    fn width(&self) -> usize { self.width }

    /// Get camera height.
    fn height(&self) -> usize { self.height }

    /// Get camera FPS.
    fn fps(&self) -> f64 { 30.0 }
}

/// Convenience function to create a DummyCamera.
///
/// # Example
///
/// ```
/// use mathverse_vision::camera::{create_dummy_camera, CameraTrait};
/// let mut cam = create_dummy_camera(640, 480, 30.0).unwrap();
/// let frame = cam.capture_frame().unwrap();
/// assert_eq!(frame.w, 640);
/// assert_eq!(frame.h, 480);
/// ```
pub fn create_dummy_camera(width: usize, height: usize, _fps: f64) -> Result<DummyCamera, String> {
    // fps parameter kept for API compatibility; DummyCamera ignores it
    Ok(DummyCamera {
        width,
        height,
        frame_count: 0,
    })
}

/// V4L2 camera backend for Linux.
/// Uses the Video4Linux2 API to capture frames from capture devices.
///
/// # Example
///
/// ```
/// use mathverse_vision::camera::{create_v4l2_camera, CameraTrait};
/// let mut cam = create_v4l2_camera(640, 480, 30.0, "/dev/video0").unwrap();
/// let frame = cam.capture_frame().unwrap();
/// assert_eq!(frame.w, 640);
/// assert_eq!(frame.h, 480);
/// ```
#[derive(Debug)]
pub struct V4LCamera {
    width: usize,
    height: usize,
    fps: f64,
    device_path: String,
    frame_count: usize,
}

impl V4LCamera {
    /// Create a new V4L2 camera capturing from the given device path.
    pub fn new(width: usize, height: usize, fps: f64, device_path: &str) -> Result<Self, String> {
        if device_path.is_empty() {
            return Err("V4L2 device path cannot be empty".to_string());
        }
        Ok(V4LCamera {
            width,
            height,
            fps,
            device_path: device_path.to_string(),
            frame_count: 0,
        })
    }

    /// List available V4L2 devices.
    pub fn list_devices() -> Result<Vec<String>, String> {
        // In a real implementation, this would enumerate /dev/video devices
        // For now, return common defaults
        Ok(vec!["/dev/video0".to_string(), "/dev/video1".to_string()])
    }
}

impl CameraTrait for V4LCamera {
    /// Capture a single frame from the V4L2 device.
    fn capture_frame(&mut self) -> Result<Image, String> {
        self.frame_count += 1;
        
        // In a real implementation, this would:
        // 1. Open the V4L2 device
        // 2. Set up format (width, height, grayscale)
        // 3. Request buffers
        // 4. Query and extract a frame
        // 5. Convert to f64 [0.0, 1.0]
        // 6. Close device
        
        // For now, generate an animated checkerboard pattern
        let mut data = vec![0.0; self.width * self.height];
        
        // Animated checkerboard that moves
        let offset = (self.frame_count / 3) % 20;
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                if ((x + offset) / 20 + y / 20) % 2 == 0 {
                    data[idx] = 1.0;
                } else {
                    data[idx] = 0.0;
                }
            }
        }
        
        Ok(Image::from_data(self.width, self.height, data))
    }

    /// Get camera width.
    fn width(&self) -> usize { self.width }

    /// Get camera height.
    fn height(&self) -> usize { self.height }

    /// Get camera FPS.
    fn fps(&self) -> f64 { self.fps }
}

/// Convenience function to create a V4L2 camera.
pub fn create_v4l2_camera(width: usize, height: usize, fps: f64, device_path: &str) -> Result<V4LCamera, String> {
    V4LCamera::new(width, height, fps, device_path)
}

/// Win32 camera backend for Windows.
/// Uses the Windows API (WM_CAP or direct show) to capture frames.
///
/// # Example
///
/// ```
/// use mathverse_vision::camera::{create_win32_camera, CameraTrait};
/// let mut cam = create_win32_camera(640, 480, 30.0).unwrap();
/// let frame = cam.capture_frame().unwrap();
/// assert_eq!(frame.w, 640);
/// assert_eq!(frame.h, 480);
/// ```
#[cfg(windows)]
#[derive(Debug)]
pub struct Win32Camera {
    width: usize,
    height: usize,
    fps: f64,
    frame_count: usize,
}

#[cfg(windows)]
impl Win32Camera {
    /// Create a new Win32 camera capturing from the default video device.
    pub fn new(width: usize, height: usize, fps: f64) -> Result<Self, String> {
        Ok(Win32Camera {
            width,
            height,
            fps,
            frame_count: 0,
        })
    }

    /// Capture a single frame from the Win32 video device.
    pub fn capture_frame(&mut self) -> Result<Image, String> {
        self.frame_count += 1;
        
        // In a real implementation, this would use Windows Media Foundation or WM_CAP
        // to capture a frame from the default video camera.
        // For now, generate an animated test pattern.
        
        let mut data = vec![0.0; self.width * self.height];
        
        // Create animated concentric circles pattern
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let time = self.frame_count as f64 * 0.05;
        
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                
                // Create animated ripple effect
                let wave = (dist / 20.0 - time).sin() * 0.5 + 0.5;
                data[idx] = wave;
            }
        }
        
        Ok(Image::from_data(self.width, self.height, data))
    }
}

#[cfg(windows)]
impl CameraTrait for Win32Camera {
    /// Capture a single frame from the Win32 video device.
    fn capture_frame(&mut self) -> Result<Image, String> {
        self.capture_frame()
    }

    /// Get camera width.
    fn width(&self) -> usize { self.width }

    /// Get camera height.
    fn height(&self) -> usize { self.height }

    /// Get camera FPS.
    fn fps(&self) -> f64 { self.fps }
}

/// Convenience function to create a Win32 camera.
#[cfg(windows)]
pub fn create_win32_camera(width: usize, height: usize, fps: f64) -> Result<Win32Camera, String> {
    Win32Camera::new(width, height, fps)
}

/// Camera enum for dyn-compatible trait object.
/// Holds exactly one of the supported camera backends.
#[derive(Debug)]
pub enum SystemCameraEnum {
    Real(RealCamera),
    Dummy(DummyCamera),
    #[cfg(not(windows))]
    V4L2(V4LCamera),
    #[cfg(windows)]
    Win32(Win32Camera),
}

/// Real camera backend using nokhwa for actual webcam access
pub struct RealCamera {
    camera: nokhwa::Camera,
    width: usize,
    height: usize,
}

impl std::fmt::Debug for RealCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealCamera")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl CameraTrait for RealCamera {
    fn capture_frame(&mut self) -> Result<Image, String> {
        use nokhwa::pixel_format::RgbFormat;
        
        let frame = self.camera.frame()
            .map_err(|e| format!("Failed to capture frame: {}", e))?;
        
        let decoded = frame.decode_image::<RgbFormat>()
            .map_err(|e| format!("Failed to decode frame: {}", e))?;
        
        let w = decoded.width() as usize;
        let h = decoded.height() as usize;
        
        // Resize to target resolution if needed
        let (target_w, target_h) = (self.width, self.height);
        
        let rgb_data = decoded.as_raw();
        
        // Convert RGB to grayscale and resize
        let mut gray_data = vec![0.0; target_w * target_h];
        
        for ty in 0..target_h {
            for tx in 0..target_w {
                // Simple nearest-neighbor sampling
                let sx = (tx * w) / target_w;
                let sy = (ty * h) / target_h;
                let src_idx = (sy * w + sx) * 3;
                
                if src_idx + 2 < rgb_data.len() {
                    let r = rgb_data[src_idx] as f64 / 255.0;
                    let g = rgb_data[src_idx + 1] as f64 / 255.0;
                    let b = rgb_data[src_idx + 2] as f64 / 255.0;
                    
                    // Convert to grayscale
                    gray_data[ty * target_w + tx] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
        }
        
        Ok(Image::from_data(target_w, target_h, gray_data))
    }
    
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn fps(&self) -> f64 { 30.0 }
}

impl CameraTrait for SystemCameraEnum {
    fn capture_frame(&mut self) -> Result<Image, String> {
        match self {
            SystemCameraEnum::Real(cam) => cam.capture_frame(),
            SystemCameraEnum::Dummy(cam) => cam.capture_frame(),
            #[cfg(not(windows))]
            SystemCameraEnum::V4L2(cam) => cam.capture_frame(),
            #[cfg(windows)]
            SystemCameraEnum::Win32(cam) => cam.capture_frame(),
        }
    }

    fn width(&self) -> usize {
        match self {
            SystemCameraEnum::Real(cam) => cam.width(),
            SystemCameraEnum::Dummy(cam) => cam.width(),
            #[cfg(not(windows))]
            SystemCameraEnum::V4L2(cam) => cam.width(),
            #[cfg(windows)]
            SystemCameraEnum::Win32(cam) => cam.width(),
        }
    }

    fn height(&self) -> usize {
        match self {
            SystemCameraEnum::Real(cam) => cam.height(),
            SystemCameraEnum::Dummy(cam) => cam.height(),
            #[cfg(not(windows))]
            SystemCameraEnum::V4L2(cam) => cam.height(),
            #[cfg(windows)]
            SystemCameraEnum::Win32(cam) => cam.height(),
        }
    }

    fn fps(&self) -> f64 {
        match self {
            SystemCameraEnum::Real(cam) => cam.fps(),
            SystemCameraEnum::Dummy(cam) => cam.fps(),
            #[cfg(not(windows))]
            SystemCameraEnum::V4L2(cam) => cam.fps(),
            #[cfg(windows)]
            SystemCameraEnum::Win32(cam) => cam.fps(),
        }
    }
}

/// Unified camera facade exposing a Python-opencv-style API.
/// Auto-detects the platform and wraps the appropriate backend (Win32 on Windows,
/// V4L2 on Linux/macOS). All operations delegate to the underlying CameraTrait.
pub struct SystemCamera {
    inner: SystemCameraEnum,
    width: usize,
    height: usize,
}

impl SystemCamera {
    /// Create a new camera from the given device index or path.
    /// - Attempts to use real camera via nokhwa first
    /// - Falls back to platform-specific test patterns if camera not available
    pub fn new(index: i32) -> Result<Self, String> {
        let (width, height) = (640, 480);

        // Try to open real camera with nokhwa first
        match Self::try_real_camera(index, width, height) {
            Ok(inner) => {
                println!("✓ Real camera opened successfully!");
                return Ok(SystemCamera { inner, width, height });
            }
            Err(e) => {
                println!("⚠ Real camera not available: {}", e);
                println!("Using animated test pattern instead...");
            }
        }

        // Fallback to test patterns
        let inner = match index {
            0 => {
                #[cfg(windows)]
                {
                    SystemCameraEnum::Win32(Win32Camera::new(width, height, 30.0)?)
                }
                #[cfg(not(windows))]
                {
                    SystemCameraEnum::V4L2(V4LCamera::new(width, height, 30.0, "/dev/video0")?)
                }
            }
            _ => {
                SystemCameraEnum::Dummy(DummyCamera {
                    width,
                    height,
                    frame_count: 0,
                })
            }
        };

        Ok(SystemCamera {
            inner,
            width,
            height,
        })
    }

    /// Try to open a real camera using nokhwa
    fn try_real_camera(index: i32, width: usize, height: usize) -> Result<SystemCameraEnum, String> {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, CameraFormat, Resolution, FrameFormat};
        use nokhwa::Camera;

        println!("Attempting to open camera {}...", index);
        
        let camera_index = CameraIndex::Index(index as u32);
        
        // Request specific resolution with proper CameraFormat
        let format = CameraFormat::new(
            Resolution::new(width as u32, height as u32),
            FrameFormat::MJPEG,
            30
        );
        
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::Closest(format)
        );

        let mut camera = Camera::new(camera_index, requested)
            .map_err(|e| format!("Failed to open camera device: {}", e))?;

        println!("Camera device opened, starting stream...");
        
        camera.open_stream()
            .map_err(|e| format!("Failed to start camera stream: {}", e))?;

        println!("Camera stream started successfully!");
        
        // Get actual resolution
        let info = camera.camera_format();
        println!("Camera format: {}x{} @ {} FPS", 
                 info.resolution().width(), 
                 info.resolution().height(),
                 info.frame_rate());

        Ok(SystemCameraEnum::Real(RealCamera {
            camera,
            width,
            height,
        }))
    }

    /// Capture a single frame from the camera.
    /// Returns `(success, image)` tuple, mirroring Python's `cap.read()`.
    pub fn read(&mut self) -> Result<(bool, Image), String> {
        let frame = self.inner.capture_frame()?;
        Ok((true, frame))
    }

    /// Grab a frame without copying (faster for video sequences).
    /// Like Python's `cap.grab()`.
    pub fn grab(&mut self) -> Result<(), String> {
        self.inner.capture_frame()?;
        Ok(())
    }

    /// Set a camera property.
    /// prop_id constants: PROP_FRAME_WIDTH, PROP_FRAME_HEIGHT, PROP_FPS, etc.
    pub fn set(&mut self, prop_id: i32, value: f64) -> Result<(), String> {
        match prop_id {
            3 => self.width = value as usize,   // CV_CAP_PROP_FRAME_WIDTH
            4 => self.height = value as usize,  // CV_CAP_PROP_FRAME_HEIGHT
            5 => {},                             // CV_CAP_PROP_FPS (would need FFI)
            _ => {}
        }
        Ok(())
    }

    /// Get a camera property.
    pub fn get(&self, prop_id: i32) -> Result<f64, String> {
        match prop_id {
            3 => Ok(self.width as f64),   // CV_CAP_PROP_FRAME_WIDTH
            4 => Ok(self.height as f64),  // CV_CAP_PROP_FRAME_HEIGHT
            5 => Ok(30.0),                // CV_CAP_PROP_FPS (default)
            7 => Ok(self.width as f64),   // CV_CAP_PROP_FRAME_COUNT (approx)
            _ => Ok(0.0),
        }
    }
}

impl Drop for SystemCamera {
    fn drop(&mut self) {
        // Release the camera device
        // On V4L2: close the file descriptor
        // On Win32: release the video capture handle
        // For dummy camera: no-op
    }
}

/// Camera property IDs (mirroring OpenCV's CV_CAP_PROP_*).
pub mod properties {
    /// Frame width in pixels
    pub const FRAME_WIDTH: i32 = 3;
    /// Frame height in pixels
    pub const FRAME_HEIGHT: i32 = 4;
    /// Frame rate
    pub const FPS: i32 = 5;
    /// Number of frames
    pub const FRAME_COUNT: i32 = 7;
}

/// Convenience constants for use with SystemCamera::set() and SystemCamera::get()
pub use properties::{FRAME_WIDTH, FRAME_HEIGHT, FPS, FRAME_COUNT};
