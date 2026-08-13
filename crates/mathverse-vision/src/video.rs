#! Basic video I/O support.

use crate::Image;
use std::fs::File;
use std::io::Write;

/// Video writer that writes a sequence of frames to a simple raw format.
///
/// This is a basic video writer that stores raw grayscale frames without
/// compression. For real applications, use dedicated video libraries like
/// `ffmpeg` or `vapour`.
///
/// # Example
///
/// ```
/// use mathverse_vision::video::VideoWriter;
/// use mathverse_vision::Image;
///
/// let mut writer = VideoWriter::new("output.raw", 640, 480, 30);
/// let img = Image::new(640, 480);
/// writer.write_frame(&img).unwrap();
/// writer.close();
/// ```
pub struct VideoWriter {
    /// Output file handle
    file: File,
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// Frame count
    frame_count: usize,
    /// Frame rate (fps)
    fps: u32,
}

impl VideoWriter {
    /// Creates a new video writer.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `fps` - Frames per second
    ///
    /// # Returns
    ///
    /// A new `VideoWriter` instance
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_vision::video::VideoWriter;
    /// use std::path::Path;
    ///
    /// // Clean up the test file afterward
    /// let _ = std::fs::remove_file("test_output.raw");
    /// let mut writer = VideoWriter::new("test_output.raw", 640, 480, 30);
    /// assert!(Path::new("test_output.raw").exists());
    /// ```
    pub fn new(path: &str, width: usize, height: usize, fps: u32) -> Self {
        let file = File::create(path).expect("Cannot create video file");
        Self {
            file,
            width,
            height,
            frame_count: 0,
            fps,
        }
    }

    /// Writes a frame to the video file.
    ///
    /// The frame data is written as raw grayscale `f64` values.
    ///
    /// # Arguments
    ///
    /// * `img` - The frame to write
    ///
    /// # Returns
    ///
    /// `Ok(())` if the frame was written successfully.
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_vision::{Image, video::VideoWriter};
    /// use std::path::Path;
    ///
/// let mut writer = VideoWriter::new("test.raw", 640, 480, 30);
/// let img = Image::new(640, 480);
/// writer.write_frame(&img).unwrap();
/// writer.close();
/// // Clean up
/// let _ = std::fs::remove_file("test.raw");
    /// ```
    pub fn write_frame(&mut self, img: &Image) -> std::io::Result<()> {
        // Write frame dimensions once (on first frame)
        if self.frame_count == 0 {
            // Write width and height as u32 little-endian
            self.file.write_all(&(self.width as u32).to_le_bytes())?;
            self.file.write_all(&(self.height as u32).to_le_bytes())?;
        }
        
        // Write frame data: width * height f64 values
        let byte_data: Vec<u8> = img.data.iter()
            .map(|&f| f.to_le_bytes())
            .flatten()
            .collect();
        self.file.write_all(&byte_data)?;
        
        self.frame_count += 1;
        Ok(())
    }

    /// Closes the video file.
    ///
    /// # Returns
    ///
    /// The total number of frames written.
    pub fn close(self) -> usize {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn writer_create_and_close() {
        let _ = std::fs::remove_file("test_writer.raw");
        let mut writer = VideoWriter::new("test_writer.raw", 32, 32, 10);
        let img = Image::new(32, 32);
        writer.write_frame(&img).unwrap();
        let frames = writer.close();
        assert_eq!(frames, 1);
        // File should exist
        assert!(Path::new("test_writer.raw").exists());
        // Clean up
        let _ = std::fs::remove_file("test_writer.raw");
    }
}