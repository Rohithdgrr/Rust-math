#! 8-Connectivity Connected Components
//!
//! Detects and labels connected components in binary images using 8-connectivity.
//! A pixel is connected to its 8 neighbors (horizontal, vertical, and diagonal).
//!
//! # Algorithm
//!
//! 1. Scan image pixels in row-major order
//! 2. When an unvisited foreground pixel (value > 0) is found, initiate flood fill
//! 3. Use 8-connectivity: a pixel is connected to its 8 surrounding neighbors
//!    - North, South, East, West, and the 4 diagonals
//! 4. Assign a unique component label to each connected region
//! 5. Track component statistics: area, bounding box, centroid
//! 6. Return labeled image and component list
//!
//! # Typical Usage
//!
//! ```rust
//! use mathverse_image::morphology::connected_components_8;
//! use mathverse_image::GrayImage;
//!
//! let mut img = GrayImage::new(64, 64).unwrap();
//! // Create two white squares
//! for y in 0..64 {
//!     for x in 0..64 {
//!         let in_square1 = (x > 10 && x < 20 && y > 10 && y < 20);
//!         let in_square2 = (x > 40 && x < 50 && y > 40 && y < 50);
//!         img.set(x, y, if in_square1 || in_square2 { 1.0 } else { 0.0 });
!     }
//! }
//! // Label connected components
//! let (labeled, components) = connected_components_8(&img);
//! // components[0] = Component { label: 1, area: 100, bbox: (11, 11, 20, 20), centroid: (15.5, 15.5) }
//! println!("Found {} components", components.len());
//! ```
//!
//! # Returns
//!
//! `(labeled_img, components)` where:
//! - `labeled_img`: GrayImage with each pixel labeled with its component number (1-indexed,
//!   0 = background)
//! - `components`: Vec<Component> with statistics for each connected region