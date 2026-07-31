# MathVerse Image

A production-grade Rust library for grayscale image processing, providing efficient implementations of common computer vision algorithms including convolution kernels, blur, edge detection (Canny), morphology operations, and geometric transforms.

## Features

- **Core Image Operations**: Convolution, Gaussian blur, box blur, sharpening
- **Edge Detection**: Sobel gradients and Canny edge detection with hysteresis
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Geometric Transforms**: Horizontal flip, 90° rotation, nearest-neighbor resize
- **Histogram Analysis**: 256-bin histogram computation
- **Error Handling**: Comprehensive error types with `thiserror`
- **Zero Dependencies**: Minimal external dependencies for maximum compatibility

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathverse-image = "0.1.0"
```

## Usage

### Basic Image Creation

```rust
use mathverse_image::GrayImage;

// Create a new blank image
let mut img = GrayImage::new(64, 64);

// Create from existing data
let data = vec![0.5; 64 * 64];
let img = GrayImage::from_data(64, 64, data)?;
```

### Image Processing

```rust
use mathverse_image::{GrayImage, box_blur, sharpen};

// Apply box blur
let blurred = box_blur(&img);

// Apply sharpening
let sharpened = sharpen(&img);

// Gaussian blur with custom sigma
let gaussian_blurred = img.gaussian_blur(3, 1.5);
```

### Edge Detection

```rust
use mathverse_image::canny::canny;

// Canny edge detection
let edges = canny(&img, 1.5, 0.05, 0.15);

// Sobel gradients
let (magnitude, direction) = img.sobel();
```

### Morphological Operations

```rust
use mathverse_image::morphology::{binarize, erode, dilate, open, close};

// Binarize image
let binary = binarize(&img, 0.5);

// Erosion and dilation
let eroded = erode(&binary);
let dilated = dilate(&binary);

// Opening and closing
let opened = open(&binary);
let closed = close(&binary);
```

### Geometric Transforms

```rust
// Horizontal flip
let flipped = img.flip_h();

// 90° rotation
let rotated = img.rotate90();

// Resize
let resized = img.resize(128, 128);
```

### Histogram Analysis

```rust
let histogram = img.histogram();
```

## Error Handling

The library uses a comprehensive error type:

```rust
use mathverse_image::{GrayImage, ImageError};

fn process_image() -> Result<(), ImageError> {
    let img = GrayImage::from_data(64, 64, data)?;
    // ... process image
    Ok(())
}
```

## Testing

Run the test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

You may choose either license for your use.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.

## Performance

The library is designed for performance with:
- Efficient row-major data layout
- Separable Gaussian blur implementation
- In-place operations where possible
- Minimal allocations in hot paths

## Roadmap

- [ ] Color image support
- [ ] Additional morphological structuring elements
- [ ] More interpolation methods for resize
- [ ] FFT-based convolution
- [ ] Parallel processing support
