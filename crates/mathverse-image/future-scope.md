# MathVerse Image - Future Scope

This document outlines planned enhancements for the `mathverse-image` crate, organized by priority and use case. Features are categorized for ML/YOLO, advanced computer vision, and general image processing.

---

## 🎯 ML / YOLO / AI / Object Detection (18 Core Features)

These features are critical for machine learning pipelines, OpenCV compatibility, YOLO input preprocessing, and AI image recognition systems.

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | `gaussian_blur(r, sigma)` | Noise reduction; antialiasing before downsampling; sigma-controlled Gaussian kernel | ✅ Implemented |
| 2 | `sobel()` → `(magnitude, direction)` | Gradient magnitude + direction for feature extraction; edge detection precursor | ✅ Implemented |
| 3 | `canny(img, sigma, low, high)` | Full Canny pipeline: Gaussian → Sobel → NMS → double threshold → hysteresis | ✅ Implemented |
| 4 | `threshold(t)` | Simple binary thresholding; mask creation; background/foreground separation | ✅ Implemented |
| 5 | `adaptive_threshold(block_size, c)` | Local-mean adaptive threshold; handles uneven lighting; OCR/text processing | ✅ Implemented |
| 6 | `normalize()` | **Essential**: Normalize pixel values to [0, 1] range for consistent model input | ✅ Implemented |
| 7 | `mean()` | Calculate dataset pixel mean; required for mean subtraction normalization | ✅ Implemented |
| 8 | `std_dev()` | Calculate dataset pixel standard deviation; for scaling/normalization | ✅ Implemented |
| 9 | `min_value()` / `max_value()` | Dynamic range detection; clipping threshold identification | ✅ Implemented |
| 10 | `contrast_stretch(low, high)` | Enhance low-contrast images; improve feature visibility for models | ✅ Implemented |
| 11 | `add_gaussian_noise(mean, std_dev)` | Data augmentation; simulate sensor noise; robustness training | ✅ Implemented |
| 12 | `add_salt_pepper_noise(density)` | Data augmentation; impulse noise simulation; outlier robustness | ✅ Implemented |
| 13 | `invert()` | Create complementary masks; generate negative samples; bitwise operations | ✅ Implemented |
| 14 | `gamma_correction(gamma)` | Brightness variation augmentation; simulate different lighting conditions | ✅ Implemented |
| 15 | `resize(nw, nh)` | **Critical**: Resize input to model's expected input size (e.g., 640×640 for YOLO) | ✅ Implemented |
| 16 | `flip_h()` | Data augmentation; horizontal flipping for symmetry; bounding box flip | ✅ Implemented |
| 17 | `rotate90()` | 90° rotation augmentation; orientation variation; simplified rotation | ✅ Implemented |
| 18 | `histogram()` → `[usize; 256]` | Image analysis; pixel distribution diagnosis; exposure assessment | ✅ Implemented |

---

## 🚀 Advanced Computer Vision Features (9 Planned Enhancements)

These features extend beyond basic preprocessing into proper CV algorithms, enabling higher-level perception tasks.

| # | Feature | Description | Priority |
|---|---------|-------------|----------|
| 1 | **Gaussian Pyramid** | Multi-scale image representation with successive Gaussian downsampling; octaves for scale-invariant detection; enables SIFT-like features. | High |
| 2 | **Harris Corner Detector** | Keypoint detection using second-moment matrix: `R = det(M) - k·trace(M)²`; returns corner coordinates with quality scores. | High |
| 3 | **Template Matching** | Normalized Cross-Correlation (NCC) for finding best template match within image; confidence score + location; useful for logo/object recognition. | Medium |
| 4 | **Dense Optical Flow** | Lucas-Kanade method with Gaussian pyramid; returns `(u, v)` vector field per pixel; basis for video stabilization, motion analysis. | Medium |
| 5 | **Hough Line Transform** | Standard Hough Transform with Rho/Theta parameterization; accumulator voting for line detection; returns `(rho, theta)` pairs. | Medium |
| 6 | **8-Connectivity Connected Components** | Enhance morphology module: 8-connectivity (vs current 4); bounding boxes, area, centroid per component; filter by min/max area. | Low |
| 7 | **Region Growing Segmentation** | Seed-based iterative region growing; stop on region mean difference threshold; returns segmented region mask. | Low |
| 8 | **Bicubic Resize** | Cubic convolution interpolation upgrade from bilinear; higher quality for print-quality downscaling/upsampling; ~2× slower but superior quality. | Low |
| 9 | **Integral Image Optimization** | O(1) box blur via summed-area table; already partially implemented; full optimization with edge-aware filtering. | Low |

---

## 📦 Feature Implementation Roadmap

### Phase 1: High-Priority ML Foundation (Weeks 1-2)
- [ ] Gaussian Pyramid (multi-scale representation)
- [ ] Harris Corner Detector (keypoint extraction)
- [ ] Template Matching (object location)

### Phase 2: Medium-Priority Advanced CV (Weeks 3-4)
- [ ] Dense Optical Flow (motion estimation)
- [ ] Hough Line Transform (line detection)
- [ ] 8-Connectivity CC (morphology enhancement)

### Phase 3: Low-Priority Quality-of-Life (Weeks 5-6)
- [ ] Region Growing Segmentation
- [ ] Bicubic Resize (quality upgrade)
- [ ] Integral Image Full Optimization

---

## 🛠️ Development Notes

### API Design Principles
- **Immutable operations**: All methods return new `GrayImage`, source unchanged
- **Consistent verbs**: `solve()`, `evaluate()`, `transform()` across domains
- **f64 precision**: [0, 1] range for mathematical accuracy in ML pipelines
- **No unsafe**: `#![forbid(unsafe_code)]` workspace-wide
- **SIMD-ready**: Hot paths designed for vectorization via `std::simd`

### Dependencies May Need
- `euclid` or similar for point/rect types (Harris corner)
- `paste` or custom acculumator for Hough Transform
- Additional rand features for seeded noise consistency

### Testing Strategy
- Add 5+ new tests per feature
- Compare output against OpenCV reference implementations where possible
- Property-based tests for numerical stability
- Integration tests with full pipeline (load → preprocess → detect → resize)

---

## 🚧 Current Status

| Category | Features Implemented | Features Planned |
|----------|---------------------|------------------|
| **ML/YOLO Core** | 18/18 ✅ | 0 |
| **Advanced CV** | 0/9 ❌ | 9 |
| **Total** | 18/27 ❌ | 9 |

*The crate already implements all 18 ML/YOLO core features. The 9 advanced CV features are planned for future implementation.*

---

## 📬 Contributing New Features

See `CONTRIBUTING.md` for workflow. New features typically follow this pattern:

1. Add module in `src/` (e.g., `harris.rs`, `optical_flow.rs`)
2. Add `pub mod` in `lib.rs`
3. Implement core algorithm
4. Add `#[cfg(test)]` module with 3-5 tests
5. Update `future-scope.md`
6. Add example in `examples/`
7. Run `cargo test -p mathverse-image`