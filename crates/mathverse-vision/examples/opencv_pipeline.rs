//! Headless OpenCV-style batch pipeline.
//!
//! Demonstrates the "basic features" of the crate in the order a typical
//! OpenCV tutorial would cover them — no camera or window required:
//!
//! 1. Create a synthetic image and save/load it (`io::imwrite` / `imread`)
//! 2. Color conversions (`rgb_to_gray`, `rgb_to_hsv`)
//! 3. Filters (`box_filter`, `median_blur`, `bilateral_filter`)
//! 4. Arithmetic & bitwise ops (`add_weighted`, `bitwise_and`)
//! 5. Morphology (`erode`, `dilate`, `opening`, `closing`)
//! 6. Thresholding (`binary`, `otsu`)
//! 7. Edge detection (`canny`, `sobel`)
//! 8. Contours, bounding boxes, hulls, approximation
//! 9. Template matching
//! 10. Histogram equalization & normalization
//! 11. Geometric transforms (`flip`, `rotate`, `warp_perspective`, `pyr_down`)
//! 12. Drawing overlays & text (`rectangle`, `put_text`)
//!
//! Output images are written to `output/`.

use mathverse_vision::contours::{approx_poly_dp, bounding_rect, contour_area, convex_hull, find_contours};
use mathverse_vision::drawing::{fill_rect, put_text, rectangle};
use mathverse_vision::filters::{bilateral_filter, box_filter, median_blur};
use mathverse_vision::io::{imread, imwrite, imwrite_color};
use mathverse_vision::morphology::{closing, dilate, erode, kernel_rect, opening};
use mathverse_vision::ops::{canny, histogram_equalize, histogram, normalize_minmax, sobel};
use mathverse_vision::template::{match_template, MatchMethod};
use mathverse_vision::threshold::{binary, otsu};
use mathverse_vision::transform::{flip, pyr_down, rotate, warp_perspective};
use mathverse_vision::utils::min_max;
use mathverse_vision::{Image, arithmetic::add_weighted, color::rgb_to_hsv};

fn main() -> Result<(), String> {
    std::fs::create_dir_all("output").map_err(|e| format!("create output dir: {e}"))?;
    let mut step = 0;
    let mut done = |name: &str, img: &Image| {
        step += 1;
        let path = format!("output/{step:02}_{name}.png");
        imwrite(&path, img)?;
        println!("{step:02}. {name}: {}x{}", img.w, img.h);
        Ok::<(), String>(())
    };

    // --- 1. Synthesize + image I/O --------------------------------------
    let img = synthetic_image(96, 96);
    imwrite("output/01_synthetic.png", &img)?;
    let img = imread("output/01_synthetic.png")?;
    println!("01. create + save + load (io::imwrite/imread): {}x{}", img.w, img.h);

    // --- 2. Color conversions -------------------------------------------
    let rgb = mathverse_vision::color::gray_to_rgb(&img);
    let hsv = rgb_to_hsv(&rgb);
    imwrite_color("output/02_hsv.png", &hsv)?;
    println!("02. rgb_to_hsv: {}x{}", hsv.w, hsv.h);

    // --- 3. Filters ------------------------------------------------------
    done("box_blur", &box_filter(&img, 5))?;
    done("median_blur", &median_blur(&img, 5))?;
    done("bilateral", &bilateral_filter(&img, 7, 0.2, 2.0))?;

    // --- 4. Arithmetic & bitwise -----------------------------------------
    let half = mathverse_vision::arithmetic::multiply_scalar(&img, 0.5);
    let blended = add_weighted(&img, 0.6, &half, 0.4, 0.0);
    done("blend", &blended)?;
    let mask = binary(&img, 0.5, 1.0);
    let masked = mathverse_vision::arithmetic::bitwise_and(&img, &mask);
    done("bitwise_and_mask", &masked)?;

    // --- 5. Morphology ----------------------------------------------------
    let k = kernel_rect();
    done("erode", &erode(&img, &k, 1))?;
    done("dilate", &dilate(&img, &k, 1))?;
    done("opening", &opening(&img, &k))?;
    done("closing", &closing(&img, &k))?;

    // --- 6. Thresholding ---------------------------------------------------
    let (t, otsu_img) = otsu(&img, 1.0);
    done("otsu", &otsu_img)?;
    println!("   otsu threshold: {t:.3}");
    let edges = canny(&img, 0.15, 0.45);
    let bin_edges = binary(&edges, 0.5, 1.0);

    // --- 7. Edge detection -------------------------------------------------
    done("canny", &edges)?;
    let (mag, _dir) = sobel(&img);
    done("sobel", &normalize_minmax(&mag, 0.0, 1.0))?;

    // --- 8. Contours ---------------------------------------------------------
    let contours = find_contours(&bin_edges, 8);
    println!("   find_contours: {} contours", contours.len());
    let mut overlay = img.clone();
    let mut shape_summary = Vec::new();
    for c in &contours {
        let area = contour_area(c);
        if area < 5.0 {
            continue;
        }
        let (x, y, w, h) = bounding_rect(c).unwrap();
        let hull = convex_hull(c);
        let approx = approx_poly_dp(c, 1.0, true);
        shape_summary.push(format!(
            "   contour: area={area:.1}, rect=({x},{y},{w},{h}), hull_pts={}, approx_pts={}",
            hull.len(),
            approx.len()
        ));
        rectangle(&mut overlay, (x, y), (x + w - 1, y + h - 1), 0.0, 1);
    }
    for line in &shape_summary {
        println!("{line}");
    }
    done("contours_overlay", &overlay)?;

    // --- 9. Template matching -----------------------------------------------
    // Crop a template that spans the bright square's corner so it has
    // structure (a flat template matches any flat patch under normalized
    // correlation).
    let templ = mathverse_vision::transform::crop(&img, 8, 8, 20, 20).unwrap();
    imwrite("output/09_template.png", &templ)?;
    let res = match_template(&img, &templ, MatchMethod::CCoeffNormed);
    // min_max returns (min_val, max_val, min_loc, max_loc); for the best match
    // use the max location (4th element).
    let (_, best_val, _, best_loc) = min_max(&res);
    println!("09. match_template (CCOEFF_NORMED): best={best_loc:?} score={best_val:.3}");
    let mut matched = img.clone();
    rectangle(&mut matched, (best_loc.0, best_loc.1), (best_loc.0 + 19, best_loc.1 + 19), 0.0, 2);
    done("match_template", &matched)?;

    // --- 10. Histogram equalization -------------------------------------------
    let dark = mathverse_vision::arithmetic::multiply_scalar(&img, 0.25);
    let eq = histogram_equalize(&dark);
    done("equalized", &eq)?;
    let hist = histogram(&eq);
    let lit: usize = hist.iter().sum();
    println!("   equalized histogram pixels: {lit}");

    // --- 11. Geometric transforms --------------------------------------------
    done("flip_h", &flip(&img, 1))?;
    done("rotate_45", &rotate(&img, 45.0))?;
    done("pyr_down", &pyr_down(&img))?;
    // Perspective "deskew": map the image to a slightly tilted quad.
    let h = [1.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0006, 0.0, 1.0];
    done("warp_perspective", &warp_perspective(&img, &h))?;

    // --- 12. Drawing overlays & text --------------------------------------------
    let mut annotated = img.clone();
    fill_rect(&mut annotated, (4, 4), (64, 40), 0.0);
    put_text(&mut annotated, "FPS: 30", (6, 14), 1, 1.0, 1);
    put_text(&mut annotated, "MODE: BASIC", (6, 38), 1, 1.0, 2);
    done("annotated", &annotated)?;

    println!("\nAll steps finished. Images written to output/");
    Ok(())
}

/// Builds a synthetic test image: gradient background, bright squares, a
/// circle, and noise-free flat regions (good for contours and matching).
fn synthetic_image(w: usize, h: usize) -> Image {
    let mut img = Image::new(w, h);
    // Diagonal gradient background.
    for y in 0..h {
        for x in 0..w {
            img.set(x, y, 0.15 + 0.4 * ((x + y) as f64 / (w + h) as f64));
        }
    }
    // Bright square.
    for y in 10..34 {
        for x in 10..34 {
            img.set(x, y, 0.95);
        }
    }
    // Mid square.
    for y in 60..78 {
        for x in 60..84 {
            img.set(x, y, 0.55);
        }
    }
    // Circle.
    let (cx, cy, r) = (70.0, 24.0, 10.0);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            if (d - r).abs() < 1.0 {
                img.set(x, y, 0.8);
            }
        }
    }
    img
}
