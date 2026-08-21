//! Drawing primitives on images.

use crate::Image;

/// Draws a line on an image between two points.
///
/// The line is drawn using a specified color and thickness.
/// Pixel values are overwritten; no blending is performed.
///
/// # Arguments
///
/// * `img` - Mutable reference to the image to draw on
/// * `start` - Starting point (x, y)
/// * `end` - Ending point (x, y)
/// * `color` - Line color as an f64 value (for grayscale)
/// * `thickness` - Line thickness in pixels (default 1)
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, drawing::line};
///
/// let mut img = Image::new(10, 10);
/// line(&mut img, (1, 1), (9, 9), 1.0, 1);
/// ```
pub fn line(img: &mut Image, start: (usize, usize), end: (usize, usize), color: f64, thickness: usize) {
    let (x0, y0) = (start.0 as i32, start.1 as i32);
    let (x1, y1) = (end.0 as i32, end.1 as i32);
    // Bresenham's line algorithm, all octants.
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);

    loop {
        // Draw thick line by drawing multiple pixels offset from the center.
        let t = thickness as i32;
        for ty in -t..=t {
            for tx in -t..=t {
                let px = x + tx;
                let py = y + ty;
                if px >= 0 && px < img.w as i32 && py >= 0 && py < img.h as i32 {
                    let idx = (py as usize * img.w) + (px as usize);
                    if idx < img.data.len() {
                        img.data[idx] = color;
                    }
                }
            }
        }

        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draws a rectangle on an image.
///
/// The rectangle outline is drawn using the specified color and thickness.
/// The rectangle includes the top-left corner and extends rightwards and downwards.
///
/// # Arguments
///
/// * `img` - Mutable reference to the image to draw on
/// * `top_left` - Top-left corner (x, y)
/// * `bottom_right` - Bottom-right corner (x, y)
/// * `color` - Rectangle color as an f64 value
/// * `thickness` - Outline thickness in pixels (default 1)
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, drawing::rectangle};
///
/// let mut img = Image::new(20, 20);
/// rectangle(&mut img, (2, 2), (18, 18), 1.0, 2);
/// ```
pub fn rectangle(img: &mut Image, top_left: (usize, usize), bottom_right: (usize, usize), color: f64, thickness: usize) {
    let (x0, y0) = top_left;
    let (x1, y1) = bottom_right;
    
    // Draw top and bottom edges
    line(img, (x0, y0), (x1, y0), color, thickness);
    line(img, (x0, y1), (x1, y1), color, thickness);
    
    // Draw left and right edges
    line(img, (x0, y0), (x0, y1), color, thickness);
    line(img, (x1, y0), (x1, y1), color, thickness);
}

/// Draws a circle on an image.
///
/// The circle is drawn using the midpoint circle algorithm.
/// The outline is drawn using the specified color and thickness.
///
/// # Arguments
///
/// * `img` - Mutable reference to the image to draw on
/// * `center` - Circle center (x, y)
/// * `radius` - Circle radius in pixels
/// * `color` - Circle color as an f64 value
/// * `thickness` - Outline thickness in pixels (default 1)
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, drawing::circle};
///
/// let mut img = Image::new(20, 20);
/// circle(&mut img, (10, 10), 8, 1.0, 2);
/// ```
pub fn circle(img: &mut Image, center: (usize, usize), radius: usize, color: f64, thickness: usize) {
    let cx = center.0 as i32;
    let cy = center.1 as i32;
    let r = radius as i32;
    
    // Midpoint circle algorithm
    let mut d = 1 - r;
    let mut ddx = 1;
    let mut ddy = -2 * r;
    let mut x = 0i32;
    let mut y = r;
    
    // Draw initial octant
    _draw_circle_octants(img, cx, cy, x, y, color, thickness);
    
    while x < y {
        if d < 0 {
            d += ddx;
            ddx += 2;
            x += 1;
        } else {
            d += ddx + ddy;
            ddx += 2;
            ddy += 2;
            x += 1;
            y -= 1;
        }
        _draw_circle_octants(img, cx, cy, x, y, color, thickness);
    }
}

fn _draw_circle_octants(img: &mut Image, cx: i32, cy: i32, x: i32, y: i32, color: f64, thickness: usize) {
    // All 8 octants
    draw_point_circle(img, cx + x, cy + y, color, thickness);
    draw_point_circle(img, cx - x, cy + y, color, thickness);
    draw_point_circle(img, cx + x, cy - y, color, thickness);
    draw_point_circle(img, cx - x, cy - y, color, thickness);
    draw_point_circle(img, cx + y, cy + x, color, thickness);
    draw_point_circle(img, cx - y, cy + x, color, thickness);
    draw_point_circle(img, cx + y, cy - x, color, thickness);
    draw_point_circle(img, cx - y, cy - x, color, thickness);
}

fn draw_point_circle(img: &mut Image, x: i32, y: i32, color: f64, thickness: usize) {
    if x >= 0 && x < img.w as i32 && y >= 0 && y < img.h as i32 {
        let idx = (y as usize * img.w) + (x as usize);
        if idx < img.data.len() {
            // Draw a small filled area for thickness
            let t = thickness;
            for ty in -(t as i32)..=(t as i32) {
                for tx in -(t as i32)..=(t as i32) {
                    let px = x + tx;
                    let py = y + ty;
                    if px >= 0 && px < img.w as i32 && py >= 0 && py < img.h as i32 {
                        let nidx = (py as usize * img.w) + (px as usize);
                        if nidx < img.data.len() {
                            img.data[nidx] = color;
                        }
                    }
                }
            }
        }
    }
}

/// Draws a filled rectangle.
///
/// The rectangle includes both corners. Equivalent to `cv2.rectangle` with
/// `thickness = -1`.
pub fn fill_rect(img: &mut Image, top_left: (usize, usize), bottom_right: (usize, usize), color: f64) {
    let (x0, y0) = top_left;
    let (x1, y1) = bottom_right;
    for y in y0.min(y1)..=y0.max(y1) {
        for x in x0.min(x1)..=x0.max(x1) {
            if x < img.w && y < img.h {
                img.data[y * img.w + x] = color;
            }
        }
    }
}

/// Draws a polyline through `points`.
///
/// When `closed` is true the last point is connected back to the first.
/// Equivalent to `cv2.polylines`.
pub fn polylines(
    img: &mut Image,
    points: &[(usize, usize)],
    closed: bool,
    color: f64,
    thickness: usize,
) {
    if points.len() < 2 {
        if points.len() == 1 {
            let (x, y) = points[0];
            if x < img.w && y < img.h {
                img.data[y * img.w + x] = color;
            }
        }
        return;
    }
    for pair in points.windows(2) {
        line(img, pair[0], pair[1], color, thickness);
    }
    if closed {
        line(img, *points.last().unwrap(), points[0], color, thickness);
    }
}

/// Fills a polygon defined by `points` using the even-odd scanline rule.
///
/// Equivalent to `cv2.fillPoly`. The polygon may be concave.
pub fn fill_poly(img: &mut Image, points: &[(usize, usize)], color: f64) {
    if points.len() < 3 {
        return;
    }
    // For each scanline, collect edge crossings and fill between pairs.
    let pts: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    for y in 0..img.h {
        let yf = y as f64;
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..pts.len() {
            let (x1, y1) = pts[i];
            let (x2, y2) = pts[(i + 1) % pts.len()];
            if (y1 <= yf && yf < y2) || (y2 <= yf && yf < y1) {
                let t = (yf - y1) / (y2 - y1);
                xs.push(x1 + t * (x2 - x1));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks(2) {
            if pair.len() != 2 {
                break;
            }
            let (xa, xb) = (pair[0].ceil() as usize, pair[1].floor() as usize);
            for x in xa..=xb.min(img.w - 1) {
                img.data[y * img.w + x] = color;
            }
        }
    }
}

/// Draws an ellipse outline by sampling 72 points and connecting them.
///
/// `axes` is `(radius_x, radius_y)`; `angle_deg` rotates the ellipse (in
/// degrees, counter-clockwise). Equivalent to `cv2.ellipse`.
pub fn ellipse(
    img: &mut Image,
    center: (usize, usize),
    axes: (usize, usize),
    angle_deg: f64,
    color: f64,
    thickness: usize,
) {
    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let (rx, ry) = (axes.0.max(1) as f64, axes.1.max(1) as f64);
    let a = angle_deg.to_radians();
    let (cos_a, sin_a) = (a.cos(), a.sin());
    let mut pts = Vec::with_capacity(72);
    for i in 0..72 {
        let t = i as f64 * std::f64::consts::TAU / 72.0;
        let (ex, ey) = (rx * t.cos(), ry * t.sin());
        let x = (cx + ex * cos_a - ey * sin_a).round() as i64;
        let y = (cy + ex * sin_a + ey * cos_a).round() as i64;
        if x >= 0 && y >= 0 && x < img.w as i64 && y < img.h as i64 {
            pts.push((x as usize, y as usize));
        }
    }
    polylines(img, &pts, true, color, thickness);
}

/// Draws a filled ellipse by filling the sampled outline polygon.
///
/// Equivalent to `cv2.ellipse` with `thickness = -1`.
pub fn fill_ellipse(img: &mut Image, center: (usize, usize), axes: (usize, usize), angle_deg: f64, color: f64) {
    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let (rx, ry) = (axes.0.max(1) as f64, axes.1.max(1) as f64);
    let a = angle_deg.to_radians();
    let (cos_a, sin_a) = (a.cos(), a.sin());
    let mut pts = Vec::with_capacity(72);
    for i in 0..72 {
        let t = i as f64 * std::f64::consts::TAU / 72.0;
        let (ex, ey) = (rx * t.cos(), ry * t.sin());
        let x = (cx + ex * cos_a - ey * sin_a).round() as i64;
        let y = (cy + ex * sin_a + ey * cos_a).round() as i64;
        if x >= 0 && y >= 0 && x < img.w as i64 && y < img.h as i64 {
            pts.push((x as usize, y as usize));
        }
    }
    fill_poly(img, &pts, color);
}

/// Renders text with a built-in 5×7 bitmap font.
///
/// `org` is the **bottom-left** corner of the text (OpenCV `putText`
/// convention). Supported characters are ASCII `32..=126`; unsupported
/// characters are skipped. `scale` scales the glyph size (1 = 5×7 pixels),
/// and `thickness > 1` renders bolder strokes. Pixels outside the image are
/// clipped.
pub fn put_text(img: &mut Image, text: &str, org: (usize, usize), scale: usize, color: f64, thickness: usize) {
    let scale = scale.max(1);
    let thickness = thickness.max(1);
    let glyph_w = 5 * scale;
    let advance = glyph_w + scale; // 1-pixel gap between glyphs.
    let (org_x, org_y) = (org.0 as i64, org.1 as i64);

    for (ch_idx, ch) in text.chars().enumerate() {
        let c = ch as usize;
        if !(32..=126).contains(&c) {
            continue;
        }
        let glyph = FONT_5X7[c - 32];
        let base_x = org_x + (ch_idx * advance) as i64;
        // Bottom-left origin: glyph row 0 is the top, so row r sits at
        // org_y − (6 − r) · scale … we draw from the glyph's top row down.
        let top = org_y - 7 * scale as i64;
        for col in 0..5usize {
            let byte = glyph[col];
            for row in 0..7usize {
                if byte & (1 << row) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = base_x + (col * scale + sx) as i64;
                            let py = top + (row * scale + sy) as i64;
                            if px >= 0 && py >= 0 && px < img.w as i64 && py < img.h as i64 {
                                // Bold strokes: fill a small neighborhood.
                                for ty in -(thickness as i64 / 2)..=(thickness as i64 / 2) {
                                    for tx in -(thickness as i64 / 2)..=(thickness as i64 / 2) {
                                        let bx = px + tx;
                                        let by = py + ty;
                                        if bx >= 0 && by >= 0 && bx < img.w as i64 && by < img.h as i64 {
                                            img.data[by as usize * img.w + bx as usize] = color;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 5×7 bitmap font for ASCII 32..=126 (classic `font5x7` table). Each glyph is
/// five column bytes; bit `r` (0 = LSB) is the `r`-th row from the top.
const FONT_5X7: [[u8; 5]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00], // 32 ' '
    [0x00, 0x00, 0x5F, 0x00, 0x00], // 33 '!'
    [0x00, 0x07, 0x00, 0x07, 0x00], // 34 '"'
    [0x14, 0x7F, 0x14, 0x7F, 0x14], // 35 '#'
    [0x24, 0x2A, 0x7F, 0x2A, 0x12], // 36 '$'
    [0x23, 0x13, 0x08, 0x64, 0x62], // 37 '%'
    [0x36, 0x49, 0x55, 0x22, 0x50], // 38 '&'
    [0x00, 0x05, 0x03, 0x00, 0x00], // 39 '''
    [0x00, 0x1C, 0x22, 0x41, 0x00], // 40 '('
    [0x00, 0x41, 0x22, 0x1C, 0x00], // 41 ')'
    [0x08, 0x2A, 0x1C, 0x2A, 0x08], // 42 '*'
    [0x08, 0x08, 0x3E, 0x08, 0x08], // 43 '+'
    [0x00, 0x50, 0x30, 0x00, 0x00], // 44 ','
    [0x08, 0x08, 0x08, 0x08, 0x08], // 45 '-'
    [0x00, 0x60, 0x60, 0x00, 0x00], // 46 '.'
    [0x20, 0x10, 0x08, 0x04, 0x02], // 47 '/'
    [0x3E, 0x51, 0x49, 0x45, 0x3E], // 48 '0'
    [0x00, 0x42, 0x7F, 0x40, 0x00], // 49 '1'
    [0x42, 0x61, 0x51, 0x49, 0x46], // 50 '2'
    [0x21, 0x41, 0x45, 0x4B, 0x31], // 51 '3'
    [0x18, 0x14, 0x12, 0x7F, 0x10], // 52 '4'
    [0x27, 0x45, 0x45, 0x45, 0x39], // 53 '5'
    [0x3C, 0x4A, 0x49, 0x49, 0x30], // 54 '6'
    [0x01, 0x71, 0x09, 0x05, 0x03], // 55 '7'
    [0x36, 0x49, 0x49, 0x49, 0x36], // 56 '8'
    [0x06, 0x49, 0x49, 0x29, 0x1E], // 57 '9'
    [0x00, 0x36, 0x36, 0x00, 0x00], // 58 ':'
    [0x00, 0x56, 0x36, 0x00, 0x00], // 59 ';'
    [0x00, 0x08, 0x14, 0x22, 0x41], // 60 '<'
    [0x14, 0x14, 0x14, 0x14, 0x14], // 61 '='
    [0x41, 0x22, 0x14, 0x08, 0x00], // 62 '>'
    [0x02, 0x01, 0x51, 0x09, 0x06], // 63 '?'
    [0x32, 0x49, 0x79, 0x41, 0x3E], // 64 '@'
    [0x7E, 0x11, 0x11, 0x11, 0x7E], // 65 'A'
    [0x7F, 0x49, 0x49, 0x49, 0x36], // 66 'B'
    [0x3E, 0x41, 0x41, 0x41, 0x22], // 67 'C'
    [0x7F, 0x41, 0x41, 0x22, 0x1C], // 68 'D'
    [0x7F, 0x49, 0x49, 0x49, 0x41], // 69 'E'
    [0x7F, 0x09, 0x09, 0x01, 0x01], // 70 'F'
    [0x3E, 0x41, 0x41, 0x51, 0x32], // 71 'G'
    [0x7F, 0x08, 0x08, 0x08, 0x7F], // 72 'H'
    [0x00, 0x41, 0x7F, 0x41, 0x00], // 73 'I'
    [0x20, 0x40, 0x41, 0x3F, 0x01], // 74 'J'
    [0x7F, 0x08, 0x14, 0x22, 0x41], // 75 'K'
    [0x7F, 0x40, 0x40, 0x40, 0x40], // 76 'L'
    [0x7F, 0x02, 0x04, 0x02, 0x7F], // 77 'M'
    [0x7F, 0x04, 0x08, 0x10, 0x7F], // 78 'N'
    [0x3E, 0x41, 0x41, 0x41, 0x3E], // 79 'O'
    [0x7F, 0x09, 0x09, 0x09, 0x06], // 80 'P'
    [0x3E, 0x41, 0x51, 0x21, 0x5E], // 81 'Q'
    [0x7F, 0x09, 0x19, 0x29, 0x46], // 82 'R'
    [0x46, 0x49, 0x49, 0x49, 0x31], // 83 'S'
    [0x01, 0x01, 0x7F, 0x01, 0x01], // 84 'T'
    [0x3F, 0x40, 0x40, 0x40, 0x3F], // 85 'U'
    [0x1F, 0x20, 0x40, 0x20, 0x1F], // 86 'V'
    [0x7F, 0x20, 0x18, 0x20, 0x7F], // 87 'W'
    [0x63, 0x14, 0x08, 0x14, 0x63], // 88 'X'
    [0x03, 0x04, 0x78, 0x04, 0x03], // 89 'Y'
    [0x61, 0x51, 0x49, 0x45, 0x43], // 90 'Z'
    [0x00, 0x00, 0x7F, 0x41, 0x41], // 91 '['
    [0x02, 0x04, 0x08, 0x10, 0x20], // 92 '\\'
    [0x41, 0x41, 0x7F, 0x00, 0x00], // 93 ']'
    [0x04, 0x02, 0x01, 0x02, 0x04], // 94 '^'
    [0x40, 0x40, 0x40, 0x40, 0x40], // 95 '_'
    [0x00, 0x01, 0x02, 0x04, 0x00], // 96 '`'
    [0x20, 0x54, 0x54, 0x54, 0x78], // 97 'a'
    [0x7F, 0x48, 0x44, 0x44, 0x38], // 98 'b'
    [0x38, 0x44, 0x44, 0x44, 0x20], // 99 'c'
    [0x38, 0x44, 0x44, 0x48, 0x7F], // 100 'd'
    [0x38, 0x54, 0x54, 0x54, 0x18], // 101 'e'
    [0x08, 0x7E, 0x09, 0x01, 0x02], // 102 'f'
    [0x0C, 0x52, 0x52, 0x52, 0x3E], // 103 'g'
    [0x7F, 0x08, 0x04, 0x04, 0x78], // 104 'h'
    [0x00, 0x44, 0x7D, 0x40, 0x00], // 105 'i'
    [0x20, 0x40, 0x44, 0x3D, 0x00], // 106 'j'
    [0x7F, 0x10, 0x28, 0x44, 0x00], // 107 'k'
    [0x00, 0x41, 0x7F, 0x40, 0x00], // 108 'l'
    [0x7C, 0x04, 0x18, 0x04, 0x78], // 109 'm'
    [0x7C, 0x08, 0x04, 0x04, 0x78], // 110 'n'
    [0x38, 0x44, 0x44, 0x44, 0x38], // 111 'o'
    [0x7C, 0x14, 0x14, 0x14, 0x08], // 112 'p'
    [0x08, 0x14, 0x14, 0x18, 0x7C], // 113 'q'
    [0x7C, 0x08, 0x04, 0x04, 0x08], // 114 'r'
    [0x48, 0x54, 0x54, 0x54, 0x20], // 115 's'
    [0x04, 0x3F, 0x44, 0x40, 0x20], // 116 't'
    [0x3C, 0x40, 0x40, 0x20, 0x7C], // 117 'u'
    [0x1C, 0x20, 0x40, 0x20, 0x1C], // 118 'v'
    [0x3C, 0x40, 0x30, 0x40, 0x3C], // 119 'w'
    [0x44, 0x28, 0x10, 0x28, 0x44], // 120 'x'
    [0x0C, 0x50, 0x50, 0x50, 0x3C], // 121 'y'
    [0x44, 0x64, 0x54, 0x4C, 0x44], // 122 'z'
    [0x00, 0x08, 0x36, 0x41, 0x00], // 123 '{'
    [0x00, 0x00, 0x7F, 0x00, 0x00], // 124 '|'
    [0x00, 0x41, 0x36, 0x08, 0x00], // 125 '}'
    [0x08, 0x04, 0x08, 0x10, 0x08], // 126 '~'
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_basic() {
        let mut img = Image::new(10, 10);
        line(&mut img, (1, 1), (9, 9), 1.0, 1);
        // Check that the center pixel was drawn
        assert_eq!(img.data[9 * 10 + 9], 1.0); // (9,9) should be on the line
    }

    #[test]
    fn rectangle_basic() {
        let mut img = Image::new(20, 20);
        rectangle(&mut img, (2, 2), (18, 18), 1.0, 2);
        // Check rectangle corners
        assert_eq!(img.data[2 * 20 + 2], 1.0); // top-left
        assert_eq!(img.data[18 * 20 + 2], 1.0); // bottom-left
        assert_eq!(img.data[2 * 20 + 18], 1.0); // top-right
        assert_eq!(img.data[18 * 20 + 18], 1.0); // bottom-right
    }

    #[test]
    fn circle_basic() {
        let mut img = Image::new(20, 20);
        circle(&mut img, (10, 10), 8, 1.0, 1);
        // Outline points of the radius-8 circle around (10, 10).
        assert_eq!(img.data[10 * 20 + 18], 1.0); // right
        assert_eq!(img.data[10 * 20 + 2], 1.0); // left
        assert_eq!(img.data[2 * 20 + 10], 1.0); // top
        assert_eq!(img.data[18 * 20 + 10], 1.0); // bottom
        // The center is on the outline only for degenerate radii.
        assert_eq!(img.data[10 * 20 + 10], 0.0);
    }

    #[test]
    fn fill_rect_basic() {
        let mut img = Image::new(10, 10);
        fill_rect(&mut img, (2, 2), (5, 4), 1.0);
        assert_eq!(img.data[2 * 10 + 2], 1.0);
        assert_eq!(img.data[4 * 10 + 5], 1.0);
        assert_eq!(img.data[3 * 10 + 3], 1.0);
        assert_eq!(img.data[0 * 10 + 0], 0.0);
    }

    #[test]
    fn polylines_draws_edges() {
        let mut img = Image::new(10, 10);
        let pts = [(1, 1), (8, 1), (8, 8)];
        polylines(&mut img, &pts, true, 1.0, 1);
        assert_eq!(img.data[1 * 10 + 1], 1.0);
        assert_eq!(img.data[1 * 10 + 5], 1.0); // top edge
        assert_eq!(img.data[8 * 10 + 8], 1.0); // closing edge
    }

    #[test]
    fn fill_poly_triangle() {
        let mut img = Image::new(10, 10);
        let pts = [(2, 2), (8, 2), (5, 8)];
        fill_poly(&mut img, &pts, 1.0);
        // Interior near the centroid should be filled.
        assert_eq!(img.data[4 * 10 + 5], 1.0);
        // Outside corners stay empty.
        assert_eq!(img.data[0 * 10 + 0], 0.0);
        assert_eq!(img.data[9 * 10 + 9], 0.0);
    }

    #[test]
    fn ellipse_and_fill() {
        let mut img = Image::new(30, 30);
        ellipse(&mut img, (15, 15), (10, 5), 0.0, 1.0, 1);
        // Rightmost point of the ellipse on the horizontal axis.
        assert_eq!(img.data[15 * 30 + 25], 1.0);
        assert_eq!(img.data[15 * 30 + 5], 1.0);
        let mut filled = Image::new(30, 30);
        fill_ellipse(&mut filled, (15, 15), (10, 5), 0.0, 1.0);
        // Center must be filled.
        assert_eq!(filled.data[15 * 30 + 15], 1.0);
    }

    #[test]
    fn put_text_renders_pixels() {
        let mut img = Image::new(40, 20);
        put_text(&mut img, "A", (5, 19), 1, 1.0, 1);
        // 'A' glyph column 0 is 0x7E (rows 1..6 lit): pixel at (5, 13) lit.
        assert_eq!(img.data[13 * 40 + 5], 1.0);
        // Count lit pixels: should be a non-trivial glyph.
        let lit = img.data.iter().filter(|&&v| v > 0.5).count();
        assert!(lit >= 10, "lit {lit}");
    }
}