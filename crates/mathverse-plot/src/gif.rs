//! A dependency-free GIF89a encoder for animation frames.
//!
//! matplotlib exports animations via ffmpeg; this crate has no native video
//! toolchain, so this module provides the classic animated-GIF path: each
//! frame is a raw RGBA buffer that gets palette-quantized (global palette,
//! ≤256 colors), LZW-compressed, and packed into a GIF89a stream. This is the
//! v1 encoder: no local color tables, no interlacing, no frame disposal
//! optimization — good enough for line/bar animations.

use std::collections::HashMap;

/// One animation frame: `width * height * 4` RGBA bytes and a display delay.
#[derive(Debug, Clone)]
pub struct GifFrame {
    /// RGBA pixel data (`len == width * height * 4`).
    pub rgba: Vec<u8>,
    /// Delay in hundredths of a second.
    pub delay_cs: u16,
}

impl GifFrame {
    /// Create a frame from RGBA bytes.
    pub fn new(rgba: Vec<u8>, delay_cs: u16) -> Self {
        Self { rgba, delay_cs }
    }
}

/// Encode frames into a complete GIF89a byte stream.
///
/// Returns `None` for empty frame lists, mismatched frame sizes, or a frame
/// whose buffer does not match `width * height * 4`.
#[must_use]
pub fn encode_gif(width: u16, height: u16, frames: &[GifFrame]) -> Option<Vec<u8>> {
    if frames.is_empty() {
        return None;
    }
    let w = usize::from(width);
    let h = usize::from(height);
    let pixels = w.checked_mul(h)?;
    if frames.iter().any(|f| f.rgba.len() != pixels * 4) {
        return None;
    }
    // Allow up to ~4 MP per frame (e.g. 2048×2048); GIFs beyond that are
    // impractical anyway.
    if pixels == 0 || pixels > 4_000_000 {
        return None;
    }

    // --- Build a global palette from all frames (≤ 256 colors). ---
    let mut counts: HashMap<(u8, u8, u8), u32> = HashMap::new();
    for f in frames {
        for px in f.rgba.chunks_exact(4) {
            *counts.entry((px[0], px[1], px[2])).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<((u8, u8, u8), u32)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    if ranked.len() > 256 {
        ranked.truncate(256);
    }
    let mut palette: Vec<(u8, u8, u8)> = ranked.into_iter().map(|(c, _)| c).collect();
    while palette.len() < 2 {
        palette.push((0, 0, 0));
    }

    let mut out = Vec::with_capacity(1024 + frames.len() * pixels / 2);
    // Header.
    out.extend_from_slice(b"GIF89a");
    // Logical screen descriptor.
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    // Packed: global color table flag (1), color resolution (7), sort (0),
    // size of GCT (log2(n) - 1). Background index 0, aspect 0.
    let gct_size = (palette.len() as u32)
        .next_power_of_two()
        .trailing_zeros()
        .saturating_sub(1) as u8;
    let packed = 0x80 | 0x70 | gct_size;
    out.push(packed);
    out.push(0);
    out.push(0);
    // Global color table.
    for (r, g, b) in &palette {
        out.push(*r);
        out.push(*g);
        out.push(*b);
    }

    let min_code_size = (2u8).max(gct_size + 1).min(8);

    for frame in frames {
        // Graphic control extension.
        out.push(0x21);
        out.push(0xF9);
        out.push(4);
        out.push(0x00); // no transparency, no disposal
        out.extend_from_slice(&frame.delay_cs.to_le_bytes());
        out.push(0x00); // transparent color index (unused)
        out.push(0x00);

        // Image descriptor.
        out.push(0x2C);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.push(0x00); // no local color table, not interlaced

        // Map pixels to palette indices (nearest color).
        let mut indices = Vec::with_capacity(pixels);
        for px in frame.rgba.chunks_exact(4) {
            let key = (px[0], px[1], px[2]);
            let idx = palette
                .iter()
                .position(|c| *c == key)
                .unwrap_or_else(|| nearest_palette(&palette, key));
            indices.push(idx as u8);
        }

        let compressed = lzw_compress(&indices, min_code_size);
        out.push(min_code_size);
        for chunk in compressed.chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0x00); // block terminator
    }

    out.push(0x3B); // trailer
    Some(out)
}

fn nearest_palette(palette: &[(u8, u8, u8)], c: (u8, u8, u8)) -> usize {
    palette
        .iter()
        .enumerate()
        .min_by_key(|(_, p)| {
            let dr = i32::from(p.0) - i32::from(c.0);
            let dg = i32::from(p.1) - i32::from(c.1);
            let db = i32::from(p.2) - i32::from(c.2);
            dr * dr + dg * dg + db * db
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Pack `code` (at `code_width` bits) into the LSB-first bit buffer.
fn emit_code(code: u16, code_width: u32, bits: &mut u32, n_bits: &mut u32, out: &mut Vec<u8>) {
    *bits |= u32::from(code) << *n_bits;
    *n_bits += code_width;
    while *n_bits >= 8 {
        out.push((*bits & 0xFF) as u8);
        *bits >>= 8;
        *n_bits -= 8;
    }
}

/// GIF LZW compression of an index stream. `min_code_size` is the GIF code
/// size (2..8); the working code width starts one above it and grows to 12.
fn lzw_compress(indices: &[u8], min_code_size: u8) -> Vec<u8> {
    let clear = 1u16 << min_code_size;
    let eoi = clear + 1;
    let mut dict: HashMap<(u16, u8), u16> = HashMap::new();
    let mut next_code = eoi + 1;
    let mut code_width = u32::from(min_code_size) + 1;

    let mut out = Vec::with_capacity(indices.len());
    let mut bits: u32 = 0;
    let mut n_bits: u32 = 0;

    emit_code(clear, code_width, &mut bits, &mut n_bits, &mut out);
    let mut prefix = indices[0] as u16;
    for &c in &indices[1..] {
        let key = (prefix, c);
        if let Some(&code) = dict.get(&key) {
            prefix = code;
        } else {
            emit_code(prefix, code_width, &mut bits, &mut n_bits, &mut out);
            if next_code < 4096 {
                dict.insert(key, next_code);
                next_code += 1;
                if next_code >= (1u16 << code_width) && code_width < 12 {
                    code_width += 1;
                }
            } else {
                // Dictionary full: reset.
                emit_code(clear, code_width, &mut bits, &mut n_bits, &mut out);
                dict.clear();
                next_code = eoi + 1;
                code_width = u32::from(min_code_size) + 1;
            }
            prefix = u16::from(c);
        }
    }
    emit_code(prefix, code_width, &mut bits, &mut n_bits, &mut out);
    emit_code(eoi, code_width, &mut bits, &mut n_bits, &mut out);
    if n_bits > 0 {
        out.push((bits & 0xFF) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 4];
        for px in buf.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = 255;
        }
        buf
    }

    #[test]
    fn header_trailer_and_dimensions() {
        let f1 = GifFrame::new(solid(255, 0, 0, 2, 2), 10);
        let f2 = GifFrame::new(solid(0, 0, 255, 2, 2), 10);
        let gif = encode_gif(2, 2, &[f1, f2]).unwrap();
        assert_eq!(&gif[..6], b"GIF89a");
        assert_eq!(&gif[6..8], &2u16.to_le_bytes());
        assert_eq!(&gif[8..10], &2u16.to_le_bytes());
        assert_eq!(*gif.last().unwrap(), 0x3B);
        // Two frames -> two image separators.
        assert_eq!(gif.iter().filter(|&&b| b == 0x2C).count(), 2);
        // Two graphic control extensions.
        assert_eq!(gif.iter().filter(|&&b| b == 0xF9).count(), 2);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(encode_gif(2, 2, &[]).is_none());
        let bad = GifFrame::new(vec![0; 5], 10);
        assert!(encode_gif(2, 2, &[bad]).is_none());
    }

    #[test]
    fn many_colors_quantized_to_palette() {
        // 300 distinct colors -> palette capped at 256.
        let mut buf = Vec::new();
        for i in 0..300u16 {
            buf.extend_from_slice(&[(i & 0xFF) as u8, (i >> 4) as u8, i as u8, 255]);
        }
        let gif = encode_gif(300, 1, &[GifFrame::new(buf, 5)]).unwrap();
        assert_eq!(*gif.last().unwrap(), 0x3B);
        // Logical screen descriptor claims a 256-entry GCT (packed field
        // bits 0-2 = log2(256)-1 = 7).
        assert_eq!(gif[10] & 0x07, 7);
    }

    #[test]
    fn animation_roundtrip_structure() {
        let w = 4;
        let h = 4;
        let frames: Vec<GifFrame> = (0..3)
            .map(|i| {
                let mut buf = solid(i * 40, 200, 50, w, h);
                buf[0] = 255;
                GifFrame::new(buf, (i + 1) as u16 * 5)
            })
            .collect();
        let gif = encode_gif(w as u16, h as u16, &frames).unwrap();
        assert!(gif.len() > 50);
        assert_eq!(gif.iter().filter(|&&b| b == 0x2C).count(), 3);
    }
}
