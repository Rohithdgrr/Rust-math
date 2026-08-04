use alloc::vec::Vec;

/// A compact bitmap tracking which elements are null (1 = null, 0 = valid).
///
/// All elements are valid when the bitmap is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullBitmap {
    bits: Vec<u8>,
    len: usize,
}

impl NullBitmap {
    /// Creates a bitmap of the given length where all elements are valid (non-null).
    #[must_use]
    pub fn all_valid(len: usize) -> Self {
        Self {
            bits: vec![0; Self::byte_len(len)],
            len,
        }
    }

    /// Creates a bitmap of the given length where all elements are null.
    #[must_use]
    pub fn all_null(len: usize) -> Self {
        Self {
            bits: vec![0xFF; Self::byte_len(len)],
            len,
        }
    }

    /// Creates a bitmap from an existing byte vector and length.
    ///
    /// # Panics
    ///
    /// Panics if `bits.len() < Self::byte_len(len)`.
    #[must_use]
    pub fn from_bytes(bits: Vec<u8>, len: usize) -> Self {
        assert!(
            bits.len() >= Self::byte_len(len),
            "bitmap bytes too short: need {} bytes for {len} bits, got {}",
            Self::byte_len(len),
            bits.len()
        );
        Self { bits, len }
    }

    /// Creates a bitmap from a slice of booleans (`true` = null).
    #[must_use]
    pub fn from_bools(nulls: &[bool]) -> Self {
        let len = nulls.len();
        let mut bm = Self::all_valid(len);
        for (i, &is_null) in nulls.iter().enumerate() {
            if is_null {
                bm.set_null(i);
            }
        }
        bm
    }

    /// Returns the number of elements tracked by the bitmap.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitmap tracks zero elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the element at `index` is null.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len`.
    #[must_use]
    pub fn is_null(&self, index: usize) -> bool {
        assert!(index < self.len, "index {index} out of bounds for length {}", self.len);
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        (self.bits[byte_idx] >> bit_idx) & 1 == 1
    }

    /// Returns `true` if the element at `index` is valid (non-null).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len`.
    #[must_use]
    pub fn is_valid(&self, index: usize) -> bool {
        !self.is_null(index)
    }

    /// Sets the element at `index` to null.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len`.
    pub fn set_null(&mut self, index: usize) {
        assert!(index < self.len, "index {index} out of bounds for length {}", self.len);
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    /// Sets the element at `index` to valid (non-null).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len`.
    pub fn set_valid(&mut self, index: usize) {
        assert!(index < self.len, "index {index} out of bounds for length {}", self.len);
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        self.bits[byte_idx] &= !(1 << bit_idx);
    }

    /// Returns the number of null elements.
    #[must_use]
    pub fn null_count(&self) -> usize {
        let mut count = 0;
        for &byte in &self.bits {
            count += byte.count_ones() as usize;
        }
        // The last byte may have extra bits set to 1 beyond `len`.
        // Those should not count.
        let total_bits = self.bits.len() * 8;
        let extra = total_bits - self.len;
        if extra > 0 && !self.bits.is_empty() {
            let last = self.bits.last().expect("bitmap is non-empty");
            let mask = (1u8 << (8 - extra)) - 1;
            count -= (last & !mask).count_ones() as usize;
        }
        count
    }

    /// Returns `true` if all elements are valid (no nulls).
    #[must_use]
    pub fn all_valid_flag(&self) -> bool {
        self.null_count() == 0
    }

    /// Returns the raw byte representation of the bitmap.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bits
    }

    /// Computes the number of bytes needed for `n` bits.
    const fn byte_len(n: usize) -> usize {
        (n + 7) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid_creates_empty_bitmap() {
        let bm = NullBitmap::all_valid(10);
        assert_eq!(bm.len(), 10);
        assert_eq!(bm.null_count(), 0);
        for i in 0..10 {
            assert!(bm.is_valid(i));
        }
    }

    #[test]
    fn all_null_creates_all_null_bitmap() {
        let bm = NullBitmap::all_null(5);
        assert_eq!(bm.null_count(), 5);
        for i in 0..5 {
            assert!(bm.is_null(i));
        }
    }

    #[test]
    fn from_bools() {
        let bm = NullBitmap::from_bools(&[false, true, false, true, false]);
        assert_eq!(bm.null_count(), 2);
        assert!(bm.is_valid(0));
        assert!(bm.is_null(1));
        assert!(bm.is_valid(2));
        assert!(bm.is_null(3));
        assert!(bm.is_valid(4));
    }

    #[test]
    fn set_null_and_valid() {
        let mut bm = NullBitmap::all_valid(8);
        bm.set_null(3);
        assert!(bm.is_null(3));
        assert_eq!(bm.null_count(), 1);
        bm.set_valid(3);
        assert!(bm.is_valid(3));
        assert_eq!(bm.null_count(), 0);
    }

    #[test]
    fn large_bitmap() {
        let mut bm = NullBitmap::all_valid(100);
        bm.set_null(0);
        bm.set_null(63);
        bm.set_null(99);
        assert_eq!(bm.null_count(), 3);
        assert!(bm.is_null(0));
        assert!(bm.is_null(63));
        assert!(bm.is_null(99));
    }
}
