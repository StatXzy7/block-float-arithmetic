//! The explicit 16-entry 4-bit code table and nibble packing.
//!
//! # Code table
//!
//! The low three bits select an unsigned magnitude; the most significant bit
//! is the sign:
//!
//! | code | value | | code | value |
//! |------|-------|-|------|-------|
//! | 0x0  |    +0 | | 0x8  |    -0 |
//! | 0x1  |  +0.5 | | 0x9  |  -0.5 |
//! | 0x2  |    +1 | | 0xA  |    -1 |
//! | 0x3  |  +1.5 | | 0xB  |  -1.5 |
//! | 0x4  |    +2 | | 0xC  |    -2 |
//! | 0x5  |    +3 | | 0xD  |    -3 |
//! | 0x6  |    +4 | | 0xE  |    -4 |
//! | 0x7  |    +6 | | 0xF  |    -6 |
//!
//! Encoding `-0.0` produces `0x0` (write normalization), but decoding `0x8`
//! returns `-0.0`, so the negative-zero encoding is readable.
//!
//! # Nibble packing
//!
//! Codes are packed little nibble first: the even-indexed element occupies the
//! low nibble of a byte and the odd-indexed element occupies the high nibble.
//! For an odd element count, the final high nibble is a zero-padding nibble
//! and MUST be `0x0`; decoders reject a non-zero padding nibble.

use crate::error::{QError, QResult};

/// Magnitude table indexed by the low three bits.
pub const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Sign bit mask (the most significant nibble bit).
pub const SIGN_BIT: u8 = 0x8;

/// Returns the signed value represented by a 4-bit code.
///
/// `0x8` decodes to `-0.0`; `0x0` decodes to `+0.0`.
pub fn decode_nibble(code: u8) -> f64 {
    let mag = MAGNITUDES[(code & 0x7) as usize];
    if code & SIGN_BIT != 0 {
        -mag
    } else {
        mag
    }
}

/// Maps a non-negative magnitude in table units to the low-three-bits index
/// using round-to-nearest, ties-to-smaller-magnitude.
///
/// Inputs above 6 saturate to index 7 (this happens only with finite rounding
/// noise around 6 when the block maximum defines the scale).
pub fn magnitude_index(m: f64) -> usize {
    let candidates: [(f64, usize); 7] = [
        (0.25, 0), // midpoint 0/0.5
        (0.75, 1), // midpoint 0.5/1
        (1.25, 2), // midpoint 1/1.5
        (1.75, 3), // midpoint 1.5/2
        (2.5, 4),  // midpoint 2/3
        (3.5, 5),  // midpoint 3/4
        (5.0, 6),  // midpoint 4/6
    ];
    for (midpoint, idx) in candidates {
        if m <= midpoint {
            return idx;
        }
    }
    7
}

/// Encodes a single normalized value `x` (measured in table units, i.e. already
/// divided by the block scale) into its 4-bit code.
///
/// Returns [`QError::NonFinite`] for NaN/infinite input. `-0.0` is written as
/// positive zero (`0x0`).
pub fn encode_value(x: f64) -> QResult<u8> {
    if !x.is_finite() {
        return Err(QError::NonFinite);
    }
    if x == 0.0 {
        return Ok(0);
    }
    let sign = if x < 0.0 { SIGN_BIT } else { 0 };
    Ok(sign | magnitude_index(x.abs()) as u8)
}

/// Packs an even-length slice of nibbles (values 0..16) little nibble first.
pub fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
    debug_assert!(nibbles.len() % 2 == 0, "caller pads to an even length");
    let mut out = Vec::with_capacity(nibbles.len() / 2);
    for pair in nibbles.chunks(2) {
        out.push((pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4));
    }
    out
}

/// Unpacks packed bytes into `count` nibbles.
///
/// When `count` is odd the padding high nibble of the final byte must be zero;
/// otherwise [`QError::InvalidPadding`] is returned.
pub fn unpack_nibbles(bytes: &[u8], count: usize) -> QResult<Vec<u8>> {
    let packed_need = count.div_ceil(2);
    if bytes.len() < packed_need {
        return Err(QError::Truncated);
    }
    let mut out = vec![0u8; count];
    for (i, slot) in out.iter_mut().enumerate() {
        let byte = bytes[i / 2];
        *slot = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    }
    if count % 2 == 1 {
        let last = bytes[count / 2];
        if last & 0xF0 != 0 {
            return Err(QError::InvalidPadding);
        }
    }
    Ok(out)
}
