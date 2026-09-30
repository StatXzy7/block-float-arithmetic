//! Blocked 4-bit quantized vectors with optional magnitude clipping and the
//! exact sparse residuals needed to reconstruct the original values.

use crate::code_table::{decode_nibble, encode_value, pack_nibbles, unpack_nibbles, SIGN_BIT};
use crate::error::{QError, QResult};
use crate::StorageStats;

/// One sparse residual `original - clipped`, stored at its element index.
pub type SparseEntry = (u64, f64);

const MAGIC: &[u8; 4] = b"QLP4";
const VERSION: u8 = 1;
const HEADER_FIXED: usize = 4 + 1 + 8 + 8 + 1; // magic, version, block_len, n, flag
const RESIDUAL_ENTRY: usize = 8 + 8; // u64 index + f64 value
const LENGTH_FIELDS: usize = 8 + 8; // packed_len u64 + residual_count u64

/// A vector quantized in blocks of a configurable length.
///
/// Each block is mapped so that the largest absolute value lands on code 6.
/// When a clipping threshold is set, values are magnitude-clipped first and
/// the exact residual `original - clipped` is retained sparsely; callers choose
/// per operation whether residuals participate in the arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub struct QTensor {
    n: u64,
    block_len: u64,
    scales: Vec<f64>,
    packed: Vec<u8>,
    /// `Some(t)` means absolute-value clipping at `t` was applied.
    threshold: Option<f64>,
    /// Sorted by index; every entry is non-zero.
    residuals: Vec<SparseEntry>,
}

/// Clamps `x` to `[-threshold, threshold]` by absolute value.
pub fn clip_abs(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        threshold
    } else if x < -threshold {
        -threshold
    } else {
        x
    }
}

fn check_finite(values: &[f64]) -> QResult<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(QError::NonFinite)
    }
}

impl QTensor {
    /// Quantizes `values` in blocks of `block_len`.
    ///
    /// `threshold` of `Some(t)` enables absolute-value clipping (`t` must be
    /// finite and non-negative); residuals are always stored when non-zero.
    pub fn encode(values: &[f64], block_len: u64, threshold: Option<f64>) -> QResult<Self> {
        if block_len == 0 {
            return Err(QError::InvalidBlockLen);
        }
        if let Some(t) = threshold {
            if !t.is_finite() || t < 0.0 {
                return Err(QError::InvalidThreshold);
            }
        }
        check_finite(values)?;

        let n = values.len() as u64;
        let nb = n.div_ceil(block_len) as usize;
        let mut scales = vec![0.0f64; nb];
        let mut nibbles: Vec<u8> = Vec::with_capacity(values.len());
        let mut residuals: Vec<SparseEntry> = Vec::new();

        for (b, block) in values.chunks(block_len as usize).enumerate() {
            // Clipping defines the block's representable range, so the scale is
            // derived from the largest post-clip absolute value; an unclipped
            // block therefore still maps its raw maximum to code 6.
            let clipped_block: Vec<f64> = match threshold {
                Some(t) => block.iter().map(|&x| clip_abs(x, t)).collect(),
                None => block.to_vec(),
            };
            let max_abs = clipped_block.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let scale = max_abs / 6.0;
            scales[b] = scale;
            for (offset, &x) in block.iter().enumerate() {
                let clipped = clipped_block[offset];
                let residual = x - clipped;
                if residual != 0.0 {
                    residuals.push((b as u64 * block_len + offset as u64, residual));
                }
                let code = if scale == 0.0 {
                    0
                } else {
                    encode_value(clipped / scale)?
                };
                nibbles.push(code);
            }
        }

        if nibbles.len() % 2 == 1 {
            nibbles.push(0); // documented zero-padding nibble
        }
        let packed = pack_nibbles(&nibbles);

        Ok(Self {
            n,
            block_len,
            scales,
            packed,
            threshold,
            residuals,
        })
    }

    /// Number of encoded elements.
    pub fn len(&self) -> u64 {
        self.n
    }

    /// Whether the vector contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Configured block length.
    pub fn block_len(&self) -> u64 {
        self.block_len
    }

    /// Block scales (one per block); all-zero blocks have an exact scale of 0.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Active clipping threshold, if any.
    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Sorted non-zero residuals `(index, original - clipped)`.
    pub fn residuals(&self) -> &[SparseEntry] {
        &self.residuals
    }

    /// Number of stored non-zero residuals.
    pub fn residual_count(&self) -> usize {
        self.residuals.len()
    }

    /// Raw packed nibble bytes (including any padding byte).
    pub fn packed_bytes(&self) -> &[u8] {
        &self.packed
    }

    /// Reads the 4-bit code of element `i`.
    pub fn code_at(&self, i: u64) -> u8 {
        let byte = self.packed[(i / 2) as usize];
        if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    /// Decodes codes only (4-bit approximation, no clipping residuals).
    pub fn dequant(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n as usize];
        for (i, slot) in out.iter_mut().enumerate() {
            let i = i as u64;
            let code = self.code_at(i);
            *slot = decode_nibble(code) * self.scales[(i / self.block_len) as usize];
        }
        out
    }

    /// Reconstructs the original values exactly: decoded codes plus residuals.
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut out = self.dequant();
        for &(idx, r) in &self.residuals {
            out[idx as usize] += r;
        }
        out
    }

    /// Exact storage accounting in bytes.
    pub fn storage(&self) -> StorageStats {
        let threshold_bytes = if self.threshold.is_some() { 8 } else { 0 };
        let header = HEADER_FIXED + threshold_bytes;
        let scale_bytes = self.scales.len() * 8;
        let nibble_bytes = self.packed.len();
        let residual_bytes = self.residuals.len() * RESIDUAL_ENTRY;
        StorageStats {
            header_bytes: header as u64,
            length_bytes: LENGTH_FIELDS as u64,
            scale_bytes: scale_bytes as u64,
            nibble_bytes: nibble_bytes as u64,
            residual_bytes: residual_bytes as u64,
            total_bytes: (header + LENGTH_FIELDS + scale_bytes + nibble_bytes
                + residual_bytes) as u64,
            residual_count: self.residuals.len() as u64,
        }
    }

    /// Serializes the tensor; the layout is documented in `README.md`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&self.block_len.to_le_bytes());
        buf.extend_from_slice(&self.n.to_le_bytes());
        match self.threshold {
            None => buf.push(0),
            Some(t) => {
                buf.push(1);
                buf.extend_from_slice(&t.to_le_bytes());
            }
        }
        for &s in &self.scales {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.extend_from_slice(&(self.packed.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.packed);
        buf.extend_from_slice(&(self.residuals.len() as u64).to_le_bytes());
        for &(idx, r) in &self.residuals {
            buf.extend_from_slice(&idx.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
        }
        buf
    }

    /// Parses a serialized tensor.
    ///
    /// Truncated or internally inconsistent input is rejected wholesale; no
    /// partially decoded tensor is ever returned.
    pub fn from_bytes(data: &[u8]) -> QResult<Self> {
        let cur = Cursor::new(data);
        Self::read(cur)
    }

    pub(crate) fn read(mut cur: Cursor<'_>) -> QResult<Self> {
        let magic = cur.take(4)?;
        if magic != MAGIC {
            return Err(QError::BadHeader);
        }
        if cur.take(1)?[0] != VERSION {
            return Err(QError::BadHeader);
        }
        let block_len = cur.u64()?;
        if block_len == 0 {
            return Err(QError::InvalidBlockLen);
        }
        let n = cur.u64()?;
        let threshold = match cur.take(1)?[0] {
            0 => None,
            1 => {
                let t = cur.f64()?;
                if !t.is_finite() || t < 0.0 {
                    return Err(QError::InvalidThreshold);
                }
                Some(t)
            }
            _ => return Err(QError::BadHeader),
        };

        let nb = n.div_ceil(block_len) as usize;
        let mut scales = Vec::with_capacity(nb);
        for _ in 0..nb {
            let s = cur.f64()?;
            if !s.is_finite() || s < 0.0 {
                return Err(QError::InvalidScale);
            }
            scales.push(s);
        }

        let packed_len = cur.u64()? as usize;
        let expected_packed = n.div_ceil(2) as usize;
        if packed_len != expected_packed {
            return Err(QError::CorruptBlock);
        }
        let packed = cur.take(packed_len)?.to_vec();
        let nibbles = unpack_nibbles(&packed, n as usize)?;

        for (b, chunk) in nibbles.chunks(block_len as usize).enumerate() {
            let all_zero = chunk.iter().all(|&c| c == 0);
            for &c in chunk {
                if c == SIGN_BIT {
                    return Err(QError::CorruptBlock); // -0 is never written
                }
            }
            if scales[b] == 0.0 && !all_zero {
                return Err(QError::CorruptBlock);
            }
            if scales[b] > 0.0 && all_zero {
                return Err(QError::CorruptBlock);
            }
        }

        let rc = cur.u64()? as usize;
        let mut residuals = Vec::with_capacity(rc);
        let mut prev: Option<u64> = None;
        for _ in 0..rc {
            let idx = cur.u64()?;
            let r = cur.f64()?;
            if idx >= n || !r.is_finite() || r == 0.0 {
                return Err(QError::CorruptResiduals);
            }
            if let Some(p) = prev {
                if idx <= p {
                    return Err(QError::CorruptResiduals);
                }
            }
            prev = Some(idx);
            residuals.push((idx, r));
        }
        if threshold.is_none() && !residuals.is_empty() {
            return Err(QError::CorruptResiduals);
        }
        if !cur.exhausted() {
            return Err(QError::TrailingData);
        }

        Ok(Self {
            n,
            block_len,
            scales,
            packed,
            threshold,
            residuals,
        })
    }
}

/// Minimal bounds-checked little-endian reader.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn take(&mut self, len: usize) -> QResult<&'a [u8]> {
        if self.pos.checked_add(len).map_or(true, |e| e > self.data.len()) {
            return Err(QError::Truncated);
        }
        let s = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }

    pub(crate) fn u64(&mut self) -> QResult<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }

    pub(crate) fn f64(&mut self) -> QResult<f64> {
        Ok(f64::from_le_bytes({
            let mut a = [0u8; 8];
            a.copy_from_slice(self.take(8)?);
            a
        }))
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.pos == self.data.len()
    }
}
