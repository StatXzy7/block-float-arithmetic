# qlp4 — blocked 4-bit numeric primitives (Linux CPU)

A dependency-free Rust library for studying how a 4-bit representation affects
dot products and matrix multiplication. Each block is independently scaled,
outliers can be magnitude-clipped, and the **exact clipping residuals are
stored sparsely and actually participate** in dot products and matrix
multiplication when compensation is enabled.

```bash
cargo build
cargo test
cargo run --example demo
```

## Code table (16 explicit codes)

Low three bits select a magnitude; the MSB is the sign:

| code | value | code | value |
|------|-------|------|-------|
| `0x0` | `+0`   | `0x8` | `-0`   |
| `0x1` | `+0.5` | `0x9` | `-0.5` |
| `0x2` | `+1`   | `0xA` | `-1`   |
| `0x3` | `+1.5` | `0xB` | `-1.5` |
| `0x4` | `+2`   | `0xC` | `-2`   |
| `0x5` | `+3`   | `0xD` | `-3`   |
| `0x6` | `+4`   | `0xE` | `-4`   |
| `0x7` | `+6`   | `0xF` | `-6`   |

* Decoding `0x8` returns `-0.0` (the negative-zero encoding is readable).
* Encoding `-0.0` writes `0x0` (write normalization to positive zero).
* Round-to-nearest; at an exact midpoint the smaller-magnitude code is chosen.
* Per block the largest absolute value maps to code `6`, so the block scale is
  `max_abs / 6`. With clipping enabled, `max_abs` is taken after clipping.
* An all-zero block has a deterministic scale of exactly `0.0`.

## Nibble packing and tail rules

* Element `2i` is the **low** nibble of byte `i`; element `2i+1` is the high
  nibble (little nibble first).
* An odd element count appends one zero-padding nibble in the final high
  nibble. Decoders require that padding nibble to be `0` (`InvalidPadding`
  otherwise).
* The final, shorter block uses the same block length semantics; its padding
  nibble is zero as well. Both odd nibble counts and short tail blocks
  round-trip exactly.

## Clipping residuals and computation

* Optional absolute-value threshold `t`: values are clipped to `[-t, t]`.
* For every clipped element the residual `original - clipped` is stored as a
  sorted sparse pair `(u64 index, f64 value)`.
* `dot`/`matmul(compensate: false)` use decoded 4-bit codes only.
* `dot`/`matmul(compensate: true)` add the exact residual cross terms
  `rA·dB + dA·rB + rA·rB` via a sorted merge, so the kernel equals a direct
  pure-double computation when codes are table-exact after clipping.
* The compensation flag changes arithmetic only; storage is identical either
  way. Residuals are retained whenever a threshold is configured.

## Errors and storage

* `ref_dot` / `ref_matmul` are pure-double reference kernels.
* `error_metrics` returns maximum absolute error and mean squared error.
* `QTensor::storage()` reports `header/scale/nibble/residual` byte counts;
  `total_bytes` equals `to_bytes().len()`. `Matrix::storage_total()` counts
  both the row and the column encodings used by multiplication.
* Rejected: non-finite input, zero block length, NaN/infinite/negative
  threshold or scale, matrix shape conflicts. Decoding a truncated,
  dirty-padding, trailing-byte, or internally inconsistent buffer returns an
  error — never a partially decoded tensor.

## Serialized layout (little endian)

```
"QLP4"            4 magic bytes
version           u8 (= 1)
block_len         u64
n                 u64 element count
flag              u8: 0 = no threshold, 1 = threshold present
threshold         f64 (only when flag == 1)
scales[n.blocks]  f64 per block (0.0 for all-zero blocks)
packed_len        u64 (== ceil(n/2))
packed            packed_len nibble bytes
residual_count    u64
residuals         repeated: u64 index, f64 value (sorted, unique, non-zero)
```

## Public API overview

* `code_table::{MAGNITUDES, decode_nibble, encode_value, magnitude_index, pack_nibbles, unpack_nibbles}`
* `QTensor::encode/dequant/reconstruct/to_bytes/from_bytes/storage/residuals/...`
* `dot(a, b, compensate)`, `Matrix::quantize(rows, cols, ...)`, `matmul(...)`
* `ref_dot`, `ref_matmul`, `error_metrics`

## Dependencies

Zero third-party crates (the task allows up to two). Tested with
`cargo 1.75.0`; `cargo test` runs 14 tests covering the full code table,
rounding ties, tail/odd-nibble round trips, residual numeric contributions,
and matrix-multiplication agreement with an independent reference.
