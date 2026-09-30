use nibble4::{
    code_to_value, dot, metrics, pack_nibbles, reference_dot, unpack_nibbles, Matrix, QError,
    QMatrix, QVector, QuantConfig, LOW_NIBBLE_FIRST, ODD_TAIL_PAD,
};

fn cfg(block_len: u32, threshold: f64) -> QuantConfig {
    QuantConfig::new(block_len, threshold).unwrap()
}

// 1. Code table: all sixteen nibbles decode exactly, and the encoder can
//    reach every entry from real values (neg zero only readable, never written).
#[test]
fn full_sixteen_entry_code_table() {
    let mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for idx in 0..8u8 {
        assert_eq!(code_to_value(idx), mags[idx as usize]);
        assert_eq!(code_to_value(idx | 0x08), -mags[idx as usize]);
    }
    // 0x8 is the readable negative zero; -0.0 writes back as positive zero.
    assert!(code_to_value(0x8).is_sign_negative());
    let q = QVector::encode(&[-0.0], cfg(1, f64::INFINITY)).unwrap();
    assert_eq!(q.code_bytes, vec![0x00]);
    assert_eq!(q.scales, vec![0.0]);

    // Drive each positive and negative code through the block encoder. The
    // block max abs is 6, so scale is exactly 1 and every table entry appears.
    for idx in 0..8u8 {
        let pos = [mags[idx as usize], 6.0];
        let qp = QVector::encode(&pos, cfg(2, f64::INFINITY)).unwrap();
        assert_eq!(qp.code_bytes[0] & 0x0F, idx, "positive code {}", mags[idx as usize]);
        assert_eq!(qp.decode(false)[0], mags[idx as usize]);

        let neg = [-mags[idx as usize], -6.0];
        let qn = QVector::encode(&neg, cfg(2, f64::INFINITY)).unwrap();
        let expected = if idx == 0 { 0x0 } else { 0x08 | idx };
        assert_eq!(qn.code_bytes[0] & 0x0F, expected, "negative code {}", -mags[idx as usize]);
        let want = if idx == 0 { 0.0 } else { -mags[idx as usize] };
        assert_eq!(qn.decode(false)[0], want);
    }
}

// 2. Midpoint rounding: ties land on the smaller absolute magnitude.
#[test]
fn round_nearest_tie_picks_smaller_magnitude() {
    // [0.25, 0.75, 1.25, 2.5, 3.5, 5.0] scaled by max=6 -> norm unchanged.
    let values = [0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 6.0];
    let expected_idx = [0u8, 1, 2, 4, 5, 6, 7];
    let q = QVector::encode(&values, cfg(values.len() as u32, 6.0)).unwrap();
    assert_eq!(q.scales, vec![1.0]);
    let codes = unpack_nibbles(&q.code_bytes, values.len()).unwrap();
    assert_eq!(codes, expected_idx);

    // Negative ties choose the smaller absolute magnitude too (same index).
    let neg: Vec<f64> = values.iter().map(|v| -v).collect();
    let qn = QVector::encode(&neg, cfg(values.len() as u32, 6.0)).unwrap();
    let cn = unpack_nibbles(&qn.code_bytes, neg.len()).unwrap();
    for (c, idx) in cn.iter().zip(expected_idx) {
        assert_eq!(c & 0x07, idx);
        if idx != 0 {
            assert_eq!(c & 0x08, 0x08);
        }
    }

    // Just-off-tie values resolve to the larger magnitude (block max is 6).
    let q2 = QVector::encode(&[0.251, 0.751, 6.0, 0.0], cfg(4, 6.0)).unwrap();
    let c2 = unpack_nibbles(&q2.code_bytes, 4).unwrap();
    assert_eq!(c2, vec![1, 2, 7, 0]);
}

// 3. Tail block round trip: short final block, odd code count and padding.
#[test]
fn tail_block_and_odd_codes_round_trip() {
    // block_len=4, 11 values -> full block(4) + full block(4) + tail(3).
    let values = [6.0, 0.0, 3.0, -1.5, 0.5, -6.0, 4.0, 0.0, 2.0, -0.5, 6.0];
    let q = QVector::encode(&values, cfg(4, f64::INFINITY)).unwrap();
    assert_eq!(q.scales.len(), 3);
    assert_eq!(q.scales[2], 1.0); // tail block max abs 6
    assert_eq!(q.code_bytes.len(), 6); // ceil(11/2)
    assert_eq!(q.code_bytes[5] & 0xF0, ODD_TAIL_PAD << 4);
    let decoded = q.decode(false);
    assert_eq!(decoded, values);

    // Bytes round trip too, including the declared odd padding.
    let raw = q.to_bytes();
    let q2 = QVector::from_bytes(&raw).unwrap();
    assert_eq!(q2.decode(false), values);

    // Zero block (including a zero tail) has a deterministic scale 0.
    let z = QVector::encode(&[0.0; 5], cfg(4, f64::INFINITY)).unwrap();
    assert_eq!(z.scales, vec![0.0, 0.0]);
    assert_eq!(z.decode(false), vec![0.0; 5]);

    // Empty vector round trips and needs no payload bytes.
    let e = QVector::encode(&[], cfg(4, f64::INFINITY)).unwrap();
    assert!(e.code_bytes.is_empty());
    assert_eq!(QVector::from_bytes(&e.to_bytes()).unwrap().decode(false), Vec::<f64>::new());
}

// 4. Nibble packing order and tail padding rule.
#[test]
fn nibble_pack_order_and_padding() {
    assert!(LOW_NIBBLE_FIRST);
    let codes = [0x1u8, 0x2, 0x3];
    let packed = pack_nibbles(&codes);
    assert_eq!(packed, vec![0x21, 0x03]); // low nibble first, high padded 0
    assert_eq!(unpack_nibbles(&packed, 3).unwrap(), vec![1, 2, 3]);

    // Nonzero padding on an odd tail is rejected, as is a size mismatch.
    assert_eq!(unpack_nibbles(&[0x21, 0x13], 3), Err(QError::Corrupted));
    assert_eq!(unpack_nibbles(&[0x21], 3), Err(QError::Corrupted));
    assert_eq!(unpack_nibbles(&[0x21, 0x03, 0x00], 3), Err(QError::Corrupted));
}

// 5. Sparse compensation contributes numerically to dot products, and storage
//    accounting stays separate from compute error.
#[test]
fn sparse_residual_compensation_contribution() {
    // block_len=1, threshold 6: 12 clips to 6 with residual +6; -9 -> -6/-3.
    let x = [12.0, -9.0, 3.0];
    let y = [2.0, 2.0, 2.0];
    let qx = QVector::encode(&x, cfg(1, 6.0)).unwrap();
    let qy = QVector::encode(&y, cfg(1, 6.0)).unwrap();

    // Residuals stored only for the two clipped elements.
    assert_eq!(qx.residual_idx, vec![0, 1]);
    assert_eq!(qx.residual_val, vec![6.0, -3.0]);
    assert!(qy.residual_idx.is_empty());

    let reference = reference_dot(&x, &y).unwrap();
    let on = dot(&qx, &qy, true).unwrap();
    let off = dot(&qx, &qy, false).unwrap();
    assert_eq!(reference, 12.0);
    assert_eq!(on, reference, "residuals are real operands, result exact");
    assert_eq!(off, 6.0, "clipped-only products: (6-6+3)*2");
    assert_ne!(on, off);

    // Same operands and params: only the compensation flag differs.
    let dx_on = qx.decode(true);
    let dx_off = qx.decode(false);
    assert_eq!(dx_on, x);
    assert_eq!(dx_off, [6.0, -6.0, 3.0]);

    // Storage accounting: codes stay 4 bits/value; residuals cost extra.
    let st = qx.storage_bytes();
    assert_eq!(st.total, st.header + st.scales + st.codes + st.residuals);
    assert_eq!(st.codes, 2); // ceil(3/2)
    assert!(st.residuals > 4); // count + 2*(u32+f64)
    let m_on = metrics(&x, &dx_on).unwrap();
    let m_off = metrics(&x, &dx_off).unwrap();
    assert_eq!(m_on.max_abs_error, 0.0);
    assert!(m_off.max_abs_error >= 3.0);
}

// 6. Matrix multiplication matches an independent pure-f64 reference, and the
//    two compensation modes agree on the same encoded operands.
#[test]
fn matmul_matches_independent_reference() {
    // Values 0/6/-6 map exactly onto magnitudes 0/6 at scale 1 (no clipping).
    let a = Matrix::new(
        2, 3,
        vec![
            6.0, 0.0, -6.0, //
            0.0, 6.0, 6.0,
        ],
    )
    .unwrap();
    let b = Matrix::new(
        3, 2,
        vec![
            6.0, -6.0, //
            -6.0, 0.0, //
            0.0, 6.0,
        ],
    )
    .unwrap();
    let qa = QMatrix::encode(&a, cfg(2, f64::INFINITY)).unwrap();
    let qb = QMatrix::encode(&b, cfg(2, f64::INFINITY)).unwrap();

    let reference = Matrix::reference_mul(&a, &b).unwrap();
    let off = qa.mul(&qb, false).unwrap();
    let on = qa.mul(&qb, true).unwrap();
    assert_eq!(off.data, reference.data);
    assert_eq!(on.data, reference.data);
    // Independent explicit computation of the expected entries:
    // row0: 6*6 + 0 + (-6)*0 = 36 ; 6*-6 + 0 + -6*6 = -72
    // row1: 0 + 6*-6 + 6*0 = -36 ; 0 + 0 + 6*6 = 36
    assert_eq!(reference.data, vec![36.0, -72.0, -36.0, 36.0]);

    // Shape conflicts are refused.
    let bad = Matrix::new(4, 1, vec![1.0; 4]).unwrap();
    let qbad = QMatrix::encode(&bad, cfg(2, f64::INFINITY)).unwrap();
    assert_eq!(qa.mul(&qbad, false), Err(QError::ShapeMismatch));
}

// Clipping compensation also makes an inexact (0/6/-6-style) matmul exact.
#[test]
fn matmul_clipped_compensation_matches_reference() {
    let a = Matrix::new(1, 2, vec![12.0, 0.0]).unwrap();
    let b = Matrix::new(2, 1, vec![2.0, 9.0]).unwrap();
    let qa = QMatrix::encode(&a, cfg(1, 6.0)).unwrap();
    let qb = QMatrix::encode(&b, cfg(1, 6.0)).unwrap();
    let reference = Matrix::reference_mul(&a, &b).unwrap();
    assert_eq!(qa.mul(&qb, true).unwrap().data, reference.data);
    assert_eq!(reference.data, vec![24.0]);
    // Without compensation: clipped a row [6,0] -> 12.
    assert_eq!(qa.mul(&qb, false).unwrap().data, vec![12.0]);
}

// 7. Illegal inputs and damaged buffers never yield partial results.
#[test]
fn rejects_illegal_inputs_and_damaged_buffers() {
    assert_eq!(QuantConfig::new(0, 1.0), Err(QError::BadBlockLen));
    assert_eq!(QuantConfig::new(4, f64::NAN), Err(QError::BadThreshold));
    assert_eq!(QuantConfig::new(4, -1.0), Err(QError::BadThreshold));
    assert_eq!(
        QVector::encode(&[1.0, f64::NAN], cfg(4, 1.0)),
        Err(QError::NonFinite)
    );
    assert_eq!(
        QVector::encode(&[1.0, f64::INFINITY], cfg(4, 1.0)),
        Err(QError::NonFinite)
    );

    let q = QVector::encode(&[1.0, -2.0, 3.0], cfg(2, 6.0)).unwrap();
    let raw = q.to_bytes();

    // Truncation at several cut points.
    for cut in [0, 4, 10, raw.len() - 1] {
        assert!(matches!(
            QVector::from_bytes(&raw[..cut]),
            Err(QError::Truncated) | Err(QError::Corrupted)
        ));
    }
    // Bad magic.
    let mut bad = raw.clone();
    bad[0] ^= 0xFF;
    assert_eq!(QVector::from_bytes(&bad), Err(QError::Corrupted));
    // Flip the sign bit of the first f64 scale -> negative scale rejected.
    let mut bad = raw.clone();
    bad[21 + 7] |= 0x80;
    assert_eq!(QVector::from_bytes(&bad), Err(QError::BadScale));
    // Trailing garbage is corruption.
    let mut bad = raw.clone();
    bad.push(0);
    assert_eq!(QVector::from_bytes(&bad), Err(QError::Corrupted));
    // Dot product shape conflict.
    let other = QVector::encode(&[1.0], cfg(1, 6.0)).unwrap();
    assert_eq!(dot(&q, &other, false), Err(QError::ShapeMismatch));
}

// Per-element quantization error is bounded by half the block's code step.
#[test]
fn quantization_error_within_half_code_step() {
    let values = [5.1, -0.37, 2.83, 0.0, 4.99, -1.7];
    let block_len = 3u32;
    let q = QVector::encode(&values, cfg(block_len, f64::INFINITY)).unwrap();
    let approx = q.decode(false);
    let gap_half = [0.25f64, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0];
    let codes = unpack_nibbles(&q.code_bytes, values.len()).unwrap();
    for (b, block) in values.chunks(block_len as usize).enumerate() {
        for (j, v) in block.iter().enumerate() {
            let idx = (codes[b * block_len as usize + j] & 0x07) as usize;
            let bound = gap_half[idx] * q.scales[b];
            assert!((approx[b * block_len as usize + j] - v).abs() <= bound * (1.0 + 1e-12));
        }
    }
    let m = metrics(&values, &approx).unwrap();
    assert!(m.mse >= 0.0 && m.max_abs_error > 0.0);
}

// Deterministic pseudo-random property check across many block lengths.
#[test]
fn property_roundtrip_error_bound_and_integrity() {
    let mut seed = 0x9e3779b9u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for block_len in 1..=8u32 {
        let n = 1 + (next() % 37) as usize;
        let values: Vec<f64> = (0..n)
            .map(|_| (next() as f64 / u32::MAX as f64) * 20.0 - 10.0)
            .collect();
        let q = QVector::encode(&values, cfg(block_len, f64::INFINITY)).unwrap();
        let approx = q.decode(false);

        // Per-block scale and half-step error bound, including all-zero blocks.
        for (b, block) in values.chunks(block_len as usize).enumerate() {
            let max_abs = block.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if max_abs == 0.0 {
                assert_eq!(q.scales[b], 0.0);
            } else {
                assert_eq!(q.scales[b], max_abs / 6.0);
            }
            let codes = unpack_nibbles(&q.code_bytes, n).unwrap();
            // Non-uniform table Voronoi half-gaps: gaps are .5 (0..2),
            // then 1 (2..4), then 2 (4..6).
            let gap_half = [0.25f64, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0];
            for (j, v) in block.iter().enumerate() {
                let idx = (codes[b * block_len as usize + j] & 0x07) as usize;
                let bound = gap_half[idx] * q.scales[b];
                let err = (approx[b * block_len as usize + j] - v).abs();
                assert!(err <= bound * (1.0 + 1e-12), "block {b} value {v} err {err}");
            }
        }

        // Serialize/parse integrity and storage sum.
        let again = QVector::from_bytes(&q.to_bytes()).unwrap();
        assert_eq!(again.decode(false), approx);
        let st = q.storage_bytes();
        assert_eq!(st.total, st.header + st.scales + st.codes + st.residuals);
        assert_eq!(st.total, q.to_bytes().len());
    }
}

// Tampering with a stored residual (zeroing it) must be detected.
#[test]
fn corrupted_residual_section_rejected() {
    let q = QVector::encode(&[12.0, -7.0, 1.0], cfg(1, 6.0)).unwrap();
    assert!(!q.residual_idx.is_empty());
    let mut raw = q.to_bytes();
    // Zero the first residual f64 (last 16 bytes hold two residual f64s).
    let tail = raw.len();
    for byte in &mut raw[tail - 16..tail - 8] {
        *byte = 0;
    }
    assert!(QVector::from_bytes(&raw).is_err());
}
