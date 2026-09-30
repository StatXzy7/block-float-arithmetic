use super::code_table::{
    decode_nibble, encode_value, magnitude_index, pack_nibbles, unpack_nibbles, MAGNITUDES, SIGN_BIT,
};
use super::linalg::{dot, error_metrics, ref_dot};
use super::tensor::QTensor;
use super::QError;

#[test]
fn code_table_covers_all_sixteen_entries() {
    for code in 0u8..=15 {
        let expected_sign = if code & SIGN_BIT != 0 { -1.0 } else { 1.0 };
        let expected = expected_sign * MAGNITUDES[(code & 0x7) as usize];
        assert_eq!(decode_nibble(code), expected);
    }
    // Every representable non-negative value round-trips, including 0.
    for &mag in MAGNITUDES.iter() {
        assert_eq!(decode_nibble(encode_value(mag).unwrap()), mag);
    }
    // Negative zero is readable on decode but normalized to +0 on encode.
    assert!(decode_nibble(0x8).is_sign_negative());
    assert_eq!(decode_nibble(0x8), 0.0);
    assert_eq!(encode_value(-0.0).unwrap(), 0x0);
    assert!(decode_nibble(encode_value(-0.0).unwrap()).is_sign_positive());
    for mag in MAGNITUDES {
        if mag == 0.0 {
            continue;
        }
        assert_eq!(decode_nibble(encode_value(-mag).unwrap()), -mag);
    }
}

#[test]
fn rounding_ties_pick_smaller_magnitude() {
    // Exact midpoints between consecutive table magnitudes.
    assert_eq!(magnitude_index(0.25), 0);
    assert_eq!(magnitude_index(0.75), 1);
    assert_eq!(magnitude_index(1.25), 2);
    assert_eq!(magnitude_index(1.75), 3);
    assert_eq!(magnitude_index(2.5), 4);
    assert_eq!(magnitude_index(3.5), 5);
    assert_eq!(magnitude_index(5.0), 6);
    // A value just above a midpoint rounds up; just below rounds down.
    assert_eq!(magnitude_index(0.25 + f64::EPSILON), 1);
    assert_eq!(magnitude_index(0.25 - f64::EPSILON), 0);
    assert_eq!(magnitude_index(2.5 + 2.0 * f64::EPSILON), 5);
    assert_eq!(magnitude_index(2.5 - f64::EPSILON), 4);
    // Encoding negative values applies the same magnitude rule.
    assert_eq!(encode_value(-2.5).unwrap(), SIGN_BIT | 4);
    assert_eq!(encode_value(-2.5 - 2.0 * f64::EPSILON).unwrap(), SIGN_BIT | 5);
    // Saturation past 6.
    assert_eq!(magnitude_index(6.0), 7);
    assert_eq!(magnitude_index(100.0), 7);
}

#[test]
fn block_maximum_maps_to_six() {
    let q = QTensor::encode(&[0.0, 3.0], 2, None).unwrap();
    assert_eq!(q.scales(), &[0.5]);
    assert_eq!(q.dequant(), vec![0.0, 3.0]);
    // Independent scaling per block.
    let q2 = QTensor::encode(&[0.0, 3.0, 0.0, 2.0], 2, None).unwrap();
    assert_eq!(q2.scales(), &[0.5, 2.0 / 6.0]);
}

#[test]
fn all_zero_blocks_have_deterministic_scale() {
    let q = QTensor::encode(&[0.0, -0.0, 0.0], 2, None).unwrap();
    assert_eq!(q.scales(), &[0.0, 0.0]);
    let back = q.reconstruct();
    assert_eq!(back, vec![0.0, 0.0, 0.0]);
    assert!(back.iter().all(|v| v.is_sign_positive()));
    // Serialization round-trips the deterministic scale.
    let q2 = QTensor::from_bytes(&q.to_bytes()).unwrap();
    assert_eq!(q, q2);
}

#[test]
fn odd_element_count_roundtrips_with_padding_rules() {
    let values = vec![0.0, 0.5, -1.5, 4.0, -6.0, 3.0, 2.0]; // 7: odd nibbles + tail block
    let q = QTensor::encode(&values, 3, None).unwrap();
    assert_eq!(q.len(), 7);
    assert_eq!(q.packed_bytes().len(), 4); // ceil(7/2) bytes
    // Final byte's high nibble is the documented zero padding.
    assert_eq!(q.packed_bytes()[3] & 0xF0, 0);
    let q = QTensor::from_bytes(&q.to_bytes()).unwrap();
    assert_eq!(q.reconstruct(), values);
}

#[test]
fn pack_unpack_enforces_padding_nibble() {
    let packed = pack_nibbles(&[1, 2, 3, 0]);
    assert_eq!(unpack_nibbles(&packed, 3).unwrap(), vec![1, 2, 3]);
    // Dirty padding nibble must be rejected.
    let dirty = {
        let mut p = packed.clone();
        p[1] |= 0x80;
        p
    };
    assert_eq!(unpack_nibbles(&dirty, 3), Err(QError::InvalidPadding));
    assert_eq!(unpack_nibbles(&[0xAB], 2).unwrap(), vec![0xB, 0xA]);
}

#[test]
fn clipping_residuals_contribute_to_dot() {
    // Every block has a post-clip magnitude of 6, so its scale is exactly 1
    // and codes reproduce every clipped value exactly; residuals then restore
    // the originals with no residual quantization noise.
    let a = vec![8.0, -6.0, 6.0, -7.0];
    let b = vec![6.0, 4.0, -3.0, 6.0];
    let qa = QTensor::encode(&a, 2, Some(6.0)).unwrap();
    let qb = QTensor::encode(&b, 2, Some(6.0)).unwrap();

    assert_eq!(qa.residuals(), &[(0, 2.0), (3, -1.0)]);
    assert_eq!(qb.residuals(), &[]);
    let expected = ref_dot(&a, &b).unwrap();

    let exact = dot(&qa, &qb, true).unwrap();
    let approximate = dot(&qa, &qb, false).unwrap();
    assert_eq!(exact, expected, "compensated dot must equal pure-double ref");
    assert!(
        (approximate - expected).abs() > 1e-9,
        "without compensation the clipped outlier must change the result"
    );

    // Reconstruction is exact; plain dequantization is not.
    assert_eq!(qa.reconstruct(), a);
    assert_ne!(qa.dequant(), a);
}

#[test]
fn metrics_separate_error_from_storage() {
    let x = vec![6.0, 2.0, 1.0, 0.5];
    let q = QTensor::encode(&x, 4, None).unwrap();
    let m = error_metrics(&x, &q.dequant()).unwrap();
    assert_eq!(m.max_abs, 0.0);
    assert_eq!(m.mse, 0.0);
    let stats = q.storage();
    assert_eq!(stats.total_bytes as usize, q.to_bytes().len());
    assert_eq!(stats.nibble_bytes, 2); // 4 nibbles -> 2 bytes
    assert_eq!(stats.residual_bytes, 0); // no clipping -> no residual storage
}

#[test]
fn rejects_non_finite_inputs_bad_params_and_shapes() {
    assert_eq!(QTensor::encode(&[f64::NAN], 1, None), Err(QError::NonFinite));
    assert_eq!(
        QTensor::encode(&[f64::INFINITY, 1.0], 1, None),
        Err(QError::NonFinite)
    );
    assert_eq!(QTensor::encode(&[1.0], 0, None), Err(QError::InvalidBlockLen));
    assert_eq!(
        QTensor::encode(&[1.0], 1, Some(f64::NAN)),
        Err(QError::InvalidThreshold)
    );
    assert_eq!(
        QTensor::encode(&[1.0], 1, Some(-1.0)),
        Err(QError::InvalidThreshold)
    );
    let q1 = QTensor::encode(&[1.0, 2.0], 1, None).unwrap();
    let q2 = QTensor::encode(&[1.0], 1, None).unwrap();
    assert_eq!(dot(&q1, &q2, false), Err(QError::ShapeMismatch));
}

#[test]
fn corrupted_and_truncated_buffers_never_partially_decode() {
    // No threshold: the fixed header ends exactly where the scales begin.
    let q = QTensor::encode(&[0.5, -1.5, 4.0, 6.0, -3.0], 2, None).unwrap();
    let good = q.to_bytes();

    // Any truncation point is rejected.
    for cut in 0..good.len() {
        assert!(matches!(
            QTensor::from_bytes(&good[..cut]),
            Err(QError::Truncated | QError::BadHeader)
        ));
    }
    // Appended garbage is trailing data.
    let mut plus = good.clone();
    plus.push(0);
    assert_eq!(QTensor::from_bytes(&plus), Err(QError::TrailingData));

    // Corrupt the second scale to NaN: invalid scale is rejected wholesale.
    let scale_offset = 4 + 1 + 8 + 8 + 1;
    let mut bad = good.clone();
    bad[scale_offset + 8 + 6] = 0xFF;
    bad[scale_offset + 8 + 7] = 0xFF;
    assert_eq!(QTensor::from_bytes(&bad), Err(QError::InvalidScale));

    // Corrupt the first scale to zero while codes stay non-zero.
    let mut bad = good.clone();
    for k in 0..8 {
        bad[scale_offset + k] = 0;
    }
    assert_eq!(QTensor::from_bytes(&bad), Err(QError::CorruptBlock));

    // Corrupt the padding nibble of the final (odd) nibble byte.
    let mut bad = good.clone();
    let packed_len_pos = scale_offset + q.scales().len() * 8;
    let nibble_start = packed_len_pos + 8;
    let last_byte = nibble_start + q.packed_bytes().len() - 1;
    bad[last_byte] |= 0xF0;
    assert_eq!(QTensor::from_bytes(&bad), Err(QError::InvalidPadding));
}
