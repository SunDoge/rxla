use rxla_core::vision::{NmsError, nms, nms_by_class};

#[test]
fn classes_do_not_cross_suppress_and_cap_is_global() {
    let boxes = [[0., 0., 2., 2.]; 5];
    let scores = [0.8, 0.9, 0.7, 0.9, 0.6];
    let classes = [i32::MIN, i32::MIN, i32::MAX, -1, i32::MAX];
    assert_eq!(
        nms_by_class(&boxes, &scores, &classes, 0.5, 5).unwrap(),
        [1, 3, 2]
    );
    assert_eq!(
        nms_by_class(&boxes, &scores, &classes, 0.5, 2).unwrap(),
        [1, 3]
    );
    assert_eq!(
        nms_by_class(&boxes, &scores, &[0; 5], 0.5, 5).unwrap(),
        nms(&boxes, &scores, 0.5, 5).unwrap()
    );
    // Even extreme coordinates retain exact class separation without offsets.
    let huge = [[-f32::MAX, -f32::MAX, f32::MAX, f32::MAX]; 2];
    assert_eq!(
        nms_by_class(&huge, &[1., 1.], &[i32::MIN, i32::MAX], 0.5, 2).unwrap(),
        [0, 1]
    );
    assert!(nms_by_class(&[], &[], &[], 0.5, 2).unwrap().is_empty());
    for limit in [0, 1] {
        assert!(nms_by_class(&boxes, &scores, &[], 0.5, limit).is_err());
        assert!(nms_by_class(&boxes, &scores, &classes, f32::NAN, limit).is_err());
    }
}

#[test]
fn greedy_order_limits_and_strict_threshold() {
    let boxes = [
        [0., 0., 2., 2.],
        [0., 0., 2., 2.],
        [4., 4., 5., 5.],
        [0., 0., 1., 2.],
    ];
    let scores = [0.5, 0.9, 0.8, 0.7];
    assert_eq!(nms(&boxes, &scores, 0.5, 10).unwrap(), [1, 2, 3]);
    assert_eq!(nms(&boxes, &scores, 0.49, 10).unwrap(), [1, 2]);
    assert_eq!(nms(&boxes, &scores, 1., 10).unwrap(), [1, 2, 3, 0]);
    assert_eq!(nms(&boxes, &scores, 0., 1).unwrap(), [1]);
    assert!(nms(&boxes, &scores, 0., 0).unwrap().is_empty());
    assert!(nms(&[], &[], 0.5, usize::MAX).unwrap().is_empty());
}

#[test]
fn suppressed_candidates_do_not_suppress_later_boxes() {
    // A overlaps B, B overlaps C, but A merely touches C. Once B is removed,
    // it must not suppress C (greedy NMS is not connected-component merging).
    let boxes = [[0., 0., 2., 2.], [1., 0., 3., 2.], [2., 0., 4., 2.]];
    assert_eq!(nms(&boxes, &[3., 2., 1.], 0.3, 3).unwrap(), [0, 2]);
}

#[test]
fn ties_signed_zero_degenerate_and_large_finite_coordinates() {
    let same = [[0., 0., 1., 1.]; 3];
    assert_eq!(nms(&same, &[-0., 0., -0.], 0.5, 3).unwrap(), [0]);
    assert_eq!(nms(&same, &[-2., -1., -1.], 0.5, 3).unwrap(), [1]);
    let empty = [[0.; 4]; 3];
    assert_eq!(nms(&empty, &[1.; 3], 0., 3).unwrap(), [0, 1, 2]);
    let huge = [[-f32::MAX, -f32::MAX, f32::MAX, f32::MAX]; 2];
    assert_eq!(nms(&huge, &[1., 2.], 0.5, 2).unwrap(), [1]);
}

#[test]
fn validates_even_when_no_output_requested() {
    for max_output in [0, 10] {
        assert!(nms(&[[0.; 4]], &[], 0.5, max_output).is_err());
        for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(nms(&[], &[], threshold, max_output).is_err());
        }
        for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(nms(&[[0.; 4]], &[score], 0.5, max_output).is_err());
        }
        for b in [
            [1., 0., 0., 1.],
            [0., 1., 1., 0.],
            [0., 0., f32::INFINITY, 1.],
            [f32::NAN, 0., 1., 1.],
        ] {
            assert!(nms(&[b], &[1.], 0.5, max_output).is_err());
        }
    }
}

#[test]
fn validation_failures_preserve_the_offending_field_and_index() {
    assert!(matches!(
        nms(&[[0.; 4]], &[], 0.5, 0),
        Err(NmsError::BoxScoreCount {
            boxes: 1,
            scores: 0
        })
    ));
    assert!(matches!(
        nms_by_class(&[[0.; 4]], &[1.], &[], 0.5, 0),
        Err(NmsError::BoxClassCount {
            boxes: 1,
            classes: 0
        })
    ));
    assert!(matches!(
        nms(&[], &[], 1.5, 0),
        Err(NmsError::InvalidThreshold { threshold: 1.5 })
    ));
    assert!(matches!(
        nms(&[[0.; 4], [0.; 4]], &[1., f32::INFINITY], 0.5, 0),
        Err(NmsError::InvalidScore {
            index: 1,
            value: f32::INFINITY
        })
    ));
    assert!(matches!(
        nms(&[[0.; 4], [2., 0., 1., 1.]], &[1., 1.], 0.5, 0),
        Err(NmsError::InvalidBox {
            index: 1,
            value: [2., 0., 1., 1.]
        })
    ));
}
