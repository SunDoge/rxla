use rxla_core::{Client, Graph, RotaryLayout};

#[test]
fn angles_require_matching_graph_and_half_width_broadcast() {
    let graph = Graph::default();
    let x = graph.input(&[2, 3, 4]).unwrap();
    assert!(
        x.rotary_embedding_angles(&graph.input(&[3, 4]).unwrap(), RotaryLayout::SplitHalf)
            .is_err()
    );
    assert!(
        x.rotary_embedding_angles(
            &Graph::default().input(&[3, 2]).unwrap(),
            RotaryLayout::SplitHalf
        )
        .is_err()
    );
    assert_eq!(
        x.rotary_embedding_angles(&graph.input(&[3, 2]).unwrap(), RotaryLayout::SplitHalf)
            .unwrap()
            .shape(),
        [2, 3, 4]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rope_runtime_positions_and_frequency_gradients_for_both_layouts() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for layout in [RotaryLayout::SplitHalf, RotaryLayout::Interleaved] {
        let graph = Graph::default();
        let x = graph.input(&[2, 3, 4]).unwrap();
        let positions = graph.input(&[3, 1]).unwrap();
        let frequencies = graph.input(&[2]).unwrap();
        let angles = positions
            .broadcast_to(&[3, 2])
            .unwrap()
            .mul(&frequencies.broadcast_to(&[3, 2]).unwrap())
            .unwrap();
        let y = x.rotary_embedding_angles(&angles, layout).unwrap();
        let weights: Vec<_> = (0..24).map(|i| ((i % 5) as f32 - 2.) * 0.25).collect();
        let loss = y
            .mul(&graph.constant(&[2, 3, 4], &weights).unwrap())
            .unwrap()
            .sum(&[0, 1, 2], false)
            .unwrap();
        let gradients = loss.grad(&[x, positions, frequencies]).unwrap();
        let executable = graph
            .compile_many(
                &client,
                &[
                    y,
                    gradients[0].clone(),
                    gradients[1].clone(),
                    gradients[2].clone(),
                ],
            )
            .unwrap();
        for (positions, frequencies) in [
            ([0_f32, 1., 3.], [0.125_f32, 0.5]),
            ([2., 4., 5.], [0.25, -0.125]),
        ] {
            let values: Vec<_> = (0..24).map(|i| (i as f32 - 10.) * 0.25).collect();
            let actual = executable
                .run_many(&[&values, &positions, &frequencies])
                .unwrap();
            let mut expected = [vec![0_f64; 24], vec![0.; 24], vec![0.; 3], vec![0.; 2]];
            for batch in 0..2 {
                for (p, &position) in positions.iter().enumerate() {
                    for (f, &frequency) in frequencies.iter().enumerate() {
                        let base = (batch * 3 + p) * 4;
                        let (a, b) = match layout {
                            RotaryLayout::SplitHalf => (base + f, base + f + 2),
                            RotaryLayout::Interleaved => (base + 2 * f, base + 2 * f + 1),
                        };
                        let angle = position as f64 * frequency as f64;
                        let (s, c) = angle.sin_cos();
                        let (va, vb) = (values[a] as f64, values[b] as f64);
                        let (wa, wb) = (weights[a] as f64, weights[b] as f64);
                        expected[0][a] = va * c - vb * s;
                        expected[0][b] = vb * c + va * s;
                        expected[1][a] = wa * c + wb * s;
                        expected[1][b] = wb * c - wa * s;
                        let da = wa * (-va * s - vb * c) + wb * (-vb * s + va * c);
                        expected[2][p] += da * frequency as f64;
                        expected[3][f] += da * position as f64;
                    }
                }
            }
            for (actual, expected) in actual.iter().zip(expected) {
                for (&actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        (actual as f64 - expected).abs() < 5e-6,
                        "layout={layout:?}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
