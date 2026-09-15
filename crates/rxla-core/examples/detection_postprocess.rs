//! Device candidate selection followed by explicit host filtering/NMS.
//! Synthetic class-agnostic/class-aware postprocessing, not a YOLO importer.
use rxla_core::{CacheLimits, Client, Compiler, Graph, vision};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Graph::default();
    let boxes = graph.input(&[8, 4])?; // cxcywh model output
    let scores = graph.input(&[8])?; // finite confidence scores
    let classes = graph.input_i32(&[8])?;
    let (selected_scores, ids) = scores.topk(4, 0)?;
    let selected_boxes = boxes.take(&ids, 0)?.box_cxcywh_to_xyxy()?;
    let selected_classes = classes.take_along_axis(&ids, 0)?;
    let prepared =
        graph.prepare_outputs(&[selected_boxes, selected_scores, ids, selected_classes])?;
    let executable = compiler.compile_lowered(&prepared)?;
    let coordinates = [
        [1., 1., 2., 2.],
        [1., 1., 2., 2.], // identical boxes, different source IDs
        [4.5, 4.5, 1., 1.],
        [0.5, 1., 1., 2.], // disjoint and IoU exactly .5
        [8., 8., 1., 1.],
        [10., 10., 1., 1.],
        [12., 12., 1., 1.],
        [14., 14., 1., 1.],
    ];
    let flat: Vec<_> = coordinates.into_iter().flatten().collect();
    let boxes = client.buffer(&[8, 4], &flat)?;
    // Opaque labels must remain I32, never round through F32.
    let labels = [i32::MIN, i32::MAX, i32::MIN, i32::MAX, 0, 1, 2, 3];
    let classes = client.buffer(&[8], &labels)?;
    for (scores, cutoff, expected_top, expected_kept, expected_class_kept) in [
        (
            [0.8, 0.9, 0.7, 0.6, 0.2, 0.1, 0.05, 0.],
            0.5,
            [1, 0, 2, 3],
            vec![1, 2, 3],
            vec![1, 0, 2],
        ),
        (
            [1., 1., 0.8, 0.7, 0.2, 0.1, 0.05, 0.],
            0.75,
            [0, 1, 2, 3],
            vec![0, 2],
            vec![0, 1, 2],
        ),
        ([0.1; 8], 0.5, [0, 1, 2, 3], vec![], vec![]),
    ] {
        let input = client.buffer(&[8], &scores)?;
        let output = executable.execute(&[&boxes, &input, &classes])?;
        assert_eq!(output[0].dimensions()?, [4, 4]);
        assert_eq!(output[1].dimensions()?, [4]);
        assert_eq!(output[2].dimensions()?, [4]);
        assert_eq!(output[3].dimensions()?, [4]);
        let bytes = output
            .iter()
            .map(|b| b.host_payload_bytes())
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum::<usize>();
        assert_eq!(bytes, 112); // K*4 F32 coordinates + K scores + K I32 IDs/classes
        // The only downloads are these four selected output buffers.
        let selected = output[0].to_vec::<f32>()?;
        let selected_scores = output[1].to_vec::<f32>()?;
        let original_ids = output[2].to_vec::<i32>()?;
        let selected_classes = output[3].to_vec::<i32>()?;
        assert_eq!(original_ids, expected_top);
        let mut filtered_boxes = Vec::new();
        let mut filtered_scores = Vec::new();
        let mut filtered_ids = Vec::new();
        let mut filtered_classes = Vec::new();
        for (index, &id) in original_ids.iter().enumerate() {
            let original = usize::try_from(id)?;
            let [cx, cy, w, h] = coordinates[original];
            assert_eq!(
                &selected[index * 4..index * 4 + 4],
                &[cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
            );
            assert_eq!(selected_scores[index], scores[original]);
            assert_eq!(selected_classes[index], labels[original]);
            if selected_scores[index] >= cutoff {
                filtered_boxes.push(selected[index * 4..index * 4 + 4].try_into()?);
                filtered_scores.push(selected_scores[index]);
                filtered_ids.push(id);
                filtered_classes.push(selected_classes[index]);
            }
        }
        let kept = vision::nms(&filtered_boxes, &filtered_scores, 0.5, 3)?;
        let original_kept: Vec<_> = kept.into_iter().map(|i| filtered_ids[i]).collect();
        assert_eq!(original_kept, expected_kept);
        let kept =
            vision::nms_by_class(&filtered_boxes, &filtered_scores, &filtered_classes, 0.5, 3)?;
        let original_kept: Vec<_> = kept.into_iter().map(|i| filtered_ids[i]).collect();
        assert_eq!(original_kept, expected_class_kept);
    }
    assert_eq!(compiler.stats().misses, 1);
    println!(
        "PASS: three candidate batches, stable top-k IDs and exact I32 classes, cxcywh conversion, explicit 112-byte downloads, confidence filter and both host NMS modes; one compilation."
    );
    println!(
        "Top-k truncation is explicit and may differ from NMS over all candidates. No GPU NMS, full model or throughput claim."
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_device_candidates_and_host_nms_preserve_source_indices() {
    run().unwrap();
}
