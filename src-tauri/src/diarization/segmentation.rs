use std::path::Path;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use super::types::{AudioBuffer16kMono, DiarizationError, RawTurnWindow};

const SAMPLE_RATE: u64 = 16_000;
const WINDOW_SAMPLES: usize = 160_000;
const HOP_SAMPLES: usize = 40_000;
const WINDOW_MS: u64 = WINDOW_SAMPLES as u64 * 1000 / SAMPLE_RATE;
const LOCAL_SPEAKER_SLOTS: usize = 3;
const POWERSET_CLASSES: usize = 7;

/// Per-frame activity across local speaker slots plus the overlap flag.
type FrameActivity = ([bool; LOCAL_SPEAKER_SLOTS], bool);

/// Loads the `pyannote/segmentation-3.0.onnx` model into an in-process ONNX session.
///
/// `ort` allows a single [`Environment`] per process, so the caller owns it and
/// shares it with every model session in the pipeline.
pub fn load_segmentation_session(
    environment: &Environment,
    model_path: &Path,
) -> Result<Session, DiarizationError> {
    Session::builder(environment)
        .and_then(|mut builder| builder.commit_from_file(model_path))
        .map_err(inference_error)
}

/// Maps the argmax of the 7-class powerset `{∅, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}}`
/// to per-slot activity and whether the winning class is a multi-speaker class.
pub fn decode_powerset_frame(logits: &[f32; 7]) -> ([bool; 3], bool) {
    let winner = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (idx, &logit)| {
            if logit > best.1 {
                (idx, logit)
            } else {
                best
            }
        })
        .0;

    match winner {
        1 => ([true, false, false], false),
        2 => ([false, true, false], false),
        3 => ([false, false, true], false),
        4 => ([true, true, false], true),
        5 => ([true, false, true], true),
        6 => ([false, true, true], true),
        _ => ([false, false, false], false),
    }
}

/// Runs powerset segmentation over 10 s windows with a 2.5 s hop and returns
/// per-window local speaker turns in absolute milliseconds.
pub fn segment_audio_windows(
    audio: &AudioBuffer16kMono,
    session: &mut Session,
) -> Result<Vec<RawTurnWindow>, DiarizationError> {
    let mut turns = Vec::new();
    for (start, end) in window_bounds(audio.samples.len()) {
        let mut chunk = vec![0.0_f32; WINDOW_SAMPLES];
        chunk[..end - start].copy_from_slice(&audio.samples[start..end]);

        let frames = infer_window_frames(session, chunk)?;
        if frames.is_empty() {
            continue;
        }
        let window_start_ms = start as u64 * 1000 / SAMPLE_RATE;
        let frame_ms = WINDOW_MS as f64 / frames.len() as f64;
        turns.extend(aggregate_frames(
            &frames,
            window_start_ms,
            frame_ms,
            audio.duration_ms,
        ));
    }
    Ok(turns)
}

fn infer_window_frames(
    session: &mut Session,
    chunk: Vec<f32>,
) -> Result<Vec<FrameActivity>, DiarizationError> {
    let input =
        Tensor::from_array(([1_usize, 1, WINDOW_SAMPLES], chunk)).map_err(inference_error)?;
    let outputs = session.run(ort::inputs![input]).map_err(inference_error)?;
    let (shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(inference_error)?;

    if shape.last() != Some(&(POWERSET_CLASSES as i64)) {
        return Err(DiarizationError::Inference(format!(
            "expected {POWERSET_CLASSES} powerset classes per frame, got output shape {shape:?}"
        )));
    }

    Ok(logits
        .as_chunks::<POWERSET_CLASSES>()
        .0
        .iter()
        .map(decode_powerset_frame)
        .collect())
}

fn inference_error(err: ort::Error) -> DiarizationError {
    DiarizationError::Inference(err.to_string())
}

/// Sample bounds `(start, end)` of each sliding window; `end` is clamped to the
/// audio length, and the caller zero-pads the remainder of a short final window.
fn window_bounds(total_samples: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < total_samples {
        bounds.push((start, (start + WINDOW_SAMPLES).min(total_samples)));
        if start + WINDOW_SAMPLES >= total_samples {
            break;
        }
        start += HOP_SAMPLES;
    }
    bounds
}

/// Merges contiguous active frames with identical overlap state into turns for
/// each local slot. Frames starting at or beyond `duration_ms` (zero padding)
/// are dropped and turn ends are clamped to `duration_ms`.
fn aggregate_frames(
    frames: &[FrameActivity],
    window_start_ms: u64,
    frame_ms: f64,
    duration_ms: u64,
) -> Vec<RawTurnWindow> {
    let frame_start = |idx: usize| window_start_ms + (idx as f64 * frame_ms).round() as u64;
    let audible_frames = frames
        .iter()
        .enumerate()
        .take_while(|&(idx, _)| frame_start(idx) < duration_ms);

    let mut turns = Vec::new();
    let mut open: [Option<RawTurnWindow>; LOCAL_SPEAKER_SLOTS] = Default::default();

    for (idx, (active, is_overlap)) in audible_frames {
        let start_ms = frame_start(idx);
        let end_ms = frame_start(idx + 1).min(duration_ms);
        for (slot, open_turn) in open.iter_mut().enumerate() {
            match open_turn {
                Some(turn) if active[slot] && turn.is_overlap == *is_overlap => {
                    turn.end_ms = end_ms;
                }
                _ => {
                    turns.extend(open_turn.take());
                    if active[slot] {
                        *open_turn = Some(RawTurnWindow {
                            start_ms,
                            end_ms,
                            local_speaker_slot: slot as u8,
                            is_overlap: *is_overlap,
                        });
                    }
                }
            }
        }
    }

    turns.extend(open.into_iter().flatten());
    turns.sort_by_key(|turn| (turn.start_ms, turn.local_speaker_slot));
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot(class: usize) -> [f32; POWERSET_CLASSES] {
        let mut logits = [-5.0; POWERSET_CLASSES];
        logits[class] = 0.0;
        logits
    }

    fn turn(start_ms: u64, end_ms: u64, slot: u8, is_overlap: bool) -> RawTurnWindow {
        RawTurnWindow {
            start_ms,
            end_ms,
            local_speaker_slot: slot,
            is_overlap,
        }
    }

    #[test]
    fn decode_powerset_frame_maps_each_class_to_slot_mask_and_overlap() {
        let expected = [
            ([false, false, false], false),
            ([true, false, false], false),
            ([false, true, false], false),
            ([false, false, true], false),
            ([true, true, false], true),
            ([true, false, true], true),
            ([false, true, true], true),
        ];
        for (class, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                decode_powerset_frame(&one_hot(class)),
                expected,
                "class {class}"
            );
        }
    }

    #[test]
    fn aggregate_frames_preserves_single_speaker_and_overlapping_turns() {
        // 20 s of 20 ms frames: {0} 0-5 s, silence 5-6 s, {0,1} 6-10 s, {1} 10-15 s, silence 15-20 s.
        let classes = [(250, 1), (50, 0), (200, 4), (250, 2), (250, 0)];
        let frames: Vec<FrameActivity> = classes
            .iter()
            .flat_map(|&(count, class)| std::iter::repeat_n(class, count))
            .map(|class| decode_powerset_frame(&one_hot(class)))
            .collect();
        assert_eq!(frames.len(), 1_000);

        let turns = aggregate_frames(&frames, 2_500, 20.0, 60_000);

        assert_eq!(
            turns,
            vec![
                turn(2_500, 7_500, 0, false),
                turn(8_500, 12_500, 0, true),
                turn(8_500, 12_500, 1, true),
                turn(12_500, 17_500, 1, false),
            ]
        );
    }

    #[test]
    fn aggregate_frames_clamps_trailing_frames_to_audio_duration() {
        let frames = vec![([true, false, false], false); 100];

        let turns = aggregate_frames(&frames, 10_000, 100.0, 12_340);

        assert_eq!(turns, vec![turn(10_000, 12_340, 0, false)]);
    }

    #[test]
    fn window_bounds_step_by_hop_and_clamp_final_window() {
        // 21 s of audio: windows start every 2.5 s until one reaches the end.
        let total = 21 * 16_000;
        let bounds = window_bounds(total);

        let starts: Vec<usize> = bounds.iter().map(|&(start, _)| start).collect();
        assert_eq!(starts, vec![0, 40_000, 80_000, 120_000, 160_000, 200_000]);
        assert!(starts
            .windows(2)
            .all(|pair| pair[1] - pair[0] == HOP_SAMPLES));
        assert!(bounds[..bounds.len() - 1]
            .iter()
            .all(|&(start, end)| end - start == WINDOW_SAMPLES));
        assert_eq!(bounds.last(), Some(&(200_000, total)));
    }

    #[test]
    fn window_bounds_handles_short_and_empty_audio() {
        assert_eq!(window_bounds(48_000), vec![(0, 48_000)]);
        assert_eq!(window_bounds(WINDOW_SAMPLES), vec![(0, WINDOW_SAMPLES)]);
        assert!(window_bounds(0).is_empty());
    }
}
