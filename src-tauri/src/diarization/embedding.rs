use std::f32::consts::PI;
use std::path::Path;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::types::{AudioBuffer16kMono, DiarizationError, RawTurnWindow};

pub const EMBEDDING_DIM: usize = 256;
pub const MEL_BINS: usize = 80;
pub const MIN_CLUSTERING_WINDOW_MS: u64 = 500;

const SAMPLE_RATE_HZ: f32 = 16_000.0;
const SAMPLES_PER_MS: u64 = 16;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const FFT_BINS: usize = FFT_SIZE / 2 + 1;
const MEL_LOW_HZ: f32 = 20.0;
const MEL_HIGH_HZ: f32 = 7_600.0;
const LOG_ENERGY_FLOOR: f32 = 1e-10;
const NORM_FLOOR: f32 = 1e-12;

/// Speaker embedding for one segmented turn window.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnEmbedding {
    pub window_index: usize,
    pub window: RawTurnWindow,
    pub embedding: [f32; EMBEDDING_DIM],
    /// Non-overlapping windows of at least [`MIN_CLUSTERING_WINDOW_MS`] seed the
    /// clustering; the rest are attributed to the nearest centroid afterwards.
    pub is_clustering_anchor: bool,
}

/// Produces an L2-normalized speaker embedding from 16 kHz mono samples.
pub trait EmbeddingExtractor {
    fn extract_embedding(
        &mut self,
        samples_16k: &[f32],
    ) -> Result<[f32; EMBEDDING_DIM], DiarizationError>;
}

/// In-process `wespeaker_en_voxceleb_CAM++.onnx` speaker embedding model.
pub struct WeSpeakerOnnxExtractor {
    session: Session,
}

impl WeSpeakerOnnxExtractor {
    /// Loads the WeSpeaker CAM++ model into an in-process ONNX session.
    ///
    /// `ort` allows a single [`Environment`] per process, so the caller owns it and
    /// shares it with every model session in the pipeline.
    pub fn load(environment: &Environment, model_path: &Path) -> Result<Self, DiarizationError> {
        Session::builder(environment)
            .and_then(|mut builder| builder.commit_from_file(model_path))
            .map(|session| Self { session })
            .map_err(inference_error)
    }
}

impl EmbeddingExtractor for WeSpeakerOnnxExtractor {
    fn extract_embedding(
        &mut self,
        samples_16k: &[f32],
    ) -> Result<[f32; EMBEDDING_DIM], DiarizationError> {
        let fbank = compute_log_mel_fbank_80(samples_16k);
        let num_frames = fbank.len();
        let features: Vec<f32> = fbank.into_iter().flatten().collect();

        let input = Tensor::from_array(([1_usize, num_frames, MEL_BINS], features))
            .map_err(inference_error)?;
        let outputs = self
            .session
            .run(ort::inputs![input])
            .map_err(inference_error)?;
        let (shape, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(inference_error)?;

        let embedding: [f32; EMBEDDING_DIM] = values.try_into().map_err(|_| {
            DiarizationError::Inference(format!(
                "expected a {EMBEDDING_DIM}-d speaker embedding, got output shape {shape:?}"
            ))
        })?;
        Ok(l2_normalize(embedding))
    }
}

fn inference_error(err: ort::Error) -> DiarizationError {
    DiarizationError::Inference(err.to_string())
}

/// Scales `vector` to unit Euclidean length; a zero vector stays zero.
pub fn l2_normalize(vector: [f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    let norm = vector
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(NORM_FLOOR);
    vector.map(|v| v / norm)
}

/// 80-bin log-mel filterbank over 25 ms Hamming-windowed frames with a 10 ms
/// shift, mean-normalized per mel bin across the utterance. Inputs shorter than
/// one frame are zero-padded to a single frame.
pub fn compute_log_mel_fbank_80(samples_16k: &[f32]) -> Vec<[f32; MEL_BINS]> {
    let padded_len = samples_16k.len().max(FRAME_LENGTH);
    let num_frames = 1 + (padded_len - FRAME_LENGTH) / FRAME_SHIFT;
    let window = hamming_window();
    let filters = mel_filterbank();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);

    let log_mel: Vec<[f32; MEL_BINS]> = (0..num_frames)
        .map(|frame_idx| {
            let start = frame_idx * FRAME_SHIFT;
            let mut buffer = vec![Complex::new(0.0, 0.0); FFT_SIZE];
            for (offset, (slot, weight)) in buffer.iter_mut().zip(&window).enumerate() {
                let sample = samples_16k.get(start + offset).copied().unwrap_or(0.0);
                *slot = Complex::new(sample * weight, 0.0);
            }
            fft.process(&mut buffer);

            let power: Vec<f32> = buffer[..FFT_BINS].iter().map(Complex::norm_sqr).collect();
            filters.map(|filter| {
                let energy: f32 = filter.iter().zip(&power).map(|(w, p)| w * p).sum();
                energy.max(LOG_ENERGY_FLOOR).ln()
            })
        })
        .collect();

    subtract_bin_means(log_mel)
}

fn subtract_bin_means(frames: Vec<[f32; MEL_BINS]>) -> Vec<[f32; MEL_BINS]> {
    let count = frames.len() as f32;
    let means = frames.iter().fold([0.0_f32; MEL_BINS], |acc, frame| {
        std::array::from_fn(|bin| acc[bin] + frame[bin] / count)
    });
    frames
        .into_iter()
        .map(|frame| std::array::from_fn(|bin| frame[bin] - means[bin]))
        .collect()
}

fn hamming_window() -> [f32; FRAME_LENGTH] {
    std::array::from_fn(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (FRAME_LENGTH - 1) as f32).cos())
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Triangular filters equally spaced on the mel scale between 20 Hz and 7600 Hz.
fn mel_filterbank() -> [[f32; FFT_BINS]; MEL_BINS] {
    let mel_low = hz_to_mel(MEL_LOW_HZ);
    let mel_step = (hz_to_mel(MEL_HIGH_HZ) - mel_low) / (MEL_BINS + 1) as f32;
    let bin_mels: [f32; FFT_BINS] =
        std::array::from_fn(|k| hz_to_mel(k as f32 * SAMPLE_RATE_HZ / FFT_SIZE as f32));

    std::array::from_fn(|m| {
        let left = mel_low + m as f32 * mel_step;
        let center = left + mel_step;
        let right = center + mel_step;
        bin_mels.map(|mel| {
            if mel > left && mel <= center {
                (mel - left) / mel_step
            } else if mel > center && mel < right {
                (right - mel) / mel_step
            } else {
                0.0
            }
        })
    })
}

/// Extracts an embedding for every window in input order, flagging
/// non-overlapping windows of at least [`MIN_CLUSTERING_WINDOW_MS`] as
/// clustering anchors.
pub fn extract_turn_embeddings<E: EmbeddingExtractor>(
    audio: &AudioBuffer16kMono,
    windows: &[RawTurnWindow],
    extractor: &mut E,
) -> Result<Vec<TurnEmbedding>, DiarizationError> {
    windows
        .iter()
        .enumerate()
        .map(|(window_index, window)| {
            let embedding = extractor.extract_embedding(window_samples(audio, window))?;
            let duration_ms = window.end_ms.saturating_sub(window.start_ms);
            Ok(TurnEmbedding {
                window_index,
                window: window.clone(),
                embedding,
                is_clustering_anchor: !window.is_overlap && duration_ms >= MIN_CLUSTERING_WINDOW_MS,
            })
        })
        .collect()
}

fn window_samples<'a>(audio: &'a AudioBuffer16kMono, window: &RawTurnWindow) -> &'a [f32] {
    let to_index = |ms: u64| {
        usize::try_from(ms.saturating_mul(SAMPLES_PER_MS))
            .unwrap_or(usize::MAX)
            .min(audio.samples.len())
    };
    let end = to_index(window.end_ms);
    &audio.samples[to_index(window.start_ms).min(end)..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the length of each audio slice it receives.
    struct MockExtractor {
        calls: Vec<usize>,
    }

    impl EmbeddingExtractor for MockExtractor {
        fn extract_embedding(
            &mut self,
            samples_16k: &[f32],
        ) -> Result<[f32; EMBEDDING_DIM], DiarizationError> {
            self.calls.push(samples_16k.len());
            let mut embedding = [0.0; EMBEDDING_DIM];
            embedding[0] = 1.0;
            Ok(embedding)
        }
    }

    fn sine(num_samples: usize, freq_hz: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|n| (2.0 * PI * freq_hz * n as f32 / SAMPLE_RATE_HZ).sin())
            .collect()
    }

    fn assert_fbank_shape(num_samples: usize, expected_frames: usize) {
        let fbank = compute_log_mel_fbank_80(&sine(num_samples, 440.0));

        assert_eq!(fbank.len(), expected_frames);
        assert!(fbank.iter().flatten().all(|v| v.is_finite()));
        for bin in 0..MEL_BINS {
            let mean = fbank.iter().map(|frame| frame[bin]).sum::<f32>() / fbank.len() as f32;
            assert!(mean.abs() < 1e-4, "bin {bin} mean {mean}");
        }
    }

    fn turn(start_ms: u64, end_ms: u64, is_overlap: bool) -> RawTurnWindow {
        RawTurnWindow {
            start_ms,
            end_ms,
            local_speaker_slot: 0,
            is_overlap,
        }
    }

    #[test]
    fn fbank_of_500ms_signal_has_48_mean_normalized_80_bin_frames() {
        assert_fbank_shape(8_000, 48);
    }

    #[test]
    fn fbank_of_1000ms_signal_has_98_mean_normalized_80_bin_frames() {
        assert_fbank_shape(16_000, 98);
    }

    #[test]
    fn fbank_pads_sub_frame_input_to_a_single_frame() {
        let fbank = compute_log_mel_fbank_80(&sine(100, 440.0));

        assert_eq!(fbank.len(), 1);
        assert!(fbank[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn l2_normalize_produces_unit_norm_vectors() {
        let vector: [f32; EMBEDDING_DIM] = std::array::from_fn(|i| (i as f32 * 0.37).sin() * 5.0);

        let normalized = l2_normalize(vector);

        let norm = normalized.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }

    #[test]
    fn l2_normalize_leaves_zero_vector_finite() {
        let normalized = l2_normalize([0.0; EMBEDDING_DIM]);

        assert!(normalized.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn extract_turn_embeddings_flags_long_non_overlapping_windows_as_anchors() {
        let audio = AudioBuffer16kMono {
            samples: vec![0.25; 32_000],
            duration_ms: 2_000,
        };
        let windows = vec![
            turn(0, 500, false),
            turn(500, 999, false),
            turn(1_000, 1_800, true),
            turn(1_200, 2_000, false),
            turn(1_900, 2_500, false),
        ];
        let mut extractor = MockExtractor { calls: Vec::new() };

        let embeddings = extract_turn_embeddings(&audio, &windows, &mut extractor).unwrap();

        let anchors: Vec<bool> = embeddings.iter().map(|e| e.is_clustering_anchor).collect();
        assert_eq!(anchors, vec![true, false, false, true, true]);
        let indices: Vec<usize> = embeddings.iter().map(|e| e.window_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        assert!(embeddings.iter().zip(&windows).all(|(e, w)| &e.window == w));
        // The final window extends past the audio and is clamped to its end.
        assert_eq!(extractor.calls, vec![8_000, 7_984, 12_800, 12_800, 1_600]);
    }
}
