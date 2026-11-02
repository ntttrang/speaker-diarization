use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ort::environment::Environment;
use ort::session::Session;

use super::audio::decode_to_16k_mono;
use super::clustering::diarize_turn_windows;
use super::embedding::{EmbeddingExtractor, WeSpeakerOnnxExtractor, EMBEDDING_DIM};
use super::segmentation::{load_segmentation_session, segment_window, window_bounds};
use super::types::{
    AudioBuffer16kMono, DiarizationError, DiarizationOutput, DiarizationProgress, RawTurnWindow,
};

pub const STAGE_DECODING: &str = "Decoding audio";
pub const STAGE_SEGMENTATION: &str = "Segmenting speech";
pub const STAGE_CLUSTERING: &str = "Clustering speakers";
pub const STAGE_COMPLETE: &str = "Complete";

const DECODING_START_PERCENT: f32 = 0.0;
const SEGMENTATION_START_PERCENT: f32 = 15.0;
const CLUSTERING_START_PERCENT: f32 = 70.0;
const COMPLETE_PERCENT: f32 = 100.0;

/// Environment variable overriding the directory holding the ONNX models.
pub const MODEL_DIR_ENV_VAR: &str = "DIARIZATION_MODEL_DIR";
const DEFAULT_MODEL_DIR: &str = "models";
const SEGMENTATION_MODEL_FILE: &str = "segmentation-3.0.onnx";
const EMBEDDING_MODEL_FILE: &str = "wespeaker_en_voxceleb_CAM++.onnx";
const ENVIRONMENT_NAME: &str = "diarization";

/// Cooperative cancellation flag shared between the caller and a running pipeline.
#[derive(Debug, Clone, Default)]
pub struct DiarizationCancelToken(pub Arc<AtomicBool>);

impl DiarizationCancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears a prior cancellation so the token can be reused for a new run.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Runs decoding, sliding-window segmentation and speaker clustering for the
/// recording at `path`, streaming stage progress through `on_progress`.
///
/// Cancellation is polled before every stage, every 10 s segmentation window and
/// every speaker embedding; once observed, intermediate audio, turn windows and
/// ONNX sessions are dropped and `Err(DiarizationError::Cancelled)` is returned.
pub fn execute_diarization_pipeline(
    path: &Path,
    cancel_token: &DiarizationCancelToken,
    on_progress: &(dyn Fn(DiarizationProgress) + Send + Sync),
) -> Result<DiarizationOutput, DiarizationError> {
    let mut stages = OnnxPipelineStages {
        environment: shared_environment()?,
        model_dir: model_dir(),
        segmentation: None,
    };
    run_pipeline(&mut stages, path, cancel_token, on_progress)
}

/// Pipeline steps behind a seam so the coordinator can be exercised without ONNX models.
trait PipelineStages {
    fn decode(&mut self, path: &Path) -> Result<AudioBuffer16kMono, DiarizationError>;

    fn segment_window(
        &mut self,
        audio: &AudioBuffer16kMono,
        bounds: (usize, usize),
    ) -> Result<Vec<RawTurnWindow>, DiarizationError>;

    fn cluster(
        &mut self,
        audio: &AudioBuffer16kMono,
        windows: &[RawTurnWindow],
        cancel_token: &DiarizationCancelToken,
    ) -> Result<DiarizationOutput, DiarizationError>;
}

fn run_pipeline<S: PipelineStages>(
    stages: &mut S,
    path: &Path,
    cancel_token: &DiarizationCancelToken,
    on_progress: &(dyn Fn(DiarizationProgress) + Send + Sync),
) -> Result<DiarizationOutput, DiarizationError> {
    ensure_not_cancelled(cancel_token)?;
    report(on_progress, STAGE_DECODING, DECODING_START_PERCENT);
    let audio = stages.decode(path)?;

    ensure_not_cancelled(cancel_token)?;
    report(on_progress, STAGE_SEGMENTATION, SEGMENTATION_START_PERCENT);
    let windows = segment_with_progress(stages, &audio, cancel_token, on_progress)?;

    ensure_not_cancelled(cancel_token)?;
    report(on_progress, STAGE_CLUSTERING, CLUSTERING_START_PERCENT);
    let output = stages.cluster(&audio, &windows, cancel_token)?;

    ensure_not_cancelled(cancel_token)?;
    report(on_progress, STAGE_COMPLETE, COMPLETE_PERCENT);
    Ok(output)
}

fn segment_with_progress<S: PipelineStages>(
    stages: &mut S,
    audio: &AudioBuffer16kMono,
    cancel_token: &DiarizationCancelToken,
    on_progress: &(dyn Fn(DiarizationProgress) + Send + Sync),
) -> Result<Vec<RawTurnWindow>, DiarizationError> {
    let bounds = window_bounds(audio.samples.len());
    let total = bounds.len();
    let span = CLUSTERING_START_PERCENT - SEGMENTATION_START_PERCENT;

    let mut turns = Vec::new();
    for (index, window) in bounds.into_iter().enumerate() {
        ensure_not_cancelled(cancel_token)?;
        turns.extend(stages.segment_window(audio, window)?);
        let completed = (index + 1) as f32 / total as f32;
        report(
            on_progress,
            STAGE_SEGMENTATION,
            SEGMENTATION_START_PERCENT + span * completed,
        );
    }
    Ok(turns)
}

fn ensure_not_cancelled(cancel_token: &DiarizationCancelToken) -> Result<(), DiarizationError> {
    if cancel_token.is_cancelled() {
        Err(DiarizationError::Cancelled)
    } else {
        Ok(())
    }
}

fn report(on_progress: &(dyn Fn(DiarizationProgress) + Send + Sync), stage: &str, percent: f32) {
    on_progress(DiarizationProgress {
        stage: stage.to_string(),
        percent: percent.clamp(DECODING_START_PERCENT, COMPLETE_PERCENT),
    });
}

/// Production stages backed by the pyannote segmentation and WeSpeaker ONNX models.
struct OnnxPipelineStages {
    environment: &'static Environment,
    model_dir: PathBuf,
    segmentation: Option<Session>,
}

impl PipelineStages for OnnxPipelineStages {
    fn decode(&mut self, path: &Path) -> Result<AudioBuffer16kMono, DiarizationError> {
        decode_to_16k_mono(path)
    }

    fn segment_window(
        &mut self,
        audio: &AudioBuffer16kMono,
        bounds: (usize, usize),
    ) -> Result<Vec<RawTurnWindow>, DiarizationError> {
        let session = match self.segmentation.take() {
            Some(session) => session,
            None => load_segmentation_session(
                self.environment,
                &self.model_dir.join(SEGMENTATION_MODEL_FILE),
            )?,
        };
        segment_window(audio, self.segmentation.insert(session), bounds)
    }

    fn cluster(
        &mut self,
        audio: &AudioBuffer16kMono,
        windows: &[RawTurnWindow],
        cancel_token: &DiarizationCancelToken,
    ) -> Result<DiarizationOutput, DiarizationError> {
        // Release the segmentation session before the embedding model is loaded.
        self.segmentation = None;
        let extractor = WeSpeakerOnnxExtractor::load(
            self.environment,
            &self.model_dir.join(EMBEDDING_MODEL_FILE),
        )?;
        let mut cancellable = CancellableExtractor {
            inner: extractor,
            cancel_token,
        };
        diarize_turn_windows(audio, windows, &mut cancellable)
    }
}

/// Polls the cancellation token before every embedding inference.
struct CancellableExtractor<'a, E> {
    inner: E,
    cancel_token: &'a DiarizationCancelToken,
}

impl<E: EmbeddingExtractor> EmbeddingExtractor for CancellableExtractor<'_, E> {
    fn extract_embedding(
        &mut self,
        samples_16k: &[f32],
    ) -> Result<[f32; EMBEDDING_DIM], DiarizationError> {
        ensure_not_cancelled(self.cancel_token)?;
        self.inner.extract_embedding(samples_16k)
    }
}

/// `ort` permits a single environment per process, so it is created once and
/// shared by every diarization run.
fn shared_environment() -> Result<&'static Environment, DiarizationError> {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    static INIT_LOCK: Mutex<()> = Mutex::new(());

    if let Some(environment) = ENVIRONMENT.get() {
        return Ok(environment);
    }
    let _guard = INIT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(environment) = ENVIRONMENT.get() {
        return Ok(environment);
    }
    let environment = ort::init()
        .with_name(ENVIRONMENT_NAME)
        .build()
        .map_err(|err| DiarizationError::Inference(err.to_string()))?;
    Ok(ENVIRONMENT.get_or_init(|| environment))
}

fn model_dir() -> PathBuf {
    std::env::var_os(MODEL_DIR_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::diarization::types::SpeakerSegment;

    const SAMPLES_PER_SECOND: usize = 16_000;

    /// Stand-in stages producing one single-speaker turn per window.
    struct MockStages {
        audio_seconds: usize,
        window_delay: Duration,
        segmented_windows: usize,
    }

    impl MockStages {
        fn new(audio_seconds: usize, window_delay: Duration) -> Self {
            Self {
                audio_seconds,
                window_delay,
                segmented_windows: 0,
            }
        }
    }

    impl PipelineStages for MockStages {
        fn decode(&mut self, _path: &Path) -> Result<AudioBuffer16kMono, DiarizationError> {
            Ok(AudioBuffer16kMono {
                samples: vec![0.0; self.audio_seconds * SAMPLES_PER_SECOND],
                duration_ms: self.audio_seconds as u64 * 1000,
            })
        }

        fn segment_window(
            &mut self,
            _audio: &AudioBuffer16kMono,
            (start, end): (usize, usize),
        ) -> Result<Vec<RawTurnWindow>, DiarizationError> {
            thread::sleep(self.window_delay);
            self.segmented_windows += 1;
            let to_ms = |sample: usize| (sample * 1000 / SAMPLES_PER_SECOND) as u64;
            Ok(vec![RawTurnWindow {
                start_ms: to_ms(start),
                end_ms: to_ms(end),
                local_speaker_slot: 0,
                is_overlap: false,
            }])
        }

        fn cluster(
            &mut self,
            _audio: &AudioBuffer16kMono,
            windows: &[RawTurnWindow],
            _cancel_token: &DiarizationCancelToken,
        ) -> Result<DiarizationOutput, DiarizationError> {
            let segments = windows
                .iter()
                .map(|window| SpeakerSegment {
                    speaker_label: "SPEAKER_00".to_string(),
                    start_ms: window.start_ms,
                    end_ms: window.end_ms,
                    is_overlap: window.is_overlap,
                })
                .collect();
            Ok(DiarizationOutput {
                speaker_count: 1,
                segments,
            })
        }
    }

    fn collecting_callback() -> (
        Arc<Mutex<Vec<DiarizationProgress>>>,
        impl Fn(DiarizationProgress) + Send + Sync,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        (events, move |progress| sink.lock().unwrap().push(progress))
    }

    #[test]
    fn cancel_is_visible_across_cloned_handles() {
        let token = DiarizationCancelToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());

        token.cancel();

        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());
    }

    #[test]
    fn progress_is_monotonic_and_spans_all_stages() {
        let mut stages = MockStages::new(30, Duration::ZERO);
        let token = DiarizationCancelToken::new();
        let (events, on_progress) = collecting_callback();

        let output = run_pipeline(&mut stages, Path::new("meeting.wav"), &token, &on_progress)
            .expect("pipeline should complete");

        let events = events.lock().unwrap();
        let percents: Vec<f32> = events.iter().map(|event| event.percent).collect();
        assert_eq!(percents.first(), Some(&0.0));
        assert_eq!(percents.last(), Some(&100.0));
        assert!(percents.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(percents.iter().all(|p| (0.0..=100.0).contains(p)));
        assert!(events.iter().all(|event| !event.stage.is_empty()));

        let stages_seen: Vec<&str> = events.iter().map(|event| event.stage.as_str()).collect();
        for stage in [
            STAGE_DECODING,
            STAGE_SEGMENTATION,
            STAGE_CLUSTERING,
            STAGE_COMPLETE,
        ] {
            assert!(stages_seen.contains(&stage), "missing stage {stage}");
        }

        // 30 s of audio yields 9 windows (starts 0..=20 s every 2.5 s).
        let window_events = events
            .iter()
            .filter(|event| event.stage == STAGE_SEGMENTATION)
            .count();
        assert_eq!(window_events, 1 + 9);
        assert_eq!(output.segments.len(), 9);
    }

    #[test]
    fn cancel_mid_segmentation_halts_within_50ms_without_output() {
        // 120 s of audio yields 45 windows at 10 ms each (~450 ms uncancelled).
        let mut stages = MockStages::new(120, Duration::from_millis(10));
        let token = DiarizationCancelToken::new();
        let (tx, rx) = mpsc::channel::<f32>();
        let tx = Mutex::new(tx);
        let on_progress = move |progress: DiarizationProgress| {
            let _ = tx.lock().unwrap().send(progress.percent);
        };

        let canceller_token = token.clone();
        let canceller = thread::spawn(move || {
            let mut window_events = 0;
            while let Ok(percent) = rx.recv() {
                if percent > SEGMENTATION_START_PERCENT {
                    window_events += 1;
                }
                if window_events == 3 {
                    canceller_token.cancel();
                    return Some(Instant::now());
                }
            }
            None
        });

        let result = run_pipeline(&mut stages, Path::new("meeting.wav"), &token, &on_progress);
        let returned_at = Instant::now();
        let cancelled_at = canceller.join().unwrap().expect("cancel was triggered");

        assert_eq!(result, Err(DiarizationError::Cancelled));
        assert!(returned_at.duration_since(cancelled_at) < Duration::from_millis(50));
        assert!(stages.segmented_windows < 45);
    }

    #[test]
    fn cancelled_before_start_skips_all_stages() {
        let mut stages = MockStages::new(30, Duration::ZERO);
        let token = DiarizationCancelToken::new();
        token.cancel();
        let (events, on_progress) = collecting_callback();

        let result = run_pipeline(&mut stages, Path::new("meeting.wav"), &token, &on_progress);

        assert_eq!(result, Err(DiarizationError::Cancelled));
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(stages.segmented_windows, 0);
    }

    #[test]
    fn reset_clears_prior_cancellation_for_next_run() {
        let mut stages = MockStages::new(10, Duration::ZERO);
        let token = DiarizationCancelToken::new();
        token.cancel();
        let (_events, on_progress) = collecting_callback();
        assert_eq!(
            run_pipeline(&mut stages, Path::new("meeting.wav"), &token, &on_progress),
            Err(DiarizationError::Cancelled)
        );

        token.reset();

        assert!(!token.is_cancelled());
        let output = run_pipeline(&mut stages, Path::new("meeting.wav"), &token, &on_progress)
            .expect("reset token should allow a fresh run");
        assert_eq!(output.speaker_count, 1);
    }

    #[test]
    fn cancellable_extractor_stops_before_inference() {
        struct CountingExtractor(usize);
        impl EmbeddingExtractor for CountingExtractor {
            fn extract_embedding(
                &mut self,
                _samples_16k: &[f32],
            ) -> Result<[f32; EMBEDDING_DIM], DiarizationError> {
                self.0 += 1;
                Ok([0.0; EMBEDDING_DIM])
            }
        }

        let token = DiarizationCancelToken::new();
        let mut extractor = CancellableExtractor {
            inner: CountingExtractor(0),
            cancel_token: &token,
        };
        assert!(extractor.extract_embedding(&[0.0; 16]).is_ok());

        token.cancel();

        assert_eq!(
            extractor.extract_embedding(&[0.0; 16]),
            Err(DiarizationError::Cancelled)
        );
        assert_eq!(extractor.inner.0, 1);
    }
}
