use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use tauri::ipc::Channel;
use tauri::State;

use crate::diarization::runtime::{execute_diarization_pipeline, DiarizationCancelToken};
use crate::diarization::types::{DiarizationError, DiarizationOutput, DiarizationProgress};

/// Tauri-managed state holding the cancellation token shared between an
/// in-flight diarization run and the `cancel_diarization` command.
#[derive(Default)]
pub struct DiarizationState(Mutex<DiarizationCancelToken>);

impl DiarizationState {
    /// A cloned handle onto the shared cancellation flag.
    fn token(&self) -> DiarizationCancelToken {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[tauri::command]
pub async fn run_diarization(
    path: String,
    state: State<'_, DiarizationState>,
    channel: Channel<DiarizationProgress>,
) -> Result<DiarizationOutput, DiarizationError> {
    run_diarization_with(PathBuf::from(path), state.token(), move |progress| {
        let _ = channel.send(progress);
    })
    .await
}

#[tauri::command]
pub fn cancel_diarization(state: State<'_, DiarizationState>) {
    state.token().cancel();
}

/// Resets `cancel_token` for this run and executes the diarization pipeline on
/// a blocking thread, decoupled from Tauri's IPC types so it can be exercised
/// directly in tests.
async fn run_diarization_with(
    path: PathBuf,
    cancel_token: DiarizationCancelToken,
    on_progress: impl Fn(DiarizationProgress) + Send + Sync + 'static,
) -> Result<DiarizationOutput, DiarizationError> {
    run_pipeline_on_blocking_thread(
        execute_diarization_pipeline,
        path,
        cancel_token,
        on_progress,
    )
    .await
}

/// Resets `cancel_token`, then runs `pipeline` on a blocking thread and maps a
/// panicked/cancelled task to a `DiarizationError`. Generic over `pipeline` so
/// the reset/dispatch/error-mapping behavior can be tested without the real,
/// ONNX-backed pipeline.
async fn run_pipeline_on_blocking_thread(
    pipeline: impl FnOnce(
            &Path,
            &DiarizationCancelToken,
            &(dyn Fn(DiarizationProgress) + Send + Sync),
        ) -> Result<DiarizationOutput, DiarizationError>
        + Send
        + 'static,
    path: PathBuf,
    cancel_token: DiarizationCancelToken,
    on_progress: impl Fn(DiarizationProgress) + Send + Sync + 'static,
) -> Result<DiarizationOutput, DiarizationError> {
    cancel_token.reset();
    tokio::task::spawn_blocking(move || pipeline(&path, &cancel_token, &on_progress))
        .await
        .unwrap_or_else(|join_err| Err(DiarizationError::Inference(join_err.to_string())))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn collecting_callback() -> (
        Arc<Mutex<Vec<DiarizationProgress>>>,
        impl Fn(DiarizationProgress) + Send + Sync + 'static,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        (events, move |progress| sink.lock().unwrap().push(progress))
    }

    /// Stands in for `execute_diarization_pipeline`: reports one progress
    /// event and fails, recording whether `cancel_token` was already
    /// cancelled when the pipeline started running.
    fn fake_pipeline(
        was_cancelled_on_entry: Arc<Mutex<Option<bool>>>,
    ) -> impl FnOnce(
        &Path,
        &DiarizationCancelToken,
        &(dyn Fn(DiarizationProgress) + Send + Sync),
    ) -> Result<DiarizationOutput, DiarizationError> {
        move |_path, cancel_token, on_progress| {
            *was_cancelled_on_entry.lock().unwrap() = Some(cancel_token.is_cancelled());
            on_progress(DiarizationProgress {
                stage: "Decoding audio".to_string(),
                percent: 0.0,
            });
            Err(DiarizationError::Io("no such file".to_string()))
        }
    }

    #[tokio::test]
    async fn resets_a_previously_cancelled_token_before_running() {
        let token = DiarizationCancelToken::new();
        token.cancel();
        let (_events, on_progress) = collecting_callback();
        let was_cancelled_on_entry = Arc::new(Mutex::new(None));

        let result = run_pipeline_on_blocking_thread(
            fake_pipeline(Arc::clone(&was_cancelled_on_entry)),
            PathBuf::from("does-not-exist.wav"),
            token,
            on_progress,
        )
        .await;

        // A stale cancellation from a prior run must not block this one: the
        // pipeline should run with a freshly reset token rather than the
        // caller having to clear it beforehand.
        assert_eq!(*was_cancelled_on_entry.lock().unwrap(), Some(false));
        assert!(matches!(result, Err(DiarizationError::Io(_))));
    }

    #[tokio::test]
    async fn nonexistent_path_reports_decoding_progress_then_fails() {
        let token = DiarizationCancelToken::new();
        let (events, on_progress) = collecting_callback();

        let result = run_pipeline_on_blocking_thread(
            fake_pipeline(Arc::new(Mutex::new(None))),
            PathBuf::from("does-not-exist.wav"),
            token,
            on_progress,
        )
        .await;

        assert!(matches!(result, Err(DiarizationError::Io(_))));
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].stage, "Decoding audio");
        assert_eq!(events[0].percent, 0.0);
    }

    #[test]
    fn state_token_shares_cancellation_across_handles() {
        let state = DiarizationState::default();
        let handle = state.token();
        assert!(!handle.is_cancelled());

        state.token().cancel();

        assert!(handle.is_cancelled());
    }
}
