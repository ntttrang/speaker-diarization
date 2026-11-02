use std::fmt;

use serde::{Deserialize, Serialize};

/// Normalized mono audio at 16 kHz with samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer16kMono {
    pub samples: Vec<f32>,
    pub duration_ms: u64,
}

/// A contiguous span of activity for one local (per-window) speaker slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawTurnWindow {
    pub start_ms: u64,
    pub end_ms: u64,
    pub local_speaker_slot: u8,
    pub is_overlap: bool,
}

/// A speech segment attributed to a global speaker identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub speaker_label: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_overlap: bool,
}

/// Final diarization result for a recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationOutput {
    pub speaker_count: usize,
    pub segments: Vec<SpeakerSegment>,
}

/// Progress event emitted while a diarization run is in flight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationProgress {
    pub stage: String,
    pub percent: f32,
}

/// Error shared by every stage of the diarization pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiarizationError {
    /// The input file could not be opened or read.
    Io(String),
    /// The container could not be demuxed or its audio packets could not be decoded.
    Decode(String),
    /// Sample-rate conversion failed.
    Resample(String),
    /// ONNX model loading or inference failed.
    Inference(String),
    /// The run was cancelled before completion.
    Cancelled,
}

impl fmt::Display for DiarizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Decode(msg) => write!(f, "audio decode error: {msg}"),
            Self::Resample(msg) => write!(f, "resampling error: {msg}"),
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
            Self::Cancelled => write!(f, "diarization cancelled"),
        }
    }
}

impl std::error::Error for DiarizationError {}
