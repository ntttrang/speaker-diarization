use std::fs::File;
use std::path::Path;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::types::{AudioBuffer16kMono, DiarizationError};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const RESAMPLER_CHUNK_FRAMES: usize = 1024;

/// Interleaved `f32` PCM decoded from a container's default audio track.
struct DecodedAudio {
    interleaved: Vec<f32>,
    channels: usize,
    sample_rate: u32,
}

/// Decodes an audio container into normalized 16 kHz mono samples.
pub fn decode_to_16k_mono(path: &Path) -> Result<AudioBuffer16kMono, DiarizationError> {
    let decoded = decode_interleaved(path)?;
    let mono = downmix_to_mono(&decoded.interleaved, decoded.channels);
    let resampled = resample_to_16k(&mono, decoded.sample_rate)?;
    let samples: Vec<f32> = resampled.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect();
    let duration_ms = (samples.len() as u64 * 1000) / u64::from(TARGET_SAMPLE_RATE);
    Ok(AudioBuffer16kMono {
        samples,
        duration_ms,
    })
}

fn decode_interleaved(path: &Path) -> Result<DecodedAudio, DiarizationError> {
    let file = File::open(path).map_err(|e| DiarizationError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(decode_error)?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| DiarizationError::Decode("no audio track found".to_owned()))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| DiarizationError::Decode("missing audio codec parameters".to_owned()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .map_err(decode_error)?;

    let mut decoded = DecodedAudio {
        interleaved: Vec::new(),
        channels: codec_params.channels.as_ref().map_or(1, |c| c.count()),
        sample_rate: codec_params.sample_rate.unwrap_or(TARGET_SAMPLE_RATE),
    };
    let mut packet_samples: Vec<f32> = Vec::new();

    while let Some(packet) = format.next_packet().map_err(decode_error)? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buffer) => {
                decoded.channels = buffer.spec().channels().count();
                decoded.sample_rate = buffer.spec().rate();
                buffer.copy_to_vec_interleaved(&mut packet_samples);
                decoded.interleaved.extend_from_slice(&packet_samples);
            }
            // A single malformed packet is recoverable; skip it and keep decoding.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(decode_error(err)),
        }
    }

    Ok(decoded)
}

fn decode_error(err: SymphoniaError) -> DiarizationError {
    match err {
        SymphoniaError::IoError(io) => DiarizationError::Io(io.to_string()),
        other => DiarizationError::Decode(other.to_string()),
    }
}

/// Averages all channels at each frame index into a single mono stream.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_to_16k(mono: &[f32], source_rate: u32) -> Result<Vec<f32>, DiarizationError> {
    if source_rate == TARGET_SAMPLE_RATE || mono.is_empty() {
        return Ok(mono.to_vec());
    }
    let resample_error = |e: &dyn std::fmt::Display| DiarizationError::Resample(e.to_string());

    let ratio = f64::from(TARGET_SAMPLE_RATE) / f64::from(source_rate);
    let mut resampler = Async::<f32>::new_sinc(
        ratio,
        1.0,
        &SincInterpolationParameters::default(),
        RESAMPLER_CHUNK_FRAMES,
        1,
        FixedAsync::Input,
    )
    .map_err(|e| resample_error(&e))?;

    let input = InterleavedSlice::new(mono, 1, mono.len()).map_err(|e| resample_error(&e))?;
    let output = resampler
        .process_all(&input, mono.len(), None)
        .map_err(|e| resample_error(&e))?;
    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    /// Temporary file removed on drop so failed assertions don't leak fixtures.
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(name: &str, bytes: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("diarization-audio-{}-{name}", std::process::id()));
            File::create(&path).unwrap().write_all(bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Encodes interleaved 16-bit PCM as a canonical RIFF/WAVE file.
    fn wav_bytes(sample_rate: u32, channels: u16, interleaved: &[i16]) -> Vec<u8> {
        let block_align = channels * 2;
        let data_len = (interleaved.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in interleaved {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn to_i16(value: f32) -> i16 {
        (value * f32::from(i16::MAX)).round() as i16
    }

    fn stereo_fixture(
        name: &str,
        rate: u32,
        frames: usize,
        f: impl Fn(usize) -> (f32, f32),
    ) -> TempFile {
        let interleaved: Vec<i16> = (0..frames)
            .flat_map(|i| {
                let (left, right) = f(i);
                [to_i16(left), to_i16(right)]
            })
            .collect();
        TempFile::new(name, &wav_bytes(rate, 2, &interleaved))
    }

    #[test]
    fn stereo_48k_fixture_resamples_to_16k_mono_with_matching_duration() {
        let duration_sec = 1.5;
        let frames = (48_000.0 * duration_sec) as usize;
        let fixture = stereo_fixture("stereo48k.wav", 48_000, frames, |_| (0.6, 0.2));

        let audio = decode_to_16k_mono(&fixture.0).unwrap();

        let expected_len = (duration_sec * 16_000.0) as i64;
        assert!((audio.samples.len() as i64 - expected_len).abs() <= 1);
        assert_eq!(audio.duration_ms, 1_500);
        assert!((audio.samples.len() as i64 - (audio.duration_ms * 16) as i64).abs() <= 1);

        // Away from the resampler's edge transients, the constant channels average to 0.4.
        let middle = &audio.samples[2_000..audio.samples.len() - 2_000];
        assert!(middle.iter().all(|s| (s - 0.4).abs() < 1e-2));
        assert!(audio.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
    }

    #[test]
    fn stereo_channels_are_averaged_per_frame() {
        let frames = 4_000;
        let left = |i: usize| (i as f32 / frames as f32) - 0.5;
        let right = |i: usize| if i.is_multiple_of(2) { 0.8 } else { -0.3 };
        let fixture = stereo_fixture("stereo16k.wav", 16_000, frames, |i| (left(i), right(i)));

        let audio = decode_to_16k_mono(&fixture.0).unwrap();

        assert_eq!(audio.samples.len(), frames);
        assert_eq!(audio.duration_ms, 250);
        for (i, sample) in audio.samples.iter().enumerate() {
            let expected = (left(i) + right(i)) / 2.0;
            assert!(
                (sample - expected).abs() < 1e-3,
                "frame {i}: {sample} != {expected}"
            );
        }
    }

    #[test]
    fn full_scale_input_stays_within_unit_range_after_resampling() {
        // 1 kHz full-scale square wave: sinc resampling rings past +/-1.0 without clamping.
        let fixture = stereo_fixture("square48k.wav", 48_000, 48_000, |i| {
            let level = if (i / 24) % 2 == 0 { 1.0 } else { -1.0 };
            (level, level)
        });

        let audio = decode_to_16k_mono(&fixture.0).unwrap();

        assert!(!audio.samples.is_empty());
        assert!(audio.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
    }

    #[test]
    fn corrupt_container_returns_decode_error() {
        let fixture = TempFile::new("corrupt.wav", b"definitely not an audio container");

        let err = decode_to_16k_mono(&fixture.0).unwrap_err();

        assert!(matches!(err, DiarizationError::Decode(_)), "got {err:?}");
    }

    #[test]
    fn missing_file_returns_io_error() {
        let path = std::env::temp_dir().join("diarization-audio-does-not-exist.wav");

        let err = decode_to_16k_mono(&path).unwrap_err();

        assert!(matches!(err, DiarizationError::Io(_)), "got {err:?}");
    }
}
