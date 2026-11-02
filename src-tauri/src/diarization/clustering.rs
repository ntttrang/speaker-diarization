use kodama::{linkage, Method};

use super::embedding::{
    extract_turn_embeddings, l2_normalize, EmbeddingExtractor, TurnEmbedding, EMBEDDING_DIM,
};
use super::types::{
    AudioBuffer16kMono, DiarizationError, DiarizationOutput, RawTurnWindow, SpeakerSegment,
};

pub const DEFAULT_COSINE_DISTANCE_THRESHOLD: f32 = 0.65;
pub const COLLAR_MERGE_THRESHOLD_MS: u64 = 250;

type Embedding = [f32; EMBEDDING_DIM];

/// Extracts speaker embeddings for every turn window, clusters them into global
/// speaker identities and merges collar-adjacent turns.
pub fn diarize_turn_windows<E: EmbeddingExtractor>(
    audio: &AudioBuffer16kMono,
    windows: &[RawTurnWindow],
    extractor: &mut E,
) -> Result<DiarizationOutput, DiarizationError> {
    let turn_embeddings = extract_turn_embeddings(audio, windows, extractor)?;
    Ok(cluster_and_merge_turns(
        &turn_embeddings,
        DEFAULT_COSINE_DISTANCE_THRESHOLD,
        COLLAR_MERGE_THRESHOLD_MS,
    ))
}

/// Clusters anchor embeddings with average-linkage cosine AHC cut at
/// `distance_threshold`, attributes the remaining turns to the nearest cluster
/// centroid, and merges same-speaker turns separated by less than `collar_ms`.
///
/// When no turn qualifies as an anchor, every turn seeds the clustering.
pub fn cluster_and_merge_turns(
    turn_embeddings: &[TurnEmbedding],
    distance_threshold: f32,
    collar_ms: u64,
) -> DiarizationOutput {
    let anchor_indices: Vec<usize> = {
        let flagged: Vec<usize> = (0..turn_embeddings.len())
            .filter(|&idx| turn_embeddings[idx].is_clustering_anchor)
            .collect();
        if flagged.is_empty() {
            (0..turn_embeddings.len()).collect()
        } else {
            flagged
        }
    };
    if anchor_indices.is_empty() {
        return DiarizationOutput {
            speaker_count: 0,
            segments: Vec::new(),
        };
    }

    let anchors: Vec<&TurnEmbedding> = anchor_indices
        .iter()
        .map(|&idx| &turn_embeddings[idx])
        .collect();
    let anchor_embeddings: Vec<Embedding> = anchors.iter().map(|turn| turn.embedding).collect();
    let anchor_labels = label_by_first_appearance(
        &anchors,
        &cluster_anchors(&anchor_embeddings, distance_threshold),
    );
    let speaker_count = anchor_labels.iter().max().map_or(0, |max| max + 1);
    let centroids = compute_centroids(&anchor_embeddings, &anchor_labels, speaker_count);

    let mut turn_labels: Vec<Option<usize>> = vec![None; turn_embeddings.len()];
    for (&turn_idx, &label) in anchor_indices.iter().zip(&anchor_labels) {
        turn_labels[turn_idx] = Some(label);
    }

    let segments = turn_embeddings
        .iter()
        .zip(turn_labels)
        .map(|(turn, label)| SpeakerSegment {
            speaker_label: speaker_label(
                label.unwrap_or_else(|| nearest_centroid(&turn.embedding, &centroids)),
            ),
            start_ms: turn.window.start_ms,
            end_ms: turn.window.end_ms,
            is_overlap: turn.window.is_overlap,
        })
        .collect();

    DiarizationOutput {
        speaker_count,
        segments: merge_collar_adjacent(segments, collar_ms),
    }
}

fn speaker_label(index: usize) -> String {
    format!("SPEAKER_{index:02}")
}

fn cosine_distance(a: &Embedding, b: &Embedding) -> f32 {
    let similarity: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (1.0 - similarity).clamp(0.0, 2.0)
}

/// Raw cluster id per embedding from cutting the average-linkage dendrogram at
/// `distance_threshold`. Ids are arbitrary but equal ids share a cluster.
fn cluster_anchors(embeddings: &[Embedding], distance_threshold: f32) -> Vec<usize> {
    let count = embeddings.len();
    if count < 2 {
        return (0..count).collect();
    }

    let mut condensed: Vec<f32> = (0..count)
        .flat_map(|i| (i + 1..count).map(move |j| (i, j)))
        .map(|(i, j)| cosine_distance(&embeddings[i], &embeddings[j]))
        .collect();
    let dendrogram = linkage(&mut condensed, count, Method::Average);

    // Dendrogram node `count + step` is the cluster formed at that step.
    let mut parent: Vec<usize> = (0..2 * count - 1).collect();
    for (step_idx, step) in dendrogram.steps().iter().enumerate() {
        if step.dissimilarity <= distance_threshold {
            parent[step.cluster1] = count + step_idx;
            parent[step.cluster2] = count + step_idx;
        }
    }

    (0..count)
        .map(|node| {
            let mut root = node;
            while parent[root] != root {
                root = parent[root];
            }
            root
        })
        .collect()
}

/// Renumbers raw cluster ids to `0, 1, ...` in order of each cluster's earliest
/// anchor by `(start_ms, end_ms, window_index)`.
fn label_by_first_appearance(anchors: &[&TurnEmbedding], raw_ids: &[usize]) -> Vec<usize> {
    let mut chronological: Vec<usize> = (0..anchors.len()).collect();
    chronological.sort_by_key(|&idx| {
        let turn = anchors[idx];
        (turn.window.start_ms, turn.window.end_ms, turn.window_index)
    });

    let mut first_seen: Vec<usize> = Vec::new();
    for &idx in &chronological {
        if !first_seen.contains(&raw_ids[idx]) {
            first_seen.push(raw_ids[idx]);
        }
    }
    raw_ids
        .iter()
        .map(|raw| {
            first_seen
                .iter()
                .position(|seen| seen == raw)
                .expect("every raw cluster id was recorded")
        })
        .collect()
}

fn compute_centroids(
    embeddings: &[Embedding],
    labels: &[usize],
    speaker_count: usize,
) -> Vec<Embedding> {
    (0..speaker_count)
        .map(|speaker| {
            let sum = embeddings
                .iter()
                .zip(labels)
                .filter(|&(_, &label)| label == speaker)
                .fold([0.0_f32; EMBEDDING_DIM], |acc, (embedding, _)| {
                    std::array::from_fn(|dim| acc[dim] + embedding[dim])
                });
            l2_normalize(sum)
        })
        .collect()
}

fn nearest_centroid(embedding: &Embedding, centroids: &[Embedding]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            cosine_distance(embedding, a).total_cmp(&cosine_distance(embedding, b))
        })
        .map_or(0, |(idx, _)| idx)
}

/// Sorts segments chronologically and folds each one into the immediately
/// preceding segment when it shares the same speaker label and overlap state
/// and the pause between them is shorter than `collar_ms`. Only the segment
/// directly before it is considered, so a merge never bridges over an
/// intervening segment from a different speaker.
fn merge_collar_adjacent(mut segments: Vec<SpeakerSegment>, collar_ms: u64) -> Vec<SpeakerSegment> {
    let sort_key = |segment: &SpeakerSegment| {
        (
            segment.start_ms,
            segment.end_ms,
            segment.speaker_label.clone(),
        )
    };
    segments.sort_by_key(sort_key);

    let mut merged: Vec<SpeakerSegment> = Vec::with_capacity(segments.len());
    for segment in segments {
        let mergeable = merged.last_mut().is_some_and(|prev| {
            prev.speaker_label == segment.speaker_label
                && prev.is_overlap == segment.is_overlap
                && segment.start_ms.saturating_sub(prev.end_ms) < collar_ms
        });
        if mergeable {
            let prev = merged.last_mut().expect("checked by `mergeable`");
            prev.end_ms = prev.end_ms.max(segment.end_ms);
        } else {
            merged.push(segment);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEAKER_SAMPLE_STEP: f32 = 0.1;

    /// Maps the constant sample level of a slice to a speaker basis vector:
    /// level `0.1 * (k + 1)` yields a unit vector along dimension `k`.
    struct LevelToSpeakerExtractor;

    impl EmbeddingExtractor for LevelToSpeakerExtractor {
        fn extract_embedding(
            &mut self,
            samples_16k: &[f32],
        ) -> Result<Embedding, DiarizationError> {
            let level = samples_16k.first().copied().unwrap_or(0.0);
            let speaker = (level / SPEAKER_SAMPLE_STEP).round() as usize - 1;
            Ok(speaker_embedding(speaker, 0.0))
        }
    }

    /// Unit vector along `speaker`'s axis with a small deterministic `jitter`
    /// spread over the other dimensions.
    fn speaker_embedding(speaker: usize, jitter: f32) -> Embedding {
        l2_normalize(std::array::from_fn(|dim| {
            if dim == speaker {
                1.0
            } else {
                jitter * ((dim * 7 + speaker * 13) as f32).sin()
            }
        }))
    }

    fn turn(
        window_index: usize,
        start_ms: u64,
        end_ms: u64,
        is_overlap: bool,
        embedding: Embedding,
    ) -> TurnEmbedding {
        TurnEmbedding {
            window_index,
            window: RawTurnWindow {
                start_ms,
                end_ms,
                local_speaker_slot: 0,
                is_overlap,
            },
            is_clustering_anchor: !is_overlap && end_ms - start_ms >= 500,
            embedding,
        }
    }

    fn segment(label: &str, start_ms: u64, end_ms: u64, is_overlap: bool) -> SpeakerSegment {
        SpeakerSegment {
            speaker_label: label.to_string(),
            start_ms,
            end_ms,
            is_overlap,
        }
    }

    /// Two-second anchor turns separated by 1 s pauses, cycling through
    /// `speakers` speakers `rounds` times.
    fn cycling_anchor_turns(speakers: usize, rounds: usize) -> Vec<TurnEmbedding> {
        (0..speakers * rounds)
            .map(|idx| {
                let start = idx as u64 * 3_000;
                let jitter = 0.01 * (idx / speakers + 1) as f32;
                turn(
                    idx,
                    start,
                    start + 2_000,
                    false,
                    speaker_embedding(idx % speakers, jitter),
                )
            })
            .collect()
    }

    fn labels(output: &DiarizationOutput) -> Vec<String> {
        output
            .segments
            .iter()
            .map(|segment| segment.speaker_label.clone())
            .collect()
    }

    /// `SPEAKER_00, SPEAKER_01, ...` repeating every `speakers` turns.
    fn cycling_labels(speakers: usize, turns: usize) -> Vec<String> {
        (0..turns)
            .map(|idx| speaker_label(idx % speakers))
            .collect()
    }

    fn cluster_default(turns: &[TurnEmbedding]) -> DiarizationOutput {
        cluster_and_merge_turns(
            turns,
            DEFAULT_COSINE_DISTANCE_THRESHOLD,
            COLLAR_MERGE_THRESHOLD_MS,
        )
    }

    #[test]
    fn estimates_one_speaker_without_a_target_count() {
        let output = cluster_default(&cycling_anchor_turns(1, 3));

        assert_eq!(output.speaker_count, 1);
        assert!(output
            .segments
            .iter()
            .all(|segment| segment.speaker_label == "SPEAKER_00"));
    }

    #[test]
    fn estimates_two_speakers_without_a_target_count() {
        let output = cluster_default(&cycling_anchor_turns(2, 3));

        assert_eq!(output.speaker_count, 2);
        assert_eq!(
            labels(&output),
            cycling_labels(2, 6),
            "labels follow first chronological appearance"
        );
    }

    #[test]
    fn estimates_four_speakers_without_a_target_count() {
        let output = cluster_default(&cycling_anchor_turns(4, 3));

        assert_eq!(output.speaker_count, 4);
        assert_eq!(labels(&output), cycling_labels(4, 12));
    }

    #[test]
    fn labels_follow_first_appearance_not_input_order() {
        let turns = vec![
            turn(0, 5_000, 7_000, false, speaker_embedding(0, 0.0)),
            turn(1, 0, 2_000, false, speaker_embedding(1, 0.0)),
        ];

        let output = cluster_default(&turns);

        assert_eq!(
            output.segments,
            vec![
                segment("SPEAKER_00", 0, 2_000, false),
                segment("SPEAKER_01", 5_000, 7_000, false),
            ]
        );
    }

    #[test]
    fn short_and_overlapping_turns_join_nearest_centroid_and_keep_overlap_flag() {
        let turns = vec![
            turn(0, 0, 2_000, false, speaker_embedding(0, 0.01)),
            turn(1, 3_000, 5_000, false, speaker_embedding(1, 0.01)),
            turn(2, 6_000, 8_000, false, speaker_embedding(0, 0.02)),
            turn(3, 9_000, 11_000, false, speaker_embedding(1, 0.02)),
            // Short turn closest to speaker 1.
            turn(4, 12_000, 12_300, false, speaker_embedding(1, 0.3)),
            // Overlapping pair resolved to each speaker.
            turn(5, 14_000, 16_000, true, speaker_embedding(0, 0.3)),
            turn(6, 14_000, 16_000, true, speaker_embedding(1, 0.3)),
        ];

        let output = cluster_default(&turns);

        assert_eq!(output.speaker_count, 2);
        assert_eq!(
            output.segments[4..],
            [
                segment("SPEAKER_01", 12_000, 12_300, false),
                segment("SPEAKER_00", 14_000, 16_000, true),
                segment("SPEAKER_01", 14_000, 16_000, true),
            ]
        );
    }

    #[test]
    fn collar_merges_pauses_under_250ms_and_keeps_longer_pauses() {
        let speaker = speaker_embedding(0, 0.0);
        let turns = vec![
            turn(0, 0, 1_000, false, speaker),
            turn(1, 1_100, 2_000, false, speaker),
            turn(2, 2_249, 3_000, false, speaker),
            turn(3, 3_250, 4_000, false, speaker),
            turn(4, 4_400, 5_000, false, speaker),
        ];

        let output = cluster_default(&turns);

        assert_eq!(
            output.segments,
            vec![
                segment("SPEAKER_00", 0, 3_000, false),
                segment("SPEAKER_00", 3_250, 4_000, false),
                segment("SPEAKER_00", 4_400, 5_000, false),
            ]
        );
    }

    #[test]
    fn collar_does_not_merge_different_overlap_states() {
        let speaker = speaker_embedding(0, 0.0);
        let turns = vec![
            turn(0, 0, 1_000, false, speaker),
            turn(1, 1_000, 2_000, true, speaker),
        ];

        let output = cluster_default(&turns);

        assert_eq!(
            output.segments,
            vec![
                segment("SPEAKER_00", 0, 1_000, false),
                segment("SPEAKER_00", 1_000, 2_000, true),
            ]
        );
    }

    #[test]
    fn collar_merge_does_not_bridge_over_a_different_speakers_segment() {
        let turns = vec![
            turn(0, 0, 1_000, false, speaker_embedding(0, 0.0)),
            // A short, distinct-speaker interjection inside the collar window
            // of both of speaker 0's surrounding turns. Too short to be a
            // clustering anchor itself.
            turn(1, 1_050, 1_100, false, speaker_embedding(1, 0.0)),
            turn(2, 1_150, 1_900, false, speaker_embedding(0, 0.0)),
            // A distant, long speaker-1 anchor so clustering recognizes two
            // real speakers instead of folding the interjection into
            // speaker 0's only cluster by default.
            turn(3, 5_000, 6_000, false, speaker_embedding(1, 0.0)),
        ];

        let output = cluster_default(&turns);

        assert_eq!(
            output.segments,
            vec![
                segment("SPEAKER_00", 0, 1_000, false),
                segment("SPEAKER_01", 1_050, 1_100, false),
                segment("SPEAKER_00", 1_150, 1_900, false),
                segment("SPEAKER_01", 5_000, 6_000, false),
            ],
            "speaker 0's turns must stay separate instead of merging across speaker 1's segment"
        );
    }

    #[test]
    fn no_turns_report_zero_speakers() {
        assert_eq!(
            cluster_default(&[]),
            DiarizationOutput {
                speaker_count: 0,
                segments: Vec::new(),
            }
        );
    }

    #[test]
    fn diarize_turn_windows_combines_extraction_and_clustering() {
        // Speaker A (level 0.1) for 0-3 s, speaker B (level 0.2) for 3-6 s, A again 6-8 s.
        let samples: Vec<f32> = [(48_000, 0.1), (48_000, 0.2), (32_000, 0.1)]
            .iter()
            .flat_map(|&(count, level)| std::iter::repeat_n(level, count))
            .collect();
        let audio = AudioBuffer16kMono {
            samples,
            duration_ms: 8_000,
        };
        let raw = |start_ms, end_ms, is_overlap| RawTurnWindow {
            start_ms,
            end_ms,
            local_speaker_slot: 0,
            is_overlap,
        };
        let windows = vec![
            raw(0, 1_400, false),
            raw(1_500, 2_900, false),
            raw(3_000, 4_500, false),
            raw(4_600, 4_900, false),
            raw(5_000, 6_000, true),
            raw(6_000, 8_000, false),
        ];

        let output = diarize_turn_windows(&audio, &windows, &mut LevelToSpeakerExtractor).unwrap();

        assert_eq!(output.speaker_count, 2);
        assert_eq!(
            output.segments,
            vec![
                segment("SPEAKER_00", 0, 2_900, false),
                segment("SPEAKER_01", 3_000, 4_900, false),
                segment("SPEAKER_01", 5_000, 6_000, true),
                segment("SPEAKER_00", 6_000, 8_000, false),
            ]
        );
    }
}
