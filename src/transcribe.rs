use indicatif::{MultiProgress, ProgressBar};
use log::info;
use srtlib::{Subtitle, Subtitles, Timestamp};
use std::error::Error;
use whisper_rs::{
    DtwModelPreset, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

use crate::progress::{get_active_style, get_finished_style};
use crate::vad::ActiveSpeech;

pub fn do_whisper(
    multi: &MultiProgress,
    model_path: &str,
    active_speech_list: &[ActiveSpeech],
    language: &str,
    initial_prompt_text: &str,
) -> Result<Subtitles, Box<dyn Error>> {
    // Install a hook to log any errors from the whisper C++ code.
    whisper_rs::install_logging_hooks();
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: DtwModelPreset::LargeV3Turbo,
    };

    let ctx =
        WhisperContext::new_with_params(model_path, context_param).expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(8);
    // Enable translation.
    params.set_translate(false);
    // Set the language
    params.set_language(Some(language));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // params.set_no_context(true);
    params.set_initial_prompt(initial_prompt_text);

    let mut subs = Subtitles::new();
    let mut num = 1;

    // Enable token level timestamps
    params.set_token_timestamps(true);
    let pb = multi.add(ProgressBar::new(active_speech_list.len() as u64));
    pb.set_style(get_active_style());
    pb.set_message("Transcribing Speech...");

    let st = std::time::Instant::now();
    for active_speech in active_speech_list.iter() {
        let s = active_speech.data.to_vec();
        // s.extend(vec![0.0; 16000usize]);
        state.full(params.clone(), &s).expect("failed to run model");

        // Create a file to write the transcript to.

        // Iterate through the segments of the transcript.
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        for i in 0..num_segments {
            // Get the transcribed text and timestamps for the current segment.
            let segment_text_raw = state
                .full_get_segment_text(i)
                .expect("failed to get segment");

            let mut processed_text_slice = segment_text_raw.as_str();

            // Check if the segment starts with the initial prompt
            if !initial_prompt_text.is_empty()
                && processed_text_slice.starts_with(initial_prompt_text)
            {
                // Remove the prompt
                processed_text_slice = &processed_text_slice[initial_prompt_text.len()..];
                // Trim leading common separators (like comma, space) that might follow the prompt
                // You can extend the characters in the closure as needed based on observation
                processed_text_slice = processed_text_slice
                    .trim_start_matches(|c: char| c.is_whitespace() || c == '，' || c == ',');
            }

            let segment = processed_text_slice.to_string();

            // Skip empty segments after processing
            if segment.is_empty() {
                continue;
            }

            let start_timestamp = state
                .full_get_segment_t0(i)
                .expect("failed to get start timestamp");
            let end_timestamp = state
                .full_get_segment_t1(i)
                .expect("failed to get end timestamp");
            let start_time_ms = start_timestamp * 10 + ((active_speech.start_time * 1000.0) as i64);
            let mut end_time_ms = end_timestamp * 10 + ((active_speech.start_time * 1000.0) as i64);
            if end_time_ms > (active_speech.end_time * 1000.0) as i64 {
                end_time_ms = (active_speech.end_time * 1000.0) as i64;
            }

            info!("[{}] -> [{}]: {}", start_time_ms, end_time_ms, segment);
            let start_timestamp = Timestamp::from_milliseconds(start_time_ms as u32);
            let end_timestamp = Timestamp::from_milliseconds(end_time_ms as u32);

            // Add subtitle at the end of the subs' collection.
            subs.push(Subtitle::new(num, start_timestamp, end_timestamp, segment));
            num += 1;
        }
        pb.inc(1);
    }
    let et = std::time::Instant::now();
    info!("took {}ms", (et - st).as_millis());
    pb.set_style(get_finished_style());
    pb.finish();
    multi.remove(&pb);
    Ok(subs)
}
