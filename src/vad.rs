use std::error::Error;
use std::sync::{Arc, Mutex};
use log::{error, warn};
use vad_rs::{Vad, VadStatus};


#[derive(Debug, Clone)]
pub struct ActiveSpeech {
    pub start_time: f32,
    pub end_time: f32,
    pub data: Vec<f32>,
}

impl ActiveSpeech {
    pub fn new(start_time: f32, end_time: f32, data: Vec<f32>) -> Self {
        Self {
            start_time,
            end_time,
            data,
        }
    }
}

pub fn do_vad(
    progress: Arc<Mutex<f32>>,
    target_sample_rate: u32,
    model_path: &str,
    output_samples: &mut Vec<f32>,
) -> Result<Vec<ActiveSpeech>, Box<dyn Error>> {
    let mut vad = Vad::new(model_path, target_sample_rate.try_into().unwrap())?;

    let mut is_speech = false;
    let mut start_time = 0.0;
    let mut full_audio_chunk: Vec<f32> = Vec::new();
    let silence_min_samples = 3200;
    let mut silence_samples = 0;
    let sample_rate = target_sample_rate as f32;
    let chunk_size = (0.1 * sample_rate) as usize;

    // Add 1s of silence to the end of the samples
    output_samples.extend(vec![0.0; sample_rate as usize]);
    let chunks: Vec<_> = output_samples.chunks(chunk_size).enumerate().collect();
    let mut active_speeches: Vec<ActiveSpeech> = Vec::new();
    let total = chunks.len();



    for (i, chunk) in chunks.iter() {
        {        *progress.lock().unwrap() = *i as f32 / total as f32;}

        let time = *i as f32 * chunk_size as f32 / sample_rate;

        match vad.compute(chunk) {
            Ok(result) => {
                let status = if result.prob > 0.35 {
                    VadStatus::Speech
                } else {
                    VadStatus::Silence
                };
                match status {
                    VadStatus::Speech => {
                        silence_samples = 0;
                        full_audio_chunk.extend_from_slice(chunk);
                        if !is_speech {
                            start_time = time;
                            is_speech = true;
                        }
                    }
                    VadStatus::Silence => {
                        if is_speech {
                            if silence_samples < silence_min_samples {
                                silence_samples += chunk_size;
                                full_audio_chunk.extend_from_slice(chunk);
                                continue;
                            }
                            let len = full_audio_chunk.len();
                            let duration = len as f32 / sample_rate;
                            if duration > 60.0 {
                                warn!(
                                    "Found a {:.2}s chunks at {}s-{}s which is longer than 60.0s.Forced slicing into 2s pieces...",
                                    duration, start_time, time
                                );
                                for (idx, slices) in full_audio_chunk.chunks(16000 * 2).enumerate()
                                {
                                    let new_start_time = start_time + (idx as f32) * 2.0;
                                    active_speeches.push(ActiveSpeech::new(
                                        new_start_time,
                                        new_start_time + (slices.len() as f32) / sample_rate,
                                        slices.to_vec(),
                                    ));
                                }
                            } else if duration < 1.01 {
                                warn!(
                                    "Found a {:.2}s chunks at {}s-{}s which is shorter than 1.01s.Extending to 1.01s...",
                                    duration, start_time, time
                                );
                                let padding_length = 16000 - len + 100;
                                let padding = vec![0.0; padding_length]; // 动态创建一个 Vec
                                full_audio_chunk.extend_from_slice(&padding); // 使用 extend_from_slice 扩展数据
                                active_speeches.push(ActiveSpeech::new(
                                    start_time,
                                    time,
                                    full_audio_chunk.clone(),
                                ));
                            } else {
                                active_speeches.push(ActiveSpeech::new(
                                    start_time,
                                    time,
                                    full_audio_chunk.clone(),
                                ));
                            }
                            is_speech = false;
                            silence_samples = 0;
                            full_audio_chunk.clear();
                        }
                    }
                    _ => {}
                }
            }
            Err(e) => {
                if let ort::ErrorCode::InvalidArgument = e
                    .downcast_ref::<ort::Error>()
                    .ok_or("Error downcasting error.")?
                    .code()
                {
                    warn!(
                        "Got an InvalidArgument error from ort.This might be a normal behavior at the end of the audio."
                    );
                    is_speech = true;
                } else {
                    error!("Unknown error: {:?}", e);
                }
            }
        }
    }

    Ok(active_speeches)
}
