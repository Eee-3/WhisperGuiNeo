use std::error::Error;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::format::sample::Type::Planar;
use ffmpeg_next::{
    channel_layout::ChannelLayout, format::input, software, util::frame::audio::Audio,
};
use indicatif::{MultiProgress, ProgressBar};
use indicatif_log_bridge::LogWrapper;
use log::*;
use std::path::Path;
use clap::builder::Str;
use clap::Parser;
use vad_rs::{Vad, VadStatus};
use whisper_rs::{
    DtwModelPreset, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};
use srtlib::{Timestamp, Subtitle, Subtitles};
#[derive(Parser, Debug)]
#[command(version, about=None, long_about = None)]
struct Args {
    #[arg(short='i', long)]
    input: String,
    
    #[arg(short='o', long)]
    output: String,
    #[arg(short='v', long,default_value = "./models/silero_vad.onnx")]
    vad_model:String,
    #[arg(short='w', long,default_value = "./models/ggml-large-v3-turbo.bin")]
    whisper_model:String,
}
struct ActiveSpeech {
    start_time: f32,
    end_time: f32,
    data: Vec<f32>,
}
impl ActiveSpeech {
    fn new(start_time: f32, end_time: f32, data: Vec<f32>) -> Self {
        Self {
            start_time,
            end_time,
            data,
        }
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .filter_module("whisper_rs::whisper_logging_hook", LevelFilter::Info)
            .build();
    let level = logger.filter();
    let multi = MultiProgress::new();

    LogWrapper::new(multi.clone(), logger).try_init().unwrap();
    set_max_level(level);
    let args = Args::parse();
    // 设置目标采样率
    let target_sample_rate = 16000;
    let input_path = Path::new(args.input.as_str());
    // if input_path.extension().unwrap() == "wav" {
    //     warn!("There's an unknown issue that prevent wav file from resampling.\n\
    //            Please convert this audio to any other format to continue");
    //     // return Err(Box::from("File extension not supported"));
    // }

    let mut output_samples=do_resample(&multi,target_sample_rate,input_path)?;
    let active_speeches = do_vad(&multi, target_sample_rate, &args.vad_model, &mut output_samples)?;
    let subs=do_whisper(&multi, &args.whisper_model, &active_speeches)?;
    info!("Saving subs");
    subs.write_to_file(args.output, None).unwrap();

    Ok(())
}

fn do_vad(multi: &MultiProgress, target_sample_rate: u32,model_path:&str, output_samples: &mut Vec<f32>) -> Result<Vec<ActiveSpeech>, Box<dyn Error>> {
    // println!("Output samples bytes: {:?}", output_samples);
    // Load the model
    // let model_path = "models/silero_vad.onnx";

    // Create a VAD iterator
    let mut vad = Vad::new(model_path, target_sample_rate.try_into().unwrap())?;

    // let audio=Array1::from_vec(output_samples);
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
    let pb = multi.add(ProgressBar::new(chunks.len() as u64));
    for (i, chunk) in chunks {
        pb.inc(1);
        let time = i as f32 * chunk_size as f32 / sample_rate;

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
                            // debug!("Speech detected from {:.2}s to {:.2}s", start_time, time);
                            if silence_samples < silence_min_samples {
                                silence_samples += chunk_size;
                                full_audio_chunk.extend_from_slice(chunk);
                                continue;
                            }
                            let len = full_audio_chunk.len();
                            let duration = len as f32 / sample_rate;
                            if duration > 60.0 {
                                warn!("Found a {:.2}s chunks at {}s-{}s which is longer than 60.0s.Forced slicing into 2s pieces...",duration,start_time,time);
                                for (idx, slices) in full_audio_chunk.chunks(16000 * 2).enumerate() {
                                    let new_start_time = start_time + (idx as f32) * 2.0;
                                    active_speeches.push(ActiveSpeech::new(new_start_time, new_start_time + (slices.len() as f32) / sample_rate, slices.to_vec()));
                                }
                            } else if duration<1.01{
                                warn!("Found a {:.2}s chunks at {}s-{}s which is shorter than 1.01s.Extending to 1.01s...",duration,start_time,time);
                                let padding_length = 16000 - len + 100;
                                let padding = vec![0.0; padding_length]; // 动态创建一个 Vec
                                full_audio_chunk.extend_from_slice(&padding); // 使用 extend_from_slice 扩展数据

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
                        "Got an InvalidArgument error fro ort.This might be a normal behavior at the end of the audio."
                    );
                    is_speech = true;
                } else {
                    error!("Unknown error: {:?}", e);
                }

                // error!("E:{:?}", e);
            }
        }
    }
    pb.finish();
    multi.remove(&pb);
    // for speech in active_speeches.as_slice() {
    //     debug!(
    //         " {:.2}s Speech detected from {:.2}s to {:.2}s",
    //         speech.end_time-speech.start_time,speech.start_time, speech.end_time
    //     );
    //     let spec = hound::WavSpec {
    //         channels: 1,
    //         sample_rate: target_sample_rate,
    //         bits_per_sample: 32,
    //         sample_format: hound::SampleFormat::Float,
    //     };
    //     let mut writer = hound::WavWriter::create(
    //         format!(
    //             "debug-output/output_{:.2}s-{:.2}s.wav",
    //             speech.start_time, speech.end_time
    //         ),
    //         spec,
    //     )
    //         .unwrap();
    //     for sample in speech.data.as_slice() {
    //         writer.write_sample(*sample).unwrap();
    //     }
    //     writer.finalize().unwrap();
    // }
    Ok(active_speeches)
}

fn do_resample(multi: &MultiProgress, target_sample_rate: u32, input_path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    // 打开输入文件
    let mut ictx = input(input_path).unwrap();

    // 查找音频流
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .unwrap();
    let audio_stream_index = stream.index();

    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = context.decoder().audio()?;

    // 获取原始音频信息
    let original_sample_rate = decoder.rate();
    let original_format = decoder.format();
    let original_channels = decoder.channels();
    //fill in default Channel layout if it's empty
    if decoder.channel_layout().is_empty() {
        decoder.set_channel_layout(ChannelLayout::default(original_channels as i32));
    }
    let original_channel_layout = decoder.channel_layout();
    info!("original_format: {:?}", original_format);
    info!("original_channel_layout: {:?}", original_channel_layout);
    info!("channels: {}", original_channels);
    info!("original_sample_rate: {}", original_sample_rate);

    // 创建重采样器
    let mut resampler = software::resampling::Context::get(
        original_format,
        original_channel_layout,
        original_sample_rate,
        Sample::I16(Planar),
        ChannelLayout::MONO,
        target_sample_rate,
    )
        .unwrap();

    let mut output_samples: Vec<f32> = Vec::new();
    let packets: Vec<_> = ictx.packets().collect();
    let pb = multi.add(ProgressBar::new(packets.len() as u64));

    // 读取并处理每一帧
    for (stream, packet) in packets {
        pb.inc(1);
        if stream.index() != audio_stream_index {
            continue;
        }

        decoder.send_packet(&packet).unwrap();

        let mut decoded: Audio = Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            // println!(
            //     "Frame: format={:?}, rate={}, channels={}, layout={:?}",
            //     decoded.format(),
            //     decoded.rate(),
            //     decoded.channels(),
            //     decoded.channel_layout()
            // );
            //
            // 重采样
            // let mut ctx = decoded.resampler(Sample::I16(Planar), ChannelLayout::MONO, target_sample_rate).unwrap();
            // 确保重采样器配置与数据访问一致
            let mut resampled = Audio::empty();
            resampler.run(&decoded, &mut resampled)?;
            // resampler.run(&decoded,&mut resampled)?;

            // 将重采样的音频数据转为 Vec<f32>
            // for ch in 0..resampled.channels() {
            // println!("Resampled format: {:?}", resampled.format());
            for sample in resampled.plane::<i16>(0) {
                let f32_sample = *sample as f32 / i16::MAX as f32;
                output_samples.push(f32_sample);
            }
            // }
        }
    }
    pb.finish();
    multi.remove(&pb);

    // 输出结果
    debug!("Output samples count: {}", output_samples.len());
    Ok(output_samples)
}

fn do_whisper(multi: &MultiProgress,model_path:&str,active_speech_list: &[ActiveSpeech])->Result<Subtitles, Box<dyn Error>> {
    // Install a hook to log any errors from the whisper C++ code.
    whisper_rs::install_logging_hooks();
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: DtwModelPreset::LargeV3Turbo,
    };

    let ctx = WhisperContext::new_with_params(model_path, context_param)
        .expect("failed to load model");
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
    // Set the language to translate to English.
    params.set_language(Some("zh"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // params.set_no_context(true);
    let prompt = "请输出简体中文";
    params.set_initial_prompt(prompt);


    let mut subs = Subtitles::new();
    let mut num=1;


    // Enable token level timestamps
    params.set_token_timestamps(true);
    let pb = multi.add(ProgressBar::new(active_speech_list.len() as u64));
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
        // debug!("num_segments: {}", num_segments);
        for i in 0..num_segments {
            // Get the transcribed text and timestamps for the current segment.
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment")
                .replace(prompt, "");
            let start_timestamp = state
                .full_get_segment_t0(i)
                .expect("failed to get start timestamp");
            let end_timestamp = state
                .full_get_segment_t1(i)
                .expect("failed to get end timestamp");
            let start_time_ms=start_timestamp*10+((active_speech.start_time*1000.0) as i64);
            let mut end_time_ms=end_timestamp*10+((active_speech.start_time*1000.0) as i64);
            if end_time_ms > (active_speech.end_time*1000.0) as i64 {
                end_time_ms=(active_speech.end_time*1000.0) as i64;
            }

            info!(
    "[{} - {}]: {}",
    start_time_ms, end_time_ms,  segment
);
            let start_timestamp=Timestamp::from_milliseconds(start_time_ms as u32);
            let end_timestamp=Timestamp::from_milliseconds(end_time_ms as u32);


            // Add subtitle at the end of the subs' collection.
            subs.push(Subtitle::new(num, start_timestamp, end_timestamp, segment));
            num+=1;
        }
        pb.inc(1);
    }
    let et = std::time::Instant::now();
    println!("took {}ms", (et - st).as_millis());
    pb.finish();
    multi.remove(&pb);
    Ok(subs)
}
