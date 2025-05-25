use ffmpeg_next::{
                  format::input,
                  util::frame::audio::Audio,
                  channel_layout::ChannelLayout,
                  software,
};
use std::path::Path;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::format::sample::Type::{ Planar};
use indicatif::{MultiProgress, ProgressBar};
use indicatif_log_bridge::LogWrapper;
use vad_rs::{Vad, VadStatus};
use log::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .build();
    let level = logger.filter();
    let multi = MultiProgress::new();

    LogWrapper::new(multi.clone(), logger)
        .try_init()
        .unwrap();
    log::set_max_level(level);
    let input_path = Path::new("samples/ep0音轨.wav");
    // if input_path.extension().unwrap() == "wav" {
    //     warn!("There's an unknown issue that prevent wav file from resampling.\n\
    //            Please convert this audio to any other format to continue");
    //     // return Err(Box::from("File extgension not supported"));
    // }

    // 打开输入文件
    let mut ictx = input(input_path).unwrap();

    // 查找音频流
    let stream = ictx.streams().best(ffmpeg_next::media::Type::Audio).unwrap();
    let audio_stream_index = stream.index();

    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = context.decoder().audio()?;

    // 设置目标采样率
    let target_sample_rate = 16000;

    // 获取原始音频信息
    let original_sample_rate = decoder.rate();
    let original_format=decoder.format();
    let original_channels = decoder.channels();
    //fill in default Channel layout if it's empty
    if decoder.channel_layout().is_empty(){
        decoder.set_channel_layout(ChannelLayout::default(original_channels as i32));
    }
    let original_channel_layout=decoder.channel_layout();
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


        let mut decoded:Audio= Audio::empty();
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
                let f32_sample=*sample as f32 / i16::MAX as f32;
                output_samples.push(f32_sample);
            }
            // }
        }

    }
    pb.finish();
    multi.remove(&pb);


    // 输出结果
    debug!("Output samples count: {}", output_samples.len());
    // println!("Output samples bytes: {:?}", output_samples);
    // Load the model
    let model_path = "models/silero_vad.onnx";

    // Create a VAD iterator
    let mut vad = Vad::new(model_path, target_sample_rate.try_into().unwrap())?;

    // let audio=Array1::from_vec(output_samples);
    let mut is_speech = false;
    let mut start_time = 0.0;
    let sample_rate = target_sample_rate as f32;
    let chunk_size = (0.1 * sample_rate) as usize;

    // Add 1s of silence to the end of the samples
    output_samples.extend(vec![0.0; sample_rate as usize]);
    let chunks:Vec<_>=output_samples.chunks(chunk_size).enumerate().collect();
    let pb = multi.add(ProgressBar::new(chunks.len() as u64));

    for (i, chunk) in chunks {
        pb.inc(1);
        let time = i as f32 * chunk_size as f32 / sample_rate;

        match vad.compute(chunk){
            Ok(mut result)=>{
                match result.status() {
                    VadStatus::Speech => {
                        if !is_speech {
                            start_time = time;
                            is_speech = true;
                        }
                    }
                    VadStatus::Silence => {
                        if is_speech {
                            debug!("Speech detected from {:.2}s to {:.2}s", start_time, time);
                            is_speech = false;
                        }
                    }
                    _ => {}
                }
            }
            Err(e) => {
                error!("E:{:?}", e);
            }
        }
    }
    pb.finish();
    multi.remove(&pb);


    Ok(())
}
