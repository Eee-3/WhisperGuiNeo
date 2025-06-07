use ffmpeg_next::{
    channel_layout::ChannelLayout, codec::Context, format::Sample, format::input,
    format::sample::Type::Planar, software, util::frame::audio::Audio,
};
use indicatif::{MultiProgress, ProgressBar};
use log::{debug, info};
use std::error::Error;
use std::path::Path;

use crate::progress::{get_active_style, get_finished_style};

pub fn do_resample(
    multi: &MultiProgress,
    target_sample_rate: u32,
    input_path: &Path,
) -> Result<Vec<f32>, Box<dyn Error>> {
    // 打开输入文件
    let mut ictx = input(input_path).unwrap();

    // 查找音频流
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .unwrap();
    let audio_stream_index = stream.index();

    let context = Context::from_parameters(stream.parameters())?;
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
    pb.set_style(get_active_style());
    pb.set_message("Resampling Audio...");

    // 读取并处理每一帧
    for (stream, packet) in packets {
        pb.inc(1);
        if stream.index() != audio_stream_index {
            continue;
        }

        decoder.send_packet(&packet).unwrap();

        let mut decoded: Audio = Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = Audio::empty();
            resampler.run(&decoded, &mut resampled)?;

            for sample in resampled.plane::<i16>(0) {
                let f32_sample = *sample as f32 / i16::MAX as f32;
                output_samples.push(f32_sample);
            }
        }
    }
    pb.set_style(get_finished_style());
    pb.finish();
    multi.remove(&pb);

    debug!("Output samples count: {}", output_samples.len());
    Ok(output_samples)
}
