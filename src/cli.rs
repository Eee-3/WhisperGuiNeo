use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short = 'i', long, help = "Path to the input audio file.")]
    pub input: PathBuf,

    #[arg(short = 'o', long, help = "Path to save the output SRT file.")]
    pub output: PathBuf,

    #[arg(
        short = 'v',
        long,
        default_value = "./models/silero_vad.onnx",
        help = "Path to the Silero VAD ONNX model."
    )]
    pub vad_model: PathBuf,

    #[arg(
        short = 'w',
        long,
        default_value = "./models/ggml-large-v3-turbo.bin",
        help = "Path to the Whisper GGML model file (e.g., ggml-large-v3-turbo.bin)."
    )]
    pub whisper_model: PathBuf,

    #[arg(
        long,
        default_value = "zh",
        help = "Language code for transcription (e.g., 'en', 'zh')."
    )]
    pub language: String,

    #[arg(
        long,
        default_value = "",
        help = "Initial prompt for the Whisper model to guide transcription."
    )]
    pub initial_prompt: String,

    #[arg(
        long,
        default_value = "INFO",
        help = "Logging level (e.g., TRACE, DEBUG, INFO, WARN, ERROR)."
    )]
    pub log_level: String,
}

pub fn get_args() -> Args {
    Args::parse()
}

#[cfg(debug_assertions)]
pub fn get_debug_mode_args() -> Args {
    Args {
        whisper_model: PathBuf::from("./models/ggml-large-v3-turbo.bin"),
        vad_model: PathBuf::from("./models/silero_vad.onnx"),
        input: PathBuf::from("./samples/test2_cn.wav"),
        output: PathBuf::from("./samples/test_zh.srt"),
        language: "zh".to_string(),
        initial_prompt: "请输出简体中文".to_string(),
        log_level: "debug".to_string(),
    }
}
