use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    pub model: PathBuf,

    #[arg(long)]
    pub vad_model: PathBuf,

    #[arg(long)]
    pub path: PathBuf,

    #[arg(long)]
    pub output: PathBuf,

    #[arg(long)]
    pub language: Option<String>,

    #[arg(long)]
    pub initial_prompt: Option<String>,

    #[arg(long, default_value = "info")]
    pub log_level: String,
}

pub fn get_args() -> Args {
    Args::parse()
}

#[cfg(debug_assertions)]
pub fn get_debug_mode_args() -> Args {
    Args {
        model: PathBuf::from("./models/ggml-large-v3-turbo.bin"),
        vad_model: PathBuf::from("./models/silero_vad.onnx"),
        path: PathBuf::from("./samples/test2_cn.wav"),
        output: PathBuf::from("./samples/test_zh.srt"),
        language: Some("zh".to_string()),
        initial_prompt: Some("请输出简体中文".to_string()),
        log_level: "debug".to_string(),
    }
}
