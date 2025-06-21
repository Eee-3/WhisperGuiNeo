use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "whisper_with_vad",
    version = "1.0.0",
    about = "A tool for audio transcription using VAD (Voice Activity Detection) and Whisper",
    long_about = "whisper_with_vad is a command-line tool that combines Voice Activity Detection (VAD) \
                  with OpenAI's Whisper model to efficiently transcribe audio files. It first detects \
                  speech segments using Silero VAD, then processes only those segments with Whisper \
                  for accurate transcription."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Transcribe audio file to SRT subtitle format
    Transcribe {
        #[arg(short = 'i', long, help = "Path to the input audio file.")]
        input: PathBuf,

        #[arg(short = 'o', long, help = "Path to save the output SRT file.")]
        output: PathBuf,

        #[arg(
            short = 'v',
            long,
            default_value = "./models/silero_vad.onnx",
            help = "Path to the Silero VAD ONNX model."
        )]
        vad_model: PathBuf,

        #[arg(
            short = 'w',
            long,
            default_value = "./models/ggml-large-v3-turbo.bin",
            help = "Path to the Whisper GGML model file (e.g., ggml-large-v3-turbo.bin)."
        )]
        whisper_model: PathBuf,

        #[arg(
            long,
            default_value = "zh",
            help = "Language code for transcription (e.g., 'en', 'zh')."
        )]
        language: String,

        #[arg(
            long,
            default_value = "",
            help = "Initial prompt for the Whisper model to guide transcription."
        )]
        initial_prompt: String,

        #[arg(
            long,
            default_value = "INFO",
            help = "Logging level (e.g., TRACE, DEBUG, INFO, WARN, ERROR)."
        )]
        log_level: String,
    },
    /// Show information about this tool
    About,
}

// 为了保持向后兼容性，保留原来的Args结构
#[derive(Debug)]
pub struct Args {
    pub input: PathBuf,
    pub output: PathBuf,
    pub vad_model: PathBuf,
    pub whisper_model: PathBuf,
    pub language: String,
    pub initial_prompt: String,
    pub log_level: String,
}

impl From<&Commands> for Option<Args> {
    fn from(command: &Commands) -> Self {
        match command {
            Commands::Transcribe {
                input,
                output,
                vad_model,
                whisper_model,
                language,
                initial_prompt,
                log_level,
            } => Some(Args {
                input: input.clone(),
                output: output.clone(),
                vad_model: vad_model.clone(),
                whisper_model: whisper_model.clone(),
                language: language.clone(),
                initial_prompt: initial_prompt.clone(),
                log_level: log_level.clone(),
            }),
            Commands::About => None,
        }
    }
}

pub fn get_args() -> Option<Args> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::About => {
            show_about();
            None
        }
        _ => Option::<Args>::from(&cli.command),
    }
}

fn show_about() {
    println!("🎧 whisper_with_vad v1.0.0");
    println!();
    println!("📝 Description:");
    println!("   A high-performance audio transcription tool that combines Voice Activity Detection");
    println!("   (VAD) with OpenAI's Whisper model for efficient and accurate speech-to-text conversion.");
    println!();
    println!("🔧 How it works:");
    println!("   1. 🎧 Audio Resampling - Converts input audio to 16kHz sample rate");
    println!("   2. 🗣️  Speech Detection - Uses Silero VAD to identify speech segments");
    println!("   3. 📝 Transcription - Processes speech segments with Whisper model");
    println!();
    println!("🚀 Features:");
    println!("   • Efficient processing by skipping silent segments");
    println!("   • Support for multiple languages (Chinese, English, etc.)");
    println!("   • Customizable model paths and parameters");
    println!("   • Progress tracking with visual indicators");
    println!("   • Comprehensive logging system");
    println!();
    println!("📦 Models:");
    println!("   • VAD Model: Silero VAD (ONNX format)");
    println!("   • Whisper Model: OpenAI Whisper (GGML format)");
    println!();
    println!("💡 Usage:");
    println!("   whisper_with_vad transcribe -i input.wav -o output.srt");
    println!();
    println!("🔗 For more information, run: whisper_with_vad transcribe --help");
}

#[cfg(debug_assertions)]
pub fn get_debug_mode_args() -> Args {
    Args {
        whisper_model: PathBuf::from("./models/ggml-large-v3-turbo.bin"),
        vad_model: PathBuf::from("./models/silero_vad.onnx"),
        input: PathBuf::from("./samples/test2_cn.wav"),
        output: PathBuf::from("./samples/test_zh.srt"),
        language: "zh".to_string(),
        initial_prompt: "".to_string(),
        log_level: "debug".to_string(),
    }
}
