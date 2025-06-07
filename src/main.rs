use console::{Emoji, style};
use flexi_logger::{Age, Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use indicatif::{HumanDuration, MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, info};
use std::error::Error;
use std::str::FromStr;
use std::time::Instant;

mod audio;
mod cli;
mod progress;
mod transcribe;
mod vad;

use cli::Args;
use cli::get_args;
#[cfg(debug_assertions)]
use cli::get_debug_mode_args;

#[cfg(debug_assertions)]
fn get_cli_args() -> Args {
    if std::env::args().len() == 1 {
        println!("Running in debug mode with default arguments.");
        get_debug_mode_args()
    } else {
        get_args()
    }
}

#[cfg(not(debug_assertions))]
fn get_cli_args() -> Args {
    get_args()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = get_cli_args();
    let main_start_time = Instant::now();

    // Determine base log level based on build profile
    let base_log_level = if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        args.log_level.clone()
    };

    // Parse the base log level string into a LevelFilter enum
    let base_level = LevelFilter::from_str(&base_log_level).unwrap_or(LevelFilter::Info);

    // Determine the whisper_rs log level: it should be the more restrictive (less verbose)
    // of the base level and the INFO level.
    let whisper_level = std::cmp::max(base_level, LevelFilter::Info);

    // Build the final log spec string
    let log_spec = format!("{},whisper_rs={}", base_log_level, whisper_level);

    let (main_logger, _handle) = Logger::try_with_str(&log_spec)?
        .log_to_file(FileSpec::default().directory("logs").basename("app"))
        .create_symlink("logs/latest.log")
        .duplicate_to_stdout(Duplicate::Info)
        .format_for_stdout(|w, now, record| {
            let level_style = match record.level() {
                log::Level::Error => console::style(record.level()).red().bold(),
                log::Level::Warn => console::style(record.level()).yellow(),
                log::Level::Info => console::style(record.level()).green(),
                log::Level::Debug => console::style(record.level()).blue(),
                log::Level::Trace => console::style(record.level()).magenta(),
            };
            let timestamp_str = format!("[{}]", now.format("%H:%M:%S"));
            write!(
                w,
                "{:<10} {:<5}  {:<20}  {}",
                console::style(timestamp_str).dim(),
                level_style,
                console::style(record.target()).cyan(),
                &record.args()
            )
        })
        .rotate(
            Criterion::Age(Age::Day),
            Naming::Timestamps,
            Cleanup::KeepLogFiles(7),
        )
        .build()?;

    let multi = MultiProgress::new();
    LogWrapper::new(multi.clone(), main_logger)
        .try_init()
        .expect("Failed to initialize logger");

    info!("Starting transcription for file: {}", args.input.display());
    info!("Model path: {}", args.whisper_model.display());
    info!("Language: {}", args.language);

    let header_style = ProgressStyle::with_template("{spinner:.green} {prefix:8} {wide_msg}")
        .unwrap()
        .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]);

    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(header_style.clone());
    pb.set_prefix(style("[1/3]").bold().dim().to_string());
    pb.set_message(format!("{} Resampling audio...", Emoji("🎧", "»")));
    let mut samples = audio::do_resample(&multi, 16000, &args.input)?;
    pb.finish_with_message(format!(
        "{} Resampling complete.",
        style(Emoji("✔", "✓")).green()
    ));

    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(header_style.clone());
    pb.set_prefix(style("[2/3]").bold().dim().to_string());
    pb.set_message(format!("{} Detecting speech...", Emoji("🗣️", "»")));
    let active_speeches = vad::do_vad(
        &multi,
        16000,
        &args.vad_model.to_str().unwrap(),
        &mut samples,
    )?;
    pb.finish_with_message(format!(
        "{} Speech detection complete.",
        style(Emoji("✔", "✓")).green()
    ));

    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(header_style.clone());
    pb.set_prefix(style("[3/3]").bold().dim().to_string());
    pb.set_message(format!("{} Transcribing audio...", Emoji("📝", "»")));
    let subs = transcribe::do_whisper(
        &multi,
        &args.whisper_model.to_str().unwrap(),
        &active_speeches,
        &args.language,
        &args.initial_prompt,
    )?;
    pb.finish_with_message(format!(
        "{} Transcription complete.",
        style(Emoji("✔", "✓")).green()
    ));

    info!(
        "Transcription complete. Output saved to {}",
        args.output.display()
    );
    subs.write_to_file(args.output.to_str().unwrap(), None)?;

    multi.println(format!(
        "\n{} Done in {}",
        Emoji("✨", ":-)"),
        HumanDuration(main_start_time.elapsed())
    ))?;

    Ok(())
}
