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

fn main() -> Result<(), Box<dyn Error>> {

    Ok(())
}
