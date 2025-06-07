use indicatif::ProgressStyle;

/// Returns the style for an active progress bar, mimicking Python's rich library.
pub fn get_active_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {msg:<25.bold} [{bar:40.cyan/blue}] {percent:>3}% | ETA: {eta_precise}")
        .expect("Failed to create progress style")
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        .progress_chars("=>-")
}

/// Returns the style for a finished progress bar, featuring a checkmark and green colors.
pub fn get_finished_style() -> ProgressStyle {
    ProgressStyle::with_template("  ✔ {msg:<25.green.bold} | Elapsed: {elapsed_precise}")
        .expect("Failed to create progress style for finished tasks")
}
