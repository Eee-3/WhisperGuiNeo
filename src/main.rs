#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use eframe::egui;
use eframe::egui::{Button, Id, InnerResponse, Modal, ProgressBar, RichText, TextEdit, TextStyle, ViewportBuilder};
use egui_file_dialog::FileDialog;
use flexi_logger::{Age, Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
#[allow(unused_imports)]
use log::{LevelFilter, debug, error, info, trace, warn};
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::error::Error;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread;

mod audio;
mod transcribe;
mod vad;

#[derive(Default)]
struct App {
    file_dialog: RefCell<FileDialog>,
    audio_path: FileSelectionData,
    whisper_path: FileSelectionData,
    silero_vad_path: FileSelectionData,
    output_path: FileOutputData,
    language: String,
    initial_prompt: String,
    state: Arc<Mutex<AppState>>,
    progress:Arc<Mutex<f32>>,
}
#[derive(PartialOrd, PartialEq)]
enum AppState{
    Idle,
    Resample,
    VAD,
    Whisper,
    Saving,
    Finished
}
impl Default for AppState {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Default)]
struct FileSelectionData {
    hint: String,
    path: PathBuf,
    path_string: String,
    // default_filename: String,
    ongoing: bool,
}
impl FileSelectionData {
    fn new(hint: String) -> Self {
        Self {
            hint,
            ongoing: false,
            // default_filename,
            ..Self::default()
        }
    }
}
#[derive(Default)]
struct FileOutputData {
    hint: String,
    path: PathBuf,
    path_string: String,
    default_filename: String,
    ongoing: bool,
}
impl FileOutputData {
    fn new(hint: String, default_filename: String) -> Self {
        Self {
            hint,
            ongoing: false,
            default_filename,
            ..Self::default()
        }
    }
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self::load_chinese_fonts(cc);
        Self {
            file_dialog: RefCell::new(FileDialog::new().as_modal(true).default_size([664.,200.])),
            audio_path: FileSelectionData::new("音频文件".to_string()),
            whisper_path: FileSelectionData::new("Whisper模型(ggml-*.bin)".to_string()),
            silero_vad_path: FileSelectionData::new("SileroVAD模型(silero_vad.onnx)".to_string()),
            output_path: FileOutputData::new("输出文件".to_string(), String::new()),
            language:"zh".to_string(),
            ..Self::default()
        }
    }
    fn load_chinese_fonts(cc: &eframe::CreationContext<'_>) {
        let mut fonts = egui::FontDefinitions::default();

        // Install my own font (maybe supporting non-latin characters):
        fonts.font_data.insert(
            "MiSans-Regular".to_owned(),
            Arc::from(egui::FontData::from_static(include_bytes!(
                "../assets/MiSans-Regular.ttf"
            ))),
        ); // .ttf and .otf supported

        // Put my font first (highest priority):
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "MiSans-Regular".to_owned());

        // Put my font as last fallback for monospace:
        fonts
            .families
            .get_mut(&egui::FontFamily::Monospace)
            .unwrap()
            .push("MiSans-Regular".to_owned());

        // let mut ctx = egui::CtxRef::default();
        cc.egui_ctx.set_fonts(fonts);
    }
}


impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use egui::text::{LayoutJob, TextFormat};
        catppuccin_egui::set_theme(ctx, catppuccin_egui::MACCHIATO);
        self.file_dialog.borrow_mut().update(ctx);
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            // 使用 vertical_centered，它能完美地居中其内部的每个独立控件。
            ui.vertical_centered(|ui| {
                // 第一个控件：标题。它会被居中。
                ui.heading("WhisperGuiNeo");

                // 第二个控件：我们使用 LayoutJob 将两个文本合二为一。
                let mut job = LayoutJob::default();
                let font_id = TextStyle::Body.resolve(ui.style());

                // 添加第一段普通文本
                job.append(
                    "一个使用Whisper和SileroVAD的实用语音转文字小程序 ", // 在末尾加一个空格
                    0.0,
                    TextFormat {
                        font_id: font_id.clone(),
                        color: ui.style().visuals.text_color(),
                        ..Default::default()
                    },
                );

                // 添加第二段 weak 样式的文本
                job.append(
                    "本程序使用了MiSans-Regular字体",
                    0.0,
                    TextFormat {
                        font_id: font_id.clone(),
                        color: ui.style().visuals.weak_text_color(), // 使用 weak 颜色
                        ..Default::default()
                    },
                );

                // 将这个 job 作为一个单一的 label 添加到 UI 中。现在它是一个整体，可以被轻松居中。
                ui.label(job);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            Self::file_selection(ui, &self.file_dialog, &mut self.audio_path);
            Self::file_selection(ui, &self.file_dialog, &mut self.whisper_path);
            Self::file_selection(ui, &self.file_dialog, &mut self.silero_vad_path);
            ui.separator();
            ui.horizontal(|ui| {
                ui.centered_and_justified(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("语言");
                        // ui.centered_and_justified(|ui| {
                        ui.text_edit_singleline(&mut self.language);
                        // })
                    })
                });
                ui.centered_and_justified(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Initial Prompt (Optional)");
                        // ui.centered_and_justified(|ui| {
                        ui.text_edit_singleline(&mut self.initial_prompt);
                        // })
                    })
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("输出文件").clicked() {
                    debug!("开始选择输出文件");
                    self.output_path.default_filename = self
                        .audio_path
                        .path
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    self.output_path.ongoing = true;

                    //真服了，用了refcell 结果告诉我 default filename 是 save file mode 用的
                    self.file_dialog.replace(
                        self.file_dialog
                            .take()
                            .title("打开输出文件")
                            .default_file_name(&self.output_path.default_filename)
                            .add_save_extension("SubRip Subtitle", "srt")
                            .default_save_extension("SubRip Subtitle"),
                    );
                    self.file_dialog.borrow_mut().save_file();
                };
                if self.output_path.ongoing
                    && let Some(path) = self.file_dialog.borrow_mut().take_picked()
                {
                    self.output_path.path = path.to_path_buf();
                    self.output_path.path_string =
                        self.output_path.path.to_string_lossy().into_owned();
                    debug!("保存到{}文件", self.output_path.hint);
                    self.output_path.ongoing = false;
                } else {
                    if self.output_path.path_string != self.output_path.path.to_string_lossy() {
                        debug!("检测到文件输入框变更: {}", self.output_path.path_string);
                        self.output_path.path =
                            self.output_path.path_string.clone().try_into().unwrap();
                        debug!(
                            "当前PathBuf内容 {}",
                            self.output_path.path.to_string_lossy()
                        );
                    }
                }
                ui.centered_and_justified(|ui| {
                    let file_text_edit = TextEdit::singleline(&mut self.output_path.path_string)
                        .hint_text("请选择".to_string() + &self.output_path.hint);
                    ui.add(file_text_edit)
                });
            });
            ui.separator();
            let should_start = if self.audio_path.path.is_file()
                && self.whisper_path.path.is_file()
                && self.silero_vad_path.path.is_file()
                && !self.output_path.path_string.is_empty()
                && !self.language.is_empty()
                && *self.state.lock().unwrap() == AppState::Idle
            {
                true
            } else {
                false
            };
            ui.horizontal(|ui| {
               if ui.add_enabled(
                    should_start,
                    Button::new(RichText::new("开始转录").size(14.0).strong())
                        .corner_radius(5.0)
                        .min_size([710.0, 32.8].into()),
                ).clicked(){
                   debug!("开始转录");
                   thread::spawn({
                        let state = Arc::clone(&self.state);
                        let progress = Arc::clone(&self.progress);
                       let audio_path = self.audio_path.path.clone();
                       let vad_path=self.silero_vad_path.path_string.clone();
                       let whisper_path=self.whisper_path.path_string.clone();
                       let language=self.language.clone();
                       let initial_prompt=self.initial_prompt.clone();
                       let output_path=self.output_path.path.clone();
                        move || {
                            {*progress.lock().unwrap() =  0.;}
                            {*state.lock().unwrap() = AppState::Resample;}

                            let mut resampled =audio::do_resample(progress.clone(), 16000, &audio_path).unwrap();
                            {*progress.lock().unwrap() =  1.;}

                            {*progress.lock().unwrap() =  0.;}
                            {*state.lock().unwrap() = AppState::VAD;}

                            let active=vad::do_vad(progress.clone(), 16000, &vad_path, &mut resampled).unwrap();
                            {*progress.lock().unwrap() =  1.;}

                            {*progress.lock().unwrap() =  0.;}
                            {*state.lock().unwrap() = AppState::Whisper;}

                            let subs=transcribe::do_whisper(progress.clone(), &whisper_path, &active, &language, &initial_prompt).unwrap();
                            {*progress.lock().unwrap() =  1.;}

                            {*progress.lock().unwrap() =  0.;}
                            {*state.lock().unwrap() = AppState::Saving;}

                            subs.write_to_file(&output_path, None).unwrap();
                            {*progress.lock().unwrap() =  1.;}

                            {*progress.lock().unwrap() =  0.;}
                            {*state.lock().unwrap() = AppState::Finished;}
                        }
                    });
               };
            });
            ui.separator();
                ui.label(
                    match *self.state.lock().unwrap() {
                        AppState::Idle => "空闲".to_string(),
                        AppState::VAD => "正在检测语音活动...".to_string(),
                        AppState::Whisper => "正在转录...".to_string(),
                        AppState::Resample=>"正在重采样".to_string(),
                        AppState::Saving=>"正在保存".to_string(),
                        AppState::Finished=>"完成".to_string(),
                    }
                );
            // ui.label("Label");
            ui.add(ProgressBar::new(*self.progress.lock().unwrap()).show_percentage())



            // ui.top
        });
        if *self.state.lock().unwrap()==AppState::Finished{
            let modal = Modal::new(Id::from("my_modal"));

            // What goes inside the modal
            modal.show(ctx,|ui| {
                ui.heading("转录已完成");
                ui.separator();
                ui.strong("文件已保存到".to_string()+ &self.output_path.path_string);
                ui.separator();
                ui.vertical_centered_justified(|ui| {
                    if ui.button("OK").clicked(){
                        *self.state.lock().unwrap()=AppState::Idle;
                    };
                });


            });
        }
    }
}
impl App {
    fn file_selection(
        ui: &mut egui::Ui,
        file_dialog: &RefCell<FileDialog>,
        file_selection_data: &mut FileSelectionData,
    ) -> InnerResponse<()> {
        ui.horizontal(|ui| {
            if ui
                .button("打开".to_string() + &file_selection_data.hint)
                .clicked()
            {
                debug!("开始选择{}文件", file_selection_data.hint);
                file_selection_data.ongoing = true;

                //真服了，用了refcell 结果告诉我 default filename 是 save file mode 用的
                file_dialog.replace(file_dialog.take().title(&("打开".to_string() + &file_selection_data.hint)));
                file_dialog.borrow_mut().pick_file();
            };
            //我真服了，之前匹配放前面pathbuf被“偷”走了
            if file_selection_data.ongoing
                && let Some(path) = file_dialog.borrow_mut().take_picked()
            {
                file_selection_data.path = path.to_path_buf();
                file_selection_data.path_string =
                    file_selection_data.path.to_string_lossy().into_owned();
                debug!("打开{}文件", file_selection_data.hint);
                file_selection_data.ongoing = false;
            } else {
                if file_selection_data.path_string != file_selection_data.path.to_string_lossy() {
                    debug!("检测到文件输入框变更: {}", file_selection_data.path_string);
                    file_selection_data.path =
                        file_selection_data.path_string.clone().try_into().unwrap();
                    debug!(
                        "当前PathBuf内容 {}",
                        file_selection_data.path.to_string_lossy()
                    );
                }
            }
            ui.centered_and_justified(|ui| {
                let file_text_edit = TextEdit::singleline(&mut file_selection_data.path_string)
                    .hint_text("请选择".to_string() + &file_selection_data.hint);
                ui.add(file_text_edit)
            });
        })
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    // let main_start_time = Instant::now();

    // Determine base log level based on build profile
    let base_log_level = if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        std::env::var("RUST_LOG").unwrap_or("info".to_string())
    };

    // Parse the base log level string into a LevelFilter enum
    let base_level = LevelFilter::from_str(&base_log_level).unwrap_or(LevelFilter::Info);

    // Determine the whisper_rs log level: it should be the more restrictive (less verbose)
    // of the base level and the INFO level.
    let whisper_level = std::cmp::max(base_level, LevelFilter::Info);

    // Build the final log spec string
    let log_spec = format!("{},whisper_rs={}", base_log_level, whisper_level);

    Logger::try_with_str(&log_spec)?
        .log_to_file(FileSpec::default().directory("logs").basename("app"))
        .create_symlink("logs/latest.log")
        .duplicate_to_stdout(Duplicate::Trace)
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
        .start()?;
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([725.6, 292.])
            .with_resizable(false),
        ..Default::default()
    };
    eframe::run_native(
        "WhisperGuiNeo",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )?;
    Ok(())
}
