use crate::error::{AppError, AppResult};
use name_matcher::loaders::csv_loader::{
    load_csv_preview as load_csv_preview_inner, CsvPreviewDto, CsvPreviewRequestDto,
};

#[tauri::command]
pub async fn load_csv_preview(request: CsvPreviewRequestDto) -> AppResult<CsvPreviewDto> {
    tauri::async_runtime::spawn_blocking(move || load_csv_preview_inner(&request))
        .await
        .map_err(|e| AppError::Internal(format!("CSV preview task failed: {e}")))?
        .map_err(|e| AppError::Validation(e.to_string()))
}
