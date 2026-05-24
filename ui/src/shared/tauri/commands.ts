import { invoke } from "@tauri-apps/api/core";
import type {
  AppErrorDto,
  CudaDiagnosticsDto,
  CsvPreviewDto,
  CsvPreviewRequestDto,
  DbCredentialsDto,
  DbSessionDto,
  DiffJobsRequestDto,
  DiffResultDto,
  ExcelPreviewDto,
  ExcelPreviewRequestDto,
  ExportRequestDto,
  ExportResultDto,
  ExplainPairRequestDto,
  JobSummaryDto,
  ResultPageDto,
  ResultPageRequestDto,
  ReviewDecisionDto,
  RunConfigDto,
  SaveDecisionRequestDto,
  ScoreBreakdownDto,
  SystemInfoDto,
  TableColumnsDto,
  TableInfoDto,
} from "./types";

/**
 * Wraps `invoke` to surface the structured `AppErrorDto` thrown by the
 * Rust commands. Tauri serialises Rust `Result::Err` into a JSON value;
 * we re-throw it as a typed JS object so the UI can render the right
 * error UX (toast vs modal vs inline).
 */
async function call<T>(
  cmd: string,
  args?: Record<string, unknown>,
): Promise<T> {
  try {
    return (await invoke(cmd, args)) as T;
  } catch (raw) {
    // Tauri can throw either a string (older paths) or a serialised AppErrorDto.
    if (typeof raw === "object" && raw !== null && "kind" in raw) {
      throw raw as AppErrorDto;
    }
    throw {
      kind: "internal",
      message: typeof raw === "string" ? raw : "Unknown error",
      recoverable: false,
    } satisfies AppErrorDto;
  }
}

// ---------- System / config ----------

export const systemInfo = () => call<SystemInfoDto>("system_info");
export const cudaDiagnostics = () =>
  call<CudaDiagnosticsDto>("cuda_diagnostics");
export const loadConfig = () => call<Record<string, unknown>>("load_config");
export const saveConfig = (config: Record<string, unknown>) =>
  call<void>("save_config", { config });

// ---------- Database ----------

export const connectDb = (creds: DbCredentialsDto) =>
  call<DbSessionDto>("connect_db", { creds });
export const validateDbCredentials = (creds: DbCredentialsDto) =>
  call<number>("validate_db_credentials", { creds });
export const testConnection = (sessionId: string) =>
  call<number>("test_connection", { sessionId });
export const listTables = (sessionId: string) =>
  call<TableInfoDto[]>("list_tables", { sessionId });
export const getTableColumns = (sessionId: string, table: string) =>
  call<TableColumnsDto>("get_table_columns", { sessionId, table });
export const getRowCount = (sessionId: string, table: string) =>
  call<number>("get_row_count", { sessionId, table });
export const disconnectDb = (sessionId: string) =>
  call<void>("disconnect_db", { sessionId });
export const listSessions = () => call<DbSessionDto[]>("list_sessions");

// ---------- File preview/import ----------

export const loadCsvPreview = (request: CsvPreviewRequestDto) =>
  call<CsvPreviewDto>("load_csv_preview", { request });
export const loadExcelPreview = (request: ExcelPreviewRequestDto) =>
  call<ExcelPreviewDto>("load_excel_preview", { request });

// ---------- Matching ----------

export const startMatching = (config: RunConfigDto) =>
  call<string>("start_matching", { config });
export const cancelMatching = (jobId: string) =>
  call<void>("cancel_matching", { jobId });
export const pauseMatching = (jobId: string) =>
  call<void>("pause_matching", { jobId });
export const resumeMatching = (jobId: string) =>
  call<void>("resume_matching", { jobId });
export const getMatchingStatus = (jobId: string) =>
  call<JobSummaryDto>("get_matching_status", { jobId });
export const listMatchingJobs = () =>
  call<JobSummaryDto[]>("list_matching_jobs");
export const forgetMatchingJob = (jobId: string) =>
  call<void>("forget_matching_job", { jobId });

// ---------- Results / export ----------

export const getResultsPage = (request: ResultPageRequestDto) =>
  call<ResultPageDto>("get_results_page", { request });
export const explainPair = (request: ExplainPairRequestDto) =>
  call<ScoreBreakdownDto>("explain_pair", { request });
export const saveDecision = (request: SaveDecisionRequestDto) =>
  call<ReviewDecisionDto>("save_decision", { request });
export const getDecisions = (jobId: string) =>
  call<ReviewDecisionDto[]>("get_decisions", { jobId });
export const diffJobs = (request: DiffJobsRequestDto) =>
  call<DiffResultDto>("diff_jobs", { request });
export const exportResults = (request: ExportRequestDto) =>
  call<ExportResultDto>("export_results", { request });
