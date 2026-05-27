import { create } from "zustand";
import type { SessionSide } from "@/shared/stores/connectionStore";
import type {
  ColumnMappingDto,
  CsvDelimiterDto,
  CsvEncodingDto,
  CsvImportDryRunResultDto,
  CsvImportDuplicateBehaviorDto,
  CsvImportDuplicateKeyDto,
  CsvImportIdBehaviorDto,
  CsvImportJobDto,
  CsvImportTargetModeDto,
  CsvPreviewDto,
} from "@/shared/tauri/types";

export type CsvImportStep =
  | "target"
  | "file"
  | "mapping"
  | "policies"
  | "dry-run"
  | "commit"
  | "done";

export type CsvImportStatus = "idle" | "running" | "cancelling" | "terminal";

interface CsvImportState {
  open: boolean;
  side: SessionSide;
  step: CsvImportStep;
  targetTable: string;
  targetMode: CsvImportTargetModeDto;
  filePath: string;
  encoding: CsvEncodingDto | null;
  delimiter: CsvDelimiterDto | null;
  dateFormat: string;
  preview: CsvPreviewDto | null;
  mapping: ColumnMappingDto | null;
  idBehavior: CsvImportIdBehaviorDto;
  duplicateBehavior: CsvImportDuplicateBehaviorDto;
  duplicateKey: CsvImportDuplicateKeyDto;
  batchSize: number;
  createIndexes: boolean;
  confirmedDestructive: boolean;
  dryRun: CsvImportDryRunResultDto | null;
  job: CsvImportJobDto | null;
  jobId: string | null;
  importStatus: CsvImportStatus;
  previewLoading: boolean;
  dryRunLoading: boolean;
  importRunning: boolean;
  error: string | null;
  openForSide: (side: SessionSide) => void;
  close: () => void;
  patch: (patch: Partial<CsvImportState>) => void;
  resetTransient: () => void;
}

const defaults = {
  step: "target" as CsvImportStep,
  targetTable: "",
  targetMode: "create" as CsvImportTargetModeDto,
  filePath: "",
  encoding: null,
  delimiter: null,
  dateFormat: "%Y-%m-%d",
  preview: null,
  mapping: null,
  idBehavior: "use-csv-id" as CsvImportIdBehaviorDto,
  duplicateBehavior: "skip" as CsvImportDuplicateBehaviorDto,
  duplicateKey: "id" as CsvImportDuplicateKeyDto,
  batchSize: 5000,
  createIndexes: true,
  confirmedDestructive: false,
  dryRun: null,
  job: null,
  jobId: null,
  importStatus: "idle" as CsvImportStatus,
  previewLoading: false,
  dryRunLoading: false,
  importRunning: false,
  error: null,
};

export const useCsvImportStore = create<CsvImportState>((set) => ({
  open: false,
  side: "source",
  ...defaults,
  openForSide: (side) =>
    set({
      open: true,
      side,
      ...defaults,
    }),
  close: () =>
    set({
      open: false,
      previewLoading: false,
      dryRunLoading: false,
      importRunning: false,
      importStatus: "idle",
      error: null,
    }),
  patch: (patch) => set(patch),
  resetTransient: () =>
    set({
      preview: null,
      mapping: null,
      dryRun: null,
      job: null,
      jobId: null,
      importStatus: "idle",
      error: null,
    }),
}));
