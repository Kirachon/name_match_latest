import { create } from "zustand";
import { sideMappingReady } from "@/shared/lib/columnMapping";
import type {
  ColumnMappingDto,
  CsvDelimiterDto,
  CsvEncodingDto,
  DbSessionDto,
  FilePreviewDto,
  TableColumnsDto,
  TableInfoDto,
} from "@/shared/tauri/types";

export type SessionSide = "source" | "target";
export type DataSourceMode = "database" | "file";

export interface FileSourceState {
  path: string;
  preview: FilePreviewDto | null;
  loading: boolean;
  error: string | null;
  sheetName: string | null;
  encoding: CsvEncodingDto | null;
  delimiter: CsvDelimiterDto | null;
  dateFormat: string;
}

interface SideState {
  mode: DataSourceMode;
  session: DbSessionDto | null;
  tables: TableInfoDto[];
  selectedTable: string | null;
  columnMapping: ColumnMappingDto | null;
  columns: TableColumnsDto | null;
  file: FileSourceState;
  rowCount: number | null;
  loading: boolean;
  error: string | null;
}

const emptyFile: FileSourceState = {
  path: "",
  preview: null,
  loading: false,
  error: null,
  sheetName: null,
  encoding: null,
  delimiter: null,
  dateFormat: "%Y-%m-%d",
};

const emptySide: SideState = {
  mode: "database",
  session: null,
  tables: [],
  selectedTable: null,
  columnMapping: null,
  columns: null,
  file: { ...emptyFile },
  rowCount: null,
  loading: false,
  error: null,
};

interface ConnectionStore {
  source: SideState;
  target: SideState;
  setMode: (side: SessionSide, mode: DataSourceMode) => void;
  setLoading: (side: SessionSide, loading: boolean) => void;
  setError: (side: SessionSide, error: string | null) => void;
  setSession: (side: SessionSide, session: DbSessionDto | null) => void;
  setTables: (side: SessionSide, tables: TableInfoDto[]) => void;
  setSelectedTable: (side: SessionSide, table: string | null) => void;
  setColumnMapping: (
    side: SessionSide,
    mapping: ColumnMappingDto | null,
  ) => void;
  setColumns: (side: SessionSide, columns: TableColumnsDto | null) => void;
  setFileSource: (side: SessionSide, patch: Partial<FileSourceState>) => void;
  setRowCount: (side: SessionSide, count: number | null) => void;
  resetSide: (side: SessionSide) => void;
  reset: () => void;
}

export const useConnectionStore = create<ConnectionStore>((set) => ({
  source: { ...emptySide },
  target: { ...emptySide },
  setMode: (side, mode) =>
    set(
      (s) =>
        ({
          [side]: {
            ...s[side],
            mode,
            error: null,
            ...(mode === "database"
              ? { file: { ...emptyFile } }
              : {
                  session: null,
                  tables: [],
                  selectedTable: null,
                  columnMapping: null,
                  columns: null,
                  rowCount: null,
                }),
          },
        }) as Partial<ConnectionStore>,
    ),
  setLoading: (side, loading) =>
    set(
      (s) => ({ [side]: { ...s[side], loading } }) as Partial<ConnectionStore>,
    ),
  setError: (side, error) =>
    set((s) => ({ [side]: { ...s[side], error } }) as Partial<ConnectionStore>),
  setSession: (side, session) =>
    set(
      (s) =>
        ({
          [side]: { ...s[side], session, error: null },
        }) as Partial<ConnectionStore>,
    ),
  setTables: (side, tables) =>
    set(
      (s) => ({ [side]: { ...s[side], tables } }) as Partial<ConnectionStore>,
    ),
  setSelectedTable: (side, selectedTable) =>
    set(
      (s) =>
        ({
          [side]: {
            ...s[side],
            selectedTable,
            columnMapping: null,
            columns: null,
            rowCount: null,
          },
        }) as Partial<ConnectionStore>,
    ),
  setColumnMapping: (side, columnMapping) =>
    set(
      (s) =>
        ({ [side]: { ...s[side], columnMapping } }) as Partial<ConnectionStore>,
    ),
  setColumns: (side, columns) =>
    set(
      (s) => ({ [side]: { ...s[side], columns } }) as Partial<ConnectionStore>,
    ),
  setFileSource: (side, patch) =>
    set(
      (s) =>
        ({
          [side]: { ...s[side], file: { ...s[side].file, ...patch } },
        }) as Partial<ConnectionStore>,
    ),
  setRowCount: (side, rowCount) =>
    set(
      (s) => ({ [side]: { ...s[side], rowCount } }) as Partial<ConnectionStore>,
    ),
  resetSide: (side) =>
    set(() => ({ [side]: { ...emptySide } }) as Partial<ConnectionStore>),
  reset: () =>
    set(() => ({ source: { ...emptySide }, target: { ...emptySide } })),
}));

export function readinessForRun(state: Pick<ConnectionStore, "source" | "target">): {
  ready: boolean;
  reason: string | null;
} {
  if (state.source.mode === "database" && !state.source.session)
    return { ready: false, reason: "Connect a source database" };
  if (state.source.mode === "database" && !state.source.selectedTable)
    return { ready: false, reason: "Select a source table" };
  if (state.target.mode === "database" && !state.target.session)
    return { ready: false, reason: "Connect a target database" };
  if (state.target.mode === "database" && !state.target.selectedTable)
    return { ready: false, reason: "Select a target table" };

  const source = sideMappingReady("source", state.source);
  if (!source.ok) return { ready: false, reason: source.reason };
  const target = sideMappingReady("target", state.target);
  if (!target.ok) return { ready: false, reason: target.reason };
  return { ready: true, reason: null };
}
