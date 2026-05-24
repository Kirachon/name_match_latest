import { create } from "zustand";
import type {
  DbSessionDto,
  TableColumnsDto,
  TableInfoDto,
} from "@/shared/tauri/types";

export type SessionSide = "source" | "target";

interface SideState {
  session: DbSessionDto | null;
  tables: TableInfoDto[];
  selectedTable: string | null;
  columns: TableColumnsDto | null;
  rowCount: number | null;
  loading: boolean;
  error: string | null;
}

const emptySide: SideState = {
  session: null,
  tables: [],
  selectedTable: null,
  columns: null,
  rowCount: null,
  loading: false,
  error: null,
};

interface ConnectionStore {
  source: SideState;
  target: SideState;
  setLoading: (side: SessionSide, loading: boolean) => void;
  setError: (side: SessionSide, error: string | null) => void;
  setSession: (side: SessionSide, session: DbSessionDto | null) => void;
  setTables: (side: SessionSide, tables: TableInfoDto[]) => void;
  setSelectedTable: (side: SessionSide, table: string | null) => void;
  setColumns: (side: SessionSide, columns: TableColumnsDto | null) => void;
  setRowCount: (side: SessionSide, count: number | null) => void;
  resetSide: (side: SessionSide) => void;
  reset: () => void;
}

export const useConnectionStore = create<ConnectionStore>((set) => ({
  source: { ...emptySide },
  target: { ...emptySide },
  setLoading: (side, loading) =>
    set((s) => ({ [side]: { ...s[side], loading } } as Partial<ConnectionStore>)),
  setError: (side, error) =>
    set((s) => ({ [side]: { ...s[side], error } } as Partial<ConnectionStore>)),
  setSession: (side, session) =>
    set((s) => ({
      [side]: { ...s[side], session, error: null },
    } as Partial<ConnectionStore>)),
  setTables: (side, tables) =>
    set((s) => ({ [side]: { ...s[side], tables } } as Partial<ConnectionStore>)),
  setSelectedTable: (side, selectedTable) =>
    set((s) => ({
      [side]: { ...s[side], selectedTable, columns: null, rowCount: null },
    } as Partial<ConnectionStore>)),
  setColumns: (side, columns) =>
    set((s) => ({ [side]: { ...s[side], columns } } as Partial<ConnectionStore>)),
  setRowCount: (side, rowCount) =>
    set((s) => ({ [side]: { ...s[side], rowCount } } as Partial<ConnectionStore>)),
  resetSide: (side) => set(() => ({ [side]: { ...emptySide } } as Partial<ConnectionStore>)),
  reset: () => set(() => ({ source: { ...emptySide }, target: { ...emptySide } })),
}));

export function readinessForRun(state: ConnectionStore): {
  ready: boolean;
  reason: string | null;
} {
  if (!state.source.session)
    return { ready: false, reason: "Connect a source database" };
  if (!state.source.selectedTable)
    return { ready: false, reason: "Select a source table" };
  if (!state.target.session)
    return { ready: false, reason: "Connect a target database" };
  if (!state.target.selectedTable)
    return { ready: false, reason: "Select a target table" };
  return { ready: true, reason: null };
}
