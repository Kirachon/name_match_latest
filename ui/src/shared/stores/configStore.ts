import { create } from "zustand";
import {
  type AlgorithmDto,
  type CascadeOptionsDto,
  type CascadePresetId,
  type ExportOptionsDto,
  type GpuOptionsDto,
  type MatchOptionsDto,
  type RunConfigDto,
  type StreamingOptionsDto,
  type TableSelectionDto,
  DEFAULT_EXPORT_OPTIONS,
  DEFAULT_GPU_OPTIONS,
  DEFAULT_MATCH_OPTIONS,
  DEFAULT_STREAMING_OPTIONS,
} from "@/shared/tauri/types";

export type MatchingMode = "quick" | "deep";

interface CascadeUIState {
  preset: CascadePresetId;
  levels: number[];
  fuzzy_threshold: number;
  exclusion_mode: "exclusive" | "independent";
}

const DEFAULT_CASCADE_STATE: CascadeUIState = {
  preset: "standard",
  levels: [1, 2, 3, 10, 11],
  fuzzy_threshold: 0.95,
  exclusion_mode: "exclusive",
};

interface ConfigStore {
  mode: MatchingMode;
  algorithm: AlgorithmDto;
  options: MatchOptionsDto;
  gpu: GpuOptionsDto;
  streaming: StreamingOptionsDto;
  export: ExportOptionsDto;
  cascade: CascadeUIState;
  setMode: (m: MatchingMode) => void;
  setAlgorithm: (a: AlgorithmDto) => void;
  setOptions: (patch: Partial<MatchOptionsDto>) => void;
  setGpu: (patch: Partial<GpuOptionsDto>) => void;
  setStreaming: (patch: Partial<StreamingOptionsDto>) => void;
  setExport: (patch: Partial<ExportOptionsDto>) => void;
  setCascade: (patch: Partial<CascadeUIState>) => void;
  reset: () => void;
  buildRunConfig: (
    source: TableSelectionDto,
    target: TableSelectionDto,
    geo?: { hasBarangay: boolean; hasCity: boolean },
  ) => RunConfigDto;
}

export const useConfigStore = create<ConfigStore>((set, get) => ({
  mode: "quick",
  algorithm: "fuzzy",
  options: { ...DEFAULT_MATCH_OPTIONS },
  gpu: { ...DEFAULT_GPU_OPTIONS },
  streaming: { ...DEFAULT_STREAMING_OPTIONS },
  export: { ...DEFAULT_EXPORT_OPTIONS },
  cascade: { ...DEFAULT_CASCADE_STATE },
  setMode: (mode) => set({ mode }),
  setAlgorithm: (algorithm) => set({ algorithm }),
  setOptions: (patch) => set((s) => ({ options: { ...s.options, ...patch } })),
  setGpu: (patch) => set((s) => ({ gpu: { ...s.gpu, ...patch } })),
  setStreaming: (patch) =>
    set((s) => ({ streaming: { ...s.streaming, ...patch } })),
  setExport: (patch) => set((s) => ({ export: { ...s.export, ...patch } })),
  setCascade: (patch) => set((s) => ({ cascade: { ...s.cascade, ...patch } })),
  reset: () =>
    set({
      mode: "quick",
      algorithm: "fuzzy",
      options: { ...DEFAULT_MATCH_OPTIONS },
      gpu: { ...DEFAULT_GPU_OPTIONS },
      streaming: { ...DEFAULT_STREAMING_OPTIONS },
      export: { ...DEFAULT_EXPORT_OPTIONS },
      cascade: { ...DEFAULT_CASCADE_STATE },
    }),
  buildRunConfig: (source, target, geo) => {
    const s = get();
    const cascade: CascadeOptionsDto | null =
      s.mode === "deep"
        ? {
            enabled: true,
            levels: s.cascade.levels,
            fuzzy_threshold: s.cascade.fuzzy_threshold,
            exclusion_mode: s.cascade.exclusion_mode,
            has_barangay_code: geo?.hasBarangay ?? false,
            has_city_code: geo?.hasCity ?? false,
          }
        : null;
    return {
      source,
      target,
      algorithm: s.algorithm,
      options: s.options,
      gpu: s.gpu,
      streaming: s.streaming,
      export: s.export,
      cascade,
    };
  },
}));
