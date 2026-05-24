import { create } from "zustand";

export interface UiErrorEntry {
  id: string;
  message: string;
  stack?: string;
  componentStack?: string;
  timestampMs: number;
}

interface ErrorStore {
  entries: UiErrorEntry[];
  push: (entry: Omit<UiErrorEntry, "id" | "timestampMs">) => void;
  clear: () => void;
}

export const useErrorStore = create<ErrorStore>((set) => ({
  entries: [],
  push: (entry) =>
    set((state) => ({
      entries: [
        ...state.entries.slice(-19),
        {
          ...entry,
          id: crypto.randomUUID(),
          timestampMs: Date.now(),
        },
      ],
    })),
  clear: () => set({ entries: [] }),
}));
