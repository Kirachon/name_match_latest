import { create } from "zustand";

export type ToastTone = "info" | "success" | "warn" | "error";

export interface Toast {
  id: string;
  tone: ToastTone;
  title: string;
  message?: string;
  /** Auto-dismiss in ms. `null` to require manual dismiss. */
  ttlMs?: number | null;
}

interface ToastStore {
  toasts: Toast[];
  push: (t: Omit<Toast, "id">) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  push: (t) => {
    const id = crypto.randomUUID();
    const ttl = t.ttlMs === undefined ? 4000 : t.ttlMs;
    const toast: Toast = { ...t, id, ttlMs: ttl };
    set((s) => ({ toasts: [...s.toasts, toast] }));
    if (ttl != null) {
      setTimeout(() => {
        set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) }));
      }, ttl);
    }
    return id;
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  clear: () => set({ toasts: [] }),
}));
