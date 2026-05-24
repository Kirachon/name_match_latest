import clsx from "clsx";
export const cx = clsx;

export function formatNumber(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(n);
}

export function formatBytes(mb: number): string {
  if (mb < 1024) return `${formatNumber(mb)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

export function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m ${s}s`;
}

export function formatPercent(p: number): string {
  return `${p.toFixed(1)}%`;
}

export function formatTimestamp(unixMs: number): string {
  const d = new Date(unixMs);
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}
