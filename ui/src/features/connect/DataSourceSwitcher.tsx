import {
  type DataSourceMode,
  type SessionSide,
  useConnectionStore,
} from "@/shared/stores/connectionStore";
import { cx } from "@/shared/lib/format";

const OPTIONS: Array<{ id: DataSourceMode; label: string }> = [
  { id: "database", label: "Database" },
  { id: "file", label: "File" },
];

export function DataSourceSwitcher({ side }: { side: SessionSide }) {
  const mode = useConnectionStore((s) => s[side].mode);
  const setMode = useConnectionStore((s) => s.setMode);

  return (
    <div
      role="tablist"
      aria-label={`${side} data source type`}
      className="grid grid-cols-2 gap-1 rounded-lg border border-ink-800 bg-ink-950/60 p-1"
    >
      {OPTIONS.map((option) => (
        <button
          key={option.id}
          type="button"
          role="tab"
          aria-selected={mode === option.id}
          onClick={() => setMode(side, option.id)}
          className={cx(
            "h-8 rounded-md text-xs font-medium transition-colors",
            mode === option.id
              ? "bg-accent-500/15 text-accent-200"
              : "text-ink-400 hover:text-ink-100",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
