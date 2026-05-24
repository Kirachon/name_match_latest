import { cx, formatPercent } from "@/shared/lib/format";
import type { MatchPairDto } from "@/shared/tauri/types";

export function MatchRow({
  row,
  top,
  height,
}: {
  row: MatchPairDto;
  top: number;
  height: number;
}) {
  return (
    <div
      role="row"
      className="data-row"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        transform: `translateY(${top}px)`,
        height: `${height}px`,
      }}
    >
      <div role="cell" className="font-mono text-ink-500 tabular">
        {row.row_id}
      </div>
      <div role="cell" className="tabular text-ink-400">
        {row.source_id}
      </div>
      <div role="cell" className="truncate" title={row.source_full_name}>
        {row.source_full_name}
      </div>
      <div role="cell" className="tabular text-ink-400">
        {row.source_birthdate ?? "-"}
      </div>
      <div role="cell" className="tabular text-ink-400">
        {row.target_id}
      </div>
      <div role="cell" className="truncate" title={row.target_full_name}>
        {row.target_full_name}
      </div>
      <div role="cell" className="tabular text-ink-400">
        {row.target_birthdate ?? "-"}
      </div>
      <div role="cell" className="text-right tabular">
        <ConfidencePill value={row.confidence} />
      </div>
      <div
        role="cell"
        className="text-2xs text-ink-300 truncate"
        title={row.match_method ?? undefined}
      >
        {row.matched_at_level
          ? `L${String(row.matched_at_level).padStart(2, "0")}${
              row.match_method
                ? ` - ${row.match_method.replace(/^L\d+\s*-\s*/, "")}`
                : ""
            }`
          : "-"}
      </div>
      <div
        role="cell"
        className="text-2xs text-ink-400 truncate"
        title={row.matched_fields.join(", ")}
      >
        {row.matched_fields.join(", ")}
      </div>
    </div>
  );
}

function ConfidencePill({ value }: { value: number }) {
  const tone =
    value >= 95 ? "ok" : value >= 85 ? "info" : value >= 70 ? "warn" : "danger";
  return (
    <span
      className={cx(
        "tabular px-1.5 py-0.5 rounded text-xs",
        tone === "ok" && "bg-ok-500/15 text-ok-400",
        tone === "info" && "bg-accent-500/15 text-accent-400",
        tone === "warn" && "bg-warn-500/15 text-warn-400",
        tone === "danger" && "bg-danger-500/15 text-danger-400",
      )}
    >
      {formatPercent(value)}
    </span>
  );
}
