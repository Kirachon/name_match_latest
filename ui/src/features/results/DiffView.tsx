import { Button, Pill } from "@/shared/components/primitives";
import { formatNumber, formatPercent } from "@/shared/lib/format";
import type { DiffResultDto, MatchPairDto } from "@/shared/tauri/types";

export function DiffView({
  diff,
  onClose,
}: {
  diff: DiffResultDto;
  onClose: () => void;
}) {
  return (
    <div className="border-t border-white/10 px-4 py-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone="ok">{formatNumber(diff.added.length)} added</Pill>
          <Pill tone="danger">{formatNumber(diff.removed.length)} removed</Pill>
          <Pill tone="warn">{formatNumber(diff.changed.length)} changed</Pill>
        </div>
        <Button tone="ghost" size="sm" onClick={onClose}>
          Close compare
        </Button>
      </div>
      <div className="grid xl:grid-cols-3 gap-3">
        <DiffColumn title="Added" rows={diff.added} tone="ok" />
        <DiffColumn title="Removed" rows={diff.removed} tone="danger" />
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-ink-100">Changed</h3>
          {diff.changed.slice(0, 12).map((row) => (
            <div
              key={`${row.after.source_id}:${row.after.target_id}`}
              className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2"
            >
              <div className="truncate text-sm text-ink-100">
                {row.after.source_full_name} → {row.after.target_full_name}
              </div>
              <div className="mt-1 text-xs tabular text-ink-400">
                {formatPercent(row.before.confidence)} →{" "}
                {formatPercent(row.after.confidence)} (
                {row.confidence_delta > 0 ? "+" : ""}
                {row.confidence_delta.toFixed(1)})
              </div>
            </div>
          ))}
          {diff.changed.length === 0 && <Empty />}
        </div>
      </div>
    </div>
  );
}

function DiffColumn({
  title,
  rows,
  tone,
}: {
  title: string;
  rows: MatchPairDto[];
  tone: "ok" | "danger";
}) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-ink-100">{title}</h3>
      {rows.slice(0, 12).map((row) => (
        <div
          key={`${row.source_id}:${row.target_id}`}
          className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2"
        >
          <div className="truncate text-sm text-ink-100">
            {row.source_full_name} → {row.target_full_name}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-ink-400">
            <Pill tone={tone}>{formatPercent(row.confidence)}</Pill>
            <span className="tabular">
              {row.source_id}:{row.target_id}
            </span>
          </div>
        </div>
      ))}
      {rows.length === 0 && <Empty />}
    </div>
  );
}

function Empty() {
  return <div className="text-sm text-ink-500">No rows.</div>;
}
