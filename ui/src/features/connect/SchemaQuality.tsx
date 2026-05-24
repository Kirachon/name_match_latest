import { StatusDot } from "@/shared/components/primitives";
import { useConfigStore } from "@/shared/stores/configStore";
import { algorithmMeta, type TableColumnsDto } from "@/shared/tauri/types";
import { cx, formatNumber } from "@/shared/lib/format";

export function SchemaQuality({
  columns,
  rowCount,
}: {
  columns: TableColumnsDto;
  rowCount: number | null;
}) {
  const algorithm = useConfigStore((s) => s.algorithm);
  const meta = algorithmMeta(algorithm);
  const required = meta.requiresColumns;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="section-title">Schema check</div>
        <div className="text-xs text-ink-300 tabular">
          rows: {rowCount != null ? formatNumber(rowCount) : "-"}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {(
          [
            ["id", columns.has_id],
            ["uuid", columns.has_uuid],
            ["first_name", columns.has_first_name],
            ["middle_name", columns.has_middle_name],
            ["last_name", columns.has_last_name],
            ["birthdate", columns.has_birthdate],
            ["hh_id", columns.has_hh_id],
          ] as const
        ).map(([name, ok]) => {
          const isRequired = required.includes(
            `has_${name}` as keyof TableColumnsDto,
          );
          return (
            <div
              key={name}
              className={cx(
                "flex items-center gap-2 text-xs px-2 py-1 rounded border",
                ok
                  ? "border-ok-500/30 bg-ok-500/5 text-ok-400"
                  : isRequired
                    ? "border-danger-500/30 bg-danger-500/5 text-danger-400"
                    : "border-ink-800 bg-ink-900/40 text-ink-500",
              )}
            >
              <StatusDot tone={ok ? "ok" : isRequired ? "danger" : "mute"} />
              <code className="font-mono">{name}</code>
              {isRequired && (
                <span className="ml-auto text-2xs uppercase opacity-80">
                  req
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
