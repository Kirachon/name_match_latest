import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { MatchPairDto } from "@/shared/tauri/types";
import { MatchRow } from "./MatchRow";

const COL_TEMPLATE = "70px 90px 1fr 110px 90px 1fr 110px 90px 130px 90px";

export function ResultsTable({
  rows,
  selectedRowId,
  onSelectRow,
}: {
  rows: MatchPairDto[];
  selectedRowId: number | null;
  onSelectRow: (row: MatchPairDto) => void;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 12,
  });

  return (
    <div
      ref={parentRef}
      className="h-[520px] overflow-auto border-t border-ink-800/70"
      style={{ ["--data-cols" as string]: COL_TEMPLATE }}
      role="table"
      aria-label="Match results"
    >
      <div className="data-row data-header h-9" role="row">
        <div role="columnheader" className="font-mono">
          #
        </div>
        <div role="columnheader">Source ID</div>
        <div role="columnheader">Source name</div>
        <div role="columnheader">Source DOB</div>
        <div role="columnheader">Target ID</div>
        <div role="columnheader">Target name</div>
        <div role="columnheader">Target DOB</div>
        <div role="columnheader" className="text-right">
          Confidence
        </div>
        <div role="columnheader">Level / method</div>
        <div role="columnheader">Fields</div>
      </div>
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        {rowVirtualizer.getVirtualItems().map((vi) => {
          const r = rows[vi.index];
          if (!r) return null;
          return (
            <MatchRow
              key={r.row_id}
              row={r}
              top={vi.start}
              height={vi.size}
              selected={selectedRowId === r.row_id}
              onSelect={() => onSelectRow(r)}
            />
          );
        })}
      </div>
    </div>
  );
}
