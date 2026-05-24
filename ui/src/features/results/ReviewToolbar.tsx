import { Button, Pill } from "@/shared/components/primitives";
import { formatNumber } from "@/shared/lib/format";

export function ReviewToolbar({
  total,
  accepted,
  rejected,
  pending,
  disabled,
  onNextPending,
}: {
  total: number;
  accepted: number;
  rejected: number;
  pending: number;
  disabled?: boolean;
  onNextPending: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <Pill tone="info">Review</Pill>
        <Pill tone="ok">{formatNumber(accepted)} accepted</Pill>
        <Pill tone="danger">{formatNumber(rejected)} rejected</Pill>
        <Pill tone={pending > 0 ? "warn" : "mute"}>
          {formatNumber(pending)} pending in band
        </Pill>
        <span className="text-xs text-ink-400">
          {formatNumber(total)} rows on this page
        </span>
      </div>
      <Button
        tone="secondary"
        size="sm"
        disabled={disabled || pending === 0}
        onClick={onNextPending}
      >
        Next pending
      </Button>
    </div>
  );
}
