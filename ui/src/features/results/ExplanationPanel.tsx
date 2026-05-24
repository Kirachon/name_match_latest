import { Button, Pill, SectionHeader } from "@/shared/components/primitives";
import { formatPercent } from "@/shared/lib/format";
import type { MatchPairDto, ScoreBreakdownDto } from "@/shared/tauri/types";

export function ExplanationPanel({
  row,
  breakdown,
  loading,
  error,
  onClose,
}: {
  row: MatchPairDto;
  breakdown: ScoreBreakdownDto | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  return (
    <aside className="border-l border-ink-800/70 bg-ink-950/50 p-4 min-w-[320px] max-w-[380px]">
      <SectionHeader
        title="Match explanation"
        description={`${row.source_full_name} -> ${row.target_full_name}`}
        action={
          <Button tone="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        }
      />
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <Pill tone="info">{formatPercent(row.confidence)}</Pill>
          {breakdown?.case_label && (
            <Pill tone="ok">{breakdown.case_label}</Pill>
          )}
          {loading && <Pill tone="mute">Loading</Pill>}
        </div>
        {error && <div className="help-error">{error}</div>}
        {breakdown && !breakdown.supported && (
          <div className="surface-soft p-3 text-xs text-ink-300">
            {breakdown.message ?? "Explanation is not available for this row."}
          </div>
        )}
        {breakdown?.supported && (
          <dl className="space-y-2">
            <Metric label="Levenshtein" value={breakdown.levenshtein_pct} />
            <Metric label="Jaro-Winkler" value={breakdown.jaro_winkler_pct} />
            <Metric label="Metaphone" value={breakdown.metaphone_pct} />
            <Row
              label="Birthdate"
              value={
                breakdown.birthdate_match == null
                  ? "Missing"
                  : breakdown.birthdate_match
                    ? breakdown.birthdate_swap_used
                      ? "Swap match"
                      : "Match"
                    : "No match"
              }
            />
            {breakdown.message && (
              <Row label="Note" value={breakdown.message} />
            )}
          </dl>
        )}
      </div>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value?: number | null }) {
  return (
    <Row label={label} value={value == null ? "-" : formatPercent(value)} />
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-xs text-ink-400">{label}</dt>
      <dd className="text-ink-100 tabular text-right">{value}</dd>
    </div>
  );
}
