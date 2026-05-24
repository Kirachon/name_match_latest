import {
  readinessForRun,
  useConnectionStore,
} from "@/shared/stores/connectionStore";
import { Button, Card, SectionHeader } from "@/shared/components/primitives";
import { ConnectionCard } from "./ConnectionCard";
import { DataSourceSwitcher } from "./DataSourceSwitcher";
import { FileSourceCard } from "./FileSourceCard";

export function ConnectTab({ onAdvance }: { onAdvance: () => void }) {
  const sourceMode = useConnectionStore((s) => s.source.mode);
  const targetMode = useConnectionStore((s) => s.target.mode);

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <div className="space-y-3">
        <DataSourceSwitcher side="source" />
        {sourceMode === "database" ? (
          <ConnectionCard side="source" />
        ) : (
          <FileSourceCard side="source" />
        )}
      </div>
      <div className="space-y-3">
        <DataSourceSwitcher side="target" />
        {targetMode === "database" ? (
          <ConnectionCard side="target" />
        ) : (
          <FileSourceCard side="target" />
        )}
      </div>
      <Card className="lg:col-span-2 dot-grid relative overflow-hidden">
        <div className="relative z-10 flex items-center justify-between gap-4">
          <div>
            <SectionHeader
              title="Workflow"
              description="Connect databases or preview CSV files, then map columns before configuring the run."
            />
            <ul className="text-sm text-ink-300 list-disc list-inside marker:text-accent-500 space-y-1">
              <li>
                Required columns are inferred automatically per algorithm.
              </li>
              <li>Row counts are estimates — they’re cached per session.</li>
              <li>
                CSV import preview supports encoding, delimiter, and date-format
                overrides.
              </li>
            </ul>
          </div>
          <ContinueButton onAdvance={onAdvance} />
        </div>
      </Card>
    </div>
  );
}

function ContinueButton({ onAdvance }: { onAdvance: () => void }) {
  const { ready, reason } = useConnectionStore(readinessForRun);

  return (
    <Button
      tone="primary"
      onClick={onAdvance}
      disabled={!ready}
      trailingIcon={<ArrowRight />}
      title={
        ready
          ? "Move to Configure"
          : (reason ?? "Finish source and target setup first")
      }
    >
      Continue to Configure
    </Button>
  );
}

function ArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M5 12h14M13 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
