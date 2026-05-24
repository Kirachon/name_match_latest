import { useConnectionStore } from "@/shared/stores/connectionStore";
import { Button, Card, SectionHeader } from "@/shared/components/primitives";
import { ConnectionCard } from "./ConnectionCard";

export function ConnectTab({ onAdvance }: { onAdvance: () => void }) {
  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <ConnectionCard side="source" />
      <ConnectionCard side="target" />
      <Card className="lg:col-span-2 dot-grid relative overflow-hidden">
        <div className="relative z-10 flex items-center justify-between gap-4">
          <div>
            <SectionHeader
              title="Workflow"
              description="Connect both databases, then pick the source and target tables. The Configure tab unlocks once schemas are verified."
            />
            <ul className="text-sm text-ink-300 list-disc list-inside marker:text-accent-500 space-y-1">
              <li>
                Required columns are inferred automatically per algorithm.
              </li>
              <li>Row counts are estimates — they’re cached per session.</li>
              <li>
                Passwords are dropped from memory after the pool is built. Saved
                passwords (opt-in) are stored unencrypted in the app data
                directory.
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
  const source = useConnectionStore((s) => s.source);
  const target = useConnectionStore((s) => s.target);
  const ready =
    !!source.session &&
    !!source.selectedTable &&
    !!target.session &&
    !!target.selectedTable;

  return (
    <Button
      tone="primary"
      onClick={onAdvance}
      disabled={!ready}
      trailingIcon={<ArrowRight />}
      title={
        ready
          ? "Move to Configure"
          : "Connect both databases and pick tables first"
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
