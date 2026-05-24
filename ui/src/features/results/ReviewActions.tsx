import { Button } from "@/shared/components/primitives";
import type { ReviewDecisionValue } from "@/shared/tauri/types";

export function ReviewActions({
  decision,
  disabled,
  onDecision,
}: {
  decision?: ReviewDecisionValue;
  disabled?: boolean;
  onDecision: (decision: ReviewDecisionValue) => void;
}) {
  return (
    <div className="flex items-center gap-1">
      <Button
        tone={decision === "accepted" ? "primary" : "ghost"}
        size="sm"
        disabled={disabled}
        title="Accept match"
        aria-label="Accept match"
        onClick={(event) => {
          event.stopPropagation();
          onDecision("accepted");
        }}
      >
        <span aria-hidden="true">✓</span>
      </Button>
      <Button
        tone={decision === "rejected" ? "danger" : "ghost"}
        size="sm"
        disabled={disabled}
        title="Reject match"
        aria-label="Reject match"
        onClick={(event) => {
          event.stopPropagation();
          onDecision("rejected");
        }}
      >
        <span aria-hidden="true">×</span>
      </Button>
      {decision && decision !== "pending" && (
        <Button
          tone="ghost"
          size="sm"
          disabled={disabled}
          title="Reset to pending"
          aria-label="Reset to pending"
          onClick={(event) => {
            event.stopPropagation();
            onDecision("pending");
          }}
        >
          <span aria-hidden="true">↺</span>
        </Button>
      )}
    </div>
  );
}
