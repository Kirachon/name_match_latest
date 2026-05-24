import {
  AlgorithmOrCascadeCard,
  ExportCard,
  GpuCard,
  MatchOptionsCard,
  ModeCard,
  PerformanceCard,
  StreamingCard,
  SummaryCard,
} from "./ConfigureCards";

interface ConfigureTabProps {
  onAdvance: () => void;
}

export function ConfigureTab({ onAdvance }: ConfigureTabProps) {
  return (
    <div className="grid xl:grid-cols-3 gap-5">
      <div className="xl:col-span-2 space-y-5">
        <ModeCard />
        <AlgorithmOrCascadeCard />
        <MatchOptionsCard />
        <GpuCard />
        <StreamingCard />
        <ExportCard />
      </div>
      <div className="xl:col-span-1 space-y-5">
        <SummaryCard onAdvance={onAdvance} />
        <PerformanceCard />
      </div>
    </div>
  );
}
