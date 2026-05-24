import { useConnectionStore } from "@/shared/stores/connectionStore";
import { useConfigStore } from "@/shared/stores/configStore";
import { useJobStore, useProgressStore } from "@/shared/stores/jobStore";
import { Pill, StatusDot } from "@/shared/components/primitives";
import {
  algorithmMeta,
  type SystemInfoDto,
  type JobStateDto,
} from "@/shared/tauri/types";
import { cx, formatNumber } from "@/shared/lib/format";

export function StatusRail({ system }: { system: SystemInfoDto | null }) {
  const source = useConnectionStore((s) => s.source);
  const target = useConnectionStore((s) => s.target);
  const algorithm = useConfigStore((s) => s.algorithm);
  const mode = useConfigStore((s) => s.mode);
  const cascadeLevels = useConfigStore((s) => s.cascade.levels.length);
  const gpuMode = useConfigStore((s) => s.gpu.mode);
  const exportDir = useConfigStore((s) => s.export.output_directory);
  const jobState = useJobStore((s) => s.state);
  const matches = useProgressStore((s) => s.matchesFound);

  const algoMeta = algorithmMeta(algorithm);

  return (
    <div
      className="rail-gradient border-b border-ink-800 px-4 flex items-center gap-2 h-[var(--rail-h)] shrink-0"
      role="banner"
    >
      <div className="flex items-center gap-2 mr-3 select-none">
        <div className="flex items-center justify-center h-8 w-8 rounded-md bg-accent-500/15 text-accent-400 border border-accent-500/30">
          <Logo />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-ink-50 tracking-tight">
            Name Matcher
          </div>
          <div className="text-2xs text-ink-500">
            v{system?.app_version ?? "0.1.0"} · operational
          </div>
        </div>
      </div>

      <Divider />

      <RailItem
        label="Source"
        tone={source.session ? "ok" : "mute"}
        primary={
          source.session
            ? `${source.session.username}@${source.session.host}/${source.session.database}`
            : "Not connected"
        }
        secondary={source.selectedTable ?? undefined}
      />
      <RailItem
        label="Target"
        tone={target.session ? "ok" : "mute"}
        primary={
          target.session
            ? `${target.session.username}@${target.session.host}/${target.session.database}`
            : "Not connected"
        }
        secondary={target.selectedTable ?? undefined}
      />

      <Divider />

      <RailItem
        label="Algorithm"
        tone="info"
        primary={
          mode === "deep" ? "Deep Match" : `Option ${algoMeta.optionNumber}`
        }
        secondary={
          mode === "deep"
            ? `Cascade · ${cascadeLevels} level${cascadeLevels === 1 ? "" : "s"}`
            : algoMeta.label
        }
      />
      <RailItem
        label="GPU"
        tone={
          gpuMode === "cpu" ? "mute" : system?.gpu_available ? "ok" : "warn"
        }
        primary={
          gpuMode === "cpu" ? "CPU" : gpuMode === "auto" ? "Auto" : "Force GPU"
        }
        secondary={
          system?.gpu_available
            ? (system.gpu_devices[0] ?? "GPU available")
            : gpuMode === "cpu"
              ? "Disabled by config"
              : "No CUDA detected"
        }
      />
      <RailItem
        label="Output"
        tone={exportDir ? "info" : "warn"}
        primary={exportDir || "Not set"}
      />

      <div className="ml-auto flex items-center gap-2">
        {system && (
          <div className="hidden md:flex items-center gap-2 text-2xs text-ink-400 mr-2 tabular">
            <span title="Logical CPU cores">
              CPU {formatNumber(system.cpu_cores_logical)}
            </span>
            <span aria-hidden>·</span>
            <span title="Available memory">
              RAM {formatNumber(system.memory_avail_mb)} MB
            </span>
          </div>
        )}
        <JobStatePill state={jobState} matches={matches} />
      </div>
    </div>
  );
}

function Divider() {
  return <span className="h-6 w-px bg-ink-800" aria-hidden />;
}

function RailItem({
  label,
  primary,
  secondary,
  tone,
}: {
  label: string;
  primary: string;
  secondary?: string;
  tone: "ok" | "warn" | "danger" | "info" | "mute";
}) {
  return (
    <div className="px-2.5 flex items-center gap-2 min-w-0">
      <StatusDot tone={tone} />
      <div className="min-w-0">
        <div className="text-2xs uppercase tracking-wider text-ink-500">
          {label}
        </div>
        <div
          className="text-xs text-ink-100 truncate max-w-[220px]"
          title={primary}
        >
          {primary}
        </div>
        {secondary && (
          <div
            className="text-2xs text-ink-400 truncate max-w-[220px]"
            title={secondary}
          >
            {secondary}
          </div>
        )}
      </div>
    </div>
  );
}

function JobStatePill({
  state,
  matches,
}: {
  state: JobStateDto;
  matches: number;
}) {
  const tone =
    state === "completed"
      ? "ok"
      : state === "failed"
        ? "danger"
        : state === "cancelled"
          ? "warn"
          : state === "idle"
            ? "mute"
            : "info";
  const isActive =
    state === "running" ||
    state === "starting" ||
    state === "validating" ||
    state === "cancelling";
  return (
    <Pill tone={tone}>
      <StatusDot tone={tone === "mute" ? "mute" : tone} pulse={isActive} />
      <span className={cx("uppercase")}>{state.replace(/-/g, " ")}</span>
      {state !== "idle" && (
        <span className="tabular ml-1">· {formatNumber(matches)}</span>
      )}
    </Pill>
  );
}

function Logo() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 32 32"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 22 L11 10 L16 22 L21 10 L26 22" />
    </svg>
  );
}
