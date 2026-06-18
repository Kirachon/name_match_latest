import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useConfigStore } from "@/shared/stores/configStore";
import { useConnectionStore } from "@/shared/stores/connectionStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
  Toggle,
} from "@/shared/components/primitives";
import {
  ALGORITHMS,
  type AlgorithmDto,
  algorithmMeta,
  type ComputeModeDto,
  type ExportFormatDto,
  type GpuFuzzyGateModeDto,
  type RunModeDto,
} from "@/shared/tauri/types";
import { cx, formatNumber } from "@/shared/lib/format";
import {
  anyStreamingBackendActive,
  crossSessionDbStreamingMessage,
  crossSessionStreamingBackendActive,
  maxSideRows,
  needsCrossSessionDbStreamingNotice,
  resolveEffectiveRunMode,
  rowCountForSide,
  SCALE_WARN_ROWS,
  scaleWarningLevel,
  streamingBackendActive,
} from "@/shared/runScalePolicy";
import type { TableSelectionDto } from "@/shared/tauri/types";
import { useToastStore } from "@/shared/stores/toastStore";
import { CascadePicker } from "./CascadePicker";

export function ModeCard() {
  const mode = useConfigStore((s) => s.mode);
  const setMode = useConfigStore((s) => s.setMode);
  return (
    <Card>
      <SectionHeader
        title="Matching mode"
        description="Quick Match runs a single algorithm. Deep Match runs a cascade of L1-L11 levels and combines the results."
      />
      <div
        role="radiogroup"
        aria-label="Matching mode"
        className="grid md:grid-cols-2 gap-3"
      >
        <ModeOption
          active={mode === "quick"}
          onSelect={() => setMode("quick")}
          title="Quick Match"
          subtitle="Single-pass algorithm (Options 1-7)"
          tag="Fast - exploratory"
        />
        <ModeOption
          active={mode === "deep"}
          onSelect={() => setMode("deep")}
          title="Deep Match"
          subtitle="Cascade of 11 sequential levels"
          tag="Thorough - production"
        />
      </div>
    </Card>
  );
}

function ModeOption({
  active,
  onSelect,
  title,
  subtitle,
  tag,
}: {
  active: boolean;
  onSelect: () => void;
  title: string;
  subtitle: string;
  tag: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onSelect}
      className={cx(
        "rounded-lg border p-4 text-left transition-colors surface-hover",
        active
          ? "border-accent-500/60 bg-accent-500/5 shadow-glow"
          : "border-ink-800 bg-ink-900/40",
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-base font-semibold text-ink-50">{title}</span>
        {active && <Pill tone="info">Active</Pill>}
      </div>
      <div className="text-xs text-ink-300 mb-2">{subtitle}</div>
      <div className="text-2xs uppercase tracking-wider text-ink-500">
        {tag}
      </div>
    </button>
  );
}

export function AlgorithmOrCascadeCard() {
  const mode = useConfigStore((s) => s.mode);
  if (mode === "deep") {
    return (
      <Card>
        <SectionHeader
          title="Cascade levels"
          description="Pick which levels to run. The preset auto-adapts to the columns detected on your tables; tap Custom for full control."
        />
        <CascadePicker />
      </Card>
    );
  }
  return <AlgorithmCard />;
}

function AlgorithmCard() {
  const current = useConfigStore((s) => s.algorithm);
  const setAlgorithm = useConfigStore((s) => s.setAlgorithm);

  return (
    <Card>
      <SectionHeader
        title="Algorithm"
        description="Pick the matching strategy. Determinism beats fuzziness when both tables are clean; fuzzy variants resilient against typos and middle-name gaps."
      />
      <div
        role="radiogroup"
        aria-label="Matching algorithm"
        className="grid md:grid-cols-2 gap-3"
      >
        {ALGORITHMS.map((a) => (
          <AlgorithmOption
            key={a.id}
            id={a.id}
            optionNumber={a.optionNumber}
            label={a.label}
            description={a.description}
            selected={a.id === current}
            onSelect={setAlgorithm}
          />
        ))}
      </div>
    </Card>
  );
}

function AlgorithmOption({
  id,
  optionNumber,
  label,
  description,
  selected,
  onSelect,
}: {
  id: AlgorithmDto;
  optionNumber: number;
  label: string;
  description: string;
  selected: boolean;
  onSelect: (id: AlgorithmDto) => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={() => onSelect(id)}
      className={cx(
        "text-left rounded-lg border p-3 transition-colors surface-hover",
        selected
          ? "border-accent-500/60 bg-accent-500/5 shadow-glow"
          : "border-ink-800 bg-ink-900/40",
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="text-xs uppercase tracking-wide text-ink-400">
          Option {optionNumber}
        </span>
        {selected && <Pill tone="info">Selected</Pill>}
      </div>
      <div className="text-sm font-medium text-ink-50 mb-1">{label}</div>
      <div className="text-xs text-ink-300 leading-relaxed">{description}</div>
    </button>
  );
}

export function MatchOptionsCard() {
  const mode = useConfigStore((s) => s.mode);
  const allowSwap = useConfigStore((s) => s.options.allow_birthdate_swap);
  const setOptions = useConfigStore((s) => s.setOptions);

  return (
    <Card>
      <SectionHeader
        title="Match options"
        description={
          mode === "deep"
            ? "Shared rules used by Deep Match cascade levels."
            : "Shared rules used by the selected Quick Match algorithm."
        }
      />
      <Toggle
        checked={allowSwap}
        onChange={(b) => setOptions({ allow_birthdate_swap: b })}
        label="Allow birthdate month/day swap"
        description={
          mode === "deep"
            ? "Lets fuzzy cascade levels L10/L11 treat dates like 1990-04-12 and 1990-12-04 as a possible same birthday."
            : "Useful when source data may have transposed month/day digits."
        }
      />
    </Card>
  );
}

export function GpuCard() {
  const gpu = useConfigStore((s) => s.gpu);
  const setGpu = useConfigStore((s) => s.setGpu);
  const mode = useConfigStore((s) => s.mode);
  const algorithm = useConfigStore((s) => s.algorithm);
  const meta = algorithmMeta(algorithm);
  const gpuOff = gpu.mode === "cpu";
  const isDeep = mode === "deep";

  return (
    <Card>
      <SectionHeader
        title="GPU acceleration"
        description={
          isDeep
            ? "Deep Match uses GPU acceleration mainly for fuzzy cascade levels L10/L11. Exact levels L1-L9 remain CPU-style matching."
            : meta.gpuApplicable
              ? "GPU offload is available for this algorithm. Toggle it off to force CPU."
              : "This algorithm does not benefit from GPU acceleration."
        }
        action={
          <Pill tone={gpuOff ? "mute" : "info"}>
            {gpu.mode === "cpu"
              ? "CPU"
              : gpu.mode === "auto"
                ? "Auto"
                : "Force GPU"}
          </Pill>
        }
      />
      <div className="grid md:grid-cols-3 gap-3 mb-4">
        {(["cpu", "auto", "force-gpu"] as ComputeModeDto[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setGpu({ mode: m })}
            className={cx(
              "rounded-lg border px-3 py-2 text-left text-sm transition-colors surface-hover",
              gpu.mode === m
                ? "border-accent-500/60 bg-accent-500/5"
                : "border-ink-800 bg-ink-900/40 text-ink-300",
            )}
          >
            <div className="font-medium text-ink-50">{labelMode(m)}</div>
            <div className="text-2xs text-ink-400 mt-0.5">{modeHint(m)}</div>
          </button>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <Toggle
          checked={gpu.use_hash_join}
          onChange={(b) => setGpu({ use_hash_join: b })}
          disabled={gpuOff}
          reason={gpuOff ? "Enable GPU mode to use hash-join" : undefined}
          label={
            isDeep
              ? "GPU hash-join for exact helpers"
              : "GPU hash-join (Options 1-2)"
          }
          description={
            isDeep
              ? "Available for exact matching helpers when the engine can use GPU pre-work."
              : "Accelerates exact matching with on-GPU hashing."
          }
        />
        <Toggle
          checked={gpu.use_direct_prefilter}
          onChange={(b) => setGpu({ use_direct_prefilter: b })}
          disabled={gpuOff}
          reason={gpuOff ? "Enable GPU mode to use prefilter" : undefined}
          label="GPU fuzzy direct prefilter"
          description="Cuts fuzzy candidate pairs before CPU scoring."
        />
        <Toggle
          checked={gpu.use_levenshtein_full_scoring}
          onChange={(b) => setGpu({ use_levenshtein_full_scoring: b })}
          disabled={gpuOff}
          reason={
            gpuOff ? "Enable GPU mode to use full GPU scoring" : undefined
          }
          label={
            isDeep
              ? "GPU fuzzy scoring for L10/L11"
              : "GPU Levenshtein scoring (Option 7)"
          }
          description={
            isDeep
              ? "Uses GPU scoring for fuzzy cascade levels where CUDA is available."
              : "Runs the entire weighted Levenshtein pass on the GPU."
          }
        />
        <Toggle
          checked={gpu.dynamic_tuning}
          onChange={(b) => setGpu({ dynamic_tuning: b })}
          disabled={gpuOff}
          reason={
            gpuOff ? "Enable GPU mode to allow dynamic tuning" : undefined
          }
          label="Dynamic GPU tuning"
          description="Auto-adjusts batch sizes from VRAM telemetry."
        />
      </div>
      <div className="mt-3">
        <Field
          label="L10/L11 fuzzy gate"
          help="Shadow verifies parity. Fast gate skips GPU-rejected fuzzy pairs before CPU final scoring."
        >
          <div className="grid md:grid-cols-3 gap-2">
            {(["off", "shadow", "gate-only"] as GpuFuzzyGateModeDto[]).map(
              (m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setGpu({ fuzzy_gate_mode: m })}
                  disabled={gpuOff}
                  className={cx(
                    "rounded-lg border px-3 py-2 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50 surface-hover",
                    gpu.fuzzy_gate_mode === m
                      ? "border-accent-500/60 bg-accent-500/5"
                      : "border-ink-800 bg-ink-900/40 text-ink-300",
                  )}
                >
                  <div className="font-medium text-ink-50">
                    {fuzzyGateLabel(m)}
                  </div>
                  <div className="text-2xs text-ink-400 mt-0.5">
                    {fuzzyGateHint(m)}
                  </div>
                </button>
              ),
            )}
          </div>
        </Field>
      </div>
      <div className="mt-3">
        <Field
          label="VRAM budget (MB)"
          help="Leave blank for auto. Reduce if you see VRAM OOM warnings."
        >
          <input
            type="number"
            className="input tabular"
            value={gpu.vram_budget_mb ?? ""}
            onChange={(e) =>
              setGpu({
                vram_budget_mb: e.target.value ? Number(e.target.value) : null,
              })
            }
            disabled={gpuOff}
            placeholder="auto"
            min={64}
            max={65536}
          />
        </Field>
      </div>
    </Card>
  );
}

export function StreamingCard() {
  const s = useConfigStore((st) => st.streaming);
  const set = useConfigStore((st) => st.setStreaming);
  const mode = useConfigStore((st) => st.mode);
  const algorithm = useConfigStore((st) => st.algorithm);
  const cascade = useConfigStore((st) => st.cascade);
  const source = useConnectionStore((st) => st.source);
  const target = useConnectionStore((st) => st.target);
  const srcSel = connectionSelection(source);
  const tgtSel = connectionSelection(target);
  const srcRows = rowCountForSide(srcSel);
  const tgtRows = rowCountForSide(tgtSel);
  const effective = resolveEffectiveRunMode(s.mode, srcRows, tgtRows);
  const streamingCfg = {
    source: srcSel,
    target: tgtSel,
    streaming: s,
    algorithm,
    cascade: {
      enabled: mode === "deep",
      levels: cascade.levels,
      fuzzy_threshold: cascade.fuzzy_threshold,
      exclusion_mode: cascade.exclusion_mode,
      has_barangay_code: false,
      has_city_code: false,
    },
  };
  const backendActive = streamingBackendActive(streamingCfg);
  const crossSessionActive = crossSessionStreamingBackendActive(streamingCfg);
  const anyStreamingActive = anyStreamingBackendActive(streamingCfg);
  const warn = scaleWarningLevel(Math.max(srcRows, tgtRows));
  const crossSessionNotice = needsCrossSessionDbStreamingNotice(streamingCfg);
  return (
    <Card>
      <SectionHeader
        title="Streaming"
        description="Requested mode is what you configure. Effective mode is what the backend will use for DB runs at scale."
      />
      <div className="mb-3 rounded-md border border-ink-800 bg-ink-900/40 px-3 py-2 text-xs text-ink-300 space-y-1">
        <p>
          <span className="text-ink-400">Requested mode:</span> {s.mode}
        </p>
        <p>
          <span className="text-ink-400">Effective mode:</span> {effective}
          {backendActive
            ? " (partitioned DB streaming active for algorithms 1–2)"
            : crossSessionActive
              ? " (cross-session dual-pool DB streaming active for algorithms 1–2)"
              : anyStreamingActive
                ? " (DB streaming active)"
                : effective === "streaming"
                  ? " (configured only — full streaming not active for this source/algorithm)"
                  : ""}
        </p>
        {warn !== "none" && (
          <p className="text-amber-200/90">
            {warn === "strong"
              ? "500k+ rows per side: confirm before starting and prefer export for large result sets."
              : "100k+ rows per side: streaming or import-to-DB is recommended."}
          </p>
        )}
        {crossSessionNotice && (
          <p className="text-amber-200/90">{crossSessionDbStreamingMessage()}</p>
        )}
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Mode">
          <div className="grid grid-cols-3 gap-1.5">
            {(["auto", "streaming", "in-memory"] as RunModeDto[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => set({ mode: m })}
                className={cx(
                  "rounded-md border px-2 h-8 text-xs transition-colors",
                  s.mode === m
                    ? "border-accent-500/60 bg-accent-500/5 text-ink-50"
                    : "border-ink-800 bg-ink-900/40 text-ink-300 hover:border-ink-700",
                )}
              >
                {m === "in-memory"
                  ? "In-memory"
                  : m === "auto"
                    ? "Auto"
                    : "Streaming"}
              </button>
            ))}
          </div>
        </Field>
        <Field
          label="Batch size"
          help="Recommended 5,000-50,000 for streaming runs."
        >
          <input
            type="number"
            className="input tabular"
            min={1000}
            max={200_000}
            step={1000}
            value={s.batch_size}
            onChange={(e) =>
              set({ batch_size: Math.max(1000, Number(e.target.value || 0)) })
            }
          />
        </Field>
        <Field
          label="Partition strategy"
          help="last_initial - birthyear5 - or empty for none."
        >
          <select
            className="select"
            value={s.partition_strategy ?? ""}
            onChange={(e) =>
              set({ partition_strategy: e.target.value || null })
            }
          >
            <option value="">none</option>
            <option value="last_initial">last_initial (A-Z)</option>
            <option value="birthyear5">birthyear5 (5-yr buckets)</option>
          </select>
        </Field>
        <Toggle
          checked={s.resume}
          onChange={(b) => set({ resume: b })}
          label="Resume from checkpoint"
          description="Picks up where the previous run stopped using the .nmckpt file."
        />
      </div>
    </Card>
  );
}

export function ExportCard() {
  const ex = useConfigStore((s) => s.export);
  const set = useConfigStore((s) => s.setExport);
  const algorithm = useConfigStore((s) => s.algorithm);
  const meta = algorithmMeta(algorithm);
  const pushToast = useToastStore((s) => s.push);

  async function pickDir() {
    try {
      const path = await open({ directory: true, multiple: false });
      if (typeof path === "string") set({ output_directory: path });
    } catch (err: unknown) {
      pushToast({
        tone: "error",
        title: "Folder picker failed",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    }
  }

  return (
    <Card>
      <SectionHeader
        title="Export"
        description="Output is written after the run completes. CSV is fastest; XLSX includes a Summary sheet."
      />
      <div className="grid md:grid-cols-2 gap-3">
        <Field label="Format">
          <div className="grid grid-cols-3 gap-1.5">
            {(["csv", "xlsx", "both"] as ExportFormatDto[]).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => set({ format: f })}
                className={cx(
                  "rounded-md border px-2 h-8 text-xs uppercase tracking-wide transition-colors",
                  ex.format === f
                    ? "border-accent-500/60 bg-accent-500/5 text-ink-50"
                    : "border-ink-800 bg-ink-900/40 text-ink-300 hover:border-ink-700",
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </Field>
        <Field
          label="File stem"
          help="Letters, digits, dot/underscore/hyphen only."
          required
        >
          <input
            className="input"
            value={ex.file_stem}
            onChange={(e) =>
              set({ file_stem: e.target.value.replace(/[^A-Za-z0-9._-]/g, "") })
            }
          />
        </Field>
        <Field label="Output folder" required className="md:col-span-2">
          <div className="flex gap-2">
            <input
              className="input flex-1"
              value={ex.output_directory}
              onChange={(e) => set({ output_directory: e.target.value })}
              placeholder="C:/exports/name_matcher"
            />
            <Button tone="secondary" onClick={pickDir}>
              Browse...
            </Button>
          </div>
        </Field>
        {meta.fuzzyTuneable && (
          <Field
            label="Min confidence (export only)"
            help="Rows below this score are still computed but excluded from the export."
            className="md:col-span-2"
          >
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={ex.min_confidence ?? 0}
              onChange={(e) =>
                set({
                  min_confidence: Number(e.target.value) || null,
                })
              }
              className="w-full accent-accent-500"
            />
            <div className="text-xs text-ink-300 tabular mt-1">
              &gt;= {ex.min_confidence ?? 0}%
            </div>
          </Field>
        )}
        <Field
          label="Review band"
          help="Rows in this confidence range show accept/reject controls in Results. High-confidence matches (e.g. 100%) are skipped."
          className="md:col-span-2"
        >
          <div className="grid sm:grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="text-xs text-ink-400">Minimum</span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={ex.review_band?.min_confidence ?? 70}
                onChange={(e) => {
                  const min = Number(e.target.value);
                  const max = ex.review_band?.max_confidence ?? 85;
                  set({
                    review_band: {
                      min_confidence: min,
                      max_confidence: Math.max(min, max),
                    },
                  });
                }}
                className="w-full accent-accent-500"
              />
              <div className="text-xs text-ink-300 tabular">
                {ex.review_band?.min_confidence ?? 70}%
              </div>
            </label>
            <label className="space-y-1">
              <span className="text-xs text-ink-400">Maximum</span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={ex.review_band?.max_confidence ?? 85}
                onChange={(e) => {
                  const max = Number(e.target.value);
                  const min = ex.review_band?.min_confidence ?? 70;
                  set({
                    review_band: {
                      min_confidence: Math.min(min, max),
                      max_confidence: max,
                    },
                  });
                }}
                className="w-full accent-accent-500"
              />
              <div className="text-xs text-ink-300 tabular">
                {ex.review_band?.max_confidence ?? 85}%
              </div>
            </label>
          </div>
        </Field>
      </div>
    </Card>
  );
}

export function PerformanceCard() {
  const opts = useConfigStore((s) => s.options);
  const set = useConfigStore((s) => s.setOptions);
  const source = useConnectionStore((st) => st.source);
  const target = useConnectionStore((st) => st.target);
  const maxRows = maxSideRows({
    source: connectionSelection(source),
    target: connectionSelection(target),
  });
  const showPersistHistoryWarn =
    opts.persist_result_history && maxRows >= SCALE_WARN_ROWS;
  return (
    <Card>
      <SectionHeader
        title="Performance"
        description="Auto-optimise picks rayon threads from your system profile. Override only if you know what you're doing."
      />
      <div className="space-y-4">
        <Toggle
          checked={opts.auto_optimize}
          onChange={(b) => set({ auto_optimize: b, ultra_performance: false })}
          label="Auto-optimise"
          description="Detects system, picks RAYON_NUM_THREADS automatically."
        />
        <Toggle
          checked={opts.ultra_performance}
          onChange={(b) =>
            set({
              ultra_performance: b,
              auto_optimize: b ? false : opts.auto_optimize,
              rayon_threads: b ? null : opts.rayon_threads,
              pool_size: b ? null : opts.pool_size,
            })
          }
          label="Ultra performance"
          description="Chooses the fastest safe CPU/GPU settings for this machine and workload. Respects CPU, Auto, and Force GPU choices."
        />
        <Toggle
          checked={opts.persist_result_history}
          onChange={(b) => set({ persist_result_history: b })}
          label="Persist result history"
          description="Stores person snapshots and results on disk for explanations, review decisions, restart recovery, and run diff. Leave off for faster matching start."
        />
        {showPersistHistoryWarn && (
          <p className="text-xs text-amber-200/90 -mt-2">
            {maxRows >= 500_000
              ? `${formatNumber(maxRows)}+ rows per side: persisting full result history uses substantial disk and slows startup. Prefer export for large result sets unless you need review/diff.`
              : `${formatNumber(maxRows)}+ rows per side: result history persistence adds disk I/O at this scale. Consider leaving it off unless you need review or run diff.`}
          </p>
        )}
        <Field label="Rayon threads (manual override)">
          <input
            type="number"
            className="input tabular"
            value={opts.rayon_threads ?? ""}
            onChange={(e) =>
              set({
                rayon_threads: e.target.value ? Number(e.target.value) : null,
              })
            }
            disabled={opts.ultra_performance}
            placeholder="auto"
            min={1}
            max={256}
          />
        </Field>
        <Field label="DB pool size (manual override)">
          <input
            type="number"
            className="input tabular"
            value={opts.pool_size ?? ""}
            onChange={(e) =>
              set({
                pool_size: e.target.value ? Number(e.target.value) : null,
              })
            }
            disabled={opts.ultra_performance}
            placeholder="auto"
            min={1}
            max={256}
          />
        </Field>
      </div>
    </Card>
  );
}

export function SummaryCard({ onAdvance }: { onAdvance: () => void }) {
  const mode = useConfigStore((s) => s.mode);
  const algorithm = useConfigStore((s) => s.algorithm);
  const meta = algorithmMeta(algorithm);
  const options = useConfigStore((s) => s.options);
  const cascade = useConfigStore((s) => s.cascade);
  const gpu = useConfigStore((s) => s.gpu);
  const stream = useConfigStore((s) => s.streaming);
  const ex = useConfigStore((s) => s.export);
  const source = useConnectionStore((s) => s.source);
  const target = useConnectionStore((s) => s.target);

  const ready =
    isSideReady(source) &&
    isSideReady(target) &&
    ex.output_directory.trim().length > 0;

  const [open, setOpen] = useState(true);

  return (
    <Card>
      <SectionHeader
        title="Configuration summary"
        description="Final preview before the run. Review carefully - start triggers the engine."
      />
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between text-xs uppercase tracking-wider text-ink-400 mb-2"
      >
        <span>{open ? "Hide details" : "Show details"}</span>
        <span aria-hidden>{open ? "v" : ">"}</span>
      </button>
      {open && (
        <dl className="text-sm space-y-2">
          <Row
            label="Mode"
            value={
              mode === "deep"
                ? `Deep Match - ${cascade.levels.length} level${cascade.levels.length === 1 ? "" : "s"}`
                : `Option ${meta.optionNumber} - ${meta.label}`
            }
          />
          {mode === "deep" && (
            <Row
              label="Exclusion"
              value={
                cascade.exclusion_mode === "exclusive"
                  ? "Exclusive - faster, first matched level wins"
                  : "Independent - slower, every level checks all rows"
              }
            />
          )}
          <Row
            label="Birthdate swap"
            value={options.allow_birthdate_swap ? "On" : "Off"}
          />
          <Row
            label="Source"
            value={
              source.mode === "file"
                ? source.file.preview
                  ? `${source.file.path} (preview only — ${formatNumber(source.file.preview.total_preview_rows)} rows shown)`
                  : "-"
                : source.session && source.selectedTable
                  ? `${source.session.database}.${source.selectedTable}${
                      source.rowCount != null
                        ? ` (${formatNumber(source.rowCount)} rows)`
                        : ""
                    }`
                  : "-"
            }
          />
          <Row
            label="Target"
            value={
              target.mode === "file"
                ? target.file.preview
                  ? `${target.file.path} (${formatNumber(target.file.preview.total_preview_rows)} preview rows)`
                  : "-"
                : target.session && target.selectedTable
                  ? `${target.session.database}.${target.selectedTable}${
                      target.rowCount != null
                        ? ` (${formatNumber(target.rowCount)} rows)`
                        : ""
                    }`
                  : "-"
            }
          />
          <Row
            label="GPU"
            value={gpu.mode === "cpu" ? "Off (CPU)" : labelMode(gpu.mode)}
          />
          <Row
            label="Streaming"
            value={`${stream.mode} - batch ${formatNumber(stream.batch_size)}`}
          />
          <Row
            label="Export"
            value={
              ex.output_directory
                ? `${ex.format.toUpperCase()} -> ${ex.output_directory}/${ex.file_stem}.*`
                : "-"
            }
          />
        </dl>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          tone="primary"
          disabled={!ready}
          onClick={onAdvance}
          title={ready ? "Continue to Run" : "Complete the form first"}
        >
          Continue to Run
        </Button>
        {!ready && (
          <Pill tone="warn">Pick output folder + file stem to continue</Pill>
        )}
      </div>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="text-2xs uppercase tracking-wider text-ink-500 w-24 shrink-0">
        {label}
      </dt>
      <dd className="text-ink-100 truncate">{value}</dd>
    </div>
  );
}

function isSideReady(
  side: ReturnType<typeof useConnectionStore.getState>["source"],
) {
  return side.mode === "file"
    ? !!side.file.preview && !!side.columnMapping
    : !!side.session && !!side.selectedTable;
}

function connectionSelection(
  side: ReturnType<typeof useConnectionStore.getState>["source"],
): TableSelectionDto {
  if (side.mode === "file") {
    return {
      source_kind: "file",
      session_id: "",
      table: "",
      column_mapping: side.columnMapping,
      row_count: side.file.preview?.total_preview_rows ?? null,
      file: {
        path: side.file.path,
        sheet_name: side.file.sheetName,
        encoding: side.file.encoding,
        delimiter: side.file.delimiter,
        date_format: side.file.dateFormat,
      },
    };
  }
  return {
    source_kind: "database",
    session_id: side.session!.session_id,
    table: side.selectedTable!,
    column_mapping: side.columnMapping,
    row_count: side.rowCount,
  };
}

function labelMode(m: ComputeModeDto) {
  return m === "cpu" ? "CPU" : m === "auto" ? "Auto" : "Force GPU";
}

function modeHint(m: ComputeModeDto) {
  switch (m) {
    case "cpu":
      return "Force CPU regardless of GPU availability.";
    case "auto":
      return "Use GPU when available, fall back to CPU on OOM.";
    case "force-gpu":
      return "Fail fast if CUDA is not detected.";
  }
}

function fuzzyGateLabel(m: GpuFuzzyGateModeDto) {
  switch (m) {
    case "off":
      return "Off";
    case "shadow":
      return "Shadow verify";
    case "gate-only":
      return "Fast gate";
  }
}

function fuzzyGateHint(m: GpuFuzzyGateModeDto) {
  switch (m) {
    case "off":
      return "CPU checks every fuzzy candidate.";
    case "shadow":
      return "GPU predicts skips; CPU still checks all.";
    case "gate-only":
      return "CPU checks only GPU-kept pairs.";
  }
}
