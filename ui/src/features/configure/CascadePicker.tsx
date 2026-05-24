import { useMemo } from "react";
import {
  CASCADE_LEVELS,
  CASCADE_PRESETS,
  autoSelectCascadePreset,
  type CascadeLevelMeta,
  type CascadePresetId,
} from "@/shared/tauri/types";
import { useConfigStore } from "@/shared/stores/configStore";
import { useConnectionStore } from "@/shared/stores/connectionStore";
import { Field, Pill, Toggle } from "@/shared/components/primitives";
import { cx } from "@/shared/lib/format";

/**
 * CascadePicker — Deep Match level selector.
 *
 * Layout:
 *   [Preset row]
 *   [Grouped checkbox grid]   ← only shown when preset === 'custom'
 *   [Threshold slider]
 *   [Exclusion mode]
 *   [Time-estimate hint]
 */
export function CascadePicker() {
  const cascade = useConfigStore((s) => s.cascade);
  const setCascade = useConfigStore((s) => s.setCascade);
  const sourceCols = useConnectionStore((s) => s.source.columns);
  const targetCols = useConnectionStore((s) => s.target.columns);

  const hasBarangay = useMemo(() => {
    const inRaw = (cols: typeof sourceCols) =>
      !!cols?.raw_columns?.includes("barangay_code");
    return inRaw(sourceCols) && inRaw(targetCols);
  }, [sourceCols, targetCols]);

  const hasCity = useMemo(() => {
    const inRaw = (cols: typeof sourceCols) =>
      !!cols?.raw_columns?.includes("city_code");
    return inRaw(sourceCols) && inRaw(targetCols);
  }, [sourceCols, targetCols]);

  function applyPreset(preset: CascadePresetId) {
    const def = CASCADE_PRESETS[preset];
    if (preset === "custom") {
      setCascade({ preset });
      return;
    }
    // Filter preset levels to those that are runnable on these tables.
    const allowed = def.levels.filter((id) =>
      isLevelRunnable(id, hasBarangay, hasCity),
    );
    setCascade({ preset, levels: allowed });
  }

  function toggleLevel(id: number) {
    const has = cascade.levels.includes(id);
    const next = has
      ? cascade.levels.filter((l) => l !== id).sort((a, b) => a - b)
      : [...cascade.levels, id].sort((a, b) => a - b);
    setCascade({ preset: "custom", levels: next });
  }

  // Auto-select preset on first render if user hasn't picked yet (preset still
  // matches the default 'standard' but the column data has changed).
  // We do NOT clobber an explicit user choice.
  const recommendedPreset = autoSelectCascadePreset({ hasBarangay, hasCity });

  const grouped = useMemo(() => {
    const by: Record<CascadeLevelMeta["group"], CascadeLevelMeta[]> = {
      name: [],
      barangay: [],
      city: [],
      fuzzy: [],
    };
    for (const l of CASCADE_LEVELS) by[l.group].push(l);
    return by;
  }, []);

  const enabledLevelCount = cascade.levels.length;
  const estimateLabel = describeTimeEstimate(enabledLevelCount);

  return (
    <div className="space-y-4">
      <div>
        <Field
          label="Preset"
          help={
            recommendedPreset !== cascade.preset && cascade.preset !== "custom"
              ? `Recommended for these tables: ${CASCADE_PRESETS[recommendedPreset].label}`
              : CASCADE_PRESETS[cascade.preset].description
          }
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(Object.keys(CASCADE_PRESETS) as CascadePresetId[]).map((p) => {
              const def = CASCADE_PRESETS[p];
              const isActive = cascade.preset === p;
              const isRecommended = p === recommendedPreset && p !== "custom";
              return (
                <button
                  key={p}
                  type="button"
                  onClick={() => applyPreset(p)}
                  className={cx(
                    "rounded-lg border p-3 text-left transition-colors",
                    isActive
                      ? "border-accent-500/60 bg-accent-500/5 shadow-glow"
                      : "border-ink-800 bg-ink-900/40 hover:border-ink-700",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-ink-50">
                      {def.label}
                    </span>
                    {isRecommended && !isActive && (
                      <Pill tone="info" className="text-2xs">
                        Suggested
                      </Pill>
                    )}
                  </div>
                  <div className="text-2xs text-ink-400 mt-1 leading-snug">
                    {p === "custom"
                      ? `${enabledLevelCount} level${enabledLevelCount === 1 ? "" : "s"} selected`
                      : `${def.levels.length} levels`}
                  </div>
                </button>
              );
            })}
          </div>
        </Field>
      </div>

      {cascade.preset === "custom" && (
        <div className="surface-soft p-3 space-y-3">
          <LevelGroup
            title="Name-based (no geo)"
            note="Always available."
            levels={grouped.name}
            selected={cascade.levels}
            onToggle={toggleLevel}
            disabled={false}
          />
          <LevelGroup
            title="Barangay-grouped"
            note={
              hasBarangay
                ? "Detected barangay_code on both tables."
                : "Requires barangay_code on both tables — disabled."
            }
            levels={grouped.barangay}
            selected={cascade.levels}
            onToggle={toggleLevel}
            disabled={!hasBarangay}
          />
          <LevelGroup
            title="City-grouped"
            note={
              hasCity
                ? "Detected city_code on both tables."
                : "Requires city_code on both tables — disabled."
            }
            levels={grouped.city}
            selected={cascade.levels}
            onToggle={toggleLevel}
            disabled={!hasCity}
          />
          <LevelGroup
            title="Fuzzy (Levenshtein + Jaro-Winkler)"
            note="GPU-accelerated when GPU mode is enabled."
            levels={grouped.fuzzy}
            selected={cascade.levels}
            onToggle={toggleLevel}
            disabled={false}
          />
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-3">
        <Field
          label={`Fuzzy threshold (${(cascade.fuzzy_threshold * 100).toFixed(0)}%)`}
          help="Applied to L10 / L11 fuzzy levels."
        >
          <input
            type="range"
            min={0.5}
            max={1.0}
            step={0.01}
            value={cascade.fuzzy_threshold}
            onChange={(e) =>
              setCascade({ fuzzy_threshold: Number(e.target.value) })
            }
            className="w-full accent-accent-500"
          />
        </Field>
        <Field
          label="Exclusion mode"
          help={
            cascade.exclusion_mode === "exclusive"
              ? "Recommended. Once a pair is matched in an earlier level, it is skipped in later levels."
              : "Every level checks the full dataset. This is slower and the same pair may appear more than once."
          }
        >
          <div className="grid grid-cols-2 gap-1.5">
            {(["exclusive", "independent"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setCascade({ exclusion_mode: m })}
                className={cx(
                  "rounded-md border px-2 h-8 text-xs transition-colors",
                  cascade.exclusion_mode === m
                    ? "border-accent-500/60 bg-accent-500/5 text-ink-50"
                    : "border-ink-800 bg-ink-900/40 text-ink-300 hover:border-ink-700",
                )}
              >
                {m === "exclusive" ? "Exclusive" : "Independent"}
              </button>
            ))}
          </div>
          {cascade.exclusion_mode === "independent" && (
            <div className="mt-2 rounded-md border border-warn-500/40 bg-warn-500/10 px-3 py-2 text-xs text-warn-200">
              Independent mode can take much longer on large tables because
              every selected level checks the full dataset again.
            </div>
          )}
        </Field>
      </div>

      <div className="text-2xs text-ink-400">
        {estimateLabel}{" "}
        <span className="text-ink-500">
          · L12 (Household) is excluded from cascade — use Quick Match Options 5
          / 6 for household runs.
        </span>
      </div>
    </div>
  );
}

function LevelGroup({
  title,
  note,
  levels,
  selected,
  onToggle,
  disabled,
}: {
  title: string;
  note: string;
  levels: CascadeLevelMeta[];
  selected: number[];
  onToggle: (id: number) => void;
  disabled: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <div className="section-title">{title}</div>
        <div className="text-2xs text-ink-500">{note}</div>
      </div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-1.5">
        {levels.map((l) => {
          const isOn = selected.includes(l.id);
          return (
            <Toggle
              key={l.id}
              checked={isOn && !disabled}
              onChange={() => !disabled && onToggle(l.id)}
              disabled={disabled}
              reason={
                disabled
                  ? l.requiresColumn === "barangay_code"
                    ? "Requires barangay_code"
                    : l.requiresColumn === "city_code"
                      ? "Requires city_code"
                      : undefined
                  : undefined
              }
              label={l.label}
              description={l.description}
            />
          );
        })}
      </div>
    </div>
  );
}

function isLevelRunnable(
  id: number,
  hasBarangay: boolean,
  hasCity: boolean,
): boolean {
  const meta = CASCADE_LEVELS.find((l) => l.id === id);
  if (!meta) return false;
  if (meta.requiresColumn === "barangay_code") return hasBarangay;
  if (meta.requiresColumn === "city_code") return hasCity;
  return true;
}

function describeTimeEstimate(n: number): string {
  if (n === 0) return "Pick at least one level to run.";
  if (n <= 3) return `~${n}× a single Quick Match run.`;
  if (n <= 6) return `~${n}× a single Quick Match run (mid-range cascade).`;
  return `~${n}× a single Quick Match run — full cascade.`;
}
