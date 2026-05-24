import { useEffect, useState } from "react";
import {
  connectDb,
  disconnectDb,
  getRowCount,
  getTableColumns,
  listTables,
  testConnection,
} from "@/shared/tauri/commands";
import {
  type SessionSide,
  useConnectionStore,
} from "@/shared/stores/connectionStore";
import { useConfigStore } from "@/shared/stores/configStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
  StatusDot,
  Toggle,
} from "@/shared/components/primitives";
import { cx, formatNumber } from "@/shared/lib/format";
import {
  algorithmMeta,
  type DbCredentialsDto,
  type TableColumnsDto,
} from "@/shared/tauri/types";
import {
  clearPersistedPassword,
  loadPersistedConnection,
  type PersistedConnection,
  savePersistedConnection,
} from "./persistence";

type Form = DbCredentialsDto;

const blankForm: Form = {
  host: "127.0.0.1",
  port: 3306,
  username: "root",
  password: "",
  database: "",
};

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
              <li>Required columns are inferred automatically per algorithm.</li>
              <li>Row counts are estimates — they’re cached per session.</li>
              <li>
                Passwords are dropped from memory after the pool is built.
                Saved passwords (opt-in) are stored unencrypted in the app
                data directory.
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

function ConnectionCard({ side }: { side: SessionSide }) {
  const slice = useConnectionStore((s) => s[side]);
  const setSession = useConnectionStore((s) => s.setSession);
  const setLoading = useConnectionStore((s) => s.setLoading);
  const setTables = useConnectionStore((s) => s.setTables);
  const setSelectedTable = useConnectionStore((s) => s.setSelectedTable);
  const setColumns = useConnectionStore((s) => s.setColumns);
  const setRowCount = useConnectionStore((s) => s.setRowCount);
  const setError = useConnectionStore((s) => s.setError);
  const pushToast = useToastStore((s) => s.push);

  const [form, setForm] = useState<Form>(blankForm);
  const [latency, setLatency] = useState<number | null>(null);
  const [rememberPassword, setRememberPassword] = useState(false);
  const [savedTable, setSavedTable] = useState<string | null>(null);
  const [lastConnectedMs, setLastConnectedMs] = useState<number | null>(null);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from persisted store on first mount.
  useEffect(() => {
    let cancelled = false;
    loadPersistedConnection(side)
      .then((rec) => {
        if (cancelled || !rec) {
          setHydrated(true);
          return;
        }
        setForm({
          host: rec.host,
          port: rec.port,
          username: rec.username,
          password: rec.password ?? "",
          database: rec.database,
        });
        setRememberPassword(rec.password_saved);
        setSavedTable(rec.table ?? null);
        setLastConnectedMs(rec.last_connected_unix_ms);
        setHydrated(true);
      })
      .catch(() => setHydrated(true));
    return () => {
      cancelled = true;
    };
  }, [side]);

  // Apply default file_stem from selected source table.
  const setExport = useConfigStore((s) => s.setExport);
  useEffect(() => {
    if (side === "source" && slice.selectedTable) {
      setExport({ file_stem: `matches_${slice.selectedTable}` });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slice.selectedTable]);

  async function onConnect() {
    setLoading(side, true);
    setError(side, null);
    try {
      const sess = await connectDb(form);
      setSession(side, sess);
      setLatency(sess.latency_ms ?? null);
      const tables = await listTables(sess.session_id);
      setTables(side, tables);

      // Persist on success. Save table only if we already had one selected
      // (auto-selection happens after the user picks a table).
      const rec: PersistedConnection = {
        version: 1,
        host: form.host,
        port: form.port,
        username: form.username,
        database: form.database,
        table: savedTable,
        password_saved: rememberPassword,
        password: rememberPassword ? form.password : null,
        last_connected_unix_ms: Date.now(),
      };
      savePersistedConnection(side, rec).catch(() => {
        // Non-fatal — show toast but keep session.
        pushToast({
          tone: "warn",
          title: "Could not save connection",
          message: "Local store write failed. Re-enter credentials next launch.",
        });
      });

      // Auto-select previously saved table if it still exists.
      if (savedTable && tables.some((t) => t.name === savedTable)) {
        await onSelectTable(savedTable, sess.session_id);
      }

      pushToast({
        tone: "success",
        title: `${capitalize(side)} connected`,
        message: `${tables.length} tables visible · ${
          sess.latency_ms ?? "?"
        } ms`,
      });
    } catch (err: unknown) {
      const msg = errMsg(err);
      setError(side, msg);
      // Auth failure → clear stale password so the operator isn't stuck in a
      // retry loop with an outdated stored credential.
      const looksLikeAuth = /access denied|password|auth/i.test(msg);
      if (looksLikeAuth && rememberPassword) {
        await clearPersistedPassword(side).catch(() => {});
        setRememberPassword(false);
        setForm((f) => ({ ...f, password: "" }));
        pushToast({
          tone: "warn",
          title: "Saved credentials expired",
          message: "Cleared the saved password. Please re-enter it.",
        });
      } else {
        pushToast({ tone: "error", title: "Connection failed", message: msg });
      }
    } finally {
      setLoading(side, false);
    }
  }

  async function onDisconnect() {
    if (!slice.session) return;
    try {
      await disconnectDb(slice.session.session_id);
    } catch {
      /* swallow — server may already be torn down */
    }
    useConnectionStore.getState().resetSide(side);
    setLatency(null);
  }

  async function onSelectTable(table: string, sessionIdOverride?: string) {
    const sid = sessionIdOverride ?? slice.session?.session_id;
    if (!sid) return;
    setSelectedTable(side, table);
    setColumns(side, null);
    setRowCount(side, null);
    setSavedTable(table);
    try {
      const [cols, count] = await Promise.all([
        getTableColumns(sid, table),
        getRowCount(sid, table).catch(() => null),
      ]);
      setColumns(side, cols);
      setRowCount(side, count ?? null);
      // Update persisted record with the selected table.
      loadPersistedConnection(side).then((rec) => {
        if (rec) {
          rec.table = table;
          savePersistedConnection(side, rec).catch(() => {});
        }
      });
    } catch (err: unknown) {
      pushToast({
        tone: "warn",
        title: "Schema discovery failed",
        message: errMsg(err),
      });
    }
  }

  async function onPing() {
    if (!slice.session) return;
    try {
      const ms = await testConnection(slice.session.session_id);
      setLatency(ms);
      pushToast({
        tone: "info",
        title: `${capitalize(side)} ping ok`,
        message: `${ms} ms`,
        ttlMs: 1500,
      });
    } catch (err: unknown) {
      pushToast({ tone: "error", title: "Ping failed", message: errMsg(err) });
    }
  }

  return (
    <Card className="space-y-4">
      <div className="flex items-center justify-between">
        <SectionHeader
          title={side === "source" ? "Source Database" : "Target Database"}
          description={
            side === "source"
              ? "Authoritative table; matches are written from this side."
              : "Comparison table; can live in the same or a different database."
          }
        />
        <div className="flex items-center gap-2">
          {slice.session ? (
            <Pill tone="ok">
              <StatusDot tone="ok" /> Connected
            </Pill>
          ) : (
            <Pill tone="mute">
              <StatusDot tone="mute" /> Idle
            </Pill>
          )}
        </div>
      </div>

      {hydrated && lastConnectedMs && !slice.session && (
        <LastConnectedBadge unixMs={lastConnectedMs} />
      )}

      <div className="grid grid-cols-2 gap-3">
        <Field label="Host" htmlFor={`host-${side}`} className="col-span-1">
          <input
            id={`host-${side}`}
            className="input"
            value={form.host}
            onChange={(e) => setForm({ ...form, host: e.target.value })}
            disabled={!!slice.session}
          />
        </Field>
        <Field label="Port" htmlFor={`port-${side}`}>
          <input
            id={`port-${side}`}
            type="number"
            className="input tabular"
            value={form.port}
            onChange={(e) =>
              setForm({ ...form, port: Number(e.target.value || 0) })
            }
            disabled={!!slice.session}
          />
        </Field>
        <Field label="User" htmlFor={`user-${side}`}>
          <input
            id={`user-${side}`}
            className="input"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            disabled={!!slice.session}
          />
        </Field>
        <Field label="Password" htmlFor={`pwd-${side}`}>
          <input
            id={`pwd-${side}`}
            className="input"
            type="password"
            autoComplete="off"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            disabled={!!slice.session}
          />
        </Field>
        <Field
          label="Database"
          htmlFor={`db-${side}`}
          className="col-span-2"
        >
          <input
            id={`db-${side}`}
            className="input"
            value={form.database}
            onChange={(e) => setForm({ ...form, database: e.target.value })}
            disabled={!!slice.session}
            placeholder="schema_name"
          />
        </Field>
      </div>

      {!slice.session && (
        <div
          className="surface-soft p-3 flex items-start gap-3"
          aria-describedby={`remember-help-${side}`}
        >
          <span className="text-warn-400 mt-0.5" aria-hidden>
            <WarnIcon />
          </span>
          <div className="flex-1 min-w-0">
            <Toggle
              checked={rememberPassword}
              onChange={(b) => setRememberPassword(b)}
              label={
                <span>
                  Save password locally{" "}
                  <span className="text-warn-400 font-normal">
                    (stored unencrypted)
                  </span>
                </span>
              }
              description={
                <span id={`remember-help-${side}`}>
                  Saves the password in plaintext under the app data directory.
                  Use only on trusted machines. Off by default.
                </span>
              }
            />
          </div>
        </div>
      )}

      {slice.error && (
        <div role="alert" className="help-error">
          {slice.error}
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {!slice.session ? (
          <Button tone="primary" loading={slice.loading} onClick={onConnect}>
            Connect
          </Button>
        ) : (
          <>
            <Button tone="secondary" onClick={onPing}>
              Ping {latency != null ? `· ${latency} ms` : ""}
            </Button>
            <Button tone="ghost" onClick={onDisconnect}>
              Disconnect
            </Button>
          </>
        )}
      </div>

      {slice.session && (
        <div className="surface-soft p-3 space-y-3">
          <Field label="Table" htmlFor={`table-${side}`}>
            <select
              id={`table-${side}`}
              className="select"
              value={slice.selectedTable ?? ""}
              onChange={(e) => onSelectTable(e.target.value)}
            >
              <option value="">— select a table —</option>
              {slice.tables.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
          </Field>
          {slice.selectedTable && slice.columns && (
            <SchemaQuality columns={slice.columns} rowCount={slice.rowCount} />
          )}
        </div>
      )}
    </Card>
  );
}

function LastConnectedBadge({ unixMs }: { unixMs: number }) {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const t = window.setTimeout(() => setVisible(false), 5000);
    return () => window.clearTimeout(t);
  }, []);
  if (!visible) return null;
  const d = new Date(unixMs);
  const dateStr = d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  const timeStr = d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
  return (
    <div className="text-2xs text-ink-500 animate-fade-in">
      Restored from last session · {dateStr} {timeStr}
    </div>
  );
}

function SchemaQuality({
  columns,
  rowCount,
}: {
  columns: TableColumnsDto;
  rowCount: number | null;
}) {
  const algorithm = useConfigStore((s) => s.algorithm);
  const meta = algorithmMeta(algorithm);
  const required = meta.requiresColumns;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="section-title">Schema check</div>
        <div className="text-xs text-ink-300 tabular">
          rows: {rowCount != null ? formatNumber(rowCount) : "—"}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {(
          [
            ["id", columns.has_id],
            ["uuid", columns.has_uuid],
            ["first_name", columns.has_first_name],
            ["middle_name", columns.has_middle_name],
            ["last_name", columns.has_last_name],
            ["birthdate", columns.has_birthdate],
            ["hh_id", columns.has_hh_id],
          ] as const
        ).map(([name, ok]) => {
          const isRequired = required.includes(
            `has_${name}` as keyof TableColumnsDto,
          );
          return (
            <div
              key={name}
              className={cx(
                "flex items-center gap-2 text-xs px-2 py-1 rounded border",
                ok
                  ? "border-ok-500/30 bg-ok-500/5 text-ok-400"
                  : isRequired
                    ? "border-danger-500/30 bg-danger-500/5 text-danger-400"
                    : "border-ink-800 bg-ink-900/40 text-ink-500",
              )}
            >
              <StatusDot tone={ok ? "ok" : isRequired ? "danger" : "mute"} />
              <code className="font-mono">{name}</code>
              {isRequired && (
                <span className="ml-auto text-2xs uppercase opacity-80">
                  req
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function errMsg(err: unknown): string {
  if (typeof err === "object" && err && "message" in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
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
function WarnIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
