import { LazyStore } from "@tauri-apps/plugin-store";
import type { SessionSide } from "@/shared/stores/connectionStore";
import type { ColumnMappingDto } from "@/shared/tauri/types";

/**
 * Persisted connection record. The schema is versioned so that future
 * encrypted-storage migrations can detect and upgrade older records on load.
 *
 * Storage location: managed by `tauri-plugin-store` under the app's data
 * directory (`%APPDATA%\io.namematcher.desktop\connections.json` on Windows).
 *
 * Security note: when `password_saved` is true, the password is stored
 * **plaintext** in the JSON file. The Connect tab surfaces a prominent
 * warning before this option can be selected.
 */
export interface PersistedConnection {
  version: 1;
  host: string;
  port: number;
  username: string;
  database: string;
  table?: string | null;
  column_mapping?: ColumnMappingDto | null;
  /** True when the user explicitly opted in to local password storage. */
  password_saved: boolean;
  /** Plaintext password — only present when `password_saved === true`. */
  password?: string | null;
  /** Unix ms timestamp of the last successful connect for "Last connected" UX. */
  last_connected_unix_ms: number;
}

const STORE_FILE = "connections.json";
let _store: LazyStore | null = null;

function store(): LazyStore {
  if (!_store) _store = new LazyStore(STORE_FILE);
  return _store;
}

function key(side: SessionSide): string {
  return `connections.${side}`;
}

export async function loadPersistedConnection(
  side: SessionSide,
): Promise<PersistedConnection | null> {
  try {
    const v = await store().get<PersistedConnection>(key(side));
    if (!v) return null;
    if (v.version !== 1) return null; // future-proof against schema drift
    return v;
  } catch {
    // Corrupted store / first-run / IO error — fall back to empty state
    // rather than refusing to launch.
    return null;
  }
}

export async function savePersistedConnection(
  side: SessionSide,
  rec: PersistedConnection,
): Promise<void> {
  await store().set(key(side), rec);
  await store().save();
}

export async function clearPersistedPassword(side: SessionSide): Promise<void> {
  const cur = await loadPersistedConnection(side);
  if (!cur) return;
  cur.password = null;
  cur.password_saved = false;
  await savePersistedConnection(side, cur);
}
