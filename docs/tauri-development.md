# Tauri Development Guide

This guide covers everything you need to develop, build, and ship the Tauri v2
desktop shell of `name_matcher`. It complements `docs/installation.md` (which
is end-user oriented) and the architecture in `docs/tauri-migration-plan.md`.

## Prerequisites

| Tool                 | Required version | Used for                                  |
|----------------------|------------------|-------------------------------------------|
| Rust toolchain       | 1.89.0 (pinned)  | `cargo check`, engine compilation         |
| MSVC Build Tools     | 2022             | Linking Windows binaries                  |
| Node.js              | 20.x or newer    | Vite, TypeScript, ESLint, Vitest          |
| pnpm                 | 10.x             | JS dependency manager (do not use npm)    |
| `cargo-tauri` v2     | `^2`             | `cargo tauri dev` / `build`               |
| WebView2 runtime     | Bundled / 110+   | The Tauri shell renders the UI in WebView2|
| CUDA Toolkit (opt)   | 12.x             | Only required for the `gpu` feature       |

Pin/discover versions with:

```powershell
rustc --version
cargo --version
node --version
pnpm --version
cargo tauri --version
```

Install the Tauri CLI once:

```powershell
cargo install tauri-cli --version "^2" --locked
```

## Repository Layout

```
name_match_latest/
├── Cargo.toml            # name_matcher engine + lib + 4 bins (gui gated by `gui`)
├── src/                  # Engine, db, matching, exports, run_service (NEW)
│   └── run_service/      # T3 — shared run service (used by Tauri + CLI)
├── src-tauri/            # T2 — Tauri v2 shell
│   ├── Cargo.toml        # Path-dep on the root engine
│   ├── tauri.conf.json   # v2 schema, security CSP, bundle config
│   ├── capabilities/     # Explicit allowlists for every command
│   └── src/              # AppState, AppError, command modules
└── ui/                   # T4 — Vite + React + TypeScript + Tailwind
    └── src/
        ├── app/          # EventBridge, StatusRail, TabBar, shortcuts
        ├── features/     # connect / configure / run / results
        └── shared/       # tauri commands+events+types, stores, components
```

## Day-to-Day Development

### Run the Tauri shell with hot-reload

```powershell
cd D:\GitProjects\name_match_latest
cargo tauri dev
```

This:

1. Starts Vite on `http://localhost:5173` (configured in `vite.config.ts`).
2. Compiles `src-tauri` with `tauri-build` and launches the desktop window.
3. Forwards Rust `log::*` calls to the host terminal and `LogEntryDto`
   payloads to the front-end via the `log-entry` event.

The TypeScript front-end watches `ui/src/**` and HMRs without restarting the
shell. The Rust side restarts on changes to `src-tauri/**` or `src/**`.

### Run only the front-end

For UI iteration without the desktop runtime:

```powershell
cd ui
pnpm dev      # starts Vite at http://localhost:5173
pnpm lint     # tsc --noEmit
pnpm test     # Vitest
pnpm build    # tsc + vite build (writes ui/dist)
```

### Build CPU-only release

```powershell
powershell -File scripts\windows\Build-Tauri-Cpu.ps1
```

Internally this:

1. Runs `pnpm install` (if needed) and `pnpm build` to populate `ui/dist`.
2. Calls `cargo tauri build --no-bundle` from `src-tauri/` so the EXE is
   produced even before icons / signing are finalised.

The artefact lands at:

```
src-tauri\target\release\name-matcher-tauri.exe
```

To produce a packaged MSI/NSIS installer instead, drop `--no-bundle` from
`Build-Tauri-Cpu.ps1` after generating real icons via `cargo tauri icon`.

### Build GPU release

```powershell
powershell -File scripts\windows\Build-Tauri-Gpu.ps1
```

Requires:

- `nvcc` on PATH (CUDA Toolkit 12.x).
- The runtime DLLs at `dist\gpu-dlls\` (these match the existing
  `Build-Release-Gpu.ps1` lane).

The script copies `nvrtc64_120_0.dll` and `nvrtc-builtins64_128.dll` next
to the executable so the packaged app does **not** require an installed
CUDA Toolkit on the target machine — only an NVIDIA driver new enough for
the chosen toolkit.

## Working with the Backend Contract

DTOs live in `name_matcher::run_service::dto` and are mirrored 1:1 to
`ui/src/shared/tauri/types.ts`. **Whenever you change a DTO**, update both
files and the matching Zod schema in `ui/src/shared/tauri/zod-schemas.ts`.
CI's `cargo check` / `pnpm build` will reject broken signatures, but the
human cross-check is your responsibility.

To add a new Tauri command:

1. Add a function in `src-tauri/src/commands/<area>.rs` with the
   `#[tauri::command]` attribute. Use `AppResult<T>` as the return type.
2. Re-export it from `src-tauri/src/commands/mod.rs`.
3. Register it in the `tauri::generate_handler![ ... ]` macro in
   `src-tauri/src/main.rs`.
4. Allowlist it in `src-tauri/capabilities/default.json` if it uses any
   plugin permission outside `core:default`.
5. Add a typed wrapper in `ui/src/shared/tauri/commands.ts` and (if its
   payload schema needs validation) a Zod schema next to it.

## Job Lifecycle

The plan describes 11 explicit job states. They are enforced by:

- `name_matcher::run_service::JobHandle::state()` (Rust)
- `JobStateDto` enum (TypeScript) with `JOB_STATE_TERMINAL` /
  `JOB_STATE_ACTIVE` constants.

The state transitions are emitted through the `job-state` Tauri event.
The front-end's `EventBridge` mounts exactly one listener per event type
at the app root; feature components subscribe via Zustand selectors.

## Troubleshooting

### `cargo check` fails with `Access is denied. (os error 5)` on `icu_locale_core`

The in-tree `.cargo-home/` cache is corrupt. Use a fresh CARGO_HOME:

```powershell
$env:CARGO_HOME = "C:\cargo_nm_temp"
cargo check --locked
```

### `cargo tauri build` complains about missing icons

Run `cargo tauri icon path\to\source.png` from `src-tauri/` to generate the
full icon set. The migration ships placeholder icons in `src-tauri/icons/`
that satisfy the `cargo check` build script but should be replaced before
publishing a real release.

### Front-end shows "Connect a database first" but you already connected

The connection is per-side. Both source **and** target must be connected
before the Configure tab unlocks. Check the Status Rail at the top — both
status dots should be green.

### Window opens off-screen or at an awkward size

The Tauri shell persists window geometry through `tauri-plugin-window-state`.
To reset the saved position and size, close the app and delete:

```powershell
Remove-Item "$env:APPDATA\io.namematcher.desktop\window-state.json"
```

### Cancellation is slow

Cancellation is cooperative. The engine checks the cancellation flag at DB
page boundaries, batch boundaries, GPU flush boundaries, and before each
result-store write. If a single batch is huge, the cancel can take a moment
to land. Reduce `streaming.batch_size` if this matters for your dataset.

### Logs stop after ~5,000 entries

The front-end uses a bounded ring buffer (`MAX_LOG_LINES = 5000` in
`shared/stores/jobStore.ts`) so the DOM stays light during long runs. The
backend keeps the full set; export the run summary if you need the
complete trace.

## Coexistence with the Legacy egui GUI

During the migration the legacy `gui` binary remains available behind the
`gui` feature flag:

```powershell
powershell -File scripts\windows\Build-Release-Gui.ps1            # CPU
powershell -File scripts\windows\Build-Release-Gui.ps1 -Gpu       # GPU
```

The CLI (`name_matcher` binary) is unchanged. All three front-ends share
the same engine through `name_matcher::run_service`.
