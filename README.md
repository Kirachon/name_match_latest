# name_matcher

Rust project for high-throughput name matching. Three front-ends share the
same engine:

- **`name_matcher` CLI** — scriptable, headless, default build.
- **Legacy `gui` (egui)** — preserved during the Tauri migration; gated
  behind the `gui` feature.
- **Tauri v2 desktop shell** — modern React + TypeScript front-end under
  `ui/`, backed by typed Tauri commands wired through
  `name_matcher::run_service`.

## Build

CPU-only (CLI):

```bash
cargo build --release
```

With GPU support:

```bash
cargo build --release --features gpu
```

Legacy egui GUI:

```powershell
powershell -File scripts\windows\Build-Release-Gui.ps1            # CPU
powershell -File scripts\windows\Build-Release-Gui.ps1 -Gpu       # CUDA
```

## Tauri v2 Desktop Shell

```powershell
# Dev (hot-reload front-end + Rust):
cargo install tauri-cli --version "^2" --locked
cd ui
pnpm install
cd ..
cargo tauri dev
```

Release builds:

```powershell
powershell -File scripts\windows\Build-Tauri-Cpu.ps1     # CPU
powershell -File scripts\windows\Build-Tauri-Gpu.ps1     # CUDA (requires nvcc)
```

See [`docs/tauri-development.md`](docs/tauri-development.md) for the full
workflow, the contract between Rust and TypeScript DTOs, packaging notes,
and troubleshooting.

## Performance tuning (no algorithm changes)

See [`docs/performance.md`](docs/performance.md).

### Windows helper scripts

| Script                                  | Purpose                                  |
|-----------------------------------------|------------------------------------------|
| `scripts\windows\Build-Release-Gpu.ps1` | Build the CLI with GPU support           |
| `scripts\windows\Build-Release-Gui.ps1` | Build the legacy egui GUI (CPU/GPU)      |
| `scripts\windows\Build-Tauri-Cpu.ps1`   | Build the Tauri shell (CPU)              |
| `scripts\windows\Build-Tauri-Gpu.ps1`   | Build the Tauri shell with GPU runtime   |
| `scripts\windows\Run-NameMatcher.ps1`   | Run with safe defaults (remote MySQL)    |

## CI builds (GitHub Actions)

| Workflow                          | Runs on                        | Builds                                  |
|-----------------------------------|--------------------------------|-----------------------------------------|
| `.github/workflows/ci.yml`        | `windows-latest`, `ubuntu-latest`, `[self-hosted, windows, cuda]` | CLI + legacy egui GUI |
| `.github/workflows/tauri-build.yml` | `windows-latest`, `[self-hosted, windows, cuda]` | Tauri shell (CPU + GPU)             |
| `.github/workflows/release.yml`   | `windows-latest`, `ubuntu-latest` | Release artefacts on GitHub Releases |

GPU builds require a self-hosted Windows runner with CUDA (`nvcc`)
installed. See [`docs/self_hosted_runner_windows_cuda.md`](docs/self_hosted_runner_windows_cuda.md).

## Release artefacts

Publishing a GitHub Release triggers `.github/workflows/release.yml`:

- `gui-windows-cpu-<tag>.zip` — legacy egui GUI (Windows CPU)
- `gui-linux-cpu-<tag>.tar.gz` — legacy egui GUI (Linux CPU)

Tauri release artefact naming (Phase 2 of the migration):

- `name-matcher-tauri-windows-cpu-<tag>.msi`
- `name-matcher-tauri-windows-gpu-<tag>.msi`

Until Tauri reaches stable release, both lanes are published side-by-side
so operators can fall back if the Tauri build fails inspection.

## Documentation

- [`docs/installation.md`](docs/installation.md) — install + DB setup
- [`docs/usage_guide.md`](docs/usage_guide.md) — operator workflow
- [`docs/performance.md`](docs/performance.md) — tuning knobs
- [`docs/matching_algorithms.md`](docs/matching_algorithms.md) — algorithm reference
- [`docs/tauri-development.md`](docs/tauri-development.md) — Tauri shell dev guide
- [`docs/tauri-migration-plan.md`](docs/tauri-migration-plan.md) — full migration plan + execution log
