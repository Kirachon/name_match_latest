# name_matcher

Rust project for name matching.

## Build

CPU-only:

```bash
cargo build --release
```

With GPU support:

```bash
cargo build --release --features gpu
```

## Performance tuning (no algorithm changes)

See `docs/performance.md`.

### Windows helper scripts

- Build GPU release: `powershell -File scripts\\windows\\Build-Release-Gpu.ps1`
- Run with safe defaults (remote MySQL): `powershell -File scripts\\windows\\Run-NameMatcher.ps1 -- <args>`
- Run with local MySQL preset: `powershell -File scripts\\windows\\Run-NameMatcher.ps1 -MySqlMode local -- <args>`

## CI builds (GitHub Actions)

- CPU builds run on GitHub-hosted runners by default.
- GPU builds require a self-hosted Windows runner with CUDA (`nvcc`) installed. See `docs/self_hosted_runner_windows_cuda.md`.
