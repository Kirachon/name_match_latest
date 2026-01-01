# Self-hosted GitHub Actions runner (Windows + CUDA) for GPU builds

GitHub-hosted runners do **not** include the CUDA toolkit (`nvcc`), so building with `--features gpu` requires a self-hosted runner.

## 1) Create a self-hosted runner in GitHub

In your repo:
Settings → Actions → Runners → **New self-hosted runner** → Windows.

During setup, add labels:
- `windows`
- `cuda`

The workflow expects: `runs-on: [self-hosted, windows, cuda]`.

## 2) Install prerequisites on the runner machine

- Rust toolchain (stable)
- Visual Studio Build Tools (MSVC toolchain) / C++ build tools
- NVIDIA CUDA Toolkit (so `nvcc` is available)

Make sure `nvcc` is on `PATH` (workflow runs `nvcc --version`).

## 3) Validate

Push any commit to `main` and check the Actions tab:
- “Build GUI (GPU, self-hosted Windows)” should run on your machine.
