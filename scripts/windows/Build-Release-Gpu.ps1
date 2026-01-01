param(
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
  cargo clean
}

cargo build --release --features gpu

