# Icons

Tauri v2 expects PNG / ICO / ICNS bundle icons in this directory. The current
scaffold ships **no binary icon assets**; before producing a release build,
generate them with:

```powershell
cd src-tauri
cargo tauri icon path\to\source.png
```

`cargo tauri icon` writes the full set (`32x32.png`, `128x128.png`,
`128x128@2x.png`, `icon.icns`, `icon.ico`, plus square*.png for stores).

Until icons are generated, `cargo tauri build` will fail at the bundle step
but `cargo tauri dev` works fine for development.
