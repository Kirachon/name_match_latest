# Icons

Tauri v2 expects PNG / ICO / ICNS bundle icons in this directory. The current
icon set was generated from:

```text
src-tauri/icons/source/name-matcher-icon.png
```

The original scaffold placeholders were copied to `src-tauri/icons/_placeholder/`
before generation. To regenerate the icon set from a replacement 1024x1024
source image:

```powershell
Push-Location src-tauri
cargo tauri icon icons\source\name-matcher-icon.png
Pop-Location
```

`cargo tauri icon` writes the full set (`32x32.png`, `128x128.png`,
`128x128@2x.png`, `icon.icns`, `icon.ico`, plus square*.png for stores).
