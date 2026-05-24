import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// Vite is configured for Tauri v2:
// - Fixed dev port (5173) so tauri.conf.json's beforeDevCommand can resolve it.
// - clearScreen: false so Vite errors are not swallowed by Tauri's terminal output.
// - Watch ignores src-tauri so the front-end does not rebuild on Rust changes.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: "es2022",
    minify: "esbuild",
    sourcemap: true,
  },
});
