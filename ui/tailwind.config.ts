/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Operational palette: neutral slate background, cyan accent for
        // primary actions, emerald for success, amber for caution, rose for
        // failure. Aim for AAA contrast on text, AA+ on borders.
        ink: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          850: "#172033",
          900: "#0f172a",
          950: "#0b0f17",
        },
        accent: {
          400: "#22d3ee",
          500: "#06b6d4",
          600: "#0891b2",
          700: "#0e7490",
        },
        ok: {
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
        },
        warn: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
        danger: {
          400: "#fb7185",
          500: "#f43f5e",
          600: "#e11d48",
        },
      },
      fontFamily: {
        sans: [
          "InterVariable",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.25), 0 0 20px -2px rgba(34,211,238,0.18)",
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -8px rgba(0,0,0,0.4)",
        "card-hover":
          "0 1px 0 rgba(255,255,255,0.06) inset, 0 2px 4px rgba(0,0,0,0.4), 0 12px 32px -8px rgba(0,0,0,0.5)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: 0, transform: "translateY(2px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.55 },
        },
        progress: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 180ms ease-out",
        "pulse-soft": "pulse-soft 1.6s ease-in-out infinite",
        progress: "progress 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
