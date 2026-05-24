import { useToastStore } from "@/shared/stores/toastStore";
import { cx } from "@/shared/lib/format";

export function ToastLayer() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);

  return (
    <div
      role="region"
      aria-label="Notifications"
      aria-live="polite"
      className="fixed top-3 right-3 z-50 flex flex-col gap-2 max-w-[440px]"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          className={cx(
            "toast",
            t.tone === "info" && "toast-info",
            t.tone === "success" && "toast-success",
            t.tone === "warn" && "toast-warn",
            t.tone === "error" && "toast-error",
          )}
        >
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-ink-50">{t.title}</div>
            {t.message && (
              <div className="text-xs text-ink-300 mt-0.5 break-words">
                {t.message}
              </div>
            )}
          </div>
          <button
            onClick={() => dismiss(t.id)}
            className="text-ink-500 hover:text-ink-100 -mr-1 -mt-0.5 px-1"
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
