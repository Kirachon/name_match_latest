import { useEffect, useState } from "react";
import { EventBridge } from "@/app/EventBridge";
import { StatusRail } from "@/app/StatusRail";
import { TabBar, type TabId } from "@/app/TabBar";
import { useGlobalShortcuts } from "@/app/shortcuts";
import { ConnectTab } from "@/features/connect/ConnectTab";
import { ConfigureTab } from "@/features/configure/ConfigureTab";
import { RunTab } from "@/features/run/RunTab";
import { ResultsTab } from "@/features/results/ResultsTab";
import { ToastLayer } from "@/shared/components/ToastLayer";
import { systemInfo } from "@/shared/tauri/commands";
import type { SystemInfoDto } from "@/shared/tauri/types";
import { useToastStore } from "@/shared/stores/toastStore";

export default function App() {
  const [tab, setTab] = useState<TabId>("connect");
  const [system, setSystem] = useState<SystemInfoDto | null>(null);
  const pushToast = useToastStore((s) => s.push);

  useEffect(() => {
    let cancelled = false;
    systemInfo()
      .then((s) => {
        if (!cancelled) setSystem(s);
      })
      .catch((e: unknown) => {
        pushToast({
          tone: "error",
          title: "System info unavailable",
          message:
            typeof e === "object" && e && "message" in e
              ? String((e as { message: unknown }).message)
              : String(e),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [pushToast]);

  useGlobalShortcuts({ activeTab: tab, setTab });

  return (
    <div className="h-full w-full flex flex-col bg-ink-950">
      <EventBridge />
      <StatusRail system={system} />
      <TabBar active={tab} onChange={setTab} />
      <main className="flex-1 overflow-auto">
        <div
          role="tabpanel"
          id={`tabpanel-${tab}`}
          aria-labelledby={`tab-${tab}`}
          className="mx-auto max-w-[1400px] px-6 py-6 animate-fade-in"
        >
          {tab === "connect" && (
            <ConnectTab onAdvance={() => setTab("configure")} />
          )}
          {tab === "configure" && (
            <ConfigureTab onAdvance={() => setTab("run")} />
          )}
          {tab === "run" && <RunTab onComplete={() => setTab("results")} />}
          {tab === "results" && <ResultsTab />}
        </div>
      </main>
      <ToastLayer />
    </div>
  );
}
