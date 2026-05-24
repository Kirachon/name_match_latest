import { Component, type ErrorInfo, type ReactNode, useState } from "react";
import { Button, Card, SectionHeader } from "@/shared/components/primitives";
import { useErrorStore } from "@/shared/stores/errorStore";

interface ErrorBoundaryProps {
  children: ReactNode;
  resetKey?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
  componentStack: string | null;
  resetKey?: string;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    error: null,
    componentStack: null,
    resetKey: this.props.resetKey,
  };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  static getDerivedStateFromProps(
    props: ErrorBoundaryProps,
    state: ErrorBoundaryState,
  ): Partial<ErrorBoundaryState> | null {
    if (props.resetKey !== state.resetKey) {
      return { error: null, componentStack: null, resetKey: props.resetKey };
    }
    return null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error);
    useErrorStore.getState().push({
      message: error.message,
      stack: error.stack,
      componentStack: info.componentStack ?? undefined,
    });
    this.setState({ componentStack: info.componentStack ?? null });
  }

  render() {
    if (this.state.error) {
      return (
        <ErrorFallback
          error={this.state.error}
          componentStack={this.state.componentStack}
        />
      );
    }
    return this.props.children;
  }
}

function ErrorFallback({
  error,
  componentStack,
}: {
  error: Error;
  componentStack: string | null;
}) {
  const [copyFailed, setCopyFailed] = useState(false);
  const diagnostics = [
    `Message: ${error.message}`,
    error.stack ? `Stack:\n${error.stack}` : null,
    componentStack ? `Component stack:\n${componentStack}` : null,
  ]
    .filter(Boolean)
    .join("\n\n");

  async function copyDiagnostics() {
    try {
      await navigator.clipboard.writeText(diagnostics);
      setCopyFailed(false);
    } catch {
      setCopyFailed(true);
    }
  }

  return (
    <Card className="mx-auto max-w-3xl">
      <SectionHeader
        title="Something went wrong"
        description="The workspace is still running. Reload the UI or copy diagnostics for support."
        action={
          <div className="flex flex-wrap gap-2">
            <Button tone="secondary" onClick={copyDiagnostics}>
              Copy diagnostics
            </Button>
            <Button tone="primary" onClick={() => window.location.reload()}>
              Reload UI
            </Button>
          </div>
        }
      />
      <div className="surface-soft p-3 text-sm text-ink-300">
        {error.message}
      </div>
      {copyFailed && (
        <textarea
          className="input mt-3 min-h-40 font-mono text-xs"
          readOnly
          value={diagnostics}
        />
      )}
    </Card>
  );
}
