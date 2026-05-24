import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "@/app/ErrorBoundary";
import { useErrorStore } from "@/shared/stores/errorStore";

function ThrowingChild(): JSX.Element {
  throw new Error("render exploded");
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    useErrorStore.getState().clear();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  it("renders a recovery fallback and records diagnostics", async () => {
    render(
      <ErrorBoundary resetKey="job-1">
        <ThrowingChild />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("render exploded")).toBeInTheDocument();

    await waitFor(() => {
      expect(useErrorStore.getState().entries[0]).toMatchObject({
        message: "render exploded",
      });
    });
  });

  it("copies diagnostics to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(
      <ErrorBoundary resetKey="job-1">
        <ThrowingChild />
      </ErrorBoundary>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy diagnostics" }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(
        expect.stringContaining("Message: render exploded"),
      );
    });
  });

  it("recovers when the reset key changes", () => {
    const { rerender } = render(
      <ErrorBoundary resetKey="job-1">
        <ThrowingChild />
      </ErrorBoundary>,
    );

    rerender(
      <ErrorBoundary resetKey="job-2">
        <div>Recovered tab</div>
      </ErrorBoundary>,
    );

    expect(screen.getByText("Recovered tab")).toBeInTheDocument();
  });
});
