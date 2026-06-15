import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConnectTab } from "@/features/connect/ConnectTab";

vi.mock("@/features/connect/persistence", () => ({
  clearPersistedPassword: vi.fn(),
  loadPersistedConnection: vi.fn().mockResolvedValue(null),
  savePersistedConnection: vi.fn(),
}));

vi.mock("@/shared/tauri/commands", () => ({
  connectDb: vi.fn(),
  disconnectDb: vi.fn(),
  getRowCount: vi.fn(),
  getTableColumns: vi.fn(),
  listTables: vi.fn(),
  testConnection: vi.fn(),
  validateDbCredentials: vi.fn(),
}));

describe("ConnectTab", () => {
  it("renders without triggering a recursive store update", () => {
    render(<ConnectTab onAdvance={vi.fn()} />);

    expect(
      screen.getByRole("button", { name: "Continue to Configure" }),
    ).toBeDisabled();
    expect(screen.getByText("Source Database")).toBeInTheDocument();
    expect(screen.getByText("Target Database")).toBeInTheDocument();
  });
});
