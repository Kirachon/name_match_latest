import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatNumber, formatPercent } from "./format";

describe("format helpers", () => {
  it("formats missing and invalid numbers as an em dash", () => {
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(undefined)).toBe("—");
    expect(formatNumber(Number.NaN)).toBe("—");
  });

  it("formats durations across seconds, minutes, and hours", () => {
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(75)).toBe("1m 15s");
    expect(formatDuration(3661)).toBe("1h 1m 1s");
  });

  it("formats bytes and percentages consistently", () => {
    expect(formatBytes(512)).toBe("512 MB");
    expect(formatBytes(1536)).toBe("1.5 GB");
    expect(formatPercent(12.345)).toBe("12.3%");
  });
});
