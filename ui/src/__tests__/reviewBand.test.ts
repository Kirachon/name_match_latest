import { describe, expect, it } from "vitest";
import { inReviewBand } from "@/shared/lib/reviewBand";

describe("inReviewBand", () => {
  const band = { min_confidence: 70, max_confidence: 85 };

  it("flags uncertain matches inside the default band", () => {
    expect(inReviewBand(75, band)).toBe(true);
    expect(inReviewBand(70, band)).toBe(true);
    expect(inReviewBand(85, band)).toBe(true);
  });

  it("skips high-confidence matches such as 100%", () => {
    expect(inReviewBand(100, band)).toBe(false);
    expect(inReviewBand(95, band)).toBe(false);
  });

  it("skips low-confidence matches below the band", () => {
    expect(inReviewBand(69.9, band)).toBe(false);
    expect(inReviewBand(0, band)).toBe(false);
  });
});
