import type { ReviewBandDto } from "@/shared/tauri/types";

export const DEFAULT_REVIEW_BAND: ReviewBandDto = {
  min_confidence: 70,
  max_confidence: 85,
};

export function normalizeReviewBand(
  band: ReviewBandDto | null | undefined,
): ReviewBandDto {
  return {
    min_confidence: band?.min_confidence ?? DEFAULT_REVIEW_BAND.min_confidence,
    max_confidence: band?.max_confidence ?? DEFAULT_REVIEW_BAND.max_confidence,
  };
}

/** True when a row should show manual accept/reject controls. */
export function inReviewBand(
  confidence: number,
  band: ReviewBandDto | null | undefined,
): boolean {
  const { min_confidence, max_confidence } = normalizeReviewBand(band);
  return confidence >= min_confidence && confidence <= max_confidence;
}
