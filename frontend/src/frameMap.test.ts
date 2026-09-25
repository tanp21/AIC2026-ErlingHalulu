import { describe, expect, it } from "vitest";

import { decodeFrameMap, frameTime, frameTimestampMs, resolveFrameId } from "./frameMap";

describe("exact frame map", () => {
  const mapping = decodeFrameMap({
    videoId: "V",
    timeBaseNumerator: 1,
    timeBaseDenominator: 90000,
    firstPts: 9000,
    ptsDeltas: [3003, 2997],
    frameCount: 3,
  });

  it("resolves mediaTime to an exact source frame for VFR PTS", () => {
    expect(resolveFrameId(mapping, 3003 / 90000)).toBe(1);
    expect(resolveFrameId(mapping, 6000 / 90000)).toBe(2);
  });

  it("does not guess when mediaTime is not a frame PTS", () => {
    expect(resolveFrameId(mapping, 3004 / 90000)).toBeNull();
  });

  it("converts a source frame id back to its exact presentation time", () => {
    expect(frameTime(mapping, 0)).toBe(0);
    expect(frameTime(mapping, 2)).toBe(6000 / 90000);
    expect(frameTimestampMs(mapping, 2)).toBe(67);
  });

  it("rejects malformed delta payloads", () => {
    expect(() => decodeFrameMap({
      videoId: "V", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 0, ptsDeltas: [10, 0], frameCount: 3,
    })).toThrow("strictly increasing");
  });

  it("rejects unsafe frame-map metadata", () => {
    expect(() => decodeFrameMap({
      videoId: "V", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: Number.NaN, ptsDeltas: [], frameCount: 1,
    })).toThrow("safe integers");
  });
});
