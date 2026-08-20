import { describe, expect, it } from "vitest";

import { formatTimestamp, getRelativeScoreTier, groupResultsByVideo } from "./results";
import type { SearchResult } from "./types";

function result(overrides: Partial<SearchResult>): SearchResult {
  return {
    rank: 0,
    score: 0.5,
    video_id: "K01_V001",
    frame_idx: 0,
    time: 0,
    thumb: "/thumb",
    frame: "/frame",
    video: "/video",
    source: "visual",
    scores: {},
    ...overrides,
  };
}

describe("groupResultsByVideo", () => {
  it("orders video rows by their best score and frames chronologically", () => {
    const groups = groupResultsByVideo([
      result({ video_id: "A", rank: 0, score: 0.8, time: 50 }),
      result({ video_id: "B", rank: 1, score: 0.9, time: 30 }),
      result({ video_id: "A", rank: 2, score: 0.7, time: 10 }),
      result({ video_id: "B", rank: 3, score: 0.6, time: 5 }),
    ]);

    expect(groups.map((group) => group.videoId)).toEqual(["B", "A"]);
    expect(groups[0].results.map((item) => item.time)).toEqual([5, 30]);
    expect(groups[1].results.map((item) => item.time)).toEqual([10, 50]);
  });

  it("uses original rank then video id as stable row tie breakers", () => {
    const groups = groupResultsByVideo([
      result({ video_id: "B", rank: 4, score: 0.8 }),
      result({ video_id: "A", rank: 2, score: 0.8 }),
      result({ video_id: "C", rank: 2, score: 0.8 }),
    ]);

    expect(groups.map((group) => group.videoId)).toEqual(["A", "C", "B"]);
  });
});

describe("getRelativeScoreTier", () => {
  it("maps the current result range into five levels", () => {
    expect(getRelativeScoreTier(0, 0, 100)).toBe(1);
    expect(getRelativeScoreTier(25, 0, 100)).toBe(2);
    expect(getRelativeScoreTier(50, 0, 100)).toBe(3);
    expect(getRelativeScoreTier(75, 0, 100)).toBe(4);
    expect(getRelativeScoreTier(100, 0, 100)).toBe(5);
  });

  it("uses the middle level when all results have the same score", () => {
    expect(getRelativeScoreTier(0.42, 0.42, 0.42)).toBe(3);
  });
});

describe("formatTimestamp", () => {
  it("formats multi-hour timestamps without wrapping minutes", () => {
    expect(formatTimestamp(3723)).toBe("1:02:03");
    expect(formatTimestamp(83.9)).toBe("1:23");
  });
});
