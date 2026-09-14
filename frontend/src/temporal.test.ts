import { describe, expect, it } from "vitest";

import { temporalSelection } from "./temporal";
import type { TemporalEventHit, TemporalSequence } from "./types";

function eventHit(overrides: Partial<TemporalEventHit>): TemporalEventHit {
  return {
    event_index: 0,
    rank: 5,
    score: 0.2112,
    video_id: "L30_V031",
    frame_idx: 2080,
    time: 83.2,
    thumb: "/thumb",
    frame: "/frame",
    video: "/video",
    source: "visual",
    scores: { visual_score: 0.2112, visual_rank: 6 },
    ...overrides,
  };
}

function sequence(overrides: Partial<TemporalSequence>): TemporalSequence {
  return {
    rank: 0,
    video_id: "L30_V031",
    score: 0.2576,
    visual_score: 0.2576,
    source: "visual",
    start_time: 83.2,
    end_time: 90.6,
    events: [],
    ...overrides,
  };
}

describe("temporalSelection", () => {
  it("uses the sequence's own rank, not the individual event's per-query candidate rank", () => {
    const event = eventHit({});
    const best = sequence({ rank: 0 });

    const selected = temporalSelection(event, best);

    expect(selected.rank).toBe(0);
    expect(selected.score).toBe(0.2576);
    expect(selected.scores.visual_rank).toBe(6);
  });
});
