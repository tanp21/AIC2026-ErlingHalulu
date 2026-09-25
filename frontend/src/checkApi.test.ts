import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { cachedCheck, forgetChecks, lookupCheck, queryKey, runCheck } from "./checkApi";
import type { CheckRequest, CheckResult } from "./types";

const REQUEST: CheckRequest = {
  visual_text: "Hai người cầm điện thoại", kind: "kis", video_id: "L22_V011", frame_idx: 28321,
};

function result(overrides: Partial<CheckResult> = {}): CheckResult {
  return {
    key: "abc", video_id: "L22_V011", frame_idx: 28321, time: 944, window_seconds: 300,
    shot_index: 41, covers: { start_frame: 28300, end_frame: 28400, start: 943, end: 947 },
    verdict: { score: 8, reason: "đúng" }, details: [], checks: [],
    support: { share: 0, supported: 0, partial: 0, contradicted: 0, total: 0 },
    answer: null, frames_seen: 23, model: "qwen/qwen3.7-flash", by: "owner@example.com",
    checked_at: 0, cached: false, latency_ms: 10, ...overrides,
  };
}

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return { ok, status, json: async () => data, headers: new Headers() } as Response;
}

beforeEach(() => forgetChecks());

afterEach(() => {
  forgetChecks();
  vi.restoreAllMocks();
});

describe("queryKey", () => {
  it("ignores case and spacing, so the same query is the same key", () => {
    expect(queryKey({ visual_text: "  Hai NGƯỜI   cầm ", kind: "kis" }))
      .toBe(queryKey({ visual_text: "hai người cầm", kind: "kis" }));
  });

  it("separates the kinds and the questions", () => {
    expect(queryKey({ visual_text: "x", kind: "kis" })).not.toBe(queryKey({ visual_text: "x", kind: "qa" }));
    expect(queryKey({ visual_text: "x", kind: "qa", question: "mấy con?" }))
      .not.toBe(queryKey({ visual_text: "x", kind: "qa" }));
  });

  it("keeps each channel separate -- the same words in a different channel are a different query", () => {
    expect(queryKey({ visual_text: "cá nhám", kind: "kis" }))
      .not.toBe(queryKey({ audio_text: "cá nhám", kind: "kis" }));
    expect(queryKey({ visual_text: "cá nhám", audio_text: "", kind: "kis" }))
      .toBe(queryKey({ visual_text: "cá nhám", kind: "kis" }));
  });
});

describe("the browser-side cache", () => {
  it("answers for any frame of the shot the result covers", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, frame_idx: 28350 })).not.toBeNull();
    expect(cachedCheck({ ...REQUEST, frame_idx: 28300 })).not.toBeNull();
    expect(cachedCheck({ ...REQUEST, frame_idx: 28400 })).not.toBeNull();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("does not answer for a frame outside that shot", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, frame_idx: 28299 })).toBeNull();
    expect(cachedCheck({ ...REQUEST, frame_idx: 28401 })).toBeNull();
  });

  it("answers a scrubbing player by the seconds the shot covers", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, frame_idx: undefined, time: 945 })).not.toBeNull();
    expect(cachedCheck({ ...REQUEST, frame_idx: undefined, time: 950 })).toBeNull();
  });

  it("keeps one query's answers away from another's", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, visual_text: "Một con thuyền đỏ" })).toBeNull();
    expect(cachedCheck({ ...REQUEST, kind: "qa" })).toBeNull();
  });

  it("keeps an audio-channel answer away from a visual-channel answer of the same words", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, visual_text: undefined, audio_text: "Hai người cầm điện thoại" }))
      .toBeNull();
  });

  it("keeps a narrower window's answer apart from the default one's", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck(REQUEST);

    expect(cachedCheck({ ...REQUEST, window_seconds: 60 })).toBeNull();
    expect(cachedCheck({ ...REQUEST, window_seconds: 300 })).not.toBeNull();
  });

  it("does not let a frame-exact re-read stand in for its whole shot", async () => {
    // A forced answer has no `covers`: it speaks only for the frame it was asked about.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result({ covers: null })));
    await runCheck({ ...REQUEST, force: true });

    expect(cachedCheck({ ...REQUEST, frame_idx: 28350 })).toBeNull();
  });

  it("is filled by a lookup too, so a reload does not pay again", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ status: "hit", result: result() }));

    expect(await lookupCheck(REQUEST)).toMatchObject({ status: "hit" });
    expect(cachedCheck({ ...REQUEST, frame_idx: 28350 })).not.toBeNull();
  });

  it("is not filled by a miss", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ status: "miss", result: null }));

    expect(await lookupCheck(REQUEST)).toMatchObject({ status: "miss" });
    expect(cachedCheck(REQUEST)).toBeNull();
  });
});

describe("runCheck", () => {
  it("posts the moment and the query together", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck({ ...REQUEST, window_seconds: 60, profile: "l2026-v2" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/check");
    expect(JSON.parse(String((init as RequestInit).body))).toMatchObject({
      visual_text: "Hai người cầm điện thoại", kind: "kis", video_id: "L22_V011",
      frame_idx: 28321, window_seconds: 60, profile: "l2026-v2",
    });
  });

  it("posts audio and OCR channels the same way, separately from visual", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(result()));
    await runCheck({ video_id: "L22_V011", frame_idx: 1, kind: "kis",
      audio_text: "cá nhám", ocr_text: "Nà Ní" });

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body.audio_text).toBe("cá nhám");
    expect(body.ocr_text).toBe("Nà Ní");
    expect(body.visual_text).toBeUndefined();
  });

  it("surfaces the server's own explanation", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "AI check is not configured (OPENROUTER_API_KEY is not set)" }, false, 503));

    await expect(runCheck(REQUEST)).rejects.toThrow(/OPENROUTER_API_KEY/);
  });

  it("names the identity problem when the proxy rejected the request", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({}, false, 401));
    await expect(runCheck(REQUEST)).rejects.toThrow(/Sign in to Tailscale/);
  });
});

describe("lookupCheck", () => {
  it("asks by query string and never posts", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ status: "miss", result: null }));
    await lookupCheck({ ...REQUEST, frame_idx: undefined, time: 944.5 });

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url.startsWith("/api/check/lookup?")).toBe(true);
    expect(url).toContain("time=944.5");
    expect(url).not.toContain("frame_idx");
    expect(fetchMock.mock.calls[0][1]).not.toHaveProperty("method");
  });

  it("carries whichever channels are set", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ status: "miss", result: null }));
    await lookupCheck({ video_id: "L22_V011", frame_idx: 1, kind: "kis", ocr_text: "Nà Ní" });

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("ocr_text=N");
    expect(url).not.toContain("visual_text");
  });
});
