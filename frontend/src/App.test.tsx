import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App, { VideoDialog } from "./App";
import type { SearchResponse, SearchResult } from "./types";

function hit(overrides: Partial<SearchResult>): SearchResult {
  return {
    rank: 0,
    score: 0.5,
    video_id: "K01_V001",
    frame_idx: 10,
    time: 10,
    thumb: "/thumb/K01_V001/10",
    frame: "/frame/K01_V001/10",
    video: "/video/K01_V001",
    source: "visual",
    scores: {},
    ...overrides,
  };
}

function response(results: SearchResult[], mode: SearchResponse["mode"] = "hybrid"): SearchResponse {
  return {
    results,
    latency_ms: 42,
    mode,
    index: "flat",
    index_info: { type: "FlatIP" },
  };
}

function jsonResponse(data: unknown, ok = true): Response {
  return { ok, json: async () => data } as Response;
}

function mockAuthenticatedFetch(...responses: Array<Response | Promise<Response>>) {
  let requestIndex = 0;
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    if (String(input) === "/session") {
      return Promise.resolve(jsonResponse({ login: "owner@example.com", is_admin: true }));
    }
    const response = responses[requestIndex++];
    return response instanceof Promise ? response : Promise.resolve(response);
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

beforeEach(() => {
  mockAuthenticatedFetch();
});

describe("competition search workspace", () => {
  it("shows Compare indexes to the authenticated admin", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({
      login: "owner@example.com",
      is_admin: true,
    }));

    render(<App />);

    expect(await screen.findByRole("button", { name: "Compare indexes" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/session");
  });

  it("hides Compare indexes from an authenticated teammate", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({
      login: "teammate@example.com",
      is_admin: false,
    }));

    render(<App />);

    await waitFor(() => expect(screen.queryByRole("button", { name: "Compare indexes" })).not.toBeInTheDocument());
    expect(within(screen.getByLabelText("Visual index")).getAllByRole("option")).toHaveLength(1);
  });

  it("explains when the Tailscale account is not authorized", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 403,
      json: async () => ({ detail: "not authorized" }),
    } as Response);

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Your Tailscale account is not authorized.");
  });

  it("sends separate English visual and Vietnamese transcript queries in hybrid mode", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([])));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), {
      target: { value: "four astronauts wearing black" },
    });
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), {
      target: { value: "nghiên cứu ánh sáng cực quang" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const body = JSON.parse(fetchMock.mock.calls[1][1]?.body as string);
    expect(body.visual_text).toBe("four astronauts wearing black");
    expect(body.audio_text).toBe("nghiên cứu ánh sáng cực quang");
    expect(body.text).toBeUndefined();
  });

  it("preserves each query while switching search modes", () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "a rocket launch" } });
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), { target: { value: "phóng tàu" } });

    fireEvent.click(screen.getByRole("button", { name: "Audio" }));
    expect(screen.queryByLabelText("Visual query (English)")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("phóng tàu");

    fireEvent.click(screen.getByRole("button", { name: "Visual" }));
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("a rocket launch");
    expect(screen.queryByLabelText("Transcript query (Vietnamese)")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Hybrid" }));
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("a rocket launch");
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("phóng tàu");
  });

  it("shows the exact live frame id reported from video PTS", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true,
      value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
      configurable: true,
      value: vi.fn(),
    });
    mockAuthenticatedFetch(jsonResponse({
      videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
    }));

    render(<VideoDialog result={hit({ time: 0.14 })} onClose={() => {}} />);
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0.14 });

    expect(await screen.findByText(/frame 1/)).toBeInTheDocument();
  });

  it("shows visual index controls only for modes that use visual search", () => {
    render(<App />);

    expect(screen.getByLabelText("Visual index")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Audio" }));
    expect(screen.queryByLabelText("Visual index")).not.toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Transcript evidence" })).toBeInTheDocument();
  });

  it("groups frames by video and updates transcript evidence on selection", async () => {
    mockAuthenticatedFetch(jsonResponse(response([
      hit({ video_id: "A", rank: 0, score: 0.8, time: 50, frame_idx: 50 }),
      hit({
        video_id: "B", rank: 1, score: 0.9, time: 30, frame_idx: 30,
        source: "transcript", transcript_snippet: "supporting speech", transcript_start: 25, transcript_end: 40,
      }),
      hit({ video_id: "A", rank: 2, score: 0.7, time: 10, frame_idx: 10 }),
    ])));

    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "a person speaking" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    const rows = await screen.findAllByTestId("video-row");
    expect(within(rows[0]).getByText("B")).toBeInTheDocument();
    expect(within(rows[1]).getByText("A")).toBeInTheDocument();
    const aFrames = within(rows[1]).getAllByRole("button", { name: /Select frame/ });
    expect(aFrames[0]).toHaveAccessibleName(/0:10/);
    expect(aFrames[1]).toHaveAccessibleName(/0:50/);

    fireEvent.click(within(rows[0]).getByRole("button", { name: /Select frame/ }));
    expect(screen.getByText("supporting speech")).toBeInTheDocument();
    expect(screen.getByText("0:25–0:40")).toBeInTheDocument();
  });

  it("shows a useful message when a hybrid result has no transcript match", async () => {
    mockAuthenticatedFetch(jsonResponse(response([
      hit({ source: "visual", score: 0.9 }),
    ])));

    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "mountain" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("No transcript evidence for this result.")).toBeInTheDocument();
  });

  it("reports API errors without removing the previous results", async () => {
    const fetchMock = mockAuthenticatedFetch(
      jsonResponse(response([hit({ video_id: "kept" })])),
      jsonResponse({ detail: "encoder unavailable" }, false),
    );

    render(<App />);
    const input = screen.getByLabelText("Visual query (English)");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("heading", { name: "kept" })).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("encoder unavailable");
    expect(screen.getByRole("heading", { name: "kept" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("ignores an older response when a newer search finishes first", async () => {
    let resolveFirst!: (value: Response) => void;
    const first = new Promise<Response>((resolve) => { resolveFirst = resolve; });
    mockAuthenticatedFetch(first, jsonResponse(response([hit({ video_id: "new" })])));

    render(<App />);
    const input = screen.getByLabelText("Visual query (English)");
    fireEvent.change(input, { target: { value: "old" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.change(input, { target: { value: "new" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText("new")).toBeInTheDocument();

    resolveFirst(jsonResponse(response([hit({ video_id: "old" })])));
    await waitFor(() => expect(screen.queryByText("old")).not.toBeInTheDocument());
  });

  it("searches with Enter and navigates selected frames with arrow keys", async () => {
    mockAuthenticatedFetch(jsonResponse(response([
      hit({ video_id: "A", rank: 0, score: 0.9, time: 10, frame_idx: 10 }),
      hit({ video_id: "A", rank: 1, score: 0.8, time: 20, frame_idx: 20 }),
    ])));

    render(<App />);
    const input = screen.getByLabelText("Visual query (English)");
    fireEvent.change(input, { target: { value: "keyboard search" } });
    fireEvent.keyDown(input, { key: "Enter" });

    const workspace = await screen.findByRole("main", { name: "Search results" });
    expect(await screen.findByText("frame 10 · 0:10")).toBeInTheDocument();
    fireEvent.keyDown(workspace, { key: "ArrowRight" });
    expect(screen.getByText("frame 20 · 0:20")).toBeInTheDocument();
  });
});
