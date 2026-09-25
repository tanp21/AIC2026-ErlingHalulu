import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { applyExactFrame, CollaborationPanels, QuickAddResult, TeamToolbar } from "./collab";
import type { CollabQuery, RoundSnapshot, SearchResult } from "./types";

const query: CollabQuery = {
  id: 1, round_id: 3, filename: "query-1-kis", kind: "kis", prompt: "Tìm chiếc xe màu đỏ",
  event_count: null, assignee: null, answers: [], version: 0, reviewed_by: null,
  reviewed_at: null, updated_by: null, updated_at: null, status: "todo",
};
const round: RoundSnapshot = {
  id: 3, name: "Round P1", active: true, revision: 0, created_by: "owner@example.com",
  created_at: "2026-08-20T00:00:00Z", total: 1,
  counts: { todo: 1, in_progress: 0, ready: 0, reviewed: 0 }, queries: [query],
};
const result: SearchResult = {
  rank: 0, score: 0.9, video_id: "L00_V001", frame_idx: 42, time: 1.4,
  thumb: "/thumb/L00_V001/42", frame: "/frame/L00_V001/42", video: "/video/L00_V001",
  source: "visual", scores: {},
};

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return { ok, status, json: async () => data, headers: new Headers() } as Response;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });

describe("collaboration panels", () => {
  it("asks for every TRAKE event count immediately after upload", async () => {
    const trake = { ...query, kind: "trake" as const, event_count: null, prompt: "Tìm các cảnh theo thứ tự." };
    const imported = { ...round, queries: [trake] };
    const updated = { ...trake, event_count: 3, version: 1 };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(imported)).mockResolvedValueOnce(jsonResponse(updated));
    const onRoundLoaded = vi.fn();
    render(<TeamToolbar rounds={[]} round={null} isAdmin onSelectRound={vi.fn()} onRoundLoaded={onRoundLoaded} onError={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Upload round" }));
    fireEvent.change(screen.getByLabelText("Round name"), { target: { value: "Round P1" } });
    fireEvent.change(screen.getByLabelText("Query ZIP"), { target: { files: [new File(["zip"], "queries.zip")] } });
    fireEvent.click(screen.getByRole("button", { name: "Import" }));

    expect(await screen.findByText("Tìm các cảnh theo thứ tự.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Event count query-1-kis"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Save event counts & open round" }));
    await waitFor(() => expect(onRoundLoaded).toHaveBeenCalledWith({ ...imported, queries: [updated] }));
  });

  it("saves a selected KIS result immediately", async () => {
    const updated = { ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }], version: 1, status: "ready" as const };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(updated));
    const onChange = vi.fn();
    render(<QuickAddResult query={query} result={result} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "Add frame 42" }));

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(updated));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      expected_version: 0, answers: [{ video_id: "L00_V001", frame_id: 42 }],
    });
    expect(screen.getByRole("status")).toHaveTextContent("Saved frame 42");
  });

  it("keeps the local answer limit separate from DRES player submission", async () => {
    const fullQuery = { ...query, answers: Array.from({ length: 100 }, (_, index) => ({
      video_id: `L00_V${String(index).padStart(3, "0")}`, frame_id: index,
    })) };
    render(<QuickAddResult query={fullQuery} result={result} onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Add frame 42" })).toBeDisabled();
    expect(screen.queryByText(/Submit .*DRES/)).not.toBeInTheDocument();
  });

  it("requires QA text and targets a TRAKE event", async () => {
    const qa = { ...query, kind: "qa" as const };
    const qaUpdated = { ...qa, answers: [{ video_id: "L00_V001", frame_id: 42, answer: "Hà Nội" }], version: 1 };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(qaUpdated));
    const { rerender } = render(<QuickAddResult query={qa} result={result} onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Add frame 42" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Quick add answer"), { target: { value: "Hà Nội" } });
    fireEvent.click(screen.getByRole("button", { name: "Add frame 42" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body)).answers[0].answer).toBe("Hà Nội");

    const trake = { ...query, kind: "trake" as const, event_count: 2 };
    const trakeUpdated = { ...trake, answers: [{ video_id: "L00_V001", frame_ids: [null, 42] }], version: 1 };
    fetchMock.mockResolvedValueOnce(jsonResponse(trakeUpdated));
    rerender(<QuickAddResult query={trake} result={result} onChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Quick add event"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Add frame 42" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body)).answers)
      .toEqual([{ video_id: "L00_V001", frame_ids: [null, 42] }]);
  });

  it("shows the chosen KIS frame and debounces thumbnail changes", () => {
    vi.useFakeTimers();
    const answered = { ...round, queries: [{
      ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }],
    }] };
    render(<CollaborationPanels round={answered} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    const preview = screen.getByRole("img", { name: "Preview frame 42 row 1" });
    expect(preview).toHaveAttribute("src", "/thumb/L00_V001/42?w=160");
    expect(preview).toHaveAttribute("loading", "lazy");
    fireEvent.change(screen.getByLabelText("Frame row 1"), { target: { value: "315" } });
    act(() => vi.advanceTimersByTime(399));
    expect(preview).toHaveAttribute("src", "/thumb/L00_V001/42?w=160");
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("img", { name: "Preview frame 315 row 1" }))
      .toHaveAttribute("src", "/thumb/L00_V001/315?w=160");
  });

  it("shows verification evidence for the selected search keyframe", () => {
    const result = {
      rank: 4, score: .8, video_id: "L23_V019", frame_idx: 9532, time: 380,
      thumb: "/thumb/L23_V019/9532", frame: "/frame/L23_V019/9532", video: "/video/L23_V019",
      source: "both" as const, transcript_snippet: "Martin Laas đang tăng tốc", transcript_start: 370, transcript_end: 380, matched_keywords: ["martin", "laas"],
      scores: { pe_core_rank: 3, audio_rank: 2, keyword_rank: 2 },
    };
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={result} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    expect(screen.getByText("Hybrid").nextSibling).toHaveTextContent("#5");
    expect(screen.getByText("PE-Core").nextSibling).toHaveTextContent("#3");
    expect(screen.getByText("Passage used in hybrid fusion · 6:10–6:20")).toBeInTheDocument();
    expect(screen.getByText("Match: martin, laas")).toBeInTheDocument();
  });

  it("shows one thumbnail for every TRAKE event", () => {
    const trake = { ...query, kind: "trake" as const, event_count: 2,
      answers: [{ video_id: "L00_V001", frame_ids: [10, 20] }] };
    render(<CollaborationPanels round={{ ...round, queries: [trake] }} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    expect(screen.getByRole("img", { name: "Preview E1 frame 10 row 1" }))
      .toHaveAttribute("src", "/thumb/L00_V001/10?w=160");
    expect(screen.getByRole("img", { name: "Preview E2 frame 20 row 1" }))
      .toHaveAttribute("src", "/thumb/L00_V001/20?w=160");
    fireEvent.click(screen.getByRole("button", { name: "Open video L00_V001 at frame 20" }));
    expect(screen.getByLabelText("Answer row")).toHaveValue("0");
    expect(screen.getByLabelText("TRAKE event in player")).toHaveValue("1");
  });

  it("shows an unparsed TRAKE prompt and lets the admin set its event count", async () => {
    const trake = { ...query, kind: "trake" as const, event_count: null, prompt: "Tìm các cảnh theo thứ tự." };
    const updated = { ...trake, event_count: 3, version: 1 };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(updated));
    render(<CollaborationPanels round={{ ...round, queries: [trake] }} login="owner@example.com" isAdmin selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    expect(screen.getByText("Tìm các cảnh theo thứ tự.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("TRAKE event count"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Save event count" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      expected_version: 0, event_count: 3,
    });
  });

  it("shows a fallback when a selected thumbnail cannot be loaded", () => {
    const answered = { ...round, queries: [{
      ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }],
    }] };
    render(<CollaborationPanels round={answered} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    fireEvent.error(screen.getByRole("img", { name: "Preview frame 42 row 1" }));
    expect(screen.getByRole("img", { name: "No image frame 42 row 1" })).toBeInTheDocument();
  });

  it("opens a chosen answer frame paused at its exact PTS", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
      configurable: true, value: vi.fn(),
    });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      if (String(input).startsWith("/frame-map/")) return Promise.resolve(jsonResponse({
        videoId: "L00_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
      }));
      return Promise.resolve(jsonResponse({ videoId: "L00_V001", cues: [] }));
    });
    const answered = { ...round, queries: [{
      ...query, answers: [{ video_id: "L00_V001", frame_id: 1 }],
    }] };
    render(<CollaborationPanels round={answered} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Open video L00_V001 at frame 1" }));
    await waitFor(() => expect(callback).toBeDefined());
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    expect(video).not.toHaveAttribute("autoplay");
    expect(video.currentTime).toBe(0);
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(0.04);
    expect(screen.getByLabelText("Answer row")).toHaveValue("0");
  });

  it("appends exact frames to KIS and preserves VQA text when replacing a row", () => {
    expect(applyExactFrame(query, [], "L00_V009", 315, -1, 0)).toEqual({
      answers: [{ video_id: "L00_V009", frame_id: 315 }], rowIndex: 0,
    });
    const qa = { ...query, kind: "qa" as const };
    expect(applyExactFrame(qa, [{ video_id: "L00_V001", frame_id: 10, answer: "Hà Nội" }],
      "L00_V009", 315, 0, 0)).toEqual({
      answers: [{ video_id: "L00_V009", frame_id: 315, answer: "Hà Nội" }], rowIndex: 0,
    });
  });

  it("creates a new TRAKE prediction instead of mixing events from different videos", () => {
    const trake = { ...query, kind: "trake" as const, event_count: 3 };
    expect(applyExactFrame(trake, [{ video_id: "L00_V001", frame_ids: [10, 20, null] }],
      "L00_V009", 315, 0, 2)).toEqual({
      answers: [
        { video_id: "L00_V001", frame_ids: [10, 20, null] },
        { video_id: "L00_V009", frame_ids: [null, null, 315] },
      ],
      rowIndex: 1,
    });
  });

  it("rejects exact-frame capture when creating a 101st prediction", () => {
    const full = Array.from({ length: 100 }, (_, frame_id) => ({ video_id: "L00_V001", frame_id }));
    expect(() => applyExactFrame(query, full, "L00_V009", 315, -1, 0)).toThrow("Maximum 100 answer rows");
  });

  it("rejects a destination row that disappeared from the draft", () => {
    expect(() => applyExactFrame(query, [{ video_id: "L00_V001", frame_id: 1 }],
      "L00_V009", 315, 4, 0)).toThrow("Answer row no longer exists");
  });

  it("shows only DRES submission controls when the player supports official submission", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
      configurable: true, value: vi.fn(),
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({
      videoId: "L00_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 1400, ptsDeltas: [40], frameCount: 2,
    }));
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={result} openResult={result} onCloseVideo={vi.fn()}
      onDresSubmit={vi.fn(async () => ({ message: "DRES not ready", completed: false }))}
      onSelect={vi.fn()} onQueryChange={vi.fn()} />);
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0 });
    fireEvent.pause(screen.getByLabelText("Video player"));
    expect(await screen.findByRole("button", { name: "Submit 0:00 (0 ms)" })).toBeEnabled();
    expect(screen.queryByLabelText("Answer row")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use frame 0" })).not.toBeInTheDocument();
  });

  it("shows query ownership and filters to my work", () => {
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);
    expect(screen.getByText("Tìm chiếc xe màu đỏ")).toBeInTheDocument();
    expect(screen.getAllByText("To do").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Mine" }));
    expect(within(screen.getByLabelText("Query list")).queryByText("query-1-kis")).not.toBeInTheDocument();
  });

  it("claims a query and saves a selected keyframe", async () => {
    const updated = { ...query, assignee: "mate@example.com", version: 1, status: "in_progress" };
    const ready = { ...updated, answers: [{ video_id: "L00_V001", frame_id: 42 }], version: 2, status: "ready" };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(updated))
      .mockResolvedValueOnce(jsonResponse(ready));
    const onQueryChange = vi.fn();
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={result} onSelect={vi.fn()} onQueryChange={onQueryChange} />);

    fireEvent.click(screen.getByRole("button", { name: "Assign to me" }));
    await waitFor(() => expect(onQueryChange).toHaveBeenCalledWith(updated));
    fireEvent.click(screen.getByRole("button", { name: "Add frame 42" }));
    expect(screen.getByDisplayValue("L00_V001")).toBeInTheDocument();
    expect(screen.getByDisplayValue("42")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unassign myself" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
    await waitFor(() => expect(onQueryChange).toHaveBeenCalledWith(ready));
    expect(fetchMock).toHaveBeenLastCalledWith("/api/collab/queries/1", expect.objectContaining({ method: "PATCH" }));
  });

  it("fills and saves 100 answers from search results", async () => {
    const results = Array.from({ length: 100 }, (_, index) => ({
      ...result, rank: index, frame_idx: index,
    }));
    const updated = { ...query, answers: results.map(({ video_id, frame_idx }) => ({
      video_id, frame_id: frame_idx, auto_filled: true,
    })), version: 1, status: "ready" as const };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(updated));
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={result} searchResults={results} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Fill to 100 from results" }));

    await waitFor(() => expect(screen.getAllByLabelText(/Video row/)).toHaveLength(100));
    expect(screen.getByText(/100\/100 rows/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/collab/queries/1", expect.objectContaining({ method: "PATCH" }));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body)).answers).toHaveLength(100);
  });

  it("finds similar keyframes from the existing answer before filling", async () => {
    const answered = { ...round, queries: [{ ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }] }] };
    const results = Array.from({ length: 100 }, (_, index) => ({ ...result, rank: index, frame_idx: index + 100 }));
    const search = { results, latency_ms: 1, mode: "visual", index: "flat", index_info: { type: "FlatIP" }, next_cursor: null, has_more: false, total_results: 100, truncated: false };
    const updated = { ...answered.queries[0], answers: [{ video_id: "L00_V001", frame_id: 42 }, ...results.map(({ video_id, frame_idx }) => ({ video_id, frame_id: frame_idx }))], version: 1, status: "ready" as const };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(search))
      .mockResolvedValueOnce(jsonResponse(updated));
    render(<CollaborationPanels round={answered} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Find similar & fill to 100" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[0][0]).toBe("/search/image");
    const payload = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body instanceof FormData
      ? ((fetchMock.mock.calls[0][1] as RequestInit).body as FormData).get("payload") : ""));
    expect(payload.reference).toEqual({ video_id: "L00_V001", frame_idx: 42 });
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body)).answers).toHaveLength(100);
  });

  it("prefers the saved answer's frame over live search results when a reference already exists", async () => {
    const answered = { ...round, queries: [{ ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }] }] };
    const similar = Array.from({ length: 100 }, (_, index) => ({ ...result, rank: index, frame_idx: index + 200 }));
    const search = { results: similar, latency_ms: 1, mode: "visual", index: "flat", index_info: { type: "FlatIP" }, next_cursor: null, has_more: false, total_results: 100, truncated: false };
    const liveSearchResults = Array.from({ length: 100 }, (_, index) => ({ ...result, rank: index, frame_idx: index + 900 }));
    const updated = { ...answered.queries[0], answers: [{ video_id: "L00_V001", frame_id: 42 }, ...similar.map(({ video_id, frame_idx }) => ({ video_id, frame_id: frame_idx }))], version: 1, status: "ready" as const };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(search))
      .mockResolvedValueOnce(jsonResponse(updated));
    render(<CollaborationPanels round={answered} login="mate@example.com" selectedId={1}
      selectedResult={null} searchResults={liveSearchResults} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Find similar & fill to 100" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Find similar & fill to 100" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[0][0]).toBe("/search/image");
    const savedFrames = JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body)).answers
      .map((answer: { frame_id: number }) => answer.frame_id);
    expect(savedFrames).not.toContain(900);
    expect(savedFrames).toContain(200);
  });

  it("removes only the auto-filled rows via the dedicated clear button", async () => {
    const mixed = {
      ...round,
      queries: [{
        ...query,
        answers: [
          { video_id: "L00_V001", frame_id: 42 },
          { video_id: "L00_V002", frame_id: 7, auto_filled: true },
          { video_id: "L00_V003", frame_id: 9, auto_filled: true },
        ],
      }],
    };
    const updated = { ...mixed.queries[0], answers: [{ video_id: "L00_V001", frame_id: 42 }], version: 1, status: "ready" as const };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(updated));
    render(<CollaborationPanels round={mixed} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);

    expect(screen.getAllByLabelText(/Video row/)).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "Remove auto-filled results" }));
    expect(screen.getAllByLabelText(/Video row/)).toHaveLength(1);
    expect(screen.getByDisplayValue("L00_V001")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body)).answers).toEqual([
      { video_id: "L00_V001", frame_id: 42 },
    ]);
  });

  it("keeps a dirty draft while adopting the server version after a conflict", async () => {
    const server = { ...query, version: 2, assignee: "owner@example.com", status: "in_progress" };
    const ready = { ...server, version: 3, answers: [{ video_id: "L00_V001", frame_id: 42 }], status: "ready" };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ detail: "stale query version", current: server }, false, 409))
      .mockResolvedValueOnce(jsonResponse(ready));
    render(<CollaborationPanels round={round} login="mate@example.com" selectedId={1}
      selectedResult={result} onSelect={vi.fn()} onQueryChange={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Add frame 42" }));
    fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
    expect(await screen.findByText("stale query version")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Keep draft and use server version" }));
    fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const body = JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body));
    expect(body.expected_version).toBe(2);
    expect(body.answers).toEqual([{ video_id: "L00_V001", frame_id: 42 }]);
  });

  it("disables answer editing for an inactive historical round", () => {
    const historical = { ...round, active: false, queries: [{
      ...query, answers: [{ video_id: "L00_V001", frame_id: 42 }], status: "ready" as const,
    }] };
    render(<CollaborationPanels round={historical} login="mate@example.com" selectedId={1}
      selectedResult={null} onSelect={vi.fn()} onQueryChange={vi.fn()} />);
    expect(screen.getByDisplayValue("L00_V001")).toBeDisabled();
    expect(screen.getByText("Historical round · read-only")).toBeInTheDocument();
  });
});
