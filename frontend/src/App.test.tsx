import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App, { VideoDialog } from "./App";
import { forgetChecks } from "./checkApi";
import type { CollabQuery, DresStatus, SearchResponse, SearchResult, TemporalSearchResponse } from "./types";

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
    next_cursor: null,
    has_more: false,
    total_results: results.length,
    truncated: false,
  };
}

const dialogQuery: CollabQuery = {
  id: 1, round_id: 1, filename: "query-1-kis", kind: "kis", prompt: "Find it",
  event_count: null, assignee: null, answers: [], version: 0, reviewed_by: null,
  reviewed_at: null, updated_by: null, updated_at: null, status: "todo",
};

function jsonResponse(data: unknown, ok = true): Response {
  return { ok, status: ok ? 200 : 400, json: async () => data, headers: new Headers() } as Response;
}

function mockAuthenticatedFetch(...responses: Array<Response | Promise<Response>>) {
  let requestIndex = 0;
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    if (String(input) === "/session") {
      return Promise.resolve(jsonResponse({ login: "owner@example.com", is_admin: true }));
    }
    if (String(input) === "/profiles") return Promise.resolve(jsonResponse([
      { name: "l2025", label: "L · embeddings 2025", is_default: true, count: 293239, video_count: 873, indexes: ["flat"] },
      { name: "l2026", label: "L · embeddings 2026", is_default: false, count: 289875, video_count: 873, indexes: ["flat"] },
    ]));
    if (String(input) === "/api/collab/rounds") return Promise.resolve(jsonResponse([]));
    // Background polls must not eat the queue: whether one lands before the test's own first
    // request is a timing race, and a test that queues [search, context, ...] would then hand
    // the search response to whichever poll got there first.
    if (String(input) === "/api/dres/history") return Promise.resolve(jsonResponse({ attempts: [] }));
    if (String(input) === "/api/dres/status") return Promise.resolve(jsonResponse({
      configured: false, credentials_configured: false, host: "", connected: false,
      evaluations: [], evaluation_id: null, context: null, context_valid: false,
    }));
    const response = responses[requestIndex++];
    return response instanceof Promise ? response : Promise.resolve(response);
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

beforeEach(() => {
  mockAuthenticatedFetch();
});

describe("competition search workspace", () => {
  it("shows DRES status at the end of the navigation and opens the settings popup", async () => {
    render(<App />);
    const navButton = await screen.findByRole("button", { name: "DRES · Setup required" });
    const taskPicker = screen.getByRole("combobox", { name: "DRES task type" });
    expect(taskPicker).toHaveValue("kis");
    fireEvent.change(taskPicker, { target: { value: "trake" } });
    await waitFor(() => expect(taskPicker).toHaveValue("trake"));
    fireEvent.click(navButton);
    expect(await screen.findByRole("dialog", { name: "DRES settings" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^DRES$/ })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Select current query")).not.toBeInTheDocument();
    expect(screen.queryByText(/Assignee:|Reviewed by/)).not.toBeInTheDocument();
  });

  it("keeps DRES submission out of the right panel", async () => {
    const dresStatus: DresStatus = {
      configured: true, credentials_configured: true, host: "localhost", connected: true,
      evaluations: [{ id: "eval-1", name: "Final" }], evaluation_id: "eval-1",
      context_valid: true,
      context: { kind: "kis", label: "Hall task", event_count: null, revision: 1,
        updated_by: "owner", updated_at: "2026-09-24T00:00:00Z" },
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path === "/session") return Promise.resolve(jsonResponse({ login: "owner@example.com", is_admin: true }));
      if (path === "/api/collab/rounds") return Promise.resolve(jsonResponse([]));
      if (path === "/search") return Promise.resolve(jsonResponse(response([hit({
        video_id: "L01_V001", frame_idx: 2, time: 123.456, thumb: "/thumb/L01_V001/2",
        frame: "/frame/L01_V001/2", video: "/video/L01_V001",
      })])));
      if (path.startsWith("/context/")) return Promise.resolve(jsonResponse({ video_id: "L01_V001", frame_idx: 2, frames: [] }));
      if (path === "/api/dres/status") return Promise.resolve(jsonResponse(dresStatus));
      if (path === "/api/dres/history") return Promise.resolve(jsonResponse({ attempts: [] }));
      return Promise.reject(new Error(`Unexpected request: ${path}`));
    });
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "scene" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: "Select frame L01_V001 at 2:03" }));
    expect(fetchMock.mock.calls.some(([input]) => ["/api/dres/prepare", "/api/dres/submit"].includes(String(input)))).toBe(false);
    expect(screen.queryByRole("button", { name: /^DRES$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "DRES submission" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Team results" })).not.toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Right panel mode" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Submit L01_V001 frame 2 to DRES" })).toBeInTheDocument();
    const rankingPanel = screen.getByRole("complementary", { name: "Transcript evidence" });
    expect(within(rankingPanel).queryByText("Current query")).not.toBeInTheDocument();
    expect(within(rankingPanel).queryByRole("button", { name: /Open video at/ })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => ["/api/dres/prepare", "/api/dres/submit"].includes(String(input)))).toBe(false);
  });

  it("sends the fixed competition profile with searches", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([])));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "red car" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search");
    expect(JSON.parse(call?.[1]?.body as string).profile).toBe("l2026");
    expect(screen.queryByLabelText("Embedding profile")).not.toBeInTheDocument();
  });

  it("edits temporal events and sends the dedicated request", async () => {
    const temporal: TemporalSearchResponse = {
      videos: [], total_sequences: 0, total_videos: 0, latency_ms: 25,
      faiss_latency_ms: 10, mode: "temporal", candidate_k: 2000,
      scoring_method: "raw_mean", next_cursor: null, has_more: false, truncated: false,
    };
    const fetchMock = mockAuthenticatedFetch(jsonResponse(temporal));
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));
    expect(screen.queryByLabelText("Visual index")).not.toBeInTheDocument();
    expect(screen.getAllByLabelText(/Visual event E/)).toHaveLength(2);
    fireEvent.change(screen.getByLabelText("Visual event E1"), { target: { value: "a cook cuts mushrooms" } });
    fireEvent.change(screen.getByLabelText("Visual event E2"), { target: { value: "the pan is placed on a stove" } });
    fireEvent.change(screen.getByLabelText("Temporal max gap seconds"), { target: { value: "180" } });
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), { target: { value: "món nấm" } });
    fireEvent.click(screen.getByRole("button", { name: "Add event" }));
    fireEvent.change(screen.getByLabelText("Visual event E3"), { target: { value: "the flame appears" } });
    fireEvent.click(screen.getByRole("button", { name: "Move event E3 up" }));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search/temporal")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search/temporal");
    expect(JSON.parse(call?.[1]?.body as string)).toEqual({
      events: ["a cook cuts mushrooms", "the flame appears", "the pan is placed on a stove"],
      audio_text: "món nấm", audio_method: "semantic", pe_weight: .75,
      max_gap_seconds: 180, profile: "l2026",
    });
  });

  it("renders multiple temporal sequences chronologically and opens an event", async () => {
    const event = (eventIndex: number, frame: number, time: number, score: number): SearchResult & { event_index: number } => ({
      ...hit({ frame_idx: frame, time, score, rank: frame }), event_index: eventIndex,
    });
    const temporal: TemporalSearchResponse = {
      videos: [{
        video_id: "A", best_score: 0.9, sequences: [
          { rank: 1, video_id: "A", score: 0.7, visual_score: 0.7, source: "visual", start_time: 10, end_time: 20,
            events: [event(0, 10, 10, 0.8), event(1, 20, 20, 0.6)] },
          { rank: 0, video_id: "A", score: 0.9, visual_score: 0.8, source: "both", start_time: 100, end_time: 120,
            events: [event(0, 100, 100, 0.9), event(1, 120, 120, 0.7)],
            audio_evidence: { video_id: "A", time: 110, score: 0.5, audio_rank: 1,
              transcript_snippet: "audio support", transcript_start: 105, transcript_end: 115 } },
        ],
      }], total_sequences: 2, total_videos: 1, latency_ms: 25, faiss_latency_ms: 10,
      mode: "temporal", candidate_k: 2000, scoring_method: "raw_mean",
      next_cursor: null, has_more: false, truncated: false,
    };
    mockAuthenticatedFetch(jsonResponse(temporal), jsonResponse({ detail: "unavailable" }, false));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));
    fireEvent.change(screen.getByLabelText("Visual event E1"), { target: { value: "first" } });
    fireEvent.change(screen.getByLabelText("Visual event E2"), { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    const sequences = await screen.findAllByTestId("temporal-sequence");
    expect(within(sequences[0]).getByText("0:10–0:20")).toBeInTheDocument();
    expect(within(sequences[1]).getByText("1:40–2:00")).toBeInTheDocument();
    expect(sequences[1]).toHaveClass("tier-5");
    fireEvent.click(within(sequences[1]).getByRole("button", { name: /Open event E1/ }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getAllByText("audio support")).toHaveLength(2);
  });

  it("toggles temporal results between grouped-by-video and a flat ranked list", async () => {
    const event = (eventIndex: number, frame: number, time: number, score: number): SearchResult & { event_index: number } => ({
      ...hit({ frame_idx: frame, time, score, rank: frame }), event_index: eventIndex,
    });
    const temporal: TemporalSearchResponse = {
      videos: [
        { video_id: "A", best_score: 0.9, sequences: [
          { rank: 0, video_id: "A", score: 0.9, visual_score: 0.9, source: "visual", start_time: 100, end_time: 120,
            events: [event(0, 100, 100, 0.9), event(1, 120, 120, 0.7)] },
        ] },
        { video_id: "B", best_score: 0.8, sequences: [
          { rank: 1, video_id: "B", score: 0.8, visual_score: 0.8, source: "visual", start_time: 10, end_time: 20,
            events: [event(0, 10, 10, 0.8), event(1, 20, 20, 0.6)] },
        ] },
      ], total_sequences: 2, total_videos: 2, latency_ms: 25, faiss_latency_ms: 10,
      mode: "temporal", candidate_k: 2000, scoring_method: "raw_mean",
      next_cursor: null, has_more: false, truncated: false,
    };
    mockAuthenticatedFetch(jsonResponse(temporal));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));
    fireEvent.change(screen.getByLabelText("Visual event E1"), { target: { value: "first" } });
    fireEvent.change(screen.getByLabelText("Visual event E2"), { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findAllByTestId("temporal-video")).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "Ranked list" }));

    expect(screen.queryByTestId("temporal-video")).not.toBeInTheDocument();
    const sequences = screen.getAllByTestId("temporal-sequence");
    expect(sequences).toHaveLength(2);
    expect(within(sequences[0]).getByText("A")).toBeInTheDocument();
    expect(within(sequences[1]).getByText("B")).toBeInTheDocument();
  });

  it("toggles hybrid results between a flat ranked grid and grouped-by-video", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([
      hit({ video_id: "A", frame_idx: 10, time: 10, rank: 0, score: 0.9 }),
      hit({ video_id: "B", frame_idx: 20, time: 20, rank: 1, score: 0.7 }),
    ])));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "red car" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(true));

    expect(await screen.findAllByRole("button", { name: /Select frame/ })).toHaveLength(2);
    expect(screen.queryByTestId("video-row")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Group by video" }));

    const rows = screen.getAllByTestId("video-row");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("A", { selector: "h3" })).toBeInTheDocument();
    expect(within(rows[1]).getByText("B", { selector: "h3" })).toBeInTheDocument();
  });

  it("hides index comparison when the competition profile only has flat", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({
      login: "owner@example.com",
      is_admin: true,
    }));

    render(<App />);

    await waitFor(() => expect(screen.queryByRole("button", { name: "Compare indexes" })).not.toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith("/session");
  });

  it("hides Compare indexes from an authenticated teammate", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({
      login: "teammate@example.com",
      is_admin: false,
    }));

    render(<App />);

    await waitFor(() => expect(screen.queryByRole("button", { name: "Compare indexes" })).not.toBeInTheDocument());
    expect(screen.queryByLabelText("Visual index")).not.toBeInTheDocument();
  });

  it("loads the active collaboration round into the integrated workspace", async () => {
    const summary = {
      id: 1, name: "Round P1", active: true, revision: 0, created_by: "owner@example.com",
      created_at: "2026-08-20T00:00:00Z", total: 1,
      counts: { todo: 1, in_progress: 0, ready: 0, reviewed: 0 },
    };
    const snapshot = { ...summary, queries: [{
      id: 2, round_id: 1, filename: "query-1-kis", kind: "kis",
      prompt: "Tìm chiếc xe màu đỏ", event_count: null, assignee: null, answers: [],
      version: 0, reviewed_by: null, reviewed_at: null, updated_by: null,
      updated_at: null, status: "todo",
    }] };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url === "/session") return Promise.resolve(jsonResponse({ login: "owner@example.com", is_admin: true }));
      if (url === "/api/collab/rounds") return Promise.resolve(jsonResponse([summary]));
      if (url === "/api/collab/rounds/1") return Promise.resolve(jsonResponse(snapshot));
      return Promise.resolve(jsonResponse({}));
    });

    render(<App />);

    expect(await within(screen.getByLabelText("Search controls")).findByText("Tìm chiếc xe màu đỏ")).toBeInTheDocument();
    expect(screen.queryByLabelText("Select current query")).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary", { name: "Query list" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Query management" }));
    expect(screen.getByRole("complementary", { name: "Query list" })).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Query answer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export ZIP" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload round" })).toBeInTheDocument();
  });

  it("opens the retrieval agent panel with per-teammate key fields", async () => {
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Retrieval agent" }));

    expect(screen.getByLabelText("DeepSeek API key")).toBeInTheDocument();
    expect(screen.getByLabelText("Gemini API key")).toBeInTheDocument();
    expect(screen.getByLabelText("Agent query")).toBeInTheDocument();
    expect(screen.getByLabelText("Agent mode")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run agent" })).toBeInTheDocument();
  });

  it("keeps the search workspace when switching collaboration queries", async () => {
    const summary = {
      id: 1, name: "Round P1", active: true, revision: 0, created_by: "owner@example.com",
      created_at: "2026-08-20T00:00:00Z", total: 2,
      counts: { todo: 2, in_progress: 0, ready: 0, reviewed: 0 },
    };
    const query = (id: number): CollabQuery => ({ ...dialogQuery, id, filename: `query-${id}-kis`, round_id: 1 });
    const snapshot = { ...summary, queries: [query(2), query(3)] };
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url === "/session") return Promise.resolve(jsonResponse({ login: "owner@example.com", is_admin: true }));
      if (url === "/api/collab/rounds") return Promise.resolve(jsonResponse([summary]));
      if (url === "/api/collab/rounds/1") return Promise.resolve(jsonResponse(snapshot));
      if (url === "/search") return Promise.resolve(jsonResponse(response([hit({ video_id: "kept" })])));
      return Promise.resolve(jsonResponse({ video_id: "kept", frame_idx: 10, frames: [] }));
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Query management" }));
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "red car" } });
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), { target: { value: "xe màu đỏ" } });
    fireEvent.change(screen.getByLabelText("On-screen text query"), { target: { value: "VTV" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /query-3-kis/ }));

    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("red car");
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("xe màu đỏ");
    expect(screen.getByLabelText("On-screen text query")).toHaveValue("VTV");
    expect(screen.getByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();
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
    fireEvent.change(screen.getByLabelText("On-screen text query"), {
      target: { value: "THÀNH PHỐ" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(true));
    const searchCall = fetchMock.mock.calls.find(([input]) => String(input) === "/search");
    const body = JSON.parse(searchCall?.[1]?.body as string);
    expect(body.visual_text).toBe("four astronauts wearing black");
    expect(body.audio_text).toBe("nghiên cứu ánh sáng cực quang");
    expect(body.ocr_text).toBe("THÀNH PHỐ");
    expect(body.text).toBeUndefined();
    expect(body.k).toBeUndefined();
    expect(screen.queryByText("Top k")).not.toBeInTheDocument();
  });

  it("has no separate OCR search action", () => {
    render(<App />);

    expect(screen.queryByRole("button", { name: "Ocr" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Tìm bằng OCR" })).not.toBeInTheDocument();
    expect(screen.getByText(/Search uses OCR to filter by video/)).toBeInTheDocument();
  });

  it("uses OCR-only when the main Search button has no visual or audio query", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([], "ocr")));
    render(<App />);

    fireEvent.change(screen.getByLabelText("On-screen text query"), { target: { value: "THÀNH PHỐ" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search");
    expect(JSON.parse(call?.[1]?.body as string)).toMatchObject({ mode: "ocr", ocr_text: "THÀNH PHỐ" });
  });

  it("searches with an uploaded image and adjustable image influence", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([], "visual")));
    const createObjectURL = vi.fn(() => "blob:query-image");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), {
      target: { value: "a red sports car" },
    });
    const file = new File(["png bytes"], "car.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Query image"), { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Image influence"), { target: { value: "75" } });
    fireEvent.change(screen.getByLabelText("On-screen text query"), { target: { value: "SPORT" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search/image")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search/image");
    const form = call?.[1]?.body as FormData;
    expect(form.get("image")).toBe(file);
    expect(JSON.parse(String(form.get("payload")))).toMatchObject({
      mode: "hybrid", visual_text: "a red sports car", ocr_text: "SPORT", image_weight: 0.75,
    });
    expect(createObjectURL).toHaveBeenCalledWith(file);
  });

  it("pastes a screenshot and permits an image-only search", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(response([], "visual")));
    vi.stubGlobal("URL", { ...URL, createObjectURL: () => "blob:pasted", revokeObjectURL: vi.fn() });
    render(<App />);

    const screenshot = new File(["clipboard"], "screenshot.png", { type: "image/png" });
    fireEvent.paste(screen.getByLabelText("Query image drop zone"), {
      clipboardData: { files: [screenshot] },
    });
    expect(screen.getByAltText("Query image preview")).toHaveAttribute("src", "blob:pasted");
    expect(screen.getByLabelText("Image influence")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search/image")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search/image");
    const payload = JSON.parse(String((call?.[1]?.body as FormData).get("payload")));
    expect(payload.image_weight).toBe(1);
    expect(payload.visual_text).toBeUndefined();
  });

  it("reranks immediately from a result keyframe reference", async () => {
    const first = response([hit({ video_id: "K01_V001", frame_idx: 77 })], "visual");
    const fetchMock = mockAuthenticatedFetch(jsonResponse(first), jsonResponse(response([], "visual")));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "a red car" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: /Find frames similar to K01_V001 at/ }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/search/image")).toHaveLength(1));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search/image");
    const form = call?.[1]?.body as FormData;
    expect(form.get("image")).toBeNull();
    expect(JSON.parse(String(form.get("payload")))).toMatchObject({
      visual_text: "a red car",
      image_weight: 0.5,
      reference: { video_id: "K01_V001", frame_idx: 77 },
    });
  });

  it("keeps OCR in the sidebar while switching supported search modes", () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "a rocket launch" } });
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), { target: { value: "phóng tàu" } });

    fireEvent.change(screen.getByLabelText("On-screen text query"), { target: { value: "VTV" } });
    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));
    expect(screen.getByLabelText("On-screen text query")).toHaveValue("VTV");

    fireEvent.click(screen.getByRole("button", { name: "Hybrid" }));
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("a rocket launch");
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("phóng tàu");
    expect(screen.getByLabelText("On-screen text query")).toHaveValue("VTV");
    expect(screen.queryByRole("button", { name: "Ocr" })).not.toBeInTheDocument();
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

    render(<VideoDialog result={hit({ time: 0.04 })} onClose={() => {}} />);
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0.04 });

    expect(await screen.findByText(/frame 1/)).toBeInTheDocument();
  });

  it("submits the exact paused frame from the video player", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    mockAuthenticatedFetch(jsonResponse({
      videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
    }));
    const onSubmit = vi.fn(async (_videoId: string, frameId: number, time: number) => ({ message: `Accepted frame ${frameId} at ${time}` }));
    const { container } = render(<VideoDialog result={hit({ time: 0.04 })} onClose={() => {}} originalVideo picker={{ actions: [], submit: {
      label: "Submit selected frame", kind: "kis", onSubmit,
    } }} />);
    expect(container.querySelectorAll(".dres-player-submit button")).toHaveLength(1);
    expect(screen.queryByLabelText("DRES task type in player")).not.toBeInTheDocument();
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0.04 });
    fireEvent.pause(screen.getByLabelText("Video player"));
    fireEvent.click(await screen.findByRole("button", { name: "Submit 0:00 (40 ms)" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith("K01_V001", 1, 0.04, "", "kis"));
    expect(await screen.findByText("Accepted frame 1 at 0.04")).toBeInTheDocument();
  });

  it("loads all indexed keyframes below the player and seeks by original frame PTS", async () => {
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path.startsWith("/frame-map/")) return Promise.resolve(jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
      }));
      if (path.startsWith("/context/")) return Promise.resolve(jsonResponse({
        video_id: "K01_V001", frame_idx: 1, frames: [
          { video_id: "K01_V001", frame_idx: 1, time: 0.04, thumb: "/thumb/K01_V001/1" },
          { video_id: "K01_V001", frame_idx: 2, time: 0.1, thumb: "/thumb/K01_V001/2" },
        ],
      }));
      return Promise.resolve(jsonResponse({ videoId: "K01_V001", cues: [] }));
    });
    render(<VideoDialog result={hit({ frame_idx: 1, time: 0.04 })} onClose={() => {}} originalVideo profile="l2026-v2"
      picker={{ actions: [], submit: { label: "Submit selected frame", kind: "kis", onSubmit: vi.fn() } }} />);

    const strip = await screen.findByRole("region", { name: "Keyframes in this video" });
    expect((await within(strip).findByRole("button", { name: "Show keyframe 1 at 0:00" })).querySelector("img"))
      .toHaveAttribute("src", "/thumb/K01_V001/1?w=160");
    fireEvent.click(await within(strip).findByRole("button", { name: "Show keyframe 2 at 0:00" }));
    expect((screen.getByLabelText("Video player") as HTMLVideoElement).currentTime).toBe(0.1);
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/context/K01_V001/1?radius=10&profile=l2026-v2&all_frames=true")).toBe(true);
  });

  it("adds the frame on screen as the TRAKE event, not the nearest official keyframe", async () => {
    // No frame callback arrives, so the frame comes from the playhead -- and the button has to name
    // that frame, which is 12.4 s and frame 2, not the organizer keyframe 250 at 12.5 s.
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path.startsWith("/api/dres/keyframes/")) return Promise.resolve(jsonResponse({ frames: [
        { frame_id: 100, source_frame_idx: 1, timestamp_ms: 10000 },
        { frame_id: 250, source_frame_idx: 2, timestamp_ms: 35000 },
      ] }));
      if (path.startsWith("/frame-map/")) return Promise.resolve(jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
      }));
      return Promise.resolve(jsonResponse({ videoId: "K01_V001", cues: [] }));
    });
    const onSubmit = vi.fn().mockResolvedValue({ message: "Added E1 · FRAME_ID 2", completed: false });
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo
      picker={{ actions: [], submit: { label: "Add event", kind: "trake", onSubmit },
        trakeSequence: { videoId: null, frameIds: [], onRemove: vi.fn(), onMove: vi.fn(), onOpen: vi.fn(), onSubmit: vi.fn() } }} />);
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    fireEvent.play(video);
    video.currentTime = 12.4;
    fireEvent.timeUpdate(video);

    expect(await screen.findByText("Source frame 2")).toBeInTheDocument();
    const add = await screen.findByRole("button", { name: "Add E1 · #2 · 0:12" });
    expect(add).toBeEnabled();
    fireEvent.click(add);
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith("K01_V001", 2, 12.5, "", "trake"));
    // The strip is navigation onto the organizer's keyframes, but it names the source frame each one
    // resolves to, so the number in the strip is the number that would be submitted from there.
    const strip = await screen.findByRole("region", { name: "Official TRAKE keyframes" });
    expect(within(strip).getByRole("button", { name: "Show keyframe 2 at 0:35" }).querySelector("img"))
      .toHaveAttribute("src", "/thumb/K01_V001/2?w=160");
  });

  it("names the next TRAKE event and lists the ones already staged", async () => {
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path.startsWith("/api/dres/keyframes/")) return Promise.resolve(jsonResponse({ frames: [] }));
      if (path.startsWith("/frame-map/")) return Promise.resolve(jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
      }));
      return Promise.resolve(jsonResponse({ videoId: "K01_V001", cues: [] }));
    });
    const onRemove = vi.fn();
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo
      picker={{ actions: [], submit: { label: "Add event", kind: "trake", onSubmit: vi.fn() },
        trakeSequence: { videoId: "K01_V001", frameIds: [100, 250], onRemove, onMove: vi.fn(), onOpen: vi.fn(), onSubmit: vi.fn() } }} />);
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    fireEvent.play(video);
    video.currentTime = 12.4;
    fireEvent.timeUpdate(video);

    // Two events are already staged, so the button must offer the third -- not E1 every time.
    expect(await screen.findByRole("button", { name: /^Add E3 · #/ })).toBeInTheDocument();
    expect(screen.getByText("2 TRAKE events selected")).toBeInTheDocument();

    // The staged events are manageable from the player itself, which is the point of the panel.
    fireEvent.click(screen.getByRole("button", { name: "Remove TRAKE event E1" }));
    expect(onRemove).toHaveBeenCalledWith(0);
  });

  it("numbers the next TRAKE event and lists the ones the sequence already holds", async () => {
    // A picker in TRAKE mode without its sequence counts zero events: the button reads "Add E1"
    // forever and no event is ever listed, however many were added.
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path.startsWith("/api/dres/keyframes/")) return Promise.resolve(jsonResponse({ frames: [
        { frame_id: 100, source_frame_idx: 1, timestamp_ms: 10000 },
      ] }));
      if (path.startsWith("/frame-map/")) return Promise.resolve(jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
      }));
      return Promise.resolve(jsonResponse({ videoId: "K01_V001", cues: [] }));
    });
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo startPaused
      picker={{ actions: [], submit: { label: "Add event", kind: "trake", onSubmit: vi.fn() },
        trakeSequence: { videoId: "K01_V001", frameIds: [3072], onRemove: vi.fn(), onMove: vi.fn(), onOpen: vi.fn(), onSubmit: vi.fn() } }} />);

    expect(await screen.findByText("Source frame 1")).toBeInTheDocument();
    expect(screen.getByText("1 TRAKE events selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "E1 · #3072" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add E2 · #1 · 0:10" })).toBeEnabled();
  });

  it("submits a paused video even when the decoder never names the frame", async () => {
    // A video opened from a keyframe is paused on a frame the decoder may never name exactly (the
    // seek lands between two PTS ticks). The button used to stay dead until a play/pause cycle
    // flushed a callback; the frame the playhead resolves to is enough.
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => Promise.resolve(String(input).startsWith("/frame-map/")
      ? jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
      })
      : jsonResponse({ videoId: "K01_V001", cues: [] })));
    const onSubmit = vi.fn().mockResolvedValue({ message: "accepted", completed: true });
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo startPaused
      picker={{ actions: [], submit: { label: "Submit Q&A frame", kind: "qa", needsAnswer: true, onSubmit } }} />);
    fireEvent.change(await screen.findByLabelText("DRES answer in player"), { target: { value: "5" } });

    const submit = await screen.findByRole("button", { name: "Submit 0:10 (10000 ms)" });
    expect(submit).toBeEnabled();
    fireEvent.click(submit);
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith("K01_V001", 1, 10, "5", "qa"));
  });

  it("retries a dropped frame-map request instead of opening a dead player", async () => {
    // A venue link drops connections: one failed map request used to leave the dialog unable to
    // name any frame, with every control dead until it was reopened.
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value() { return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    let mapCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path.startsWith("/frame-map/")) {
        mapCalls += 1;
        if (mapCalls === 1) return Promise.reject(new Error("dropped"));
        return Promise.resolve(jsonResponse({
          videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
          firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
        }));
      }
      return Promise.resolve(jsonResponse({ videoId: "K01_V001", cues: [] }));
    });
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo startPaused
      picker={{ actions: [], submit: { label: "Submit Q&A frame", kind: "qa", needsAnswer: true, onSubmit: vi.fn() } }} />);

    expect(await screen.findByRole("button", { name: "Submit 0:10 (10000 ms)" }, { timeout: 3000 })).toBeInTheDocument();
    expect(mapCalls).toBe(2);
  });

  it("says why a Q&A submission is held back until the answer is typed", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => Promise.resolve(String(input).startsWith("/frame-map/")
      ? jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 0, ptsDeltas: [10000, 2500], frameCount: 3,
      })
      : jsonResponse({ videoId: "K01_V001", cues: [] })));
    render(<VideoDialog result={hit({ time: 10 })} onClose={() => {}} originalVideo startPaused
      picker={{ actions: [], submit: { label: "Submit Q&A frame", kind: "qa", needsAnswer: true, onSubmit: vi.fn() } }} />);
    await waitFor(() => expect(callback).toBeDefined());
    await act(async () => callback?.(0, { mediaTime: 10 }));

    expect(await screen.findByText("Type the answer to submit it.")).toBeInTheDocument();
    const submit = screen.getByRole("button", { name: "Submit 0:10 (10000 ms)" });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByLabelText("DRES answer in player"), { target: { value: "5 người" } });

    expect(screen.queryByText("Type the answer to submit it.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Submit 0:10 (10000 ms)" })).toBeEnabled();
  });

  it("steps by source frames and sends the frame the stepper names", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", { configurable: true, value: vi.fn() });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => Promise.resolve(String(input).startsWith("/frame-map/")
      ? jsonResponse({
        videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
        firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
      })
      : jsonResponse({ videoId: "K01_V001", cues: [] })));
    const onSubmit = vi.fn().mockResolvedValue({ message: "Added E1 · FRAME_ID 2", completed: false });
    render(<VideoDialog result={hit({ time: 0.04 })} onClose={() => {}} originalVideo startPaused
      picker={{ actions: [], submit: { label: "Add event", kind: "trake", onSubmit },
        trakeSequence: { videoId: null, frameIds: [], onRemove: vi.fn(), onMove: vi.fn(), onOpen: vi.fn(), onSubmit: vi.fn() } }} />);
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    await waitFor(() => expect(callback).toBeDefined());
    await act(async () => callback?.(0, { mediaTime: 0.04 }));

    expect(await screen.findByText("Source frame 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Forward 1 frame" }));
    expect(video.currentTime).toBeCloseTo(0.1, 6);
    fireEvent.seeked(video);
    await act(async () => callback?.(0, { mediaTime: 0.1 }));

    expect(await screen.findByText("Source frame 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add E1 · #2 · 0:00" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith("K01_V001", 2, 0.1, "", "trake"));
  });

  it("uses the original video source for DRES exact-frame picking", () => {
    render(<VideoDialog result={hit({})} onClose={() => {}} originalVideo
      picker={{ actions: [{ label: "Use frame for DRES", onPick: vi.fn() }] }} />);
    expect(screen.getByLabelText("Video player")).toHaveAttribute("src", "/video/K01_V001?original=1");
  });

  it("highlights the transcript cue at the video's current time", async () => {
    const scrollIntoView = vi.fn();
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true, value: scrollIntoView,
    });
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true, value: scrollTo,
    });
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.startsWith("/transcript/")) return Promise.resolve(jsonResponse({
        videoId: "K01_V001",
        cues: [
          { start: 0, end: 2, text: "Mở đầu." },
          { start: 2, end: 5, text: "Nội dung hiện tại." },
          { start: 5, end: 8, text: "Nội dung tiếp theo." },
          { start: 8, end: 10, text: "Dòng thứ tư." },
          { start: 10, end: 12, text: "Dòng cuối." },
        ],
      }));
      return Promise.resolve(jsonResponse({}, false));
    });

    render(<VideoDialog result={hit({ time: 3 })} onClose={() => {}} />);
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 3 });
    fireEvent.timeUpdate(video);

    const current = await screen.findByText("Nội dung hiện tại.");
    expect(current.closest("li")).toHaveClass("active");
    await waitFor(() => expect(scrollTo).toHaveBeenCalled());
    expect(scrollIntoView).not.toHaveBeenCalled();
    scrollTo.mockClear();
    video.currentTime = 6;
    fireEvent.timeUpdate(video);
    expect(screen.getByText("Nội dung tiếp theo.").closest("li")).toHaveClass("active");
    expect(current.closest("li")).not.toHaveClass("active");
    expect(scrollTo).toHaveBeenCalled();
    expect(scrollIntoView).not.toHaveBeenCalled();
    scrollTo.mockClear();
    fireEvent.wheel(screen.getByRole("complementary", { name: "Video transcript" }));
    video.currentTime = 9;
    fireEvent.timeUpdate(video);
    expect(scrollTo).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Follow video" }));
    expect(scrollTo).toHaveBeenCalled();
    const transcript = screen.getByRole("complementary", { name: "Video transcript" });
    fireEvent.keyDown(transcript, { key: " " });
    fireEvent.click(screen.getByRole("button", { name: "Follow video" }));
    transcript.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab" });
    expect(screen.getByRole("button", { name: "Close video" })).toHaveFocus();
    video.currentTime = 20;
    fireEvent.timeUpdate(video);
    expect(screen.getByText("Dòng cuối.")).toBeInTheDocument();
    expect(screen.getByText("Mở đầu.")).toBeInTheDocument();
  });

  it("keeps the video usable when transcript loading fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "unavailable" }, false));

    render(<VideoDialog result={hit({})} onClose={() => {}} />);

    expect(screen.getByLabelText("Video player")).toBeInTheDocument();
    expect(await screen.findByText("Failed to load transcript.")).toBeInTheDocument();
  });

  it("aborts the media request when the video dialog closes", () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ videoId: "K01_V001", cues: [] }));
    const pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    const load = vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => {});
    const { unmount } = render(<VideoDialog result={hit({})} onClose={() => {}} />);
    const video = screen.getByLabelText("Video player");

    unmount();

    expect(pause).toHaveBeenCalled();
    expect(video).not.toHaveAttribute("src");
    expect(load).toHaveBeenCalled();
  });

  it("captures an exact frame only after the video is paused", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
      configurable: true, value: vi.fn(),
    });
    mockAuthenticatedFetch(jsonResponse({
      videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
    }));
    const onCapture = vi.fn(() => 0);
    render(<VideoDialog result={hit({ time: 0.04 })} onClose={() => {}} capture={{
      query: dialogQuery, answers: [], active: true, onCapture,
    }} />);
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0.04 });
    const useButton = await screen.findByRole("button", { name: "Use frame 1" });
    expect(useButton).toBeDisabled();
    fireEvent.pause(screen.getByLabelText("Video player"));
    fireEvent.click(useButton);
    expect(onCapture).toHaveBeenCalledWith(1, -1, 0);
  });

  it("steps by source frames and clamps at the end of the frame map", async () => {
    let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined;
    Object.defineProperty(HTMLVideoElement.prototype, "requestVideoFrameCallback", {
      configurable: true, value(fn: typeof callback) { callback = fn; return 1; },
    });
    Object.defineProperty(HTMLVideoElement.prototype, "cancelVideoFrameCallback", {
      configurable: true, value: vi.fn(),
    });
    mockAuthenticatedFetch(jsonResponse({
      videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 100, ptsDeltas: [40, 60], frameCount: 3,
    }));
    render(<VideoDialog result={hit({ time: 0.04 })} onClose={() => {}} capture={{
      query: dialogQuery, answers: [], active: true, onCapture: vi.fn(() => 0),
    }} />);
    await waitFor(() => expect(callback).toBeDefined());
    callback?.(0, { mediaTime: 0.04 });
    const video = screen.getByLabelText("Video player") as HTMLVideoElement;
    fireEvent.pause(video);
    fireEvent.click(screen.getByRole("button", { name: "Forward 10 frames" }));
    expect(video.currentTime).toBe(0.1);
    expect(screen.getByRole("button", { name: "Forward 1 frame" })).toBeDisabled();
    await act(async () => callback?.(0, { mediaTime: 0.04 }));
    expect(screen.getByRole("button", { name: "Use frame 1" })).toBeDisabled();
    fireEvent.seeked(video);
    expect(screen.getByRole("button", { name: "Use frame 1" })).toBeDisabled();
    await act(async () => callback?.(0, { mediaTime: 0.04 }));
    expect(screen.getByRole("button", { name: "Use frame 1" })).toBeDisabled();
    await act(async () => callback?.(0, { mediaTime: 0.1 }));
    expect(await screen.findByRole("button", { name: "Use frame 2" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Forward 1 frame" }));
    expect(screen.getByRole("button", { name: "Use frame 2" })).toBeEnabled();
  });

  it("resets a selected destination when answer rows are replaced", async () => {
    mockAuthenticatedFetch(jsonResponse({
      videoId: "K01_V001", timeBaseNumerator: 1, timeBaseDenominator: 1000,
      firstPts: 0, ptsDeltas: [40], frameCount: 2,
    }));
    const firstAnswers = [
      { video_id: "K01_V001", frame_id: 1 },
      { video_id: "K01_V002", frame_id: 2 },
    ];
    const { rerender } = render(<VideoDialog result={hit({})} onClose={() => {}} capture={{
      query: dialogQuery, answers: firstAnswers, active: true, onCapture: vi.fn(() => 0),
    }} />);
    fireEvent.change(screen.getByLabelText("Answer row"), { target: { value: "1" } });
    expect(screen.getByLabelText("Answer row")).toHaveValue("1");
    rerender(<VideoDialog result={hit({})} onClose={() => {}} capture={{
      query: dialogQuery, answers: [...firstAnswers].reverse(), active: true, onCapture: vi.fn(() => 0),
    }} />);
    expect(screen.getByLabelText("Answer row")).toHaveValue("-1");
  });

  it("keeps verification visible while switching search modes", () => {
    render(<App />);

    expect(screen.queryByLabelText("Visual index")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));
    expect(screen.queryByLabelText("Visual index")).not.toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Transcript evidence" })).toBeInTheDocument();
  });

  it("shows ranking provenance inside the video dialog", () => {
    render(<VideoDialog result={hit({ source: "both", rank: 2, transcript_snippet: "fusion passage",
      transcript_start: 10, transcript_end: 20, scores: { pe_core_rank: 4, audio_rank: 3, semantic_rank: 3 } })} onClose={() => {}} />);

    expect(screen.getByRole("region", { name: "Ranking evidence" })).toHaveTextContent("Passage used in hybrid fusion");
    expect(screen.getByRole("region", { name: "Ranking evidence" })).toHaveTextContent("fusion passage");
    expect(screen.getByRole("region", { name: "Ranking evidence" })).toHaveTextContent("Hybrid #3 · PE-Core #4 · Audio #3 · Semantic #3");
  });

  it("shows verification ranks and audio evidence for the selected keyframe", async () => {
    mockAuthenticatedFetch(jsonResponse(response([
      hit({ source: "both", rank: 4, transcript_snippet: "matched passage", audio_method: "keyword",
        matched_keywords: ["mưa"], scores: { pe_core_rank: 3, visual_rank: 3, audio_rank: 2, keyword_rank: 2 } }),
    ])));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "rain" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: /Select frame/ }));

    expect(screen.getByText("Hybrid rank").nextSibling).toHaveTextContent("#5");
    expect(screen.getByText("PE-Core").nextSibling).toHaveTextContent("#3");
    expect(screen.getByText("Transcript match")).toBeInTheDocument();
    expect(screen.getByText("mưa")).toBeInTheDocument();
  });

  it("does not carry a ranked frame's transcript evidence to a neighbouring context frame", async () => {
    mockAuthenticatedFetch(
      jsonResponse(response([hit({ source: "both", transcript_snippet: "ranked passage", scores: { semantic_rank: 1 } })])),
      jsonResponse({ video_id: "K01_V001", frame_idx: 11, frames: [{ video_id: "K01_V001", frame_idx: 11, time: 11, thumb: "/thumb/K01_V001/11", frame: "/frame/K01_V001/11", video: "/video/K01_V001" }] }),
    );
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "rain" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("ranked passage");
    fireEvent.click(await screen.findByRole("button", { name: "Select frame 11" }));

    expect(screen.getByText("No transcript match for this result.")).toBeInTheDocument();
    expect(screen.queryByText("ranked passage")).not.toBeInTheDocument();
  });

  it("updates transcript evidence when a result is selected", async () => {
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

    const frames = await screen.findAllByRole("button", { name: /Select frame/ });
    expect(frames).toHaveLength(3);
    fireEvent.click(frames[1]);
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

    expect(await screen.findByText("No transcript match for this result.")).toBeInTheDocument();
  });

  it("reports API errors without removing the previous results", async () => {
    const fetchMock = mockAuthenticatedFetch(
      jsonResponse(response([hit({ video_id: "kept" })])),
      jsonResponse({ video_id: "kept", frame_idx: 10, frames: [] }),
      jsonResponse({ detail: "encoder unavailable" }, false),
    );

    render(<App />);
    const input = screen.getByLabelText("Visual query (English)");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("encoder unavailable");
    expect(screen.getByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/search")).toHaveLength(2);
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

  it("automatically appends the next result page near the scroll boundary", async () => {
    let observeCallback: IntersectionObserverCallback | undefined;
    class FakeIntersectionObserver {
      constructor(callback: IntersectionObserverCallback) { observeCallback = callback; }
      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() { return []; }
      readonly root = null;
      readonly rootMargin = "0px";
      readonly thresholds = [0];
    }
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    const first = {
      ...response([hit({ video_id: "first", rank: 0 })], "visual"),
      next_cursor: "page-2", has_more: true, total_results: 2, truncated: false,
    };
    const second = {
      ...response([hit({ video_id: "second", rank: 1 })], "visual"),
      next_cursor: null, has_more: false, total_results: 2, truncated: false,
    };
    const fetchMock = mockAuthenticatedFetch(
      jsonResponse(first),
      jsonResponse({ video_id: "first", frame_idx: 10, frames: [] }),
      jsonResponse(second),
    );

    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "city street" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("button", { name: /Select frame first/ })).toBeInTheDocument();
    expect(observeCallback).toBeDefined();

    await act(async () => observeCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver));

    expect(await screen.findByRole("button", { name: /Select frame second/ })).toBeInTheDocument();
    const nextCall = fetchMock.mock.calls.find(([input]) => String(input) === "/search/next");
    expect(JSON.parse(nextCall?.[1]?.body as string)).toEqual({ cursor: "page-2" });
  });

  it("keeps loaded results and retries a failed next page", async () => {
    let observeCallback: IntersectionObserverCallback | undefined;
    class FakeIntersectionObserver {
      constructor(callback: IntersectionObserverCallback) { observeCallback = callback; }
      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() { return []; }
      readonly root = null;
      readonly rootMargin = "0px";
      readonly thresholds = [0];
    }
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    const first = {
      ...response([hit({ video_id: "kept", rank: 0 })], "visual"),
      next_cursor: "retry-page", has_more: true, total_results: 2, truncated: false,
    };
    const second = {
      ...response([hit({ video_id: "recovered", rank: 1 })], "visual"),
      next_cursor: null, has_more: false, total_results: 2, truncated: false,
    };
    mockAuthenticatedFetch(
      jsonResponse(first),
      jsonResponse({ video_id: "kept", frame_idx: 10, frames: [] }),
      jsonResponse({ detail: "temporary paging failure" }, false),
      jsonResponse(second),
    );

    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "night market" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();
    await act(async () => observeCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver));

    expect(await screen.findByRole("alert")).toHaveTextContent("temporary paging failure");
    expect(screen.getByRole("button", { name: /Select frame kept/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: /Select frame recovered/ })).toBeInTheDocument();
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

describe("DeepSeek query refinement", () => {
  const refinement = {
    summary_vi: "tìm cảnh người đàn ông đội mũ vàng",
    visual: [
      { text: "a man wearing a yellow helmet", tokens: 6, primary: true },
      { text: "a man standing near a construction site", tokens: 7, primary: false },
    ],
    transcript: ["người đàn ông đội mũ vàng"],
    ocr: ["CÔNG TRƯỜNG"],
    events: [],
    lookups: [],
  };

  it("shows suggestions from /search/refine and only fills the box a chip is applied to", async () => {
    const fetchMock = mockAuthenticatedFetch(jsonResponse(refinement));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), {
      target: { value: "một người đàn ông đội mũ vàng" },
    });
    fireEvent.click(screen.getByRole("button", { name: "✨ Refine with AI" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search/refine")).toBe(true));
    const call = fetchMock.mock.calls.find(([input]) => String(input) === "/search/refine");
    expect(JSON.parse(call?.[1]?.body as string)).toEqual({ text: "một người đàn ông đội mũ vàng", kind: "kis" });

    expect(await screen.findByText("tìm cảnh người đàn ông đội mũ vàng")).toBeInTheDocument();
    const transcriptChip = screen.getByRole("button", { name: "người đàn ông đội mũ vàng" });

    // Applying a suggestion only fills the box; it never searches on its own.
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(false);

    fireEvent.click(transcriptChip);

    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("người đàn ông đội mũ vàng");
    expect(screen.getByRole("button", { name: "Transcript for event E1" })).toHaveAttribute("aria-pressed", "true");
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/search")).toBe(false);

    // Switching back to the visual tab must show the original visual text, not the
    // transcript suggestion that was just applied -- each channel keeps its own text.
    fireEvent.click(screen.getByRole("button", { name: "Visual text for event E1" }));
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("một người đàn ông đội mũ vàng");

    // And applying the visual suggestion afterwards must not touch the transcript text.
    fireEvent.click(screen.getByRole("button", { name: "a man wearing a yellow helmet" }));
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("a man wearing a yellow helmet");
    fireEvent.click(screen.getByRole("button", { name: "Transcript for event E1" }));
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("người đàn ông đội mũ vàng");
  });

  it("separates a scene's best-match captions from its less distinctive ones", async () => {
    // A secondary caption describes a real but generic moment of the same clip: it
    // still applies fine, it just must never look as trustworthy as the best match.
    mockAuthenticatedFetch(jsonResponse(refinement));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "✨ Refine with AI" }));

    const bestMatch = await screen.findByRole("button", { name: "a man wearing a yellow helmet" });
    const secondary = screen.getByRole("button", { name: "a man standing near a construction site" });
    expect(bestMatch.className).not.toContain("refine-chip-muted");
    expect(secondary.className).toContain("refine-chip-muted");
    expect(screen.getByText("Other moments in this clip")).toBeInTheDocument();
  });

  it("warns on a visual suggestion over PE-Core's 72-token limit and reports a failed refine", async () => {
    const longRefinement = {
      ...refinement,
      visual: [{ text: "an extremely long caption", tokens: 90, primary: true }],
    };
    mockAuthenticatedFetch(jsonResponse(longRefinement));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "cảnh dài" } });
    fireEvent.click(screen.getByRole("button", { name: "✨ Refine with AI" }));

    const chip = await screen.findByRole("button", { name: "an extremely long caption" });
    expect(chip).toHaveAttribute("title", expect.stringContaining("72-token limit"));
  });

  it("reports a failed refine without touching the query boxes", async () => {
    mockAuthenticatedFetch(jsonResponse({ detail: "query refinement unavailable: down" }, false));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "cảnh lỗi" } });
    fireEvent.click(screen.getByRole("button", { name: "✨ Refine with AI" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("query refinement unavailable: down");
    expect(screen.getByLabelText("Visual query (English)")).toHaveValue("cảnh lỗi");
  });
});

describe("checking a moment with AI", () => {
  // The browser-side cache is module-level on purpose -- it lives as long as the page does --
  // so each test starts from an empty one.
  beforeEach(() => forgetChecks());

  const CHECK = {
    key: "k1", video_id: "L01_V001", frame_idx: 2, time: 123.456, window_seconds: 300,
    shot_index: 7, covers: { start_frame: 0, end_frame: 40, start: 120, end: 128 },
    verdict: { score: 9, reason: "đúng cảnh" },
    details: [{ text: "hai người cầm điện thoại", meaning_en: "two people holding a phone", distinctiveness: 0.9, weight: 0.92 }],
    checks: [{ detail: 0, status: "supported" as const, source: "vision" as const, at: 123.456, quote: "", note: "thấy rõ" }],
    support: { share: 1, supported: 1, partial: 0, contradicted: 0, total: 1 },
    answer: null, frames_seen: 21, model: "qwen/qwen3.7-flash", by: "owner@example.com",
    checked_at: 0, cached: false, latency_ms: 6000,
  };
  const searchHit = hit({
    video_id: "L01_V001", frame_idx: 2, time: 123.456, thumb: "/thumb/L01_V001/2",
    frame: "/frame/L01_V001/2", video: "/video/L01_V001",
  });

  function mockCheckFetch(available: boolean, onCheck?: () => Response) {
    return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const path = String(input);
      if (path === "/session") return Promise.resolve(jsonResponse({
        login: "owner@example.com", is_admin: true,
        check: { available, model: "qwen/qwen3.7-flash", window_seconds: 300 },
      }));
      if (path === "/api/collab/rounds") return Promise.resolve(jsonResponse([]));
      if (path === "/search") return Promise.resolve(jsonResponse(response([searchHit])));
      if (path.startsWith("/context/")) return Promise.resolve(jsonResponse({
        video_id: "L01_V001", frame_idx: 2,
        frames: [
          { video_id: "L01_V001", frame_idx: 2, pts: 0, time: 123.456, thumb: "/thumb/L01_V001/2", frame: "/frame/L01_V001/2", video: "/video/L01_V001" },
          { video_id: "L01_V001", frame_idx: 30, pts: 0, time: 125.0, thumb: "/thumb/L01_V001/30", frame: "/frame/L01_V001/30", video: "/video/L01_V001" },
        ],
      }));
      if (path.startsWith("/api/check/lookup")) return Promise.resolve(jsonResponse({ status: "miss", result: null }));
      if (path === "/api/check") return Promise.resolve(onCheck ? onCheck() : jsonResponse(CHECK));
      if (path === "/api/dres/status") return Promise.resolve(jsonResponse({
        configured: false, credentials_configured: false, host: "", connected: false,
        evaluations: [], evaluation_id: null, context: null, context_valid: false,
      }));
      if (path === "/api/dres/history") return Promise.resolve(jsonResponse({ attempts: [] }));
      return Promise.reject(new Error(`Unexpected request: ${path}`));
    });
  }

  async function searchAndSelect() {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "two people with a phone" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: "Select frame L01_V001 at 2:03" }));
  }

  it("offers the check beside Play video and shows the verdict in the right panel", async () => {
    const fetchMock = mockCheckFetch(true);
    await searchAndSelect();

    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));

    const panel = screen.getByRole("complementary", { name: "Transcript evidence" });
    expect(await within(panel).findByText("9/10")).toBeInTheDocument();
    expect(within(panel).getByText("đúng cảnh")).toBeInTheDocument();
    expect(within(panel).getByText("hai người cầm điện thoại", { exact: false })).toBeInTheDocument();
    const posted = fetchMock.mock.calls.find(([input]) => String(input) === "/api/check");
    expect(JSON.parse(String(posted?.[1]?.body))).toMatchObject({
      visual_text: "two people with a phone", kind: "kis", video_id: "L01_V001", frame_idx: 2,
    });
  });

  it("hides the button when the deployment has no OpenRouter key", async () => {
    mockCheckFetch(false);
    await searchAndSelect();

    expect(screen.queryByRole("button", { name: "✓ Check by AI" })).not.toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "▶ Play video" })).toBeInTheDocument();
  });

  it("reads another frame of the same shot from the cache instead of asking again", async () => {
    // The server keys a check by its shot, so the second frame is already answered.
    const fetchMock = mockCheckFetch(true);
    await searchAndSelect();
    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));
    await screen.findByText("9/10");

    fireEvent.click(screen.getByRole("button", { name: /^Select frame 30/ }));

    expect(await screen.findByText("9/10")).toBeInTheDocument();
    expect(screen.getByText("from the cache")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/api/check")).toHaveLength(1);
  });

  it("badges the filmstrip frames one answer already speaks for", async () => {
    mockCheckFetch(true);
    await searchAndSelect();
    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));
    await screen.findByText("9/10");

    expect(await screen.findByRole("button", { name: /^Select frame 30 · checked$/ })).toBeInTheDocument();
  });

  it("keeps checking the query that found these results after switching the box to an empty field", async () => {
    // Reported bug: typed a visual query, searched, then tapped the "Transcript" tab for
    // event E1 (nothing typed there yet) -- the shared textbox goes empty, but the results
    // grid still shows the VISUAL query's matches, so the check must keep asking about that,
    // not about the now-empty box.
    const fetchMock = mockCheckFetch(true);
    await searchAndSelect();

    fireEvent.click(screen.getByRole("button", { name: "Transcript for event E1" }));
    expect(screen.getByLabelText("Transcript query (Vietnamese)")).toHaveValue("");

    fireEvent.click(screen.getByRole("button", { name: "✓ Check by AI" }));

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    const posted = await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input]) => String(input) === "/api/check");
      expect(call).toBeDefined();
      return call!;
    });
    expect(JSON.parse(String(posted[1]?.body))).toMatchObject({
      visual_text: "two people with a phone", kind: "kis",
    });
  });

  it("reports a failed check without clearing the selection", async () => {
    mockCheckFetch(true, () => jsonResponse({ detail: "the AI check failed: upstream is down" }, false));
    await searchAndSelect();

    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("upstream is down");
    expect(screen.getByText("Selected frame 2")).toBeInTheDocument();
  });

  it("checks against the transcript, not a scene description, when the search was audio", async () => {
    // A spoken-word query is words to find near the moment, not something a frame shows --
    // sending it as "visual_text" would have the checker judge it as a scene description.
    const fetchMock = mockCheckFetch(true);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Transcript for event E1" }));
    fireEvent.change(screen.getByLabelText("Transcript query (Vietnamese)"), { target: { value: "cá nhám" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: "Select frame L01_V001 at 2:03" }));

    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));

    const posted = await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input]) => String(input) === "/api/check");
      expect(call).toBeDefined();
      return call!;
    });
    expect(JSON.parse(String(posted[1]?.body))).toMatchObject({ audio_text: "cá nhám" });
    expect(JSON.parse(String(posted[1]?.body)).visual_text).toBeUndefined();
  });

  it("checks against the on-screen text, not a scene description, when the search was OCR", async () => {
    const fetchMock = mockCheckFetch(true);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "OCR for event E1" }));
    fireEvent.change(screen.getByLabelText("OCR query"), { target: { value: "Nà Ní" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.click(await screen.findByRole("button", { name: "Select frame L01_V001 at 2:03" }));

    fireEvent.click(await screen.findByRole("button", { name: "✓ Check by AI" }));

    const posted = await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input]) => String(input) === "/api/check");
      expect(call).toBeDefined();
      return call!;
    });
    expect(JSON.parse(String(posted[1]?.body))).toMatchObject({ ocr_text: "Nà Ní" });
    expect(JSON.parse(String(posted[1]?.body)).visual_text).toBeUndefined();
  });
});
