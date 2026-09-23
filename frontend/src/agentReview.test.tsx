import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { captionsAround, clearEvidenceCache, isFurniture, ocrAround, speechAround } from "./agentEvidence";
import { AgentPanel } from "./agentPanel";
import type { AgentRunResponse } from "./types";

function jsonResponse(data: unknown, init: Partial<Response> = {}): Response {
  return { ok: true, status: 200, json: async () => data, headers: new Headers(), ...init } as Response;
}

const KIS: AgentRunResponse = {
  run_id: "abc", status: "running", query_id: "abc", kind: "kis", query: "thuyền đỏ trên sông Hương",
  elapsed: 4, result: null, usage: null, error: null,
  review: { decisions: {}, scores: {}, answer: null, stopped: false, decided: false, judge: "human", finished: false, sequence: null, pending_commands: 0 },
  progress: {
    stage: "awaiting_review", step: 0, elapsed: 4, answer: null, sequences: [],
    total_candidates: 120, details: ["thuyền màu đỏ", "trên sông Hương", "có hai người"], summarizing: false,
    lookups: [{ question: "sông Hương ở đâu", answer: "Huế", error: null, sources: [] }],
    plan: {
      summary_vi: "tìm cảnh", scenes: ["thuyền đỏ trên sông"], target_scene: 0, question_vi: null, events: [],
      keywords: { vi: ["thuyền đỏ", "sông Hương"], en: ["a red boat on a river", "a woman in a white apron beside a vase"] },
    },
    candidates: [{
      id: "C3", video_id: "L21_V010", frame_idx: 2000, time: 80, judging: false, verdict: null, sources: ["visual", "audio"],
      context: {}, channels: [{ kind: "visual", source: "r1.v0", rank: 3 }, { kind: "audio", source: "r1.a0", rank: 1 }],
      fused: 0.03, pinned: false,
      summary: { summary: "Lời thoại nhắc thuyền đỏ.", checks: [
        { detail: 0, status: "supported", source: "speech", quote: "thuyền đỏ" },
        { detail: 1, status: "supported", source: "ocr", quote: "SÔNG HƯƠNG" },
        { detail: 2, status: "not_mentioned", source: null, quote: "" },
      ] },
    }, {
      id: "C7", video_id: "L22_V003", frame_idx: 10912, time: 436.5, judging: false, verdict: null, sources: ["visual"],
      context: {}, channels: [{ kind: "visual", source: "r1.v0", rank: 9 }], fused: 0.01, pinned: false, summary: null,
    }],
  },
};

const TRAKE: AgentRunResponse = {
  ...KIS, kind: "trake", query: "E1: mở cửa\nE2: bước ra",
  progress: { ...KIS.progress!, plan: { ...KIS.progress!.plan!, events: ["opens a door", "steps out"] },
    sequences: [{ video_id: "L21_V010", frames: [2000, 2100], event_scores: null, refined: false, source: "local", times: [80, 84] }] },
};

const QA: AgentRunResponse = {
  ...KIS, kind: "qa",
  progress: { ...KIS.progress!, suggested_answers: [{ answer: "hai muỗng", source: "speech", quote: "hai muỗng nước tương", cid: "C3", confidence: 8 }] },
};

interface Server { bodies: unknown[] }

function serve(run: AgentRunResponse, overrides: Record<string, () => Response> = {}): Server {
  const bodies: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    for (const [prefix, respond] of Object.entries(overrides)) if (url.startsWith(prefix)) return respond();
    if (url === "/api/agent/runs" && method === "POST") { bodies.push(JSON.parse(String(init?.body))); return jsonResponse(run, { status: 202 }); }
    if (url.endsWith("/review")) { bodies.push(JSON.parse(String(init?.body))); return jsonResponse(run); }
    if (url === "/api/agent/runs/abc") return jsonResponse(run);
    if (url.startsWith("/transcript/")) return jsonResponse({ videoId: "x", cues: [
      { start: 10, end: 20, text: "xin chào các bạn" },
      { start: 75, end: 85, text: "chiếc thuyền đỏ đang xuôi dòng" },
      { start: 300, end: 310, text: "quá xa để hiện" },
    ] });
    if (url.startsWith("/ocr/")) return jsonResponse({ videoId: "x", shots: [
      { shot_idx: 4, frame_idx: 1900, start: 60, end: 100, time: 80, text: "SÔNG HƯƠNG - HUẾ" },
    ] });
    if (url.startsWith("/captions/")) return jsonResponse({ videoId: "x", captions: [
      { shot_idx: 2, window_idx: 0, frame_idx: null, start: 40, end: 50, time: 45, text: "The video opens on a plain black screen with a logo." },
      { shot_idx: 3, window_idx: 0, frame_idx: null, start: 50, end: 60, time: 55, text: "A woman wearing a white apron stands beside a vase of red ginger flowers. She smiles." },
      { shot_idx: 4, window_idx: 0, frame_idx: null, start: 60, end: 100, time: 80, text: "A red boat drifts down a river." },
      { shot_idx: 9, window_idx: 0, frame_idx: null, start: 600, end: 610, time: 605, text: "A far away shot." },
    ] });
    if (url.startsWith("/context/")) return jsonResponse({ video_id: "L21_V010", frame_idx: 2000, frames: [
      { video_id: "L21_V010", frame_idx: 1950, time: 78, thumb: "/thumb/L21_V010/1950", frame: "", video: "" },
      { video_id: "L21_V010", frame_idx: 2000, time: 80, thumb: "/thumb/L21_V010/2000", frame: "", video: "" },
      { video_id: "L21_V010", frame_idx: 2050, time: 82, thumb: "/thumb/L21_V010/2050", frame: "", video: "" },
    ] });
    return jsonResponse({ detail: "not found" }, { ok: false, status: 404 });
  });
  return { bodies };
}

async function startHuman(run: AgentRunResponse = KIS, overrides: Record<string, () => Response> = {}) {
  const server = serve(run, overrides);
  render(<AgentPanel />);
  fireEvent.change(screen.getByLabelText("DeepSeek API key"), { target: { value: "dsk" } });
  fireEvent.change(screen.getByLabelText("Agent judge"), { target: { value: "human" } });
  fireEvent.change(screen.getByLabelText("Agent query"), { target: { value: run.query } });
  fireEvent.click(screen.getByRole("button", { name: "Run agent" }));
  await screen.findByLabelText("Human review");
  return server;
}

const sent = (server: Server) => server.bodies.slice(1);

afterEach(() => { cleanup(); clearEvidenceCache(); vi.restoreAllMocks(); });

describe("human review workspace", () => {
  it("starts a run judged by the person, with no Gemini key, and takes over the page", async () => {
    const server = await startHuman();

    expect(server.bodies[0]).toMatchObject({ judge: "human", gemini_api_key: "" });
    expect(screen.queryByLabelText("Agent query")).not.toBeInTheDocument();
    expect(screen.getByText("Waiting for your review")).toBeInTheDocument();
    expect(screen.getByText(/120 candidates found, 2 shown/)).toBeInTheDocument();
    expect(within(screen.getByLabelText("Query details")).getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByText("Web notes (1)")).toBeInTheDocument();
  });

  it("shows on each card which channels found it, and how many details its words support", async () => {
    await startHuman();
    const card = screen.getByLabelText("Candidate C3");

    expect(within(card).getByText("V3")).toBeInTheDocument();
    expect(within(card).getByText("A1")).toBeInTheDocument();
    expect(within(card).getByText("2/3 details")).toBeInTheDocument();
    expect(within(screen.getByLabelText("Candidate C7")).queryByText(/details/)).not.toBeInTheDocument();
  });

  it("reads the words around the selected candidate with the query's words lit up", async () => {
    await startHuman();

    const speech = await screen.findByLabelText("Speech around this moment");
    await within(speech).findByText("thuyền đỏ");
    expect(within(speech).getByText("thuyền đỏ").tagName).toBe("MARK");
    expect(within(speech).queryByText(/quá xa/)).not.toBeInTheDocument();  // outside ±30 s
    const ocr = screen.getByLabelText("Text on screen");
    expect(await within(ocr).findByText("SÔNG HƯƠNG")).toHaveClass("hit-phrase");
    await waitFor(() => expect(screen.getByLabelText("Shot captions")).toHaveTextContent("A red boat drifts down a river."));
    const checks = within(screen.getByLabelText("Query details checked"));
    expect(checks.getByText(/Lời thoại nhắc thuyền đỏ/)).toBeInTheDocument();
    expect(checks.getAllByLabelText("supported")).toHaveLength(2);
    expect(checks.getByLabelText("not mentioned")).toBeInTheDocument();
  });

  it("shows the captions of the shots before the frame too, read from the windows route", async () => {
    // A query describes consecutive scenes and a candidate is one frame of one of them: the
    // apron and the vase are in the shot before the frame that shows the boat. The panel
    // used to show only the shot holding the frame, and asked the route that needs the index.
    const server = await startHuman();

    const captions = await screen.findByLabelText("Shot captions");
    await waitFor(() => expect(captions).toHaveTextContent("white apron"));
    expect(captions).toHaveTextContent("A red boat drifts down a river.");
    expect(captions).not.toHaveTextContent("far away shot");  // 500 s away
    expect(captions).not.toHaveTextContent("plain black screen");  // matches nothing: folded away
    expect(within(captions).getByText(/this shot/)).toBeInTheDocument();
    const urls = (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls.map((call) => String(call[0]));
    expect(urls).toContain("/captions/L21_V010/windows");
    expect(urls).not.toContain("/captions/L21_V010");
    expect(server.bodies.length).toBe(1);
  });

  it("folds the shots that match nothing behind one button", async () => {
    await startHuman();
    const captions = await screen.findByLabelText("Shot captions");
    await waitFor(() => expect(captions).toHaveTextContent("white apron"));

    fireEvent.click(within(captions).getByRole("button", { name: "Show 1 other shot around this frame" }));
    expect(captions).toHaveTextContent("plain black screen");
    fireEvent.click(within(captions).getByRole("button", { name: "Hide shots that match nothing" }));
    expect(captions).not.toHaveTextContent("plain black screen");
  });

  it("lights the words of a caption and opens the whole thing on request", async () => {
    await startHuman();
    const captions = await screen.findByLabelText("Shot captions");
    await waitFor(() => expect(within(captions).getAllByRole("button", { name: /Show full caption/ })).toHaveLength(1));

    expect(captions.querySelectorAll("mark").length).toBeGreaterThan(0);
    expect(captions).not.toHaveTextContent("She smiles.");  // the sentence that matches nothing stays folded
    fireEvent.click(within(captions).getByRole("button", { name: /Show full caption/ }));
    expect(captions).toHaveTextContent("She smiles.");
  });

  it("marks a detail the words only half say as close, with the reason, and never as a match", async () => {
    // Reported: a plate of two shrimp and a vase of pink flowers were ticked for "4 con X" and
    // "hoa riềng tía". Those are partial: the right subject, but not the count or the kind.
    await startHuman({ ...KIS, progress: { ...KIS.progress!, candidates: [{
      ...KIS.progress!.candidates[0],
      summary: { summary: "x", checks: [
        { detail: 0, status: "supported", source: "caption", at: 55, quote: "white apron", note: "đúng tạp dề trắng" },
        { detail: 1, status: "partial", source: "caption", at: 60, quote: "A small vase with pink flowers", note: "hoa hồng, không phải riềng tía" },
        { detail: 2, status: "contradicted", source: "caption", at: 61, quote: "holds two cooked shrimp", note: "2 con, không phải 4" },
      ] },
    }] } });

    const card = screen.getByLabelText("Candidate C3");
    expect(within(card).getByText("1/3 details")).toBeInTheDocument();   // only the full match is counted
    expect(within(card).getByText(/1 close/)).toBeInTheDocument();
    const checks = within(screen.getByLabelText("Query details checked"));
    expect(checks.getByLabelText("supported")).toBeInTheDocument();
    expect(checks.getByLabelText("partial")).toHaveTextContent("~");
    expect(checks.getByLabelText("contradicted")).toHaveTextContent("✗");
    expect(checks.getByText("hoa hồng, không phải riềng tía")).toBeInTheDocument();
    expect(checks.getByText("2 con, không phải 4")).toBeInTheDocument();
  });

  it("dims a detail the server says nearly every candidate satisfies, and says so", async () => {
    // A woman in an apron, in a cooking show: true of almost every candidate, so no help in choosing.
    await startHuman({ ...KIS, progress: { ...KIS.progress!, common_details: [true, false, false] } });

    const details = within(await screen.findByLabelText("Query details checked"));
    const common = details.getAllByTitle(/does not tell them apart/);
    expect(common).toHaveLength(1);
    expect(common[0].closest("li")).toHaveTextContent("thuyền màu đỏ");   // detail 0 only
    expect(details.getByText("trên sông Hương").closest("li")).not.toHaveClass("common");
  });

  describe("order", () => {
    // The searches only find candidates and any can be wrong. C3 leads retrieval (agent_rank 1) but its
    // words say nothing of the query; C7 is where the details are said. The server sends the list in
    // EVIDENCE order (C7 first) and the view shows it as it is.
    const said = (status: "supported" | "not_mentioned" | "contradicted") => ({ summary: "x", checks: [0, 1, 2].map((detail) => ({
      detail, status, source: status === "not_mentioned" ? null : "caption" as const, at: status === "not_mentioned" ? null : 80, quote: "", note: "",
    })) });
    const inEvidenceOrder = (extra: Partial<AgentRunResponse> = {}, summarizing = false): AgentRunResponse => ({
      ...KIS, ...extra,
      progress: { ...KIS.progress!, summarizing, candidates: [
        { ...KIS.progress!.candidates[1], agent_rank: 2, summary: said("supported"), evidence: { score: 1, share: 1, tier: 0 } },
        { ...KIS.progress!.candidates[0], agent_rank: 1, summary: said("not_mentioned"), evidence: { score: 0, share: 0, tier: 1 } },
      ] },
    });
    const shownOrder = () => screen.getAllByLabelText(/^Candidate C/).map((card) => card.getAttribute("aria-label"));

    it("shows the list as the server ordered it: by what the words support, not by retrieval", async () => {
      await startHuman(inEvidenceOrder());

      expect(shownOrder()).toEqual(["Candidate C7", "Candidate C3"]);
      expect(within(screen.getByLabelText("Candidate C7")).getByText(/#1 C7/)).toHaveAttribute("title", "Retrieval alone put this 2");
      expect((screen.getByLabelText("Candidate order") as HTMLSelectElement).value).toBe("evidence");
    });

    it("fills the bar from the evidence and says how many details are close or in conflict", async () => {
      const run = inEvidenceOrder();
      run.progress!.candidates[0] = { ...run.progress!.candidates[0], evidence: { score: 0.4, share: 0.6, video: 0.8, tier: 0 },
        summary: { summary: "x", checks: [
          { detail: 0, status: "supported", source: "caption", at: 1, quote: "a", note: "" },
          { detail: 1, status: "partial", source: "caption", at: 1, quote: "b", note: "" },
          { detail: 2, status: "contradicted", source: "caption", at: 1, quote: "c", note: "" },
        ] } };
      await startHuman(run);

      const card = screen.getByLabelText("Candidate C7");
      expect(within(card).getByText("1/3 details")).toBeInTheDocument();
      expect(within(card).getByText(/1 close/)).toBeInTheDocument();
      expect(within(card).getByText(/1 conflict/)).toBeInTheDocument();
      expect(card.querySelector(".review-match-bar i")).toHaveStyle({ width: "60%" });
      expect(card.querySelector(".review-match")).toHaveAttribute("title", expect.stringContaining("this moment 60%, its video 80%"));
    });

    it("says the order will follow the evidence once the checklist has been read", async () => {
      await startHuman(inEvidenceOrder({}, true));
      expect(screen.getByText(/ordered by evidence once the checklist is read/)).toBeInTheDocument();
    });

    it("can show retrieval's own order for comparison, and back", async () => {
      await startHuman(inEvidenceOrder());

      fireEvent.change(screen.getByLabelText("Candidate order"), { target: { value: "retrieval" } });
      expect(shownOrder()).toEqual(["Candidate C3", "Candidate C7"]);
      fireEvent.change(screen.getByLabelText("Candidate order"), { target: { value: "evidence" } });
      expect(shownOrder()).toEqual(["Candidate C7", "Candidate C3"]);
    });

    it("keeps what the reviewer settled where it belongs in either order", async () => {
      await startHuman(inEvidenceOrder({ review: { ...KIS.review!, decisions: { C3: "accept" } } }));
      fireEvent.change(screen.getByLabelText("Candidate order"), { target: { value: "retrieval" } });
      expect(shownOrder()).toEqual(["Candidate C3", "Candidate C7"]);   // accepted leads
      cleanup();
      clearEvidenceCache();
      vi.restoreAllMocks();

      await startHuman(inEvidenceOrder({ review: { ...KIS.review!, decisions: { C7: "reject" } } }));
      fireEvent.change(screen.getByLabelText("Candidate order"), { target: { value: "retrieval" } });
      expect(shownOrder()).toEqual(["Candidate C3", "Candidate C7"]);   // rejected trails
    });

    it("submits the order that is on the screen, so the export matches what was reviewed", async () => {
      const server = await startHuman(inEvidenceOrder());

      fireEvent.click(screen.getByRole("button", { name: /^Submit/ }));
      await waitFor(() => expect(sent(server)).toHaveLength(1));
      expect(sent(server)[0]).toEqual({ finish: true, order: ["C7", "C3"] });

      fireEvent.change(screen.getByLabelText("Candidate order"), { target: { value: "retrieval" } });
      fireEvent.click(screen.getByRole("button", { name: "Stop" }));
      await waitFor(() => expect(sent(server)).toHaveLength(2));
      expect(sent(server)[1]).toEqual({ stop: true, order: ["C3", "C7"] });   // whatever order was on screen
    });
  });

  it("shows where in the video each supported detail was found", async () => {
    await startHuman({ ...KIS, progress: { ...KIS.progress!, candidates: [{
      ...KIS.progress!.candidates[0],
      summary: { summary: "x", checks: [{ detail: 0, status: "supported", source: "caption", at: 55, quote: "white apron" }] },
    }] } });
    expect(within(screen.getByLabelText("Query details checked")).getByText(/caption 0:55/)).toBeInTheDocument();
  });

  it("says so when a video has no OCR rather than showing an empty box", async () => {
    await startHuman(KIS, { "/ocr/": () => jsonResponse({ detail: "ocr data unavailable" }, { ok: false, status: 503 }) });

    expect(await screen.findByText("OCR is not available for this video.")).toBeInTheDocument();
    // the rest still loads
    expect(await within(screen.getByLabelText("Speech around this moment")).findByText("thuyền đỏ")).toBeInTheDocument();
  });

  it("decides with the keyboard on the selected candidate and moves with J and K", async () => {
    const server = await startHuman();
    await screen.findByLabelText("Speech around this moment");

    fireEvent.keyDown(window, { key: "a" });
    fireEvent.keyDown(window, { key: "j" });
    fireEvent.keyDown(window, { key: "5" });
    fireEvent.keyDown(window, { key: "k" });
    fireEvent.keyDown(window, { key: "r" });

    await waitFor(() => expect(sent(server)).toHaveLength(3));
    expect(sent(server)).toEqual([
      { candidate_id: "C3", decision: "accept" },
      { candidate_id: "C7", score: 5 },
      { candidate_id: "C3", decision: "reject" },
    ]);
  });

  it("does not treat typing in a field as a shortcut", async () => {
    const server = await startHuman();
    const hint = screen.getByLabelText("Hint for another search round");

    fireEvent.keyDown(hint, { key: "a" });
    fireEvent.change(hint, { target: { value: "thuyền phải màu đỏ" } });

    expect(sent(server)).toEqual([]);
  });

  it("clicking a decision twice withdraws it", async () => {
    const server = await startHuman({ ...KIS, review: { ...KIS.review!, decisions: { C7: "accept" } } });

    fireEvent.click(screen.getByRole("button", { name: "Accept C7" }));

    await waitFor(() => expect(sent(server)).toHaveLength(1));
    expect(sent(server)[0]).toEqual({ candidate_id: "C7", decision: "clear" });
  });

  it("asks for another round with the reviewer's hint, and submits when done", async () => {
    const server = await startHuman();

    fireEvent.change(screen.getByLabelText("Hint for another search round"), { target: { value: "  phải màu đỏ " } });
    fireEvent.click(screen.getByRole("button", { name: "Search more" }));
    fireEvent.click(screen.getByRole("button", { name: /^Submit/ }));

    await waitFor(() => expect(sent(server)).toHaveLength(2));
    expect(sent(server)).toEqual([{ command: { type: "more", hint: "phải màu đỏ" } }, { finish: true, order: ["C3", "C7"] }]);
  });

  it("will not ask for another round while the agent is still searching", async () => {
    await startHuman({ ...KIS, progress: { ...KIS.progress!, stage: "searching" } });
    expect(screen.getByRole("button", { name: "Search more" })).toBeDisabled();
  });

  it("adds a neighbouring keyframe to the candidates and searches inside the video", async () => {
    const server = await startHuman();

    fireEvent.click(await screen.findByLabelText("Add frame 2050 to candidates"));
    fireEvent.click(screen.getByRole("button", { name: "Search this video" }));
    fireEvent.click(screen.getByRole("button", { name: "Neighbours" }));

    await waitFor(() => expect(sent(server)).toHaveLength(3));
    expect(sent(server)).toEqual([
      { command: { type: "add_frame", video_id: "L21_V010", frame_idx: 2050, time: 82 } },
      { command: { type: "scope", video_id: "L21_V010" } },
      { command: { type: "neighbours", cid: "C3" } },
    ]);
    expect(screen.queryByLabelText("Add frame 2000 to candidates")).not.toBeInTheDocument();  // it is the candidate
  });

  it("offers answers read from the words and sends the one the person picks", async () => {
    const server = await startHuman(QA);

    fireEvent.click(screen.getByRole("button", { name: /hai muỗng/ }));

    await waitFor(() => expect(sent(server)).toHaveLength(1));
    expect(sent(server)[0]).toEqual({ answer: "hai muỗng" });
  });

  it("hides rejected candidates and groups by video on request", async () => {
    await startHuman({ ...KIS, review: { ...KIS.review!, decisions: { C3: "reject" } } });

    fireEvent.click(screen.getByLabelText("Hide rejected"));
    expect(screen.queryByLabelText("Candidate C3")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Candidate C7")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Hide rejected"));
    fireEvent.click(screen.getByLabelText("Group by video"));
    expect(screen.getByRole("heading", { name: "L21_V010" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "L22_V003" })).toBeInTheDocument();
  });

  it("assembles a TRAKE path event by event and sends it once every event has a frame", async () => {
    const server = await startHuman(TRAKE);

    fireEvent.click(await screen.findByRole("button", { name: "Set frame 2000 as E1" }));
    expect(sent(server)).toEqual([]);  // one event of two: not a path yet
    fireEvent.click(screen.getByRole("button", { name: "View frame 2050" }));
    fireEvent.click(screen.getByRole("button", { name: "Set frame 2050 as E2" }));

    await waitFor(() => expect(sent(server)).toHaveLength(1));
    expect(sent(server)[0]).toEqual({ sequence: { video_id: "L21_V010", frames: [2000, 2050], times: [80, 82] } });
    expect(screen.getByAltText("E1: frame 2000")).toBeInTheDocument();
  });

  it("starts a TRAKE path from one the agent found", async () => {
    const server = await startHuman(TRAKE);

    fireEvent.click(screen.getByRole("button", { name: "Use L21_V010 2000 → 2100" }));

    await waitFor(() => expect(sent(server)).toHaveLength(1));
    expect(sent(server)[0]).toEqual({ sequence: { video_id: "L21_V010", frames: [2000, 2100], times: [80, 84] } });
  });
});

describe("words around a moment", () => {
  const cues = [
    { start: 0, end: 10, text: "a" }, { start: 50, end: 60, text: "b" }, { start: 100, end: 110, text: "c" },
  ];

  it("keeps cues within the window and marks the one being spoken", () => {
    expect(speechAround(cues, 55, 10).map((cue) => [cue.text, cue.active])).toEqual([["b", true]]);
    expect(speechAround(cues, 105, 60).map((cue) => cue.text)).toEqual(["b", "c"]);
  });

  it("finds the OCR shots around the moment, keeping the current one and dropping blank ones", () => {
    const shot = (shot_idx: number, start: number, end: number, text: string) =>
      ({ shot_idx, frame_idx: shot_idx, start, end, time: (start + end) / 2, text });
    const shots = [shot(1, 0, 10, "một"), shot(2, 10, 20, " "), shot(3, 20, 30, "ba"), shot(4, 30, 40, "bốn"), shot(5, 500, 510, "xa")];
    expect(ocrAround(shots, 25).map((s) => [s.shot_idx, s.current])).toEqual([[1, false], [3, true], [4, false]]);
    expect(ocrAround(shots, 500).map((s) => s.shot_idx)).toEqual([5]);  // only what is near it
    expect(ocrAround([], 5)).toEqual([]);
  });

  it("drops the channel's own logo and clock from the on-screen text", () => {
    expect(["HTV Online", "HD 06:30:39 giây", "hit online", "ONLINE"].map(isFurniture)).toEqual([true, true, true, true]);
    expect(["HTV Online - Món ngon", "Nghêu hấp sả", "GIÁ 15.000đ", ""].map(isFurniture)).toEqual([false, false, false, false]);
    const shot = (shot_idx: number, text: string) => ({ shot_idx, frame_idx: shot_idx, start: shot_idx * 10, end: shot_idx * 10 + 9, time: shot_idx * 10 + 5, text });
    const shots = [shot(1, "HTV Online"), shot(2, "NGHÊU HẤP SẢ"), shot(3, "HD 06:30:39 giây"), shot(4, "GIÁ 15.000đ")];
    expect(ocrAround(shots, 25).map((s) => s.shot_idx)).toEqual([2, 4]);
  });

  it("finds the caption windows around the moment in time order, the one holding it marked", () => {
    const caption = (shot_idx: number, start: number, end: number) =>
      ({ shot_idx, window_idx: 0, frame_idx: null, start, end, time: (start + end) / 2, text: `shot ${shot_idx}` });
    const captions = [caption(3, 40, 50), caption(1, 0, 10), caption(2, 10, 30), caption(4, 90, 99)];
    expect(captionsAround(captions, 25).map((c) => [c.shot_idx, c.current])).toEqual([[1, false], [2, true], [3, false]]);
    expect(captionsAround(captions, 500).map((c) => c.shot_idx)).toEqual([4]);  // nearest, when nothing holds it
    expect(captionsAround([], 3)).toEqual([]);
  });
});
