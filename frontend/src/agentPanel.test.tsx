import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentPanel } from "./agentPanel";
import type { AgentRunResponse } from "./types";

function jsonResponse(data: unknown, init: Partial<Response> = {}): Response {
  return { ok: true, status: 200, json: async () => data, headers: new Headers(), ...init } as Response;
}

const running: AgentRunResponse = {
  run_id: "abc", status: "running", query_id: "abc", kind: "kis", query: "a red boat",
  elapsed: 1, result: null, usage: null, error: null,
};

const live: AgentRunResponse = {
  ...running,
  kind: "qa",
  progress: {
    stage: "judging", step: 0, elapsed: 6, answer: null, sequences: [],
    plan: {
      summary_vi: "tìm cảnh", scenes: ["hai người cầm điện thoại", "trả lời phỏng vấn"],
      target_scene: 1, question_vi: "Người đó nói gì?", events: [],
    },
    candidates: [{
      id: "C3", video_id: "L21_V010", frame_idx: 23484, time: 939.4, judging: true,
      verdict: null, sources: ["caption", "visual"],
      context: { speech: "tôi rất vui", caption: "A young woman speaks while a woman walks past." },
    }, {
      id: "C7", video_id: "L22_V003", frame_idx: 10912, time: 436.5, judging: false,
      verdict: { score: 8, reason: "hai người cầm điện thoại", by: "gemini" }, sources: ["visual"],
      context: {},
    }],
  },
  review: { decisions: {}, answer: null, stopped: false, decided: false },
};

const done: AgentRunResponse = {
  ...running,
  status: "done",
  elapsed: 12,
  result: {
    query_id: "abc", kind: "kis", query: "a red boat", answer: "42", answer_evidence: "read",
    sequences: [], notes: "check it", elapsed: 12, error: null,
    ranking: [{
      id: "c0", video_id: "L01_V001", frame_idx: 10, time: 0.4, fused: 0.9,
      verdict: { score: 8, reason: "the boat matches" }, evidence: [],
    }],
  },
};

function fillForm(query = "a red boat") {
  fireEvent.change(screen.getByLabelText("DeepSeek API key"), { target: { value: "dsk" } });
  fireEvent.change(screen.getByLabelText("Gemini API key"), { target: { value: "gmk" } });
  fireEvent.change(screen.getByLabelText("Agent query"), { target: { value: query } });
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });

describe("agent panel", () => {
  it("sends the teammate's own keys with the run request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(running, { status: 202 }));
    render(<AgentPanel />);
    fillForm();

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock.mock.calls[0][0]).toBe("/api/agent/runs");
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      text: "a red boat", kind: "kis", mode: "fast", judge: "model",
      deepseek_api_key: "dsk", gemini_api_key: "gmk",
    });
  });

  it("sends the judge the user selected, so human review is not silently run as Gemini", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(running, { status: 202 }));
    render(<AgentPanel />);
    fillForm();
    fireEvent.change(screen.getByLabelText("Agent judge"), { target: { value: "human" } });

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body)).judge).toBe("human");
  });

  it("sends the selected mode", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(running, { status: 202 }));
    render(<AgentPanel />);
    fillForm();
    fireEvent.change(screen.getByLabelText("Agent mode"), { target: { value: "deep" } });

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body)).mode).toBe("deep");
  });

  it("starts with no keys at all, so the owner does not have to type them", async () => {
    // The deployment holds the owner's keys and the server decides whether this run
    // may spend them. The panel used to demand them up front, which is what blocked
    // a local-vision deployment -- and the owner's own blank-field runs.
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(running, { status: 202 }));
    render(<AgentPanel />);
    fireEvent.change(screen.getByLabelText("Agent query"), { target: { value: "a red boat" } });

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body));
    expect(body.deepseek_api_key).toBe("");
    expect(body.gemini_api_key).toBe("");
  });

  it("shows which key the server says is missing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(
      { detail: "a DeepSeek API key is required" }, { ok: false, status: 422 }));
    render(<AgentPanel />);
    fireEvent.change(screen.getByLabelText("Agent query"), { target: { value: "a red boat" } });

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    expect(await screen.findByRole("alert"))
      .toHaveTextContent("a DeepSeek API key is required");
  });

  it("shows the run already in progress instead of a dead-end error", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/agent/runs/abc") return jsonResponse(live);
      if (url === "/api/agent/runs/active") return jsonResponse({ run: live });
      return jsonResponse({ detail: "an agent run is already in progress" }, { ok: false, status: 409 });
    });
    render(<AgentPanel />);
    fillForm("a blue car");

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    expect(await screen.findByRole("status")).toHaveTextContent(/already have a run in progress/);
    expect(await screen.findByLabelText("Live review")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop and use the current ranking" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.slice(0, 2).map((call) => call[0]))
      .toEqual(["/api/agent/runs", "/api/agent/runs/active"]);
  });

  it("keeps the conflict message when nothing is actually running", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => String(input) === "/api/agent/runs/active"
      ? jsonResponse({ run: null })
      : jsonResponse({ detail: "an agent run is already in progress" }, { ok: false, status: 409 }));
    render(<AgentPanel />);
    fillForm();

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("an agent run is already in progress");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("polls until the run finishes and renders the verified ranking", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(running, { status: 202 }))
      .mockResolvedValueOnce(jsonResponse(done));
    render(<AgentPanel />);
    fillForm();

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText(/still planning and searching/)).toBeInTheDocument();

    await act(async () => { vi.advanceTimersByTime(1000); await Promise.resolve(); });

    expect(fetchMock.mock.calls[1][0]).toBe("/api/agent/runs/abc");
    expect(screen.getByText("8/10")).toBeInTheDocument();
    expect(screen.getByText(/the boat matches/)).toBeInTheDocument();
    expect(screen.getByText("Answer:")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Candidate 1: L01_V001 frame 10" }))
      .toHaveAttribute("src", "/thumb/L01_V001/10?w=320");
  });

  async function startLive() {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(live, { status: 202 }));
    render(<AgentPanel />);
    fillForm();
    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));
    await screen.findByLabelText("Live review");
    return fetchMock;
  }

  it("shows what Gemini is judging, with the same context, while it judges", async () => {
    const fetchMock = await startLive();

    expect(screen.getByText("Gemini is judging candidates")).toBeInTheDocument();
    expect(screen.getByText("Gemini is judging…")).toBeInTheDocument();
    expect(screen.getByText("tôi rất vui")).toBeInTheDocument();
    expect(screen.getByText("A young woman speaks while a woman walks past.")).toBeInTheDocument();
    expect(screen.getByText(/trả lời phỏng vấn/)).toHaveTextContent("(frame to submit)");
    expect(screen.getByText("Người đó nói gì?")).toBeInTheDocument();
    expect(screen.getByText("8/10")).toBeInTheDocument();
  });

  it("sends an acceptance, and a second click withdraws it", async () => {
    const fetchMock = await startLive();
    const accepted = { ...live, review: { decisions: { C3: "accept" as const }, answer: null, stopped: false, decided: true } };
    fetchMock.mockResolvedValueOnce(jsonResponse(accepted));

    fireEvent.click(screen.getByRole("button", { name: "Accept C3" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Accept C3" }))
      .toHaveAttribute("aria-pressed", "true"));
    expect(fetchMock.mock.calls[1][0]).toBe("/api/agent/runs/abc/review");
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body)))
      .toEqual({ candidate_id: "C3", decision: "accept" });
    expect(screen.getByText(/your decision is being applied/)).toBeInTheDocument();

    fetchMock.mockResolvedValueOnce(jsonResponse(live));
    fireEvent.click(screen.getByRole("button", { name: "Accept C3" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(JSON.parse(String((fetchMock.mock.calls[2][1] as RequestInit).body)))
      .toEqual({ candidate_id: "C3", decision: "clear" });
  });

  it("sends a typed Q&A answer and a stop", async () => {
    const fetchMock = await startLive();
    fetchMock.mockResolvedValue(jsonResponse(live));

    fireEvent.change(screen.getByLabelText("Reviewer answer"), { target: { value: " 2.15 " } });
    fireEvent.click(screen.getByRole("button", { name: "Use this answer" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole("button", { name: "Stop and use the current ranking" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));

    const bodies = fetchMock.mock.calls.slice(1).map((call) => JSON.parse(String((call[1] as RequestInit).body)));
    expect(bodies).toEqual([{ answer: "2.15" }, { stop: true }]);
  });

  it("marks a reviewer's verdict in the finished ranking", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse({
      ...done,
      result: { ...done.result!, ranking: [{
        ...done.result!.ranking[0], verdict: { score: 10, reason: "Accepted by the reviewer", by: "human" },
      }] },
    }, { status: 202 }));
    render(<AgentPanel />);
    fillForm();
    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    expect(await screen.findByText("Reviewer")).toBeInTheDocument();
    expect(screen.getByText("10/10")).toBeInTheDocument();
  });
});
