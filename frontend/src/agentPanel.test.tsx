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
      text: "a red boat", kind: "kis", mode: "fast",
      deepseek_api_key: "dsk", gemini_api_key: "gmk",
    });
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

  it("refuses to start without both keys", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    render(<AgentPanel />);
    fireEvent.change(screen.getByLabelText("Agent query"), { target: { value: "a red boat" } });

    fireEvent.click(screen.getByRole("button", { name: "Run agent" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Both API keys are required.");
    expect(fetchMock).not.toHaveBeenCalled();
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
    expect(screen.getByText(/working/)).toBeInTheDocument();

    await act(async () => { vi.advanceTimersByTime(2000); await Promise.resolve(); });

    expect(fetchMock.mock.calls[1][0]).toBe("/api/agent/runs/abc");
    expect(screen.getByText("8/10")).toBeInTheDocument();
    expect(screen.getByText(/the boat matches/)).toBeInTheDocument();
    expect(screen.getByText("Answer:")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Candidate 1: L01_V001 frame 10" }))
      .toHaveAttribute("src", "/thumb/L01_V001/10?w=320");
  });
});
