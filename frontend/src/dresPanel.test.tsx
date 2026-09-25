import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DresPanel } from "./dresPanel";
import type { DresAttempt, DresCandidate, DresPrepared, DresStatus, SearchResult } from "./types";

const api = vi.hoisted(() => ({
  dresStatus: vi.fn(), dresHistory: vi.fn(), dresPrepare: vi.fn(), dresSubmit: vi.fn(),
  dresConnect: vi.fn(), dresRelogin: vi.fn(), dresReset: vi.fn(), dresRefresh: vi.fn(),
  dresSelectEvaluation: vi.fn(), dresSetContext: vi.fn(),
}));
vi.mock("./dresApi", () => ({ ...api, DresApiError: class extends Error {} }));

const status: DresStatus = {
  configured: true, credentials_configured: true, host: "localhost", connected: true,
  evaluations: [{ id: "eval-1", name: "Final" }], evaluation_id: "eval-1",
  context_valid: true,
  context: { kind: "kis", label: "Task on hall screen", event_count: null, revision: 1,
    updated_by: "owner", updated_at: "2026-09-24T00:00:00Z" },
};
const prepared: DresPrepared = {
  evaluation_id: "eval-1", context_revision: 1, context_label: "Task on hall screen",
  kind: "kis", video_id: "L01_V001", frame_idx: 2, frame_ids: null, timestamp_ms: 123456,
  payload: { answerSets: [{ answers: [{ mediaItemName: "L01_V001", start: "123456", end: "123456" }] }] },
  preview: '{"mediaItemName":"L01_V001","start":"123456","end":"123456"}',
  fingerprint: "a".repeat(64),
};
const attempt: DresAttempt = {
  id: 1, evaluation_id: "eval-1", context_revision: 1, context_label: "Task on hall screen",
  task_type: "kis", payload: prepared.payload, fingerprint: prepared.fingerprint, video_id: "L01_V001",
  frame_idx: 2, frame_ids: null, actor: "member", submitted_at: "2026-09-24T00:00:00Z",
  status: "accepted", http_status: 200, latency_ms: 10, server_message: "OK",
};

beforeEach(() => {
  api.dresStatus.mockResolvedValue(status);
  api.dresHistory.mockResolvedValue([]);
  api.dresConnect.mockResolvedValue(status);
  api.dresPrepare.mockResolvedValue(prepared);
  api.dresSubmit.mockResolvedValue(attempt);
  api.dresSetContext.mockResolvedValue(status.context);
  api.dresSelectEvaluation.mockResolvedValue(status);
});
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function panel(candidate: DresCandidate | null = { video_id: "L01_V001", frame_idx: 2 }, selected: SearchResult | null = null) {
  return render(<DresPanel selected={selected} candidate={candidate} onCandidate={vi.fn()} onPick={vi.fn()}
    onKind={vi.fn()} isAdmin={false} pickError="" />);
}

describe("official DRES panel", () => {
  it("offers direct submission for the currently selected keyframe", async () => {
    const selected: SearchResult = { rank: 0, score: 1, video_id: "L01_V001", frame_idx: 2, time: 123.456,
      thumb: "/thumb", frame: "/frame", video: "/video", source: "visual", scores: {} };
    panel(null, selected);
    const button = await screen.findByRole("button", { name: "Submit selected frame" });
    fireEvent.click(button);
    await waitFor(() => expect(api.dresPrepare).toHaveBeenCalledWith({ video_id: "L01_V001", frame_idx: 2 }));
    await waitFor(() => expect(api.dresSubmit).toHaveBeenCalledTimes(1));
    expect(api.dresSubmit.mock.calls[0][1].payload.answerSets[0].answers[0]).toEqual({
      mediaItemName: "L01_V001", start: "123456", end: "123456",
    });
  });

  it("lets the direct Q&A action include the entered answer without a staged candidate", async () => {
    api.dresStatus.mockResolvedValue({ ...status, context: { ...status.context!, kind: "qa" } });
    const selected: SearchResult = { rank: 0, score: 1, video_id: "L01_V001", frame_idx: 2, time: 123.456,
      thumb: "/thumb", frame: "/frame", video: "/video", source: "visual", scores: {} };
    panel(null, selected);
    fireEvent.change(await screen.findByLabelText("DRES Q&A answer"), { target: { value: "sample answer" } });
    fireEvent.click(screen.getByRole("button", { name: "Submit selected frame" }));
    await waitFor(() => expect(api.dresPrepare).toHaveBeenCalledWith({
      video_id: "L01_V001", frame_idx: 2, answer: "sample answer",
    }));
    await waitFor(() => expect(api.dresSubmit).toHaveBeenCalledTimes(1));
  });

  it("connects using backend credentials and never asks for a password or session ID", async () => {
    render(<DresPanel selected={null} candidate={null} onCandidate={vi.fn()} onPick={vi.fn()}
      onKind={vi.fn()} isAdmin pickError="" settingsOpen />);
    await screen.findByRole("button", { name: "Connect" });
    expect(screen.queryByLabelText(/DRES username|DRES password|session id/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    await waitFor(() => expect(api.dresConnect).toHaveBeenCalledTimes(1));
  });

  it("automatically syncs the current app query into DRES task context", async () => {
    api.dresStatus.mockResolvedValue({ ...status, context_valid: false, context: null });
    render(<DresPanel selected={null} candidate={null} onCandidate={vi.fn()} onPick={vi.fn()}
      onKind={vi.fn()} isAdmin={false} pickError=""
      settingsOpen
      suggestedTask={{ kind: "kis", label: "query-p2-2-kis", event_count: null }} />);
    await waitFor(() => expect(api.dresSetContext).toHaveBeenCalledWith({
      expected_revision: 0, kind: "kis", label: "query-p2-2-kis", event_count: null,
    }));
    expect(screen.queryByRole("button", { name: "Use current query" })).not.toBeInTheDocument();
  });

  it("offers evaluation fallback only when multiple active evaluations need it", async () => {
    api.dresStatus.mockResolvedValue({ ...status, evaluation_id: null, context_valid: false, context: null,
      evaluations: [{ id: "qa-1", name: "QA Test" }, { id: "kis-1", name: "KIS Test" }] });
    render(<DresPanel selected={null} candidate={null} onCandidate={vi.fn()} onPick={vi.fn()}
      onKind={vi.fn()} isAdmin={false} pickError=""
      settingsOpen
      suggestedTask={{ kind: "kis", label: "query-p2-2-kis", event_count: null }} />);
    const select = await screen.findByLabelText("DRES evaluation fallback");
    expect(select).toHaveValue("__auto__");
    expect(screen.getByRole("option", { name: /KIS Test/ })).toBeInTheDocument();
  });

  it("submits the exact KIS timestamp after one explicit click", async () => {
    panel();
    await screen.findByRole("button", { name: "Submit selected frame" });
    expect(api.dresSubmit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Submit selected frame" }));
    await waitFor(() => expect(api.dresSubmit).toHaveBeenCalledTimes(1));
    expect(api.dresSubmit.mock.calls[0][0]).toEqual({ video_id: "L01_V001", frame_idx: 2 });
    await screen.findByText(/ACCEPTED · HTTP 200/);
  });

  it("blocks a second click while the first POST is in flight", async () => {
    let resolve!: (value: DresAttempt) => void;
    api.dresSubmit.mockReturnValue(new Promise<DresAttempt>((done) => { resolve = done; }));
    panel();
    const button = await screen.findByRole("button", { name: "Submit selected frame" });
    fireEvent.click(button); fireEvent.click(button);
    await waitFor(() => expect(api.dresSubmit).toHaveBeenCalledTimes(1));
    resolve(attempt);
    await screen.findByText(/ACCEPTED · HTTP 200/);
  });

  it("treats a lost browser response as uncertain and requires an explicit retry", async () => {
    api.dresSubmit.mockRejectedValue(new TypeError("network failed"));
    panel();
    fireEvent.click(await screen.findByRole("button", { name: "Submit selected frame" }));
    await screen.findByText(/Submission uncertain/i);
    expect(screen.getByRole("button", { name: "Submit selected frame" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /retry after checking DRES/i })).toBeEnabled();
    expect(api.dresSubmit).toHaveBeenCalledTimes(1);
  });

  it("preserves the Q&A answer exactly in the preview request", async () => {
    api.dresStatus.mockResolvedValue({ ...status, context: { ...status.context!, kind: "qa" } });
    api.dresPrepare.mockResolvedValue({ ...prepared, kind: "qa",
      payload: { answerSets: [{ answers: [{ text: "QA-  A-B  -L01_V001-123456" }] }] },
      preview: "QA-  A-B  -L01_V001-123456" });
    panel();
    await screen.findByLabelText("DRES Q&A answer");
    fireEvent.change(screen.getByLabelText("DRES Q&A answer"), { target: { value: "  A-B  " } });
    fireEvent.click(screen.getByRole("button", { name: "Submit selected frame" }));
    await waitFor(() => expect(api.dresPrepare).toHaveBeenCalledWith({
      video_id: "L01_V001", frame_idx: 2, answer: "  A-B  ",
    }));
  });

  it("reorders TRAKE FRAME_IDs without converting them to timestamps", async () => {
    api.dresStatus.mockResolvedValue({ ...status, context: { ...status.context!, kind: "trake", event_count: 3 } });
    api.dresPrepare.mockResolvedValue({ ...prepared, kind: "trake", frame_idx: null, timestamp_ms: null,
      frame_ids: [100, 400, 250], payload: { answerSets: [{ answers: [{ text: "TR-L01_V001-100,400,250" }] }] },
      preview: "TR-L01_V001-100,400,250" });
    const onOpenFrame = vi.fn();
    function Harness() {
      const [candidate, setCandidate] = useState<DresCandidate>({ video_id: "L01_V001", frame_ids: [100, 250, 400] });
      return <DresPanel selected={null} candidate={candidate} onCandidate={(next) => next && setCandidate(next)}
        onPick={vi.fn()} onKind={vi.fn()} isAdmin={false} pickError="" onOpenFrame={onOpenFrame} />;
    }
    render(<Harness />);
    fireEvent.click(await screen.findByRole("button", { name: "E2 · FRAME_ID 250" }));
    expect(onOpenFrame).toHaveBeenCalledWith("L01_V001", 250);
    await screen.findByLabelText("Move frame 400 up");
    fireEvent.click(screen.getByLabelText("Move frame 400 up"));
    fireEvent.click(screen.getByRole("button", { name: "Submit event sequence" }));
    await waitFor(() => expect(api.dresPrepare).toHaveBeenCalledWith({ video_id: "L01_V001", frame_ids: [100, 400, 250] }));
  });

  it("keeps the navigation task choice without changing the shared context on settings load", async () => {
    const oldStatus = { ...status, context: { ...status.context!, kind: "trake" as const, event_count: 2 } };
    api.dresStatus.mockResolvedValue(oldStatus);
    const onKind = vi.fn();
    render(<DresPanel selected={null} candidate={null} onCandidate={vi.fn()} onPick={vi.fn()}
      onKind={onKind} selectedKind="kis" externalStatus={oldStatus} isAdmin={false} pickError=""
      showSubmission={false} />);
    await waitFor(() => expect(api.dresStatus).toHaveBeenCalled());
    expect(onKind).not.toHaveBeenCalled();
    expect(api.dresSetContext).not.toHaveBeenCalled();
  });
});
