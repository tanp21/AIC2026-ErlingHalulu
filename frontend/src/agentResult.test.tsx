import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AgentResultView } from "./agentResult";
import type { AgentRunResult } from "./types";

function jsonResponse(data: unknown): Response {
  return { ok: true, status: 200, json: async () => data, headers: new Headers() } as Response;
}

/** `/context/{video}/{frame}`: the keyframes around a frame, one every 25 frames. */
function neighbours(input: RequestInfo | URL) {
  const [, , video, frame] = String(input).split("?")[0].split("/");
  const at = Number(frame);
  return jsonResponse({ video_id: video, frame_idx: at, frames: [-25, 0, 25].map((offset) => ({
    video_id: video, frame_idx: at + offset, time: (at + offset) / 25, thumb: "", frame: "", video: "",
  })) });
}

const kis: AgentRunResult = {
  query_id: "q", kind: "kis", query: "a red boat", answer: null, answer_evidence: null,
  sequences: [], notes: "", elapsed: 70, error: null, judge: "qwen",
  strategy: ["Chấm điểm: Qwen xem khung hình của từng ứng viên"],
  lookups: [{ question: "địa danh (tra theo mô tả)", answer: "Cầu Rồng, Đà Nẵng" }],
  ranking: [{
    id: "C1", video_id: "L01_V001", frame_idx: 100, time: 4, fused: 0.9, evidence: [],
    verdict: { score: 9, reason: "thuyền đỏ khớp", by: "qwen", frames_seen: 11, checks: [
      { text: "thuyền màu đỏ", status: "supported", source: "vision", note: "thấy rõ" },
      { text: "hai người chèo", status: "partial", source: "vision", note: "chỉ thấy một người" },
    ] },
    context: { speech: "chiếc thuyền đỏ" },
  }, {
    id: "C2", video_id: "L02_V002", frame_idx: 500, time: 20, fused: 0.5, evidence: [],
    verdict: { score: 4, reason: "thuyền xanh", by: "qwen" },
  }],
};

beforeEach(() => { vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => neighbours(input)); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function view(result: AgentRunResult, onSubmit = vi.fn().mockResolvedValue({ message: "ACCEPTED", completed: true })) {
  const onOpen = vi.fn();
  render(<AgentResultView result={result} downloading={false} onDownload={() => undefined}
    onOpen={onOpen} onSubmit={onSubmit} />);
  return { onSubmit, onOpen };
}

describe("agent result view", () => {
  it("shows the best answer at full size with what the judge checked, and the strategy", () => {
    view(kis);

    const inspector = screen.getByRole("article", { name: "Answer 1 in detail" });
    expect(within(inspector).getByRole("img", { name: "L01_V001 frame 100" })).toHaveAttribute("src", "/frame/L01_V001/100");
    expect(within(inspector).getByText("thuyền đỏ khớp")).toBeInTheDocument();
    expect(within(inspector).getByText(/from 11 keyframes around the moment/)).toBeInTheDocument();
    expect(within(inspector).getByText("thuyền màu đỏ")).toBeInTheDocument();
    expect(within(inspector).getByText(/chỉ thấy một người/)).toBeInTheDocument();
    expect(within(inspector).getByText("chiếc thuyền đỏ")).toBeInTheDocument();
    expect(screen.getByText("Cầu Rồng, Đà Nẵng")).toBeInTheDocument();
  });

  it("switches the inspector to another answer from the list", () => {
    view(kis);

    fireEvent.click(screen.getByRole("button", { name: /Answer 2: L02_V002/ }));

    expect(screen.getByRole("article", { name: "Answer 2 in detail" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "L02_V002 frame 500" })).toBeInTheDocument();
  });

  it("submits a nearby keyframe when one is picked instead of the agent's", async () => {
    const { onSubmit } = view(kis);

    fireEvent.click(await screen.findByRole("button", { name: /Use frame 125 / }));
    expect(screen.getByText(/moved from the agent's pick/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Submit this frame to DRES" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith({ kind: "kis", video_id: "L01_V001", frame_idx: 125 }));
    expect(await screen.findByText("ACCEPTED")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Submitted" })).toBeDisabled();
  });

  it("submits a Q&A answer with its frame, and not without one", async () => {
    const { onSubmit } = view({ ...kis, kind: "qa", answer: "Đèo Tà Pứa", answer_evidence: "read from C1" });

    expect(screen.getByLabelText("Answer to submit")).toHaveValue("Đèo Tà Pứa");
    fireEvent.change(screen.getByLabelText("Answer to submit"), { target: { value: " " } });
    expect(screen.getByRole("button", { name: "Submit frame and answer to DRES" })).toBeDisabled();

    fireEvent.change(screen.getByLabelText("Answer to submit"), { target: { value: "Đèo Bứa " } });
    fireEvent.click(screen.getByRole("button", { name: "Submit frame and answer to DRES" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(
      { kind: "qa", video_id: "L01_V001", frame_idx: 100, answer: "Đèo Bứa" }));
  });

  it("shows a TRAKE path event by event, and submits its frames in order", async () => {
    const { onSubmit } = view({
      ...kis, kind: "trake", ranking: [], events: ["người múa lân nhảy lên", "lân ngã xuống"],
      sequences: [{ video_id: "L24_V010", frames: [1000, 1300], times: [40, 52], score: 1, event_scores: [9, 6], refined: true }],
    });

    const inspector = screen.getByRole("article", { name: "Path 1 in detail" });
    expect(within(inspector).getByText("người múa lân nhảy lên")).toBeInTheDocument();
    expect(within(inspector).getByText("lân ngã xuống")).toBeInTheDocument();
    expect(within(inspector).getByText("9/10")).toBeInTheDocument();

    fireEvent.click(within(inspector).getAllByRole("button", { name: "Adjust" })[1]);
    fireEvent.click(await screen.findByRole("button", { name: /Use frame 1325 / }));
    fireEvent.click(screen.getByRole("button", { name: "Submit this sequence to DRES" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(
      { kind: "trake", video_id: "L24_V010", frame_ids: [1000, 1325] }));
  });

  it("refuses a TRAKE sequence whose events are out of order", async () => {
    view({
      ...kis, kind: "trake", ranking: [], events: ["a", "b"],
      sequences: [{ video_id: "L24_V010", frames: [1000, 1020], times: [40, 40.8], score: 1 }],
    });

    fireEvent.click(screen.getAllByRole("button", { name: "Adjust" })[0]);
    fireEvent.click(await screen.findByRole("button", { name: /Use frame 1025 / }));

    expect(screen.getByText(/not in time order/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Submit this sequence to DRES" })).toBeDisabled();
  });
});
