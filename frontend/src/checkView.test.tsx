import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CheckPanel, verdictTier } from "./checkView";
import type { CheckResult, CheckState } from "./types";

function result(overrides: Partial<CheckResult> = {}): CheckResult {
  return {
    key: "abc",
    video_id: "L22_V011",
    frame_idx: 28321,
    time: 944.0,
    window_seconds: 300,
    shot_index: 41,
    covers: { start_frame: 28300, end_frame: 28400, start: 943.0, end: 947.0 },
    verdict: { score: 8, reason: "đúng cảnh cầm điện thoại" },
    details: [
      { text: "hai người cầm điện thoại", meaning_en: "two people holding a phone", distinctiveness: 0.9, weight: 0.92 },
      { text: "trên phố", meaning_en: "on a street", distinctiveness: 0.2, weight: 0.36 },
      { text: "có xe buýt đỏ", meaning_en: "a red bus", distinctiveness: 0.8, weight: 0.84 },
      { text: "buổi tối", meaning_en: "at night", distinctiveness: 0.5, weight: 0.6 },
    ],
    checks: [
      { detail: 0, status: "supported", source: "vision", at: 944.0, quote: "", note: "hai người cùng nhìn điện thoại" },
      { detail: 1, status: "partial", source: "speech", at: 940.0, quote: "Chúng tôi đang đi trên phố", note: "chỉ nói chung chung" },
      { detail: 2, status: "contradicted", source: "vision", at: 944.0, quote: "", note: "xe buýt màu xanh" },
      { detail: 3, status: "not_mentioned", source: null, at: null, quote: "", note: "" },
    ],
    support: { share: 0.42, supported: 1, partial: 1, contradicted: 1, total: 4 },
    answer: null,
    frames_seen: 23,
    model: "qwen/qwen3.7-flash",
    by: "owner@example.com",
    checked_at: 1750000000,
    cached: false,
    latency_ms: 6200,
    ...overrides,
  };
}

function done(overrides: Partial<CheckResult> = {}): CheckState {
  return { status: "done", result: result(overrides), error: null, asked: null };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("the AI check panel", () => {
  it("renders nothing until a check has been asked for", () => {
    const { container } = render(
      <CheckPanel state={{ status: "idle", result: null, error: null, asked: null }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("says what it is doing while it looks", () => {
    render(<CheckPanel state={{ status: "loading", result: null, error: null, asked: null }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/Looking at the frames/);
  });

  it("shows a failure as an alert rather than an empty panel", () => {
    render(<CheckPanel state={{ status: "error", result: null, error: "the AI check failed", asked: null }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("the AI check failed");
  });

  it("leads with the overall verdict and its reason", () => {
    render(<CheckPanel state={done()} />);
    expect(screen.getByText("8/10")).toBeInTheDocument();
    expect(screen.getByText("đúng cảnh cầm điện thoại")).toBeInTheDocument();
  });

  it("marks each detail with its own verdict", () => {
    render(<CheckPanel state={done()} />);
    const marks = screen.getAllByLabelText(/^(supported|close|contradicted|not mentioned)$/);
    expect(marks.map((mark) => mark.textContent)).toEqual(["✓", "~", "✗", "·"]);
  });

  it("counts only support, and names conflicts separately", () => {
    // A contradiction is shown but lowers nothing: captions and judges get colours, counts and
    // identities wrong often enough that a conflict is as often their mistake as the moment's.
    render(<CheckPanel state={done()} />);
    expect(screen.getByText(/1\/4 details/)).toHaveTextContent("1/4 details · 1 close · 1 conflict");
    expect(screen.getByLabelText("1 of 4 details supported")).toBeInTheDocument();
  });

  it("shows each verdict's reason and the words it was read from", () => {
    render(<CheckPanel state={done()} />);
    expect(screen.getByText("hai người cùng nhìn điện thoại")).toBeInTheDocument();
    expect(screen.getByText(/Chúng tôi đang đi trên phố/)).toBeInTheDocument();
    expect(screen.getByText("speech", { exact: false })).toBeInTheDocument();
  });

  it("lights the query's own words inside a quote", () => {
    render(<CheckPanel state={done()} keywords={["phố"]} />);
    expect(screen.getByText("phố").tagName).toBe("MARK");
  });

  it("marks the details that would actually narrow the search down", () => {
    render(<CheckPanel state={done()} />);
    // 0.9 and 0.8 are telling; 0.2 ("on a street") and 0.5 are not.
    expect(screen.getAllByText("telling")).toHaveLength(2);
  });

  it("says which frames the answer speaks for and where it came from", () => {
    render(<CheckPanel state={done({ cached: true })} />);
    expect(screen.getByText(/23 frames · ±5 min/)).toBeInTheDocument();
    expect(screen.getByText(/shot 41/)).toBeInTheDocument();
    expect(screen.getByText("from the cache")).toBeInTheDocument();
  });

  it("offers a frame-exact re-read, because a shot's answer is shared by its frames", () => {
    const onRecheck = vi.fn();
    render(<CheckPanel state={done()} onRecheck={onRecheck} />);
    fireEvent.click(screen.getByRole("button", { name: "Re-check this exact frame" }));
    expect(onRecheck).toHaveBeenCalledOnce();
  });

  it("lets the player jump to the second a verdict was read at", () => {
    const onSeek = vi.fn();
    render(<CheckPanel state={done()} variant="inline" onSeek={onSeek} />);
    fireEvent.click(screen.getByRole("button", { name: "15:40" }));
    expect(onSeek).toHaveBeenCalledWith(940);
  });

  describe("a Q&A answer", () => {
    it("is shown with where it was read and how sure the model is", () => {
      render(<CheckPanel state={done({
        answer: { text: "nghêu", confidence: 8, source: "speech", quote: "xếp hai con nghêu lên đĩa" },
      })} />);
      expect(screen.getByText("nghêu")).toBeInTheDocument();
      expect(screen.getByText(/heard in the speech · confidence 8\/10/)).toBeInTheDocument();
      expect(screen.getByText(/xếp hai con nghêu/)).toBeInTheDocument();
    });

    it("says the moment does not answer rather than showing a guess", () => {
      render(<CheckPanel state={done({
        answer: { text: "", confidence: 0, source: "unknown", quote: "" },
      })} />);
      expect(screen.getByText("This moment does not answer the question.")).toBeInTheDocument();
    });

    it("is absent for a query that is not a question", () => {
      render(<CheckPanel state={done()} />);
      expect(screen.queryByRole("region", { name: "Answer" })).not.toBeInTheDocument();
    });
  });
});

describe("verdictTier", () => {
  it("matches the tiers the agent's own verdicts use", () => {
    expect(verdictTier(9)).toBe("good");
    expect(verdictTier(6)).toBe("good");
    expect(verdictTier(5)).toBe("mid");
    expect(verdictTier(3)).toBe("bad");
  });
});
