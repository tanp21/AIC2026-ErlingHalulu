import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClipStudio, shotAt } from "./clips";
import type { Shot } from "./types";

const TIME_BASE_DEN = 90000;
const DELTA = 3003;

const shots: Shot[] = [
  { index: 0, startFrame: 0, endFrame: 9, start: 0, end: 10 * DELTA / TIME_BASE_DEN, keyframes: [] },
  { index: 1, startFrame: 10, endFrame: 24, start: 10 * DELTA / TIME_BASE_DEN, end: 25 * DELTA / TIME_BASE_DEN,
    keyframes: [{ frameId: 15, time: 15 * DELTA / TIME_BASE_DEN, thumb: "/thumb/L21_V001/15", frame: "/frame/L21_V001/15" }] },
  { index: 2, startFrame: 30, endFrame: 49, start: 30 * DELTA / TIME_BASE_DEN, end: 50 * DELTA / TIME_BASE_DEN,
    keyframes: [{ frameId: 35, time: 35 * DELTA / TIME_BASE_DEN, thumb: "/thumb/L21_V001/35", frame: "/frame/L21_V001/35" }] },
];

const frameMapPayload = {
  videoId: "L21_V001",
  timeBaseNumerator: 1,
  timeBaseDenominator: TIME_BASE_DEN,
  firstPts: 0,
  ptsDeltas: Array.from({ length: 99 }, () => DELTA),
  frameCount: 100,
};

function json(data: unknown, init: Partial<Response> = {}): Response {
  return { ok: true, status: 200, json: async () => data, headers: new Headers(), ...init } as Response;
}

let frameCallback: VideoFrameRequestCallback | null = null;
let writeText: ReturnType<typeof vi.fn>;

function presentFrame(frameId: number) {
  const callback = frameCallback;
  frameCallback = null;
  act(() => {
    callback?.(0, { mediaTime: frameId * DELTA / TIME_BASE_DEN } as VideoFrameCallbackMetadata);
  });
}

beforeEach(() => {
  frameCallback = null;
  writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
  Object.defineProperty(HTMLMediaElement.prototype, "requestVideoFrameCallback", {
    configurable: true,
    value(callback: VideoFrameRequestCallback) { frameCallback = callback; return 1; },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "cancelVideoFrameCallback", {
    configurable: true,
    value() {},
  });
  Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
    configurable: true, writable: true, value: 0,
  });
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url === "/videos") {
      return json([
        { video_id: "K01_V001", group: "K01", year: "2025", fps: null,
          indexed: false, has_shots: true },
        { video_id: "L21_V001", group: "L21", year: "2026", fps: 30,
          indexed: true, has_shots: true },
        { video_id: "L21_V002", group: "L21", year: "2026", fps: 30,
          indexed: true, has_shots: false },
      ]);
    }
    if (url === "/shots/L21_V001/labels") {
      return json({ videoId: "L21_V001", labelRevision: 2, labels: {
        "1": { state: "excluded", reason: "presenter", reviewed_by: "owner@example.com" },
      }});
    }
    if (url.startsWith("/shots/L21_V001")) {
      const selectedSource = url.includes("keyframe_source=l2026") ? "l2026" : "l2025";
      return json({ videoId: "L21_V001", fps: 30, frameCount: 100,
        labelRevision: 1,
        keyframeSource: selectedSource, keyframeSources: [
          { name: "l2025", label: "L 2025" }, { name: "l2026", label: "L 2026" },
        ], shots: shots.map((shot) => shot.index === 1 ? { ...shot, reviewFrames: [10, 17, 24],
          label: { state: "suggested", reason: "presenter", confidence: .96 } } : shot) });
    }
    if (url === "/frame-map/L21_V001") return json(frameMapPayload);
    if (url === "/transcript/L21_V001") return json({ videoId: "L21_V001", cues: [
      { start: 0.1, end: 0.2, text: "shot đầu" },
      { start: 0.4, end: 0.7, text: "lời trong shot hai" },
      { start: 1.2, end: 1.5, text: "shot cuối" },
    ] });
    if (url.startsWith("/clip/")) {
      return {
        ok: true, status: 200, blob: async () => new Blob(["clip"]),
        headers: new Headers({ "Content-Disposition": 'attachment; filename="clip.mp4"' }),
      } as unknown as Response;
    }
    throw new Error(`unexpected fetch ${url}`);
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:clip");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
});

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("shotAt", () => {
  it("finds the shot holding a frame, including its edges", () => {
    expect(shotAt(shots, 0)).toBe(0);
    expect(shotAt(shots, 9)).toBe(0);
    expect(shotAt(shots, 10)).toBe(1);
    expect(shotAt(shots, 24)).toBe(1);
    expect(shotAt(shots, 49)).toBe(2);
  });

  it("reports no shot for gaps, negatives and unknown frames", () => {
    expect(shotAt(shots, 27)).toBe(-1);
    expect(shotAt(shots, 50)).toBe(-1);
    expect(shotAt(shots, -1)).toBe(-1);
    expect(shotAt(shots, null)).toBe(-1);
    expect(shotAt([], 0)).toBe(-1);
  });
});

describe("clip studio", () => {
  async function openVideo() {
    render(<ClipStudio />);
    fireEvent.click(await screen.findByRole("button", { name: /L21_V001/ }));
    await screen.findByLabelText("Video L21_V001");
    await waitFor(() => expect(screen.getByText("3 shot")).toBeInTheDocument());
    await waitFor(() => expect(frameCallback).not.toBeNull());
  }

  it("flags videos whose shot boundaries have not been detected yet", async () => {
    render(<ClipStudio />);

    const missing = await screen.findByRole("button", { name: /L21_V002/ });
    expect(missing).toHaveTextContent("no shots");
    expect(await screen.findByRole("button", { name: /L21_V001/ }))
      .not.toHaveTextContent("no shots");
  });

  it("lists videos that shots exist for but the search index does not cover", async () => {
    render(<ClipStudio />);

    const unindexed = await screen.findByRole("button", { name: /K01_V001/ });
    expect(unindexed).not.toHaveTextContent("no shots");
    expect(within(unindexed).getByTitle("Not in search index")).toBeInTheDocument();
  });

  it("narrows the list to one group", async () => {
    render(<ClipStudio />);
    await screen.findByRole("button", { name: /K01_V001/ });

    fireEvent.change(screen.getByLabelText("Filter by group"), { target: { value: "K01" } });

    expect(screen.getByRole("button", { name: /K01_V001/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /L21_V001/ })).not.toBeInTheDocument();
  });

  it("starts from the search profile and switches keyframe sources", async () => {
    render(<ClipStudio profile="l2025" />);
    fireEvent.click(await screen.findByRole("button", { name: /L21_V001/ }));
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledWith(
      "/shots/L21_V001?keyframe_source=l2025"
    ));

    fireEvent.change(await screen.findByLabelText("Keyframe source"), { target: { value: "l2026" } });

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledWith(
      "/shots/L21_V001?keyframe_source=l2026"
    ));
  });

  it("names the export button after the shot under the playhead", async () => {
    await openVideo();
    presentFrame(15);

    await waitFor(() => expect(screen.getByRole("button", { name: /Export shot #1/ }))
      .toBeEnabled());
  });

  it("keeps export disabled while the video is playing", async () => {
    await openVideo();
    presentFrame(15);
    fireEvent.play(screen.getByLabelText("Video L21_V001"));

    expect(screen.getByRole("button", { name: /Export shot #1/ })).toBeDisabled();
    expect(screen.getByText("Pause the video to export the shot you're viewing.")).toBeInTheDocument();
  });

  it("downloads the shot under the playhead, not the one that was clicked", async () => {
    await openVideo();
    presentFrame(35);

    const button = await screen.findByRole("button", { name: /Export shot #2/ });
    await act(async () => { fireEvent.click(button); });

    expect(globalThis.fetch).toHaveBeenCalledWith("/clip/L21_V001/2");
    expect(await screen.findByText("Downloaded clip.mp4")).toBeInTheDocument();
  });

  it("reports the server error when a clip cannot be cut", async () => {
    await openVideo();
    presentFrame(15);
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      json({ detail: "ffmpeg failed" }, { ok: false, status: 500 })
    );

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /Export shot #1/ }));
    });

    expect(await screen.findByRole("alert")).toHaveTextContent("ffmpeg failed");
  });

  it("seeks to the first frame of a shot picked from the strip", async () => {
    await openVideo();
    const video = screen.getByLabelText("Video L21_V001") as HTMLVideoElement;

    fireEvent.click(screen.getByRole("button", { name: /#2/ }));

    expect(video.currentTime).toBeCloseTo(30 * DELTA / TIME_BASE_DEN);
  });

  it("shows keyframes and only transcript cues overlapping the current shot", async () => {
    await openVideo();
    presentFrame(15);

    expect(await screen.findByRole("img", { name: "Frame 15" })).toBeInTheDocument();
    const shotTranscript = screen.getByLabelText("Transcript for shot");
    expect(within(shotTranscript).getByText("lời trong shot hai")).toBeInTheDocument();
    expect(within(shotTranscript).queryByText("shot cuối")).not.toBeInTheDocument();
  });

  it("shows and copies the complete transcript below the shot dashboard", async () => {
    await openVideo();
    const full = screen.getByLabelText("Full transcript to copy");

    expect(within(full).getByText("shot đầu lời trong shot hai shot cuối")).toBeInTheDocument();
    await act(async () => { fireEvent.click(within(full).getByRole("button", { name: "Copy all" })); });

    expect(writeText).toHaveBeenCalledWith("shot đầu lời trong shot hai shot cuối");
    expect(within(full).getByRole("button", { name: "Copied" })).toBeInTheDocument();
  });

  it("lets the owner confirm an automatic exclusion", async () => {
    render(<ClipStudio isAdmin />);
    fireEvent.click(await screen.findByRole("button", { name: /L21_V001/ }));
    await waitFor(() => expect(frameCallback).not.toBeNull());
    presentFrame(15);

    expect((await screen.findAllByText(/suggested · presenter/)).length).toBeGreaterThan(0);
    expect(screen.getByRole("option", { name: "Outro" })).toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Exclude from index" })); });

    expect(globalThis.fetch).toHaveBeenCalledWith("/shots/L21_V001/labels", expect.objectContaining({
      method: "PATCH",
    }));
    expect(await screen.findByText("Marked as excluded from index")).toBeInTheDocument();
  });
});
