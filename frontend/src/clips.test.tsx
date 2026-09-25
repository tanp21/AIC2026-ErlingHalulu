import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ClipStudio, captionAt, containedVideoRect, detectionsAt, shotAt } from "./clips";
import type { CaptionWindow, Shot } from "./types";

const TIME_BASE_DEN = 90000;
const DELTA = 3003;

const shots: Shot[] = [
  { index: 0, startFrame: 0, endFrame: 9, start: 0, end: 10 * DELTA / TIME_BASE_DEN, keyframes: [] },
  { index: 1, startFrame: 10, endFrame: 24, start: 10 * DELTA / TIME_BASE_DEN, end: 25 * DELTA / TIME_BASE_DEN,
    keyframes: [{ frameId: 15, time: 15 * DELTA / TIME_BASE_DEN, thumb: "/thumb/L21_V001/15", frame: "/frame/L21_V001/15" }] },
  { index: 2, startFrame: 30, endFrame: 49, start: 30 * DELTA / TIME_BASE_DEN, end: 50 * DELTA / TIME_BASE_DEN,
    keyframes: [{ frameId: 35, time: 35 * DELTA / TIME_BASE_DEN, thumb: "/thumb/L21_V001/35", frame: "/frame/L21_V001/35" }] },
];

const shotSeconds = (frame: number) => frame * DELTA / TIME_BASE_DEN;

const captions: CaptionWindow[] = [
  { shot_idx: 0, window_idx: 0, window_count: 1, frame_idx: null, start: shotSeconds(0), end: shotSeconds(10),
    time: shotSeconds(5), text: "A chef plates a dish." },
  { shot_idx: 1, window_idx: 0, window_count: 2, frame_idx: null, start: shotSeconds(10), end: shotSeconds(20),
    time: shotSeconds(15), text: "A woman walks past the first stall." },
  { shot_idx: 1, window_idx: 1, window_count: 2, frame_idx: null, start: shotSeconds(16), end: shotSeconds(24),
    time: shotSeconds(20), text: "The camera pans to a second stall." },
  { shot_idx: 2, window_idx: 0, window_count: 1, frame_idx: null, start: shotSeconds(30), end: shotSeconds(49),
    time: shotSeconds(40), text: "Two boats cross the harbour." },
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
let captionScenario: "captions" | "partial" | "none" | "error";

function presentFrame(frameId: number) {
  const callback = frameCallback;
  frameCallback = null;
  act(() => {
    callback?.(0, { mediaTime: frameId * DELTA / TIME_BASE_DEN } as VideoFrameCallbackMetadata);
  });
}

beforeEach(() => {
  frameCallback = null;
  captionScenario = "captions";
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
    if (url === "/objects/L21_V001/detections") return json({ video_id: "L21_V001", frames: [
      { frame_idx: 15, time: shotSeconds(15), detections: [
        { label: "person", confidence: .91, box: [.1, .15, .3, .8] },
      ] },
    ] });
    if (url === "/shots/L21_V001/labels") {
      return json({ videoId: "L21_V001", labelRevision: 2, labels: {
        "1": { state: "excluded", reason: "presenter", reviewed_by: "owner@example.com" },
      }});
    }
    if (url === "/captions/L21_V001/windows") {
      if (captionScenario === "error") {
        return json({ detail: "caption data unavailable" }, { ok: false, status: 503 });
      }
      const payload = captionScenario === "captions" ? captions
        : captionScenario === "partial" ? captions.filter((caption) => caption.shot_idx === 0) : [];
      return json({ videoId: "L21_V001", available: payload.length > 0, captions: payload });
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

describe("captionAt", () => {
  const windows: CaptionWindow[] = [
    { shot_idx: 1, window_idx: 0, frame_idx: null, start: 10, end: 20, time: 15, text: "first" },
    { shot_idx: 1, window_idx: 1, frame_idx: null, start: 16, end: 24, time: 20, text: "second" },
    { shot_idx: 2, window_idx: 0, frame_idx: null, start: 30, end: 40, time: 35, text: "third" },
  ];

  it("returns nothing when the video has no windows", () => {
    expect(captionAt([], 5)).toBeNull();
  });

  it("picks the window holding the time, deepest one where windows overlap", () => {
    expect(captionAt(windows, 12)?.text).toBe("first");
    expect(captionAt(windows, 23)?.text).toBe("second");
    expect(captionAt(windows, 17)?.text).toBe("first");
    expect(captionAt(windows, 19)?.text).toBe("second");
  });

  it("includes the window bounds and falls back to the nearest window", () => {
    expect(captionAt(windows, 10)?.text).toBe("first");
    expect(captionAt(windows, 24)?.text).toBe("second");
    expect(captionAt(windows, 27)?.text).toBe("second");
    expect(captionAt(windows, 900)?.text).toBe("third");
    expect(captionAt([windows[2], windows[1], windows[0]], 12)?.text).toBe("first");
  });
});

describe("detectionsAt", () => {
  const samples = [
    { frame_idx: 10, time: 1, detections: [] },
    { frame_idx: 20, time: 2, detections: [] },
  ];

  it("chooses the closest detector sample and hides stale boxes", () => {
    expect(detectionsAt(samples, 1.4)?.frame_idx).toBe(10);
    expect(detectionsAt(samples, 1.6)?.frame_idx).toBe(20);
    expect(detectionsAt(samples, 4)).toBeNull();
  });
});

describe("containedVideoRect", () => {
  it("fits the overlay to portrait video inside a landscape stage", () => {
    expect(containedVideoRect(1200, 600, 900, 1600)).toEqual({
      left: 431.25, top: 0, width: 337.5, height: 600,
    });
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

  it("draws detections over the matching playback frame and can hide them", async () => {
    await openVideo();
    presentFrame(15);

    expect(await screen.findByRole("img", { name: "Object detections near frame 15" })).toBeInTheDocument();
    expect(screen.getByText("person 91%")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Object detection"));
    expect(screen.queryByRole("img", { name: "Object detections near frame 15" })).not.toBeInTheDocument();
  });

  it("requests fullscreen on the overlay stage from the external player action", async () => {
    const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "requestFullscreen");
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", { configurable: true, value: requestFullscreen });
    try {
      await openVideo();
      fireEvent.click(screen.getByRole("button", { name: "Fullscreen video" }));
      expect(requestFullscreen).toHaveBeenCalledOnce();
      expect(requestFullscreen.mock.contexts[0]).toHaveClass("clip-video-stage");
    } finally {
      if (previous) Object.defineProperty(HTMLElement.prototype, "requestFullscreen", previous);
      else Reflect.deleteProperty(HTMLElement.prototype, "requestFullscreen");
    }
  });

  it("downloads the shot under the playhead, not the one that was clicked", async () => {
    await openVideo();
    presentFrame(35);

    const button = await screen.findByRole("button", { name: /Export shot #2/ });
    await act(async () => { fireEvent.click(button); });

    expect(globalThis.fetch).toHaveBeenCalledWith("/clip/L21_V001/2");
    expect(await screen.findByText("Downloaded clip.mp4")).toBeInTheDocument();
  });

  it("shows the caption of the shot under the playhead below the keyframes", async () => {
    await openVideo();
    presentFrame(15);

    const caption = screen.getByLabelText("Shot caption");
    expect(await within(caption).findByText("A woman walks past the first stall.")).toBeInTheDocument();
    expect(within(caption).getByText(/Shot #1 · window 1\/2/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Caption for shot")).not.toBeInTheDocument();

    presentFrame(35);

    expect(await within(caption).findByText("Two boats cross the harbour.")).toBeInTheDocument();
    expect(within(caption).queryByText("A woman walks past the first stall.")).not.toBeInTheDocument();
  });

  it("follows the caption window the playhead sits deepest inside", async () => {
    await openVideo();
    presentFrame(17);

    const caption = screen.getByLabelText("Shot caption");
    expect(await within(caption).findByText("A woman walks past the first stall.")).toBeInTheDocument();

    presentFrame(19);

    expect(await within(caption).findByText("The camera pans to a second stall.")).toBeInTheDocument();
    expect(within(caption).queryByText("A woman walks past the first stall.")).not.toBeInTheDocument();
  });

  it("copies the whole caption of the shot under the playhead", async () => {
    await openVideo();
    presentFrame(15);
    const caption = screen.getByLabelText("Shot caption");

    await act(async () => {
      fireEvent.click(await within(caption).findByRole("button", { name: "Copy" }));
    });

    expect(writeText).toHaveBeenCalledWith("A woman walks past the first stall.");
    expect(within(caption).getByRole("button", { name: "Copied" })).toBeInTheDocument();
  });

  it("disables the caption copy button when there is nothing to copy", async () => {
    captionScenario = "none";
    await openVideo();
    presentFrame(15);

    expect(within(screen.getByLabelText("Shot caption")).getByRole("button", { name: "Copy" }))
      .toBeDisabled();
  });

  it("says the video has no captions when the endpoint returns none", async () => {
    captionScenario = "none";
    await openVideo();
    presentFrame(15);

    expect(await within(screen.getByLabelText("Shot caption")).findByText("This video has no captions."))
      .toBeInTheDocument();
  });

  it("says a shot has no caption when the video only captioned other shots", async () => {
    captionScenario = "partial";
    await openVideo();
    presentFrame(15);

    const caption = screen.getByLabelText("Shot caption");
    expect(await within(caption).findByText("This shot has no caption.")).toBeInTheDocument();
    expect(within(caption).queryByText("A chef plates a dish.")).not.toBeInTheDocument();
  });

  it("reports a caption that could not be loaded", async () => {
    captionScenario = "error";
    await openVideo();
    presentFrame(15);

    const caption = screen.getByLabelText("Shot caption");
    expect(await within(caption).findByText("Failed to load caption.")).toBeInTheDocument();
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
