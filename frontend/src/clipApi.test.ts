import { afterEach, describe, expect, it, vi } from "vitest";

import { copyKeyframe, downloadClip, fetchShots, fetchVideoList, updateShotLabels } from "./clipApi";

function response(data: unknown, init: Partial<Response> = {}): Response {
  return { ok: true, status: 200, json: async () => data, headers: new Headers(), ...init } as Response;
}

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("clip API", () => {
  it("lists videos and shots", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(response([{ video_id: "L21_V001" }]))
      .mockResolvedValueOnce(response({ videoId: "L21_V001", shots: [] }));

    expect(await fetchVideoList()).toEqual([{ video_id: "L21_V001" }]);
    await fetchShots("L21_V001");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/videos");
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/shots/L21_V001");
  });

  it("requests shots from the selected keyframe source", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response({ videoId: "L21_V001", shots: [] })
    );

    await fetchShots("L21_V001", "l2026");

    expect(fetchMock).toHaveBeenCalledWith("/shots/L21_V001?keyframe_source=l2026");
  });

  it("sends an optimistic shot-label update", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response({ videoId: "L21_V001", labelRevision: 4, labels: {} })
    );

    await updateShotLabels("L21_V001", 3, [
      { shot_index: 12, state: "excluded", reason: "presenter" },
    ]);

    expect(fetchMock).toHaveBeenCalledWith("/shots/L21_V001/labels", expect.objectContaining({
      method: "PATCH",
      body: JSON.stringify({ expected_revision: 3, updates: [
        { shot_index: 12, state: "excluded", reason: "presenter" },
      ] }),
    }));
  });

  it("surfaces the server detail when a clip cannot be cut", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response({ detail: "shot 7 is outside L21_V001" }, { ok: false, status: 422 })
    );

    await expect(downloadClip("L21_V001", 7)).rejects.toThrow("shot 7 is outside L21_V001");
  });

  it("saves the clip under the filename the server chose", async () => {
    const headers = new Headers({
      "Content-Disposition": 'attachment; filename="L21_V001_shot0012_f000450-000612.mp4"',
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, status: 200, headers, blob: async () => new Blob(["clip"]),
    } as unknown as Response);
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:clip");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    expect(await downloadClip("L21_V001", 12)).toBe("L21_V001_shot0012_f000450-000612.mp4");
    expect(click).toHaveBeenCalled();
  });

  it("copies a full keyframe as PNG", async () => {
    const write = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { write } });
    vi.stubGlobal("ClipboardItem", class { constructor(public data: Record<string, Blob>) {} });
    vi.stubGlobal("createImageBitmap", vi.fn().mockResolvedValue({ width: 1920, height: 1080, close: vi.fn() }));
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true, blob: async () => new Blob(["jpeg"], { type: "image/jpeg" }),
    } as Response);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      { drawImage: vi.fn() } as unknown as CanvasRenderingContext2D
    );
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((callback) => {
      callback(new Blob(["png"], { type: "image/png" }));
    });

    await copyKeyframe("/frame/L21_V001/15");

    expect(write).toHaveBeenCalledOnce();
  });
});
