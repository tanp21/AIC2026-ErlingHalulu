import type { ObjectDetectionsResponse, ShotLabelReason, ShotLabelsResponse, ShotsResponse, VideoSummary } from "./types";

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : "Request failed");
  }
  return data as T;
}

export async function fetchVideoList(): Promise<VideoSummary[]> {
  return readJson<VideoSummary[]>(await fetch("/videos"));
}

export async function fetchObjectDetections(videoId: string): Promise<ObjectDetectionsResponse> {
  return readJson<ObjectDetectionsResponse>(await fetch(`/objects/${encodeURIComponent(videoId)}/detections`));
}

export async function fetchShots(videoId: string, keyframeSource?: string): Promise<ShotsResponse> {
  const query = keyframeSource
    ? `?keyframe_source=${encodeURIComponent(keyframeSource)}`
    : "";
  return readJson<ShotsResponse>(await fetch(`/shots/${encodeURIComponent(videoId)}${query}`));
}

export async function updateShotLabels(
  videoId: string,
  expectedRevision: number,
  updates: { shot_index: number; state: "excluded" | "included" | null; reason?: ShotLabelReason }[],
): Promise<ShotLabelsResponse> {
  return readJson<ShotLabelsResponse>(await fetch(`/shots/${encodeURIComponent(videoId)}/labels`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: expectedRevision, updates }),
  }));
}

export async function copyKeyframe(url: string): Promise<void> {
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error("Browser does not support copying images to the clipboard");
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to load keyframe");
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d");
  if (!context) { bitmap.close(); throw new Error("Failed to process keyframe"); }
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const png = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
    (blob) => blob ? resolve(blob) : reject(new Error("Failed to process keyframe")),
    "image/png",
  ));
  await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
}

export async function downloadClip(videoId: string, shotIndex: number): Promise<string> {
  const response = await fetch(`/clip/${encodeURIComponent(videoId)}/${shotIndex}`);
  if (!response.ok) await readJson(response);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename = disposition.match(/filename="([^"]+)"/)?.[1]
    ?? `${videoId}_shot${shotIndex}.mp4`;
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
  return filename;
}
