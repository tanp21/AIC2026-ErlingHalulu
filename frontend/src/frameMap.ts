export interface FrameMapPayload {
  videoId: string;
  timeBaseNumerator: number;
  timeBaseDenominator: number;
  firstPts: number;
  ptsDeltas: number[];
  frameCount: number;
}

export interface FrameMap extends FrameMapPayload {
  pts: number[];
}

export function decodeFrameMap(payload: FrameMapPayload): FrameMap {
  if (![payload.timeBaseNumerator, payload.timeBaseDenominator, payload.firstPts, payload.frameCount]
    .every(Number.isSafeInteger)) {
    throw new Error("frame map metadata must be safe integers");
  }
  if (payload.timeBaseNumerator <= 0 || payload.timeBaseDenominator <= 0 || payload.frameCount <= 0) {
    throw new Error("invalid time base");
  }
  const pts = [payload.firstPts];
  for (const delta of payload.ptsDeltas) {
    if (!Number.isSafeInteger(delta) || delta <= 0) throw new Error("PTS must be strictly increasing");
    pts.push(pts[pts.length - 1] + delta);
  }
  if (pts.length !== payload.frameCount) throw new Error("frame count does not match PTS map");
  return { ...payload, pts };
}

export function resolveFrameId(mapping: FrameMap, mediaTime: number): number | null {
  const target = mapping.firstPts + Math.round(mediaTime * mapping.timeBaseDenominator / mapping.timeBaseNumerator);
  let low = 0;
  let high = mapping.pts.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const pts = mapping.pts[middle];
    if (pts === target) return middle;
    if (pts < target) low = middle + 1;
    else high = middle - 1;
  }
  return null;
}

export function frameTime(mapping: FrameMap, frameId: number): number {
  return (mapping.pts[frameId] - mapping.firstPts) * mapping.timeBaseNumerator / mapping.timeBaseDenominator;
}

export async function fetchFrameMap(videoId: string): Promise<FrameMap> {
  const response = await fetch(`/frame-map/${encodeURIComponent(videoId)}`);
  if (!response.ok) throw new Error("Exact frame unavailable");
  return decodeFrameMap(await response.json() as FrameMapPayload);
}
