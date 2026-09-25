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

/**
 * The frame on screen at `mediaTime`: the exact PTS match when the decoder pinned one, otherwise the
 * nearest frame -- while a video plays `mediaTime` usually falls between two PTS values, and an event
 * picked then still has to name one frame.
 */
export function frameIdAt(mapping: FrameMap, mediaTime: number): number | null {
  const exact = resolveFrameId(mapping, mediaTime);
  if (exact !== null) return exact;
  if (!Number.isFinite(mediaTime)) return null;
  const target = mapping.firstPts + mediaTime * mapping.timeBaseDenominator / mapping.timeBaseNumerator;
  let low = 0;
  let high = mapping.pts.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (mapping.pts[middle] < target) low = middle + 1;
    else high = middle;
  }
  if (low > 0 && Math.abs(mapping.pts[low] - target) >= Math.abs(mapping.pts[low - 1] - target)) return low - 1;
  return low;
}

export function frameTime(mapping: FrameMap, frameId: number): number {
  return (mapping.pts[frameId] - mapping.firstPts) * mapping.timeBaseNumerator / mapping.timeBaseDenominator;
}

export function frameTimestampMs(mapping: FrameMap, frameId: number): number {
  if (!Number.isSafeInteger(frameId) || frameId < 0 || frameId >= mapping.frameCount)
    throw new Error("frame is outside the original video");
  const delta = BigInt(mapping.pts[frameId] - mapping.firstPts);
  const numerator = delta * BigInt(mapping.timeBaseNumerator) * 1000n;
  const denominator = BigInt(mapping.timeBaseDenominator);
  return Number((2n * numerator + denominator) / (2n * denominator));
}

export async function fetchFrameMap(videoId: string): Promise<FrameMap> {
  const response = await fetch(`/frame-map/${encodeURIComponent(videoId)}`);
  if (!response.ok) throw new Error("Exact frame unavailable");
  return decodeFrameMap(await response.json() as FrameMapPayload);
}

/**
 * `fetchFrameMap`, retried: without the map the player cannot name a frame at all, so one dropped
 * request must not leave a dialog where nothing can be submitted. A venue link that serves maps from
 * another machine drops connections under load.
 */
export async function fetchFrameMapWithRetry(videoId: string, attempts = 3, delayMs = 500): Promise<FrameMap> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await fetchFrameMap(videoId);
    } catch (error) {
      if (attempt >= attempts) throw error;
      await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
    }
  }
}
