import type { DresAttempt, DresCandidate, DresContext, DresPrepared, DresStatus, QueryKind } from "./types";

export class DresApiError extends Error {
  constructor(message: string, readonly status: number, readonly category = "not_sent") {
    super(message);
  }
}

async function read<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new DresApiError(
    typeof data?.detail === "string" ? data.detail : "DRES request failed",
    response.status, typeof data?.category === "string" ? data.category : "not_sent",
  );
  if (data === null) throw new DresApiError("DRES API returned malformed JSON", 502);
  return data as T;
}

function post<T>(path: string, payload?: unknown): Promise<T> {
  return fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: payload === undefined ? "{}" : JSON.stringify(payload),
  }).then(read<T>);
}

export async function dresStatus(): Promise<DresStatus> {
  const data = await fetch("/api/dres/status").then(read<DresStatus>);
  if (typeof data?.configured !== "boolean" || typeof data.credentials_configured !== "boolean"
    || typeof data.context_valid !== "boolean" || !Array.isArray(data.evaluations))
    throw new DresApiError("DRES API returned an unexpected status response", 502);
  return data;
}
export function dresConnect(): Promise<DresStatus> { return post("/api/dres/connect"); }
export function dresRelogin(): Promise<DresStatus> { return post("/api/dres/relogin"); }
export function dresReset(): Promise<DresStatus> { return post("/api/dres/reset"); }
export function dresRefresh(): Promise<DresStatus> { return post("/api/dres/evaluations/refresh"); }
export function dresSelectEvaluation(evaluation_id: string | null): Promise<DresStatus> {
  return post("/api/dres/evaluation", { evaluation_id });
}
export function dresSetContext(context: {
  expected_revision: number; kind: QueryKind; label: string; event_count: number | null;
}): Promise<DresContext> { return post("/api/dres/context", context); }
export function dresPrepare(candidate: DresCandidate): Promise<DresPrepared> {
  return post("/api/dres/prepare", { candidate });
}
export async function dresFrameTimestamp(videoId: string, frameIdx: number): Promise<number> {
  const data = await fetch(`/api/dres/frame/${encodeURIComponent(videoId)}/${frameIdx}`).then(read<{ timestamp_ms: number }>);
  if (!Number.isSafeInteger(data.timestamp_ms) || data.timestamp_ms < 0)
    throw new DresApiError("DRES frame map returned an invalid timestamp", 502);
  return data.timestamp_ms;
}
export async function dresOfficialFrameId(videoId: string, frameIdx: number): Promise<number> {
  const data = await fetch(`/api/dres/frame-id/${encodeURIComponent(videoId)}/${frameIdx}`).then(read<{ frame_id: number }>);
  if (!Number.isSafeInteger(data.frame_id) || data.frame_id < 0)
    throw new DresApiError("DRES frame map returned an invalid FRAME_ID", 502);
  return data.frame_id;
}
export type OfficialKeyframe = { frame_id: number; source_frame_idx: number; timestamp_ms: number };
export async function dresOfficialKeyframes(videoId: string): Promise<OfficialKeyframe[]> {
  const data = await fetch(`/api/dres/keyframes/${encodeURIComponent(videoId)}`).then(read<{ frames: OfficialKeyframe[] }>);
  if (!Array.isArray(data.frames) || data.frames.some((frame) =>
    !Number.isSafeInteger(frame.frame_id) || frame.frame_id < 0
    || !Number.isSafeInteger(frame.source_frame_idx) || frame.source_frame_idx < 0
    || !Number.isSafeInteger(frame.timestamp_ms) || frame.timestamp_ms < 0))
    throw new DresApiError("DRES returned an invalid official TRAKE keyframe map", 502);
  return data.frames;
}
export function nearestOfficialKeyframe(frames: OfficialKeyframe[], timestampMs: number): OfficialKeyframe | null {
  if (!frames.length || !Number.isFinite(timestampMs)) return null;
  return frames.reduce((best, frame) =>
    Math.abs(frame.timestamp_ms - timestampMs) < Math.abs(best.timestamp_ms - timestampMs) ? frame : best);
}
export function dresSubmit(candidate: DresCandidate, prepared: DresPrepared, retry_uncertain = false): Promise<DresAttempt> {
  return post("/api/dres/submit", {
    candidate, fingerprint: prepared.fingerprint,
    expected_evaluation_id: prepared.evaluation_id,
    expected_context_revision: prepared.context_revision,
    retry_uncertain,
  });
}
export function dresHistory(): Promise<DresAttempt[]> {
  return fetch("/api/dres/history").then(read<{ attempts: DresAttempt[] }>).then((data) => {
    if (!Array.isArray(data.attempts)) throw new DresApiError("DRES API returned unexpected history", 502);
    return data.attempts;
  });
}
