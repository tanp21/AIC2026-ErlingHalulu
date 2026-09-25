import type { CheckLookup, CheckRequest, CheckResult } from "./types";

/**
 * "Check by AI" for one moment of one query.
 *
 * The server caches an answer by its SHOT, and every answer says which frames it `covers`. So
 * this module keeps the same answers in the browser, indexed by the frames they speak for:
 * clicking along the filmstrip inside one shot renders the answer with no request at all, and
 * the strip can badge the frames that already have one. A reload falls back to `lookupCheck`,
 * which asks the server's cache without ever calling a model.
 */

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401) throw new Error("Sign in to Tailscale to access this workspace.");
    if (response.status === 403) throw new Error("Your Tailscale account is not authorized.");
    const message = typeof (data as { detail?: unknown })?.detail === "string"
      ? (data as { detail: string }).detail : "The AI check failed";
    throw new CheckApiError(message, response.status);
  }
  return data as T;
}

export class CheckApiError extends Error {
  constructor(message: string, readonly status = 0) {
    super(message);
    this.name = "CheckApiError";
  }
}

function fold(text: string | undefined): string {
  return (text ?? "").trim().replace(/\s+/g, " ").toLowerCase();
}

/** What identifies a query for caching -- the same fields the server hashes, one per channel. */
export function queryKey(request: Pick<CheckRequest, "visual_text" | "audio_text" | "ocr_text" | "kind" | "question">): string {
  return [request.kind, fold(request.visual_text), fold(request.audio_text), fold(request.ocr_text),
    fold(request.question)].join("");
}

interface Cached { result: CheckResult; window: number }

const cache = new Map<string, Cached[]>();

function bucketKey(query: string, videoId: string): string {
  return `${query}${videoId}`;
}

/** Remember an answer under every frame and second its shot covers. */
export function remember(request: CheckRequest, result: CheckResult): void {
  if (result.covers === null) return;   // a forced, frame-exact answer speaks only for itself
  const key = bucketKey(queryKey(request), result.video_id);
  const held = cache.get(key) ?? [];
  cache.set(key, [...held.filter((entry) => entry.result.key !== result.key),
    { result, window: result.window_seconds }]);
}

/**
 * An answer already held for this moment, or null.
 *
 * A frame matches when it is inside the shot the answer covers; a time matches the same way.
 * `covers.start_frame` is null for a video with no shot boundaries, where the server buckets by
 * time instead -- then only the seconds can decide.
 */
export function cachedCheck(request: CheckRequest): CheckResult | null {
  const held = cache.get(bucketKey(queryKey(request), request.video_id));
  if (!held) return null;
  const window = request.window_seconds ?? 0;
  for (const { result, window: had } of held) {
    if (window && window !== had) continue;
    const covers = result.covers;
    if (!covers) continue;
    if (request.frame_idx !== undefined && covers.start_frame !== null && covers.end_frame !== null) {
      if (request.frame_idx >= covers.start_frame && request.frame_idx <= covers.end_frame) return result;
      continue;
    }
    const at = request.time;
    if (at !== undefined && at >= covers.start && at < covers.end) return result;
  }
  return null;
}

export function forgetChecks(): void {
  cache.clear();
}

export async function runCheck(request: CheckRequest, signal?: AbortSignal): Promise<CheckResult> {
  const response = await fetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  const result = await readJson<CheckResult>(response);
  remember(request, result);
  return result;
}

export async function lookupCheck(request: CheckRequest, signal?: AbortSignal): Promise<CheckLookup> {
  const params = new URLSearchParams({ kind: request.kind, video_id: request.video_id });
  if (request.visual_text) params.set("visual_text", request.visual_text);
  if (request.audio_text) params.set("audio_text", request.audio_text);
  if (request.ocr_text) params.set("ocr_text", request.ocr_text);
  if (request.question) params.set("question", request.question);
  if (request.frame_idx !== undefined) params.set("frame_idx", String(request.frame_idx));
  if (request.time !== undefined) params.set("time", String(request.time));
  if (request.window_seconds) params.set("window_seconds", String(request.window_seconds));
  if (request.profile) params.set("profile", request.profile);
  const found = await readJson<CheckLookup>(await fetch(`/api/check/lookup?${params}`, { signal }));
  if (found.status === "hit" && found.result) remember(request, found.result);
  return found;
}
