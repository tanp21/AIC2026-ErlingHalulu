import type { CompareResult, ImageSearchRequest, IndexProfile, SearchRequest, SearchResponse, Session, TemporalSearchRequest, TemporalSearchResponse, TranscriptResponse, SearchResult } from "./types";

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401) throw new Error("Sign in to Tailscale to access this workspace.");
    if (response.status === 403) throw new Error("Your Tailscale account is not authorized.");
    const message = typeof data?.detail === "string" ? data.detail : "Request failed";
    throw new Error(message);
  }
  return data as T;
}

export async function searchVideos(request: SearchRequest, signal: AbortSignal): Promise<SearchResponse> {
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  return readJson<SearchResponse>(response);
}

export async function searchVideosByImage(
  request: ImageSearchRequest,
  image: File | null,
  signal: AbortSignal,
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("payload", JSON.stringify(request));
  if (image) form.append("image", image);
  const response = await fetch("/search/image", { method: "POST", body: form, signal });
  return readJson<SearchResponse>(response);
}

export async function searchNextPage(cursor: string, signal: AbortSignal): Promise<SearchResponse> {
  const response = await fetch("/search/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cursor }),
    signal,
  });
  return readJson<SearchResponse>(response);
}

export async function searchTemporal(request: TemporalSearchRequest, signal: AbortSignal): Promise<TemporalSearchResponse> {
  const response = await fetch("/search/temporal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  return readJson<TemporalSearchResponse>(response);
}

export async function searchTemporalNextPage(cursor: string, signal: AbortSignal): Promise<TemporalSearchResponse> {
  const response = await fetch("/search/temporal/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cursor }),
    signal,
  });
  return readJson<TemporalSearchResponse>(response);
}

export async function compareIndexes(text: string): Promise<CompareResult[]> {
  const response = await fetch("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, k: 50 }),
  });
  const data = await readJson<{ results: CompareResult[] }>(response);
  return data.results;
}

export async function fetchSession(): Promise<Session> {
  const response = await fetch("/session");
  return readJson<Session>(response);
}

export async function fetchProfiles(): Promise<IndexProfile[]> {
  const response = await fetch("/profiles");
  return readJson<IndexProfile[]>(response);
}

export async function fetchTranscript(videoId: string): Promise<TranscriptResponse> {
  const response = await fetch(`/transcript/${encodeURIComponent(videoId)}`);
  return readJson<TranscriptResponse>(response);
}

export async function fetchKeyframeContext(videoId: string, frameId: number, profile?: string): Promise<{ video_id: string; frame_idx: number; frames: SearchResult[] }> {
  const params = new URLSearchParams({ radius: "10" });
  if (profile) params.set("profile", profile);
  const response = await fetch(`/context/${encodeURIComponent(videoId)}/${frameId}?${params}`, {
    cache: "no-store",
  });
  return readJson(response);
}
