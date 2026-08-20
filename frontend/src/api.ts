import type { CompareResult, SearchRequest, SearchResponse, Session } from "./types";

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

export async function compareIndexes(text: string, k: number): Promise<CompareResult[]> {
  const response = await fetch("/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, k }),
  });
  const data = await readJson<{ results: CompareResult[] }>(response);
  return data.results;
}

export async function fetchSession(): Promise<Session> {
  const response = await fetch("/session");
  return readJson<Session>(response);
}
