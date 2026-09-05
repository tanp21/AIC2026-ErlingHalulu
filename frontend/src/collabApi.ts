import type { AnswerDraft, CollabQuery, ExportPreview, RoundSnapshot, RoundSummary } from "./types";

export class CollabApiError extends Error {
  current?: CollabQuery;
  preview?: ExportPreview;
}

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    const error = new CollabApiError(typeof data?.detail === "string" ? data.detail : "Request failed");
    error.current = data?.current;
    error.preview = data?.current?.missing ? data.current : undefined;
    throw error;
  }
  return data as T;
}

export async function fetchRounds(): Promise<RoundSummary[]> {
  return readJson(await fetch("/api/collab/rounds"));
}

export async function fetchRound(id: number, etag?: string): Promise<{ round: RoundSnapshot; etag: string } | null> {
  const response = await fetch(`/api/collab/rounds/${id}`, {
    headers: etag ? { "If-None-Match": etag } : {},
  });
  if (response.status === 304) return null;
  return { round: await readJson<RoundSnapshot>(response), etag: response.headers.get("ETag") ?? "" };
}

export async function uploadRound(name: string, file: File): Promise<RoundSnapshot> {
  return readJson(await fetch(`/api/collab/rounds?name=${encodeURIComponent(name)}`, {
    method: "POST", headers: { "Content-Type": "application/zip" }, body: file,
  }));
}

export async function activateRound(id: number): Promise<RoundSnapshot> {
  return readJson(await fetch(`/api/collab/rounds/${id}/active`, { method: "PUT" }));
}

export interface QueryPatch {
  expected_version: number;
  assignee?: string | null;
  answers?: AnswerDraft[];
  event_count?: number;
}

export async function patchQuery(id: number, patch: QueryPatch): Promise<CollabQuery> {
  return readJson(await fetch(`/api/collab/queries/${id}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
  }));
}

export async function reviewQuery(id: number, expectedVersion: number, reviewed: boolean): Promise<CollabQuery> {
  return readJson(await fetch(`/api/collab/queries/${id}/review`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_version: expectedVersion, reviewed }),
  }));
}

export async function fetchExportPreview(id: number): Promise<ExportPreview> {
  return readJson(await fetch(`/api/collab/rounds/${id}/export-preview`));
}

export async function downloadExport(id: number, allowIncomplete: boolean): Promise<{ blob: Blob; filename: string }> {
  const suffix = allowIncomplete ? "?allow_incomplete=true" : "";
  const response = await fetch(`/api/collab/rounds/${id}/export${suffix}`);
  if (!response.ok) await readJson(response);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename = disposition.match(/filename="([^"]+)"/)?.[1] ?? "submission.zip";
  return { blob: await response.blob(), filename };
}
