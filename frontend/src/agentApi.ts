import type { AgentRunRequest, AgentRunResponse } from "./types";

export class AgentApiError extends Error {}

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new AgentApiError(typeof data?.detail === "string" ? data.detail : "Request failed");
  }
  return data as T;
}

export async function startAgentRun(request: AgentRunRequest): Promise<AgentRunResponse> {
  return readJson(await fetch("/api/agent/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }));
}

export async function fetchAgentRun(runId: string): Promise<AgentRunResponse> {
  return readJson(await fetch(`/api/agent/runs/${runId}`));
}

export async function downloadAgentSubmission(runId: string): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`/api/agent/runs/${runId}/submission.zip`);
  if (!response.ok) await readJson(response);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename = disposition.match(/filename="([^"]+)"/)?.[1] ?? "submission.zip";
  return { blob: await response.blob(), filename };
}
