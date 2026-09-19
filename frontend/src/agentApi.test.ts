import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentApiError, fetchAgentRun, startAgentRun } from "./agentApi";

function response(data: unknown, init: Partial<Response> = {}): Response {
  return {
    ok: true,
    status: 200,
    json: async () => data,
    headers: new Headers(),
    ...init,
  } as Response;
}

afterEach(() => vi.restoreAllMocks());

describe("agent API", () => {
  it("posts the query and the teammate's own keys", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response({ run_id: "a", status: "running" }, { status: 202 })
    );

    await startAgentRun({
      text: "một chiếc thuyền đỏ", kind: "kis", mode: "fast",
      deepseek_api_key: "dsk", gemini_api_key: "gmk",
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/agent/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "một chiếc thuyền đỏ", kind: "kis", mode: "fast",
        deepseek_api_key: "dsk", gemini_api_key: "gmk",
      }),
    });
  });

  it("surfaces the server error detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response({ detail: "an agent run is already in progress" }, { ok: false, status: 409 })
    );

    await expect(fetchAgentRun("a")).rejects.toBeInstanceOf(AgentApiError);
    await expect(fetchAgentRun("a")).rejects.toThrow("an agent run is already in progress");
  });
});
