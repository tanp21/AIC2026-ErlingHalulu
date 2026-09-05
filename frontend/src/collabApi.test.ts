import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchRound, patchQuery, uploadRound } from "./collabApi";

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

describe("collaboration API", () => {
  it("uses ETag and recognizes an unchanged round", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      response(null, { status: 304, ok: false })
    );
    const result = await fetchRound(7, '"round-7-2"');
    expect(result).toBeNull();
    expect(fetchMock).toHaveBeenCalledWith("/api/collab/rounds/7", {
      headers: { "If-None-Match": '"round-7-2"' },
    });
  });

  it("keeps the current query attached to a stale update error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response(
      { detail: "stale query version", current: { id: 4, version: 2 } },
      { status: 409, ok: false },
    ));
    await expect(patchQuery(4, { expected_version: 1, assignee: "mate@example.com" }))
      .rejects.toMatchObject({ message: "stale query version", current: { id: 4, version: 2 } });
  });

  it("uploads the ZIP as a raw request body", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(response({ id: 1 }));
    const file = new File(["zip"], "queries.zip", { type: "application/zip" });
    await uploadRound("Round P1", file);
    expect(fetchMock).toHaveBeenCalledWith("/api/collab/rounds?name=Round%20P1", {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: file,
    });
  });
});
