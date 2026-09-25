import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("uses object positions as the first event's query or filter", async () => {
  const requests: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input);
    const data = path === "/session" ? { login: "owner@example.com", is_admin: true }
      : path === "/profiles" ? [{ name: "l2026-v2", label: "L/M/S", is_default: true,
                                  count: 590325, video_count: 2000, indexes: ["flat"] }]
      : path === "/api/collab/rounds" ? []
      : path === "/api/dres/status" ? { configured: false, connected: false, evaluations: [] }
      : path === "/objects/catalog" ? { labels: ["person", "car"], profiles: [], alignment_max_seconds: 0.25 }
      : path === "/search" ? (() => { requests.push(JSON.parse(String(init?.body))); return {
          results: [], latency_ms: 1, mode: "object", index: "flat", index_info: { type: "Object positions" },
          next_cursor: null, has_more: false, total_results: 0, truncated: false,
        }; })()
      : path === "/search/temporal/image" ? (() => {
          requests.push(JSON.parse(String((init?.body as FormData).get("payload"))));
          return { videos: [], total_sequences: 0, latency_ms: 1, mode: "temporal" };
        })()
      : {};
    return { ok: true, status: 200, headers: new Headers(), json: async () => data } as Response;
  });

  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Object for event E1" }));
  fireEvent.click(screen.getByRole("button", { name: "Draw object positions" }));
  const objectDialog = within(screen.getByRole("dialog", { name: "Object event E1 editor" }));
  await waitFor(() => expect(objectDialog.getByRole("button", { name: "Select person" })).toBeInTheDocument());
  fireEvent.click(objectDialog.getByRole("button", { name: "+ Add person" }));
  fireEvent.click(objectDialog.getByRole("button", { name: "Use positions" }));
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(requests).toHaveLength(1));
  expect(requests[0]).toMatchObject({ mode: "object", profile: "l2026-v2",
    object_query: { clauses: [{ label: "person", region: [0.35, 0.35, 0.65, 0.65], min_count: 1 }] } });

  fireEvent.click(screen.getByRole("button", { name: "Visual text for event E1" }));
  fireEvent.change(screen.getByLabelText("Visual query (English)"), { target: { value: "street" } });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(requests).toHaveLength(2));
  expect(requests[1]).toMatchObject({ mode: "visual", visual_text: "street",
    object_query: { clauses: [{ label: "person", region: [0.35, 0.35, 0.65, 0.65] }] } });

  fireEvent.click(screen.getByRole("button", { name: "Add event" }));
  fireEvent.click(screen.getByRole("button", { name: "OCR for event E2" }));
  fireEvent.change(screen.getByLabelText("OCR query E2"), { target: { value: "shop" } });
  fireEvent.click(screen.getByRole("button", { name: "Add object filter to E2" }));
  const filterDialog = within(screen.getByRole("dialog", { name: "Object position filter · E2 editor" }));
  await waitFor(() => expect(filterDialog.getByRole("button", { name: "Select car" })).toBeInTheDocument());
  fireEvent.click(filterDialog.getByRole("button", { name: "Select car" }));
  fireEvent.click(filterDialog.getByRole("button", { name: "+ Add car" }));
  fireEvent.click(filterDialog.getByRole("button", { name: "Use positions" }));
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(requests).toHaveLength(3));
  expect(requests[2]).toMatchObject({ events: [
    { kind: "text", text: "street", object_filter: { clauses: [{ label: "person" }] } },
    { kind: "ocr", text: "shop", object_filter: { clauses: [{ label: "car" }] } },
  ] });
});
