import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { ObjectCanvas, emptyObjectQuery } from "./objectCanvas";
import type { ObjectQuery } from "./types";

afterEach(() => vi.restoreAllMocks());

it("builds a positioned object query with minimum count and confidence", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true, json: async () => ({ labels: ["person", "car"], profiles: [], alignment_max_seconds: 0.25 }),
  } as Response);
  let query: ObjectQuery = emptyObjectQuery();
  const onChange = vi.fn((next: ObjectQuery) => { query = next; view.rerender(<ObjectCanvas query={query} onChange={onChange} />); });
  const view = render(<ObjectCanvas query={query} onChange={onChange} />);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Draw object positions" }));
  const dialog = within(screen.getByRole("dialog", { name: "Object positions editor" }));
  await waitFor(() => expect(dialog.getByRole("button", { name: "Select person" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(dialog.getByRole("button", { name: "+ Add person" }));
  expect(dialog.getByRole("button", { name: /person position 1/ })).toBeInTheDocument();
  fireEvent.change(dialog.getByLabelText("Minimum count for object 1"), { target: { value: "2" } });
  fireEvent.change(dialog.getByLabelText("Minimum object confidence"), { target: { value: "60" } });
  fireEvent.keyDown(dialog.getByRole("button", { name: /person position 1/ }), { key: "ArrowRight" });
  fireEvent.click(dialog.getByRole("button", { name: "Use positions" }));
  expect(query.clauses[0].min_count).toBe(2);
  expect(query.clauses[0].region[0]).toBeGreaterThan(0.35);
  expect(query.min_confidence).toBe(0.6);
  const viewScope = within(view.container);
  expect(viewScope.queryByRole("dialog")).not.toBeInTheDocument();
  expect(viewScope.getByRole("img", { name: "Object positions preview" })).toBeInTheDocument();
});

it("chooses a class from the searchable icon grid", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true, json: async () => ({ labels: ["person", "car"], profiles: [], alignment_max_seconds: 0.25 }),
  } as Response);
  let query: ObjectQuery = emptyObjectQuery();
  const onChange = vi.fn((next: ObjectQuery) => { query = next; view.rerender(<ObjectCanvas query={query} onChange={onChange} />); });
  const view = render(<ObjectCanvas query={query} onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", { name: "Draw object positions" }));
  const canvas = within(screen.getByRole("dialog", { name: "Object positions editor" }));
  await waitFor(() => expect(canvas.getByRole("button", { name: "Select car" })).toBeInTheDocument());
  fireEvent.change(canvas.getByLabelText("Find object class"), { target: { value: "car" } });
  expect(canvas.getByRole("button", { name: "Select car" })).toBeInTheDocument();
  expect(canvas.queryByRole("button", { name: "Select person" })).not.toBeInTheDocument();
  fireEvent.click(canvas.getByRole("button", { name: "Select car" }));
  fireEvent.click(canvas.getByRole("button", { name: "+ Add car" }));
  fireEvent.click(canvas.getByRole("button", { name: "Use positions" }));
  expect(query.clauses[0].label).toBe("car");
  expect(within(view.container).getByRole("img", { name: "Object positions preview" })).toBeInTheDocument();
});
