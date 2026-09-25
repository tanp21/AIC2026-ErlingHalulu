import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { fetchObjectCatalog } from "./api";
import type { ObjectClause, ObjectQuery } from "./types";

export const emptyObjectQuery = (): ObjectQuery => ({ clauses: [], min_confidence: 0.35 });

const OBJECT_ICONS: Record<string, string> = {
  airplane: "✈️", apple: "🍎", backpack: "🎒", banana: "🍌", "baseball bat": "🏏",
  "baseball glove": "🧤", bear: "🐻", bed: "🛏️", bench: "🪑", bicycle: "🚲", bird: "🐦",
  boat: "⛵", book: "📕", bottle: "🍼", bowl: "🥣", broccoli: "🥦", bus: "🚌", cake: "🎂",
  car: "🚗", carrot: "🥕", cat: "🐈", "cell phone": "📱", chair: "🪑", clock: "🕒",
  couch: "🛋️", cow: "🐄", cup: "☕", "dining table": "🍽️", dog: "🐕", donut: "🍩",
  elephant: "🐘", "fire hydrant": "🚒", fork: "🍴", frisbee: "🥏", giraffe: "🦒",
  "hair drier": "💨", handbag: "👜", horse: "🐎", "hot dog": "🌭", keyboard: "⌨️",
  kite: "🪁", knife: "🔪", laptop: "💻", microwave: "♨️", motorcycle: "🏍️", mouse: "🖱️",
  orange: "🍊", oven: "🔥", "parking meter": "🅿️", person: "🧍", pizza: "🍕",
  "potted plant": "🪴", refrigerator: "🧊", remote: "🎛️", sandwich: "🥪", scissors: "✂️",
  sheep: "🐑", sink: "🚰", skateboard: "🛹", skis: "🎿", snowboard: "🏂", spoon: "🥄",
  "sports ball": "⚽", "stop sign": "🛑", suitcase: "🧳", surfboard: "🏄", "teddy bear": "🧸",
  "tennis racket": "🎾", tie: "👔", toaster: "🍞", toilet: "🚽", toothbrush: "🪥",
  "traffic light": "🚦", train: "🚆", truck: "🚚", tv: "📺", umbrella: "☂️", vase: "🏺",
  "wine glass": "🍷", zebra: "🦓",
};

type Drag =
  | { kind: "draw"; index: number; x: number; y: number }
  | { kind: "move" | "resize"; index: number; x: number; y: number; region: ObjectClause["region"] };

const clamp = (value: number) => Math.max(0, Math.min(1, value));

function ObjectQueryEditor({ query, onChange, title = "Object positions" }: {
  query: ObjectQuery;
  onChange(query: ObjectQuery): void;
  title?: string;
}) {
  const [labels, setLabels] = useState<string[]>([]);
  const [label, setLabel] = useState(query.clauses.at(-1)?.label ?? "person");
  const [labelFilter, setLabelFilter] = useState("");
  const [error, setError] = useState("");
  const canvas = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);

  useEffect(() => {
    let active = true;
    fetchObjectCatalog().then((catalog) => {
      if (!active) return;
      setLabels(catalog.labels);
      setLabel((current) => catalog.labels.includes(current) ? current : catalog.labels[0] ?? "person");
    }).catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "Object index unavailable"); });
    return () => { active = false; };
  }, []);

  const point = (event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = canvas.current!.getBoundingClientRect();
    return { x: clamp((event.clientX - bounds.left) / bounds.width),
             y: clamp((event.clientY - bounds.top) / bounds.height) };
  };
  const updateClause = (index: number, changes: Partial<ObjectClause>) => onChange({
    ...query,
    clauses: query.clauses.map((clause, current) => current === index ? { ...clause, ...changes } : clause),
  });
  const add = (x = 0.5, y = 0.5) => {
    if (query.clauses.length >= 6) return;
    onChange({ ...query, clauses: [...query.clauses, {
      label, region: [clamp(x - 0.15), clamp(y - 0.15), clamp(x + 0.15), clamp(y + 0.15)], min_count: 1,
    }] });
  };
  const visibleLabels = labels.filter((item) => item.includes(labelFilter.trim().toLowerCase()));
  const startDraw = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || query.clauses.length >= 6) return;
    const { x, y } = point(event);
    add(x, y);
    drag.current = { kind: "draw", index: query.clauses.length, x, y };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    const active = drag.current;
    if (!active) return;
    const { x, y } = point(event);
    if (active.kind === "draw") {
      if (Math.abs(x - active.x) < 0.02 && Math.abs(y - active.y) < 0.02) return;
      updateClause(active.index, { region: [
        Math.min(x, active.x), Math.min(y, active.y),
        Math.min(1, Math.max(x, active.x, Math.min(x, active.x) + 0.02)),
        Math.min(1, Math.max(y, active.y, Math.min(y, active.y) + 0.02)),
      ] });
      return;
    }
    const [x0, y0, x1, y1] = active.region;
    if (active.kind === "move") {
      const dx = Math.max(-x0, Math.min(1 - x1, x - active.x));
      const dy = Math.max(-y0, Math.min(1 - y1, y - active.y));
      updateClause(active.index, { region: [x0 + dx, y0 + dy, x1 + dx, y1 + dy] });
    } else {
      updateClause(active.index, { region: [x0, y0, Math.max(x0 + 0.02, x), Math.max(y0 + 0.02, y)] });
    }
  };
  const startBox = (event: React.PointerEvent, index: number, kind: "move" | "resize") => {
    event.stopPropagation();
    const { x, y } = point(event as React.PointerEvent<HTMLDivElement>);
    drag.current = { kind, index, x, y, region: query.clauses[index].region };
    canvas.current?.setPointerCapture(event.pointerId);
  };
  const keyMove = (event: React.KeyboardEvent, index: number) => {
    const step = 1 / 7;
    const direction = event.key === "ArrowLeft" ? [-step, 0]
      : event.key === "ArrowRight" ? [step, 0]
      : event.key === "ArrowUp" ? [0, -step]
      : event.key === "ArrowDown" ? [0, step] : null;
    if (!direction) return;
    event.preventDefault();
    const [x0, y0, x1, y1] = query.clauses[index].region;
    const [dx, dy] = direction;
    if (event.shiftKey) {
      updateClause(index, { region: [x0, y0,
        Math.max(x0 + 0.02, clamp(x1 + dx)), Math.max(y0 + 0.02, clamp(y1 + dy))] });
    } else {
      const mx = Math.max(-x0, Math.min(1 - x1, dx));
      const my = Math.max(-y0, Math.min(1 - y1, dy));
      updateClause(index, { region: [x0 + mx, y0 + my, x1 + mx, y1 + my] });
    }
  };

  return <section className="object-query" aria-label={title}>
    <div className="object-editor-left">
      <div className="object-query-toolbar">
        <label className="object-label-filter">Find object
          <input aria-label="Find object class" type="search" placeholder="Search classes…" value={labelFilter}
            onChange={(event) => setLabelFilter(event.target.value.toLowerCase())} />
        </label>
        <button type="button" className="object-add-button" onClick={() => add()} disabled={query.clauses.length >= 6}>
          + Add {label}
        </button>
      </div>
      <div className="object-label-grid" role="group" aria-label="Choose object class">
        {(labels.length ? visibleLabels : [label]).map((item) => <button key={item} type="button"
          className={`object-label-tile${item === label ? " selected" : ""}`} aria-label={`Select ${item}`}
          aria-pressed={item === label} onClick={() => setLabel(item)}>
          <span className="object-label-icon" aria-hidden="true">{OBJECT_ICONS[item] ?? "🔎"}</span>
          <span className="object-label-name">{item}</span>
        </button>)}
        {!visibleLabels.length && labels.length > 0 && <small className="object-no-labels">No matching classes</small>}
      </div>
    </div>
    <div className="object-editor-right">
      <div ref={canvas} className="object-canvas" role="group" aria-label="Draw object positions"
        onPointerDown={startDraw} onPointerMove={move} onPointerUp={() => { drag.current = null; }}
        onPointerCancel={() => { drag.current = null; }}>
        {query.clauses.map((clause, index) => {
          const [x0, y0, x1, y1] = clause.region;
          return <button type="button" key={index} className="object-box"
            aria-label={`${clause.label} position ${index + 1}; arrow keys move, Shift and arrow keys resize`}
            style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}
            onPointerDown={(event) => startBox(event, index, "move")}
            onKeyDown={(event) => keyMove(event, index)}>
            <span>{clause.label}{clause.min_count > 1 ? ` ×${clause.min_count}` : ""}</span>
            <span className="object-box-resize" aria-hidden="true" onPointerDown={(event) => startBox(event, index, "resize")} />
          </button>;
        })}
      </div>
      <small className="object-editor-hint">Draw a box, or add the selected object and move it. A detection matches when its center is inside the box.</small>
      {query.clauses.map((clause, index) => <div className="object-clause-row" key={index}>
        <span className="object-clause-label"><span aria-hidden="true">{OBJECT_ICONS[clause.label] ?? "🔎"}</span>{clause.label}</span>
        <label>At least <input type="number" min="1" max="10" aria-label={`Minimum count for object ${index + 1}`}
          value={clause.min_count} onChange={(event) => updateClause(index, { min_count: Math.max(1, Math.min(10, Number(event.target.value) || 1)) })} /></label>
        <button type="button" aria-label={`Remove object ${index + 1}`}
          onClick={() => onChange({ ...query, clauses: query.clauses.filter((_, current) => current !== index) })}>×</button>
      </div>)}
      <label className="object-confidence">Minimum confidence · {Math.round(query.min_confidence * 100)}%
        <input type="range" min="25" max="95" step="5" aria-label="Minimum object confidence"
          value={Math.round(query.min_confidence * 100)}
          onChange={(event) => onChange({ ...query, min_confidence: Number(event.target.value) / 100 })} />
      </label>
      {error && <small role="alert">{error}</small>}
    </div>
  </section>;
}

export function ObjectCanvas({ query, onChange, title = "Object positions", buttonLabel = "Draw object positions" }: {
  query: ObjectQuery;
  onChange(query: ObjectQuery): void;
  title?: string;
  buttonLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(query);

  function openEditor() {
    setDraft({ ...query, clauses: query.clauses.map((clause) => ({ ...clause, region: [...clause.region] as ObjectClause["region"] })) });
    setOpen(true);
  }

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return <section className="object-query-preview" aria-label={title}>
    {query.clauses.length === 0 ? <button type="button" className="object-open-button" onClick={openEditor}>
      <span aria-hidden="true">＋</span> {buttonLabel}
    </button> : <>
      <div className="object-preview-heading">
        <div><strong>{query.clauses.length} object position{query.clauses.length === 1 ? "" : "s"}</strong>
          <span>Confidence {Math.round(query.min_confidence * 100)}%</span></div>
        <button type="button" onClick={openEditor}>Edit</button>
        <button type="button" className="object-preview-clear" aria-label="Clear object positions"
          onClick={() => onChange(emptyObjectQuery())}>×</button>
      </div>
      <div className="object-query-preview-canvas" aria-label="Object positions preview" role="img">
        {query.clauses.map((clause, index) => {
          const [x0, y0, x1, y1] = clause.region;
          return <span key={index} className="object-preview-box"
            style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}>
            {clause.label}{clause.min_count > 1 ? ` ×${clause.min_count}` : ""}
          </span>;
        })}
      </div>
      <div className="object-preview-labels">{query.clauses.map((clause, index) => <span key={index}>
        <i aria-hidden="true">{OBJECT_ICONS[clause.label] ?? "◇"}</i>{clause.label}
      </span>)}</div>
    </>}
    {open && createPortal(<div className="object-editor-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) setOpen(false);
    }}>
      <section className="object-editor-dialog" role="dialog" aria-modal="true" aria-label={`${title} editor`}>
        <header className="object-editor-heading">
          <div><span className="eyebrow">OBJECT SEARCH</span><h2>{title}</h2>
            <p>Choose an object, then draw its position box on the canvas.</p></div>
          <button type="button" aria-label="Close object editor" onClick={() => setOpen(false)}>×</button>
        </header>
        <ObjectQueryEditor query={draft} onChange={setDraft} title={title} />
        <footer className="object-editor-footer">
          <button type="button" className="object-editor-cancel" onClick={() => setOpen(false)}>Cancel</button>
          <button type="button" className="object-editor-apply" disabled={!draft.clauses.length}
            onClick={() => { onChange(draft); setOpen(false); }}>Use positions</button>
        </footer>
      </section>
    </div>, document.body)}
  </section>;
}
