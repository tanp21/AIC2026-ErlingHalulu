import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { compareIndexes, fetchSession, searchVideos } from "./api";
import { formatTimestamp, getRelativeScoreTier, groupResultsByVideo } from "./results";
import { fetchFrameMap, resolveFrameId } from "./frameMap";
import type {
  CompareResult,
  IndexName,
  SearchMode,
  SearchResponse,
  SearchResult,
  VideoGroup,
} from "./types";

const MODE_OPTIONS: SearchMode[] = ["hybrid", "visual", "audio"];

interface SearchSidebarProps {
  collapsed: boolean;
  visualQuery: string;
  audioQuery: string;
  mode: SearchMode;
  index: IndexName;
  k: number;
  nprobe: number;
  ef: number;
  loading: boolean;
  isAdmin: boolean;
  compareResults: CompareResult[];
  onCollapse(): void;
  onVisualQuery(value: string): void;
  onAudioQuery(value: string): void;
  onMode(value: SearchMode): void;
  onIndex(value: IndexName): void;
  onK(value: number): void;
  onNprobe(value: number): void;
  onEf(value: number): void;
  onSubmit(event: FormEvent): void;
  onCompare(): void;
}

function SearchSidebar(props: SearchSidebarProps) {
  const hasVisual = props.mode !== "audio";
  if (props.collapsed) {
    return (
      <aside className="search-sidebar collapsed" aria-label="Search controls">
        <button className="panel-toggle" onClick={props.onCollapse} aria-label="Expand search controls">›</button>
      </aside>
    );
  }

  return (
    <aside className="search-sidebar" aria-label="Search controls">
      <div className="brand-row">
        <div>
          <span className="eyebrow">AIC HCMC</span>
          <h1>Search workspace</h1>
        </div>
        <button className="panel-toggle" onClick={props.onCollapse} aria-label="Collapse search controls">‹</button>
      </div>

      <form onSubmit={props.onSubmit}>
        {props.mode !== "audio" && (
          <label className="field query-field">
            <span>Visual query (English)</span>
            <textarea
              aria-label="Visual query (English)"
              value={props.visualQuery}
              onChange={(event) => props.onVisualQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Describe the visible scene in English…"
              rows={4}
              autoFocus
            />
          </label>
        )}

        {props.mode !== "visual" && (
          <label className="field query-field">
            <span>Transcript query (Vietnamese)</span>
            <textarea
              aria-label="Transcript query (Vietnamese)"
              value={props.audioQuery}
              onChange={(event) => props.onAudioQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Nhập nội dung lời nói cần tìm bằng tiếng Việt…"
              rows={4}
              autoFocus={props.mode === "audio"}
            />
          </label>
        )}

        <fieldset className="mode-switch">
          <legend>Search mode</legend>
          <div>
            {MODE_OPTIONS.map((option) => (
              <button
                type="button"
                key={option}
                aria-pressed={props.mode === option}
                className={props.mode === option ? "active" : ""}
                onClick={() => props.onMode(option)}
              >
                {option[0].toUpperCase() + option.slice(1)}
              </button>
            ))}
          </div>
        </fieldset>

        {hasVisual && (
          <label className="field">
            <span>Visual index</span>
            <select
              aria-label="Visual index"
              value={props.index}
              onChange={(event) => props.onIndex(event.target.value as IndexName)}
            >
              <option value="flat">Flat · exact</option>
              {props.isAdmin && <option value="ivf">IVF · fast</option>}
              {props.isAdmin && <option value="hnsw">HNSW · fast</option>}
            </select>
          </label>
        )}

        <div className="parameter-grid">
          <label className="field">
            <span>Top k</span>
            <input type="number" min="1" max="200" value={props.k} onChange={(e) => props.onK(Number(e.target.value))} />
          </label>
          {hasVisual && props.index === "ivf" && (
            <label className="field">
              <span>nprobe</span>
              <input type="number" min="1" value={props.nprobe} onChange={(e) => props.onNprobe(Number(e.target.value))} />
            </label>
          )}
          {hasVisual && props.index === "hnsw" && (
            <label className="field">
              <span>efSearch</span>
              <input type="number" min="1" value={props.ef} onChange={(e) => props.onEf(Number(e.target.value))} />
            </label>
          )}
        </div>

        <button className="search-button" type="submit" aria-label="Search">
          <span>{props.loading ? "Searching…" : "Search"}</span>
          <kbd>Enter</kbd>
        </button>
        {hasVisual && props.isAdmin && (
          <button className="compare-button" type="button" onClick={props.onCompare}>Compare indexes</button>
        )}
      </form>

      {props.compareResults.length > 0 && (
        <section className="compare-results" aria-label="Index comparison">
          <h2>Index comparison</h2>
          {props.compareResults.map((row) => (
            <div className="compare-row" key={row.index}>
              <strong>{row.index}</strong><span>{row.latency_ms} ms</span><span>{row.recall} recall</span>
            </div>
          ))}
        </section>
      )}

      <div className="shortcut-hint"><span>Navigate</span><kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd><span>Open</span><kbd>Enter</kbd></div>
    </aside>
  );
}

interface KeyframeCardProps {
  result: SearchResult;
  selected: boolean;
  tier: number;
  onSelect(): void;
  onOpen(): void;
}

function KeyframeCard({ result, selected, tier, onSelect, onOpen }: KeyframeCardProps) {
  const time = formatTimestamp(result.time);
  return (
    <article className={`keyframe-card tier-${tier}${selected ? " selected" : ""}`}>
      <button
        className="frame-select"
        type="button"
        aria-label={`Select frame ${result.video_id} at ${time}`}
        aria-pressed={selected}
        onClick={onSelect}
      >
        <img src={result.thumb} alt="" loading="lazy" />
        <span className="frame-overlay"><time>{time}</time><span className={`source source-${result.source}`}>{result.source}</span></span>
      </button>
      <footer>
        <div className="score-block" aria-label={`Similarity level ${tier} of 5, raw score ${result.score.toFixed(4)}`}>
          <span className="score-bars" aria-hidden="true">
            {[1, 2, 3, 4, 5].map((level) => <i key={level} className={level <= tier ? "filled" : ""} />)}
          </span>
          <strong>{result.score.toFixed(4)}</strong>
        </div>
        <button type="button" className="play-icon" onClick={onOpen} aria-label={`Open video ${result.video_id} at ${time}`}>▶</button>
      </footer>
    </article>
  );
}

interface ResultsWorkspaceProps {
  groups: VideoGroup[];
  results: SearchResult[];
  selected: SearchResult | null;
  loading: boolean;
  searched: boolean;
  status: string;
  error: string;
  warning: string;
  onSelect(result: SearchResult): void;
  onOpen(result: SearchResult): void;
}

function ResultsWorkspace(props: ResultsWorkspaceProps) {
  const scores = props.results.map((result) => result.score);
  const min = scores.length ? Math.min(...scores) : 0;
  const max = scores.length ? Math.max(...scores) : 0;

  function handleKeys(event: KeyboardEvent<HTMLElement>) {
    if (!props.selected || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Enter"].includes(event.key)) return;
    if (event.key === "Enter") {
      event.preventDefault();
      props.onOpen(props.selected);
      return;
    }
    const groupIndex = props.groups.findIndex((group) => group.videoId === props.selected?.video_id);
    const group = props.groups[groupIndex];
    const itemIndex = group.results.findIndex((result) => result.frame_idx === props.selected?.frame_idx);
    let target = props.selected;
    if (event.key === "ArrowLeft") target = group.results[Math.max(0, itemIndex - 1)];
    if (event.key === "ArrowRight") target = group.results[Math.min(group.results.length - 1, itemIndex + 1)];
    if (event.key === "ArrowUp" && groupIndex > 0) target = props.groups[groupIndex - 1].results[Math.min(itemIndex, props.groups[groupIndex - 1].results.length - 1)];
    if (event.key === "ArrowDown" && groupIndex < props.groups.length - 1) target = props.groups[groupIndex + 1].results[Math.min(itemIndex, props.groups[groupIndex + 1].results.length - 1)];
    event.preventDefault();
    props.onSelect(target);
  }

  return (
    <main className="results-workspace" tabIndex={0} onKeyDown={handleKeys} aria-label="Search results">
      <header className="results-header">
        <div><span className="eyebrow">Results</span><h2>{props.status || "Ready to search"}</h2></div>
        {props.loading && <span className="loading-indicator" role="status">Searching</span>}
      </header>
      {props.error && <div className="message error" role="alert">{props.error}</div>}
      {props.warning && <div className="message warning">{props.warning}</div>}

      {!props.searched && !props.loading && (
        <div className="empty-state"><span>⌕</span><h3>Find the exact moment</h3><p>Enter a query, then scan matching videos row by row.</p></div>
      )}
      {props.searched && !props.loading && props.groups.length === 0 && (
        <div className="empty-state"><span>∅</span><h3>No matches found</h3><p>Try broader wording or a different search mode.</p></div>
      )}

      <div className="video-groups">
        {props.groups.map((group, rowIndex) => (
          <section className="video-row" data-testid="video-row" key={group.videoId}>
            <header>
              <div><span className="row-rank">{String(rowIndex + 1).padStart(2, "0")}</span><h3>{group.videoId}</h3></div>
              <div className="row-stats"><span>{group.results.length} hit{group.results.length === 1 ? "" : "s"}</span><strong>max {group.maxScore.toFixed(4)}</strong></div>
            </header>
            <div className="frame-strip">
              {group.results.map((result) => (
                <KeyframeCard
                  key={`${result.video_id}-${result.frame_idx}-${result.rank}`}
                  result={result}
                  selected={props.selected?.video_id === result.video_id && props.selected.frame_idx === result.frame_idx}
                  tier={getRelativeScoreTier(result.score, min, max)}
                  onSelect={() => props.onSelect(result)}
                  onOpen={() => props.onOpen(result)}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}

function TranscriptPanel({ selected, onOpen }: { selected: SearchResult | null; onOpen(result: SearchResult): void }) {
  return (
    <aside className="transcript-panel" aria-label="Transcript evidence">
      <header><span className="eyebrow">Verification</span><h2>Transcript evidence</h2></header>
      {!selected ? (
        <div className="transcript-empty"><span>TXT</span><p>Select a keyframe to inspect supporting speech.</p></div>
      ) : (
        <>
          <section className="selection-summary">
            <img src={selected.thumb} alt="Selected keyframe" />
            <div><strong>{selected.video_id}</strong><span>frame {selected.frame_idx} · {formatTimestamp(selected.time)}</span></div>
          </section>
          <section className="evidence-body">
            <span className={`source source-${selected.source}`}>{selected.source}</span>
            {selected.transcript_snippet ? (
              <>
                <blockquote>{selected.transcript_snippet}</blockquote>
                {selected.transcript_start !== undefined && selected.transcript_end !== undefined && (
                  <time>{formatTimestamp(selected.transcript_start)}–{formatTimestamp(selected.transcript_end)}</time>
                )}
              </>
            ) : <p className="no-evidence">No transcript evidence for this result.</p>}
          </section>
          <dl className="score-details">
            <div><dt>Fused score</dt><dd>{selected.score.toFixed(4)}</dd></div>
            {selected.scores.visual_score !== undefined && <div><dt>Visual</dt><dd>{selected.scores.visual_score.toFixed(4)}</dd></div>}
            {selected.scores.audio_score !== undefined && <div><dt>Audio</dt><dd>{selected.scores.audio_score.toFixed(4)}</dd></div>}
            {selected.scores.visual_rank !== undefined && <div><dt>Visual rank</dt><dd>#{selected.scores.visual_rank}</dd></div>}
            {selected.scores.audio_rank !== undefined && <div><dt>Audio rank</dt><dd>#{selected.scores.audio_rank}</dd></div>}
          </dl>
          <button className="open-video" type="button" onClick={() => onOpen(selected)}>Open video at {formatTimestamp(selected.time)} <span>▶</span></button>
        </>
      )}
    </aside>
  );
}

export function VideoDialog({ result, onClose }: { result: SearchResult; onClose(): void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [liveFrameId, setLiveFrameId] = useState<number | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== "function") return;
    let active = true;
    let callbackId: number | undefined;
    fetchFrameMap(result.video_id).then((mapping) => {
      const update: VideoFrameRequestCallback = (_now, metadata) => {
        if (!active) return;
        setLiveFrameId(resolveFrameId(mapping, metadata.mediaTime));
        callbackId = video.requestVideoFrameCallback(update);
      };
      if (active) callbackId = video.requestVideoFrameCallback(update);
    }).catch(() => setLiveFrameId(null));
    return () => {
      active = false;
      if (callbackId !== undefined) video.cancelVideoFrameCallback(callbackId);
    };
  }, [result.video_id]);

  return (
    <div className="dialog-backdrop" role="dialog" aria-modal="true" aria-label={`Video ${result.video_id}`} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="video-dialog">
        <header><div><strong>{result.video_id}</strong><span>{formatTimestamp(result.time)} · {liveFrameId === null ? "Exact frame unavailable" : `frame ${liveFrameId}`}</span></div><button onClick={onClose} aria-label="Close video">×</button></header>
        <video
          ref={videoRef}
          src={result.video}
          controls
          autoPlay
          onLoadedMetadata={() => { if (videoRef.current) videoRef.current.currentTime = result.time; }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [visualQuery, setVisualQuery] = useState("");
  const [audioQuery, setAudioQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [index, setIndex] = useState<IndexName>("flat");
  const [k, setK] = useState(100);
  const [nprobe, setNprobe] = useState(64);
  const [ef, setEf] = useState(64);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [openResult, setOpenResult] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [sessionError, setSessionError] = useState("");
  const requestRef = useRef<{ id: number; controller: AbortController } | null>(null);

  const groups = useMemo(() => groupResultsByVideo(response?.results ?? []), [response]);
  const showTranscript = mode !== "visual";

  async function runSearch(event: FormEvent) {
    event.preventDefault();
    const visualText = visualQuery.trim();
    const audioText = audioQuery.trim();
    if ((mode === "visual" && !visualText) || (mode === "audio" && !audioText) || (mode === "hybrid" && !visualText && !audioText)) {
      setError("Enter a query for the selected search mode.");
      return;
    }
    requestRef.current?.controller.abort();
    const id = (requestRef.current?.id ?? 0) + 1;
    const controller = new AbortController();
    requestRef.current = { id, controller };
    setLoading(true);
    setError("");
    try {
      const data = await searchVideos({
        mode, index, k, nprobe, ef,
        visual_text: mode === "audio" ? undefined : visualText,
        audio_text: mode === "visual" ? undefined : audioText,
      }, controller.signal);
      if (requestRef.current?.id !== id) return;
      setResponse(data);
      setSelected(data.results.length ? data.results.reduce((best, item) => item.score > best.score ? item : best) : null);
    } catch (searchError) {
      if (requestRef.current?.id !== id || (searchError instanceof DOMException && searchError.name === "AbortError")) return;
      setError(searchError instanceof Error ? searchError.message : "Search failed");
    } finally {
      if (requestRef.current?.id === id) setLoading(false);
    }
  }

  async function runCompare() {
    if (!visualQuery.trim()) {
      setError("Enter a visual query.");
      return;
    }
    setError("");
    try {
      setCompareResults(await compareIndexes(visualQuery.trim(), k));
    } catch (compareError) {
      setError(compareError instanceof Error ? compareError.message : "Comparison failed");
    }
  }

  useEffect(() => {
    let active = true;
    fetchSession().then((session) => {
      if (active) setIsAdmin(session.is_admin);
    }).catch((sessionFetchError) => {
      if (!active) return;
      const message = sessionFetchError instanceof Error ? sessionFetchError.message : "Unable to verify your access";
      setSessionError(message === "Request failed" ? "Unable to verify your access" : message);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setOpenResult(null);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const status = response
    ? `${response.results.length} keyframes · ${groups.length} videos · ${response.latency_ms} ms`
    : "";

  return (
    <div className={`app-shell${sidebarCollapsed ? " search-collapsed" : ""}${showTranscript ? " with-transcript" : ""}`}>
      <SearchSidebar
        collapsed={sidebarCollapsed}
        visualQuery={visualQuery}
        audioQuery={audioQuery}
        mode={mode}
        index={index}
        k={k}
        nprobe={nprobe}
        ef={ef}
        loading={loading}
        isAdmin={isAdmin}
        compareResults={compareResults}
        onCollapse={() => setSidebarCollapsed((value) => !value)}
        onVisualQuery={setVisualQuery}
        onAudioQuery={setAudioQuery}
        onMode={setMode}
        onIndex={setIndex}
        onK={setK}
        onNprobe={setNprobe}
        onEf={setEf}
        onSubmit={runSearch}
        onCompare={runCompare}
      />
      <ResultsWorkspace
        groups={groups}
        results={response?.results ?? []}
        selected={selected}
        loading={loading}
        searched={response !== null}
        status={status}
        error={sessionError || error}
        warning={response?.warning ?? ""}
        onSelect={setSelected}
        onOpen={setOpenResult}
      />
      {showTranscript && <TranscriptPanel selected={selected} onOpen={setOpenResult} />}
      {openResult && <VideoDialog result={openResult} onClose={() => setOpenResult(null)} />}
    </div>
  );
}
