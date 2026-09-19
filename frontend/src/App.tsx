import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { compareIndexes, fetchKeyframeContext, fetchSession, searchNextPage, searchTemporal, searchTemporalNextPage, searchVideos, searchVideosByImage } from "./api";
import { AgentPanel } from "./agentPanel";
import { fetchRound, fetchRounds, patchQuery } from "./collabApi";
import { CollaborationPanels, QuickAddResult, TeamToolbar } from "./collab";
import { AppTabs, ClipStudio } from "./clips";
import type { AppTab } from "./clips";
import { formatTimestamp, getRelativeScoreTier, groupResultsByVideo } from "./results";
import { LoadingOverlay } from "./loadingOverlay";
import { VideoDialog } from "./videoDialog";
import { TemporalPrimaryFields, TemporalSidebarFields, temporalSelection, TemporalWorkspace } from "./temporal";
import type { ResultsViewMode } from "./temporal";
import { QueryBox } from "./queryBox";
export { VideoDialog } from "./videoDialog";
import type {
  CompareResult,
  ClassicSearchMode,
  IndexName,
  IndexProfile,
  SearchMode,
  SearchResponse,
  SearchResult,
  TemporalEventHit,
  TemporalSearchResponse,
  TemporalSequence,
  CollabQuery,
  RoundSnapshot,
  RoundSummary,
  AudioMethod,
  VideoGroup,
  AnswerDraft,
} from "./types";

const MODE_OPTIONS: SearchMode[] = ["hybrid", "temporal"];

type QueryImage =
  | { kind: "upload"; file: File; preview: string }
  | { kind: "reference"; result: SearchResult; preview: string };

interface SearchSidebarProps {
  query: CollabQuery | null;
  visualQuery: string;
  ocrQuery: string;
  mode: SearchMode;
  temporalEvents: string[];
  temporalMaxGap: number;
  index: IndexName;
  profile: string;
  profiles: IndexProfile[];
  nprobe: number;
  ef: number;
  isAdmin: boolean;
  compareResults: CompareResult[];
  queryImage: QueryImage | null;
  imageWeight: number;
  loading: boolean;
  onOcrQuery(value: string): void;
  onMode(value: SearchMode): void;
  onTemporalEvents(events: string[]): void;
  onTemporalMaxGap(value: number): void;
  onIndex(value: IndexName): void;
  onProfile(value: string): void;
  onNprobe(value: number): void;
  onEf(value: number): void;
  onCompare(): void;
  onQueryImage(file: File): void;
  onClearQueryImage(): void;
  onImageWeight(value: number): void;
}

function SearchSidebar(props: SearchSidebarProps) {
  const hasVisualIndex = props.mode === "hybrid";
  const activeProfile = props.profiles.find((item) => item.name === props.profile) ?? props.profiles[0];
  const availableIndexes = activeProfile?.indexes ?? ["flat"];

  return (
    <aside className="search-sidebar" aria-label="Search controls">
      <fieldset className="mode-switch" aria-label="Search mode">
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

      <section className="sidebar-working-query">
        <span className="eyebrow">Current query</span>
        {props.query ? <><div><strong>{props.query.filename}</strong><i className={`query-kind kind-${props.query.kind}`}>{props.query.kind}</i></div>
          <p>{props.query.prompt}</p></> : <p>No team query selected.</p>}
      </section>

      {props.mode === "temporal" && <TemporalSidebarFields events={props.temporalEvents} maxGap={props.temporalMaxGap}
        onEvents={props.onTemporalEvents} onMaxGap={props.onTemporalMaxGap} />}

      <div>
        <section className="sidebar-ocr">
          <label className="field"><span>OCR query</span><textarea aria-label="On-screen text query" value={props.ocrQuery}
            onChange={(event) => props.onOcrQuery(event.target.value)} rows={3}
            placeholder="Signs, subtitles, program names…" /></label>
          <small>Search uses OCR to filter by video and keep only the shot containing the text; if only OCR is entered, it searches OCR independently.</small>
        </section>

        {hasVisualIndex && (
          <section className="image-query">
            <span>Query image</span>
            <div
              className={`image-drop-zone${props.queryImage ? " has-image" : ""}`}
              aria-label="Query image drop zone"
              tabIndex={0}
              onPaste={(event) => {
                const file = event.clipboardData.files[0];
                if (file) props.onQueryImage(file);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const file = event.dataTransfer.files[0];
                if (file) props.onQueryImage(file);
              }}
            >
              {props.queryImage ? (
                <img src={props.queryImage.preview} alt="Query image preview" />
              ) : <p>Drop or paste a screenshot</p>}
              <label className="image-picker">
                Choose image
                <input
                  type="file"
                  aria-label="Query image"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) props.onQueryImage(file);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              {props.queryImage && (
                <button type="button" className="image-clear" onClick={props.onClearQueryImage} aria-label="Clear query image">×</button>
              )}
            </div>
            {props.queryImage && (
              <label className="field image-weight">
                <span>Image influence <strong>{props.visualQuery.trim() ? props.imageWeight : 100}%</strong></span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  aria-label="Image influence"
                  value={props.visualQuery.trim() ? props.imageWeight : 100}
                  disabled={!props.visualQuery.trim()}
                  onChange={(event) => props.onImageWeight(Number(event.target.value))}
                />
              </label>
            )}
          </section>
        )}

        {hasVisualIndex && availableIndexes.length > 1 && (
          <label className="field">
            <span>Visual index</span>
            <select
              aria-label="Visual index"
              value={props.index}
              onChange={(event) => props.onIndex(event.target.value as IndexName)}
            >
              {availableIndexes.includes("flat") && <option value="flat">Flat · exact</option>}
              {props.isAdmin && availableIndexes.includes("ivf") && <option value="ivf">IVF · fast</option>}
              {props.isAdmin && availableIndexes.includes("hnsw") && <option value="hnsw">HNSW · fast</option>}
            </select>
          </label>
        )}

        {hasVisualIndex && props.index !== "flat" && <div className="parameter-grid">
          {props.index === "ivf" && (
            <label className="field">
              <span>nprobe</span>
              <input type="number" min="1" value={props.nprobe} onChange={(e) => props.onNprobe(Number(e.target.value))} />
            </label>
          )}
          {props.index === "hnsw" && (
            <label className="field">
              <span>efSearch</span>
              <input type="number" min="1" value={props.ef} onChange={(e) => props.onEf(Number(e.target.value))} />
            </label>
          )}
        </div>}

        {hasVisualIndex && props.isAdmin && availableIndexes.length > 1 && (
          <button className="compare-button" type="button" onClick={props.onCompare}>Compare indexes</button>
        )}
      </div>

      {props.compareResults.length > 0 && (
        <section className="compare-results" aria-label="Index comparison">
          <h2>Index comparison</h2>
          {props.compareResults.map((row) => (
            <div className="compare-row" key={row.index}>
              <strong>{row.index}</strong><span>{row.latency_ms} ms</span><span>{row.recall} recall@50</span>
            </div>
          ))}
        </section>
      )}

      <div className="shortcut-hint"><span>Navigate</span><kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd><span>Open</span><kbd>Enter</kbd></div>
    </aside>
  );
}

function SearchComposer({ mode, visualQuery, audioQuery, temporalEvents, audioMethod, peWeight,
  loading, onVisualQuery, onAudioQuery, onTemporalEvents, onAudioMethod, onPeWeight, onSubmit }: {
  mode: SearchMode;
  visualQuery: string;
  audioQuery: string;
  temporalEvents: string[];
  audioMethod: AudioMethod;
  peWeight: number;
  loading: boolean;
  onVisualQuery(value: string): void;
  onAudioQuery(value: string): void;
  onTemporalEvents(events: string[]): void;
  onAudioMethod(value: AudioMethod): void;
  onPeWeight(value: number): void;
  onSubmit(event: FormEvent): void;
}) {
  const submitOnEnter = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };
  return <section className="search-composer" aria-label="Search query">
    <form onSubmit={onSubmit}>
      <div className={`composer-fields mode-${mode}`}>
        {mode !== "audio" && mode !== "temporal" && <QueryBox
          label="Visual query (English)"
          ariaLabel="Visual query (English)"
          value={visualQuery}
          onChange={onVisualQuery}
          onKeyDown={submitOnEnter}
          placeholder="Describe the visible scene in English…"
          autoFocus
          footer={<>
            <span>PE-Core influence</span>
            <input type="range" min="0" max="100" aria-label="PE-Core influence" value={peWeight}
              onChange={(event) => onPeWeight(Number(event.target.value))} />
            <strong>{peWeight}%</strong>
          </>}
        />}
        {mode !== "visual" && mode !== "temporal" && <QueryBox
          label="Transcript query (Vietnamese)"
          ariaLabel="Transcript query (Vietnamese)"
          value={audioQuery}
          onChange={onAudioQuery}
          onKeyDown={submitOnEnter}
          placeholder="Enter the spoken content to search for…"
          footer={<>
            <span>Audio search</span>
            <select aria-label="Audio search" value={audioMethod} onChange={(event) => onAudioMethod(event.target.value as AudioMethod)}>
              <option value="semantic">Semantic</option>
              <option value="keyword">Keyword</option>
            </select>
          </>}
        />}
        {mode === "temporal" && <TemporalPrimaryFields events={temporalEvents} audioQuery={audioQuery}
          audioMethod={audioMethod} peWeight={peWeight}
          onEvents={onTemporalEvents} onAudioQuery={onAudioQuery}
          onAudioMethod={onAudioMethod} onPeWeight={onPeWeight} />}
      </div>
      <button className="search-button composer-submit icon-only" type="submit" aria-label="Search">
        {loading ? <span className="spinner" aria-hidden="true" /> : <>
          <span className="search-icon" aria-hidden="true">⌕</span>
          <kbd>Enter</kbd>
        </>}
      </button>
    </form>
  </section>;
}

interface KeyframeCardProps {
  result: SearchResult;
  selected: boolean;
  tier: number;
  onSelect(): void;
  onOpen(): void;
  onSimilar(): void;
}

function KeyframeCard({ result, selected, tier, onSelect, onOpen, onSimilar }: KeyframeCardProps) {
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
        <span className="frame-video-id">{result.video_id}</span>
        <span className="frame-overlay"><time>{time}</time><span className={`source source-${result.source}`}>{result.source}</span></span>
      </button>
      <footer>
        <div className="score-block" aria-label={`Similarity level ${tier} of 5, raw score ${result.score.toFixed(4)}`}>
          <span className="score-bars" aria-hidden="true">
            {[1, 2, 3, 4, 5].map((level) => <i key={level} className={level <= tier ? "filled" : ""} />)}
          </span>
          <strong>{result.score.toFixed(4)}</strong>
        </div>
        <button type="button" className="similar-icon" onClick={onSimilar} aria-label={`Find frames similar to ${result.video_id} at ${time}`}>≈</button>
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
  loadingMore: boolean;
  hasMore: boolean;
  searched: boolean;
  error: string;
  moreError: string;
  warning: string;
  truncated: boolean;
  profile?: string;
  viewMode: ResultsViewMode;
  onViewMode(mode: ResultsViewMode): void;
  onSelect(result: SearchResult): void;
  onOpen(result: SearchResult): void;
  onLoadMore(): void;
  onSimilar(result: SearchResult): void;
}

function ResultsWorkspace(props: ResultsWorkspaceProps) {
  const workspaceRef = useRef<HTMLElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const scores = props.results.map((result) => result.score);
  const min = scores.length ? Math.min(...scores) : 0;
  const max = scores.length ? Math.max(...scores) : 0;

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !props.hasMore || props.loadingMore || props.moreError) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries.some((entry) => entry.isIntersecting)) props.onLoadMore(); },
      { root: workspaceRef.current, rootMargin: "0px 0px 400px 0px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [props.hasMore, props.loadingMore, props.moreError, props.onLoadMore]);

  function handleKeys(event: KeyboardEvent<HTMLElement>) {
    if (!props.selected || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Enter"].includes(event.key)) return;
    if (event.key === "Enter") {
      event.preventDefault();
      props.onOpen(props.selected);
      return;
    }
    const itemIndex = props.results.findIndex((result) => result.video_id === props.selected?.video_id && result.frame_idx === props.selected?.frame_idx);
    let target = props.selected;
    const columns = 5;
    if (event.key === "ArrowLeft") target = props.results[Math.max(0, itemIndex - 1)];
    if (event.key === "ArrowRight") target = props.results[Math.min(props.results.length - 1, itemIndex + 1)];
    if (event.key === "ArrowUp") target = props.results[Math.max(0, itemIndex - columns)];
    if (event.key === "ArrowDown") target = props.results[Math.min(props.results.length - 1, itemIndex + columns)];
    event.preventDefault();
    props.onSelect(target);
  }

  return (
    <main ref={workspaceRef} className="results-workspace" tabIndex={0} onKeyDown={handleKeys} aria-label="Search results">
      {props.error && <div className="message error" role="alert">{props.error}</div>}
      {props.warning && <div className="message warning">{props.warning}</div>}

      {!props.searched && !props.loading && (
        <div className="empty-state"><span>⌕</span><h3>Find the exact moment</h3><p>Enter a query, then scan matching videos row by row.</p></div>
      )}
      {props.searched && !props.loading && props.results.length === 0 && (
        <div className="empty-state"><span>∅</span><h3>No matches found</h3><p>Try broader wording or a different search mode.</p></div>
      )}

      <div className={`results-body${props.loading ? " is-loading" : ""}`}>
        <LoadingOverlay active={props.loading} sticky />
        {props.viewMode === "grouped" ? (
          <div className="video-groups">{props.groups.map((group, index) => (
            <section className="video-row" data-testid="video-row" key={group.videoId}>
              <header><div><span className="row-rank">{String(index + 1).padStart(2, "0")}</span><h3>{group.videoId}</h3></div>
                <div className="row-stats"><span>{group.results.length} frame{group.results.length === 1 ? "" : "s"}</span><strong>best {group.maxScore.toFixed(4)}</strong></div></header>
              <div className="frame-strip">{group.results.map((result) => (
                <KeyframeCard
                  key={`${result.video_id}-${result.frame_idx}-${result.rank}`}
                  result={result}
                  selected={props.selected?.video_id === result.video_id && props.selected.frame_idx === result.frame_idx}
                  tier={getRelativeScoreTier(result.score, min, max)}
                  onSelect={() => props.onSelect(result)}
                  onOpen={() => props.onOpen(result)}
                  onSimilar={() => props.onSimilar(result)}
                />
              ))}</div>
            </section>
          ))}</div>
        ) : (
          <div className="results-grid">
                {props.results.map((result) => (
                  <KeyframeCard
                    key={`${result.video_id}-${result.frame_idx}-${result.rank}`}
                    result={result}
                    selected={props.selected?.video_id === result.video_id && props.selected.frame_idx === result.frame_idx}
                    tier={getRelativeScoreTier(result.score, min, max)}
                    onSelect={() => props.onSelect(result)}
                    onOpen={() => props.onOpen(result)}
                    onSimilar={() => props.onSimilar(result)}
                  />
                ))}
          </div>
        )}
        <KeyframeContext
          selected={props.selected} profile={props.profile} suppressOwnLoading={props.loading}
          onSelect={props.onSelect} onOpen={props.onOpen} />
      </div>
      {props.searched && props.results.length > 0 && (
        <div ref={sentinelRef} className="load-more-sentinel">
          {props.loadingMore && <span className="loading-indicator" role="status">Loading more</span>}
          {props.moreError && <><span role="alert">{props.moreError}</span><button type="button" onClick={props.onLoadMore}>Retry</button></>}
          {!props.hasMore && !props.loadingMore && !props.moreError && (
            <span>{props.truncated ? "Reached the 10,000-result search depth." : "All results loaded."}</span>
          )}
        </div>
      )}
    </main>
  );
}

function KeyframeContext({ selected, profile, suppressOwnLoading, onSelect, onOpen }: { selected: SearchResult | null; profile?: string; suppressOwnLoading?: boolean; onSelect(result: SearchResult): void; onOpen(result: SearchResult): void }) {
  const [frames, setFrames] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const stripRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!selected) { setFrames([]); setLoading(false); return; }
    const controller = new AbortController();
    setLoading(true);
    fetchKeyframeContext(selected.video_id, selected.frame_idx, profile).then((data) => {
      if (!controller.signal.aborted) {
        setFrames(data.frames.map((frame) => ({ ...frame, rank: 0, score: 0, source: "visual", scores: {} })));
        setLoading(false);
      }
    }).catch(() => { if (!controller.signal.aborted) { setFrames([]); setLoading(false); } });
    return () => controller.abort();
  }, [selected?.video_id, selected?.frame_idx, profile]);
  useEffect(() => {
    stripRef.current?.querySelector<HTMLElement>("[aria-current='true']")?.scrollIntoView({
      behavior: "smooth", block: "nearest", inline: "center",
    });
  }, [frames, selected?.frame_idx]);
  if (!selected) return null;
  const showOwnLoading = loading && !suppressOwnLoading;
  return <section className="keyframe-context" aria-label="Neighbouring keyframes">
    <div><strong>{selected.video_id} <b>Selected frame {selected.frame_idx}</b></strong><button type="button" onClick={() => onOpen(selected)}>▶ Play video</button></div>
    <div className={`results-body context-frames-body${showOwnLoading ? " is-loading" : ""}`}>
      <LoadingOverlay active={showOwnLoading} />
      <div ref={stripRef} className="context-frames">
      {!loading && !frames.length && <span className="context-loading">No neighbouring keyframes.</span>}
      {frames.map((frame) => {
        const active = frame.frame_idx === selected.frame_idx;
        return <button key={frame.frame_idx} type="button" aria-current={active ? "true" : undefined}
          aria-label={`${active ? "Selected " : "Select "}frame ${frame.frame_idx}`}
          onClick={() => onSelect({ ...frame, rank: selected.rank, score: 0, source: "visual", scores: {}, context_only: true })} className={active ? "active" : ""}>
          <img src={`${frame.thumb}?w=160`} alt="" /><span>{formatTimestamp(frame.time)}</span><em>Frame {frame.frame_idx}</em>
          {active && <i aria-hidden="true">✓</i>}
        </button>;
      })}
      </div>
    </div>
  </section>;
}

function TranscriptPanel({ selected, query, canAdd, onOpen, onQueryChange }: {
  selected: SearchResult | null;
  query: CollabQuery | null;
  canAdd: boolean;
  onOpen(result: SearchResult): void;
  onQueryChange(query: CollabQuery): void;
}) {
  const [view, setView] = useState<"ranking" | "team">("ranking");
  return (
    <aside className="transcript-panel" aria-label="Transcript evidence">
      <div className="evidence-view-switch" role="group" aria-label="Right panel mode">
        <div>
          <button type="button" className={view === "ranking" ? "active" : ""} aria-pressed={view === "ranking"}
            onClick={() => setView("ranking")}>Ranking &amp; matched content</button>
          <button type="button" className={view === "team" ? "active" : ""} aria-pressed={view === "team"}
            onClick={() => setView("team")}>Team results</button>
        </div>
      </div>
      {view === "team" ? <TeamQueryResults query={query} editable={canAdd} onQueryChange={onQueryChange} /> : !selected ? (
        <div className="transcript-empty"><span>RANK</span><p>Select a keyframe to see ranking and matched content.</p></div>
      ) : (
        <>
          <section className="selection-summary">
            <img src={selected.thumb} alt="Selected keyframe" />
            <div><strong>{selected.video_id}</strong><span>frame {selected.frame_idx} · {formatTimestamp(selected.time)}</span></div>
          </section>
          <section className="ranking-groups" aria-label="Ranking details">
            <div className="ranking-group overall"><h3>Overall</h3><dl>
              {!selected.context_only && <div><dt>{selected.source === "both" ? "Hybrid rank" : "Result rank"}</dt><dd>#{selected.rank + 1}</dd></div>}
              <div><dt>Fused score</dt><dd>{selected.score.toFixed(4)}</dd></div>
              {selected.scores.sequence_visual_score !== undefined && <div><dt>Sequence</dt><dd>{selected.scores.sequence_visual_score.toFixed(4)}</dd></div>}
            </dl></div>
            {(selected.scores.visual_score !== undefined || selected.scores.visual_rank != null || selected.scores.pe_core_rank != null || selected.scores.naflex_rank != null) &&
              <div className="ranking-group visual"><h3>Visual</h3><dl>
                {selected.scores.visual_score !== undefined && <div><dt>Score</dt><dd>{selected.scores.visual_score.toFixed(4)}</dd></div>}
                {selected.scores.visual_rank != null && <div><dt>Visual rank</dt><dd>#{selected.scores.visual_rank}</dd></div>}
                {selected.scores.pe_core_rank != null && <div><dt>PE-Core</dt><dd>#{selected.scores.pe_core_rank}</dd></div>}
                {selected.scores.naflex_rank != null && <div><dt>NaFlex</dt><dd>#{selected.scores.naflex_rank}</dd></div>}
              </dl></div>}
            {(selected.scores.audio_score !== undefined || selected.scores.audio_rank != null || selected.scores.keyword_rank != null || selected.scores.semantic_rank != null || selected.scores.bm25_rank != null || selected.scores.vector_rank != null) &&
              <div className="ranking-group audio"><h3>Audio</h3><dl>
                {selected.scores.audio_score !== undefined && <div><dt>Score</dt><dd>{selected.scores.audio_score.toFixed(4)}</dd></div>}
                {selected.scores.audio_rank != null && <div><dt>Audio rank</dt><dd>#{selected.scores.audio_rank}</dd></div>}
                {selected.scores.keyword_rank != null && <div><dt>Keyword</dt><dd>#{selected.scores.keyword_rank}</dd></div>}
                {selected.scores.semantic_rank != null && <div><dt>Semantic</dt><dd>#{selected.scores.semantic_rank}</dd></div>}
                {selected.scores.bm25_rank != null && <div><dt>BM25</dt><dd>#{selected.scores.bm25_rank}</dd></div>}
                {selected.scores.vector_rank != null && <div><dt>Vector</dt><dd>#{selected.scores.vector_rank}</dd></div>}
              </dl></div>}
            {(selected.scores.ocr_score !== undefined || selected.scores.ocr_rank != null) &&
              <div className="ranking-group ocr"><h3>OCR</h3><dl>
                {selected.scores.ocr_score !== undefined && <div><dt>Score</dt><dd>{selected.scores.ocr_score.toFixed(4)}</dd></div>}
                {selected.scores.ocr_rank != null && <div><dt>OCR rank</dt><dd>#{selected.scores.ocr_rank}</dd></div>}
              </dl></div>}
          </section>
          <section className="match-evidence">
            <article className="evidence-card ocr-evidence"><header><span>OCR match</span><b>ON-SCREEN</b></header>
              {selected.ocr_snippet ? <><blockquote>{selected.ocr_snippet}</blockquote>
                {selected.ocr_start !== undefined && selected.ocr_end !== undefined && <time>{formatTimestamp(selected.ocr_start)}–{formatTimestamp(selected.ocr_end)}</time>}</>
                : <p>No OCR match for this result.</p>}
              {selected.ocr_matched_keywords?.length ? <div className="keyword-chips">{selected.ocr_matched_keywords.map((word) => <i key={word}>{word}</i>)}</div> : null}
            </article>
            <article className="evidence-card transcript-evidence"><header><span>Transcript match</span><b>{selected.audio_method ?? "AUDIO"}</b></header>
              {selected.transcript_snippet ? <><blockquote>{selected.transcript_snippet}</blockquote>
                {selected.transcript_start !== undefined && selected.transcript_end !== undefined && <time>{formatTimestamp(selected.transcript_start)}–{formatTimestamp(selected.transcript_end)}</time>}
                {selected.matched_keywords?.length ? <div className="keyword-chips">{selected.matched_keywords.map((word) => <i key={word}>{word}</i>)}</div> : null}</>
                : <p>No transcript match for this result.</p>}
            </article>
          </section>
          {canAdd && <QuickAddResult query={query} result={selected} onChange={onQueryChange} />}
          <button className="open-video" type="button" onClick={() => onOpen(selected)}>Open video at {formatTimestamp(selected.time)} <span>▶</span></button>
        </>
      )}
    </aside>
  );
}

function TeamQueryResults({ query, editable, onQueryChange }: {
  query: CollabQuery | null;
  editable: boolean;
  onQueryChange(query: CollabQuery): void;
}) {
  const [answers, setAnswers] = useState<AnswerDraft[]>(() => structuredClone(query?.answers ?? []));
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setAnswers(structuredClone(query?.answers ?? []));
    setDirty(false);
    setError("");
  }, [query?.id, query?.version]);

  if (!query) return <div className="team-results-empty">No current query selected.</div>;
  const activeQuery = query;

  function changeRow(index: number, patch: Record<string, unknown>) {
    setAnswers((rows) => rows.map((row, current) => current === index ? { ...row, ...patch } as AnswerDraft : row));
    setDirty(true);
  }

  function addRow() {
    const answer: AnswerDraft = activeQuery.kind === "trake"
      ? { video_id: "", frame_ids: Array(activeQuery.event_count ?? 0).fill(null) }
      : activeQuery.kind === "qa" ? { video_id: "", frame_id: null, answer: "" } : { video_id: "", frame_id: null };
    setAnswers((rows) => [...rows, answer]);
    setDirty(true);
  }

  async function save() {
    setBusy(true); setError("");
    try {
      const updated = await patchQuery(activeQuery.id, { expected_version: activeQuery.version, answers });
      onQueryChange(updated);
      setDirty(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to save answer");
    } finally { setBusy(false); }
  }

  return <section className="team-results-compact" aria-label="Team's current results">
    <header><strong>{answers.length} results</strong><span>{query.filename}</span></header>
    {!answers.length && <div className="team-results-empty compact">The team hasn't saved results for this query yet.</div>}
    <div>{answers.map((answer, rowIndex) => {
      const row = answer as unknown as Record<string, unknown>;
      const frames = query.kind === "trake"
        ? row.frame_ids as Array<number | null>
        : [row.frame_id as number | null];
      return <article key={rowIndex}>
        <header><b>#{rowIndex + 1}</b><input aria-label={`Video result ${rowIndex + 1}`} value={String(row.video_id ?? "")}
          disabled={!editable} placeholder="Video ID" onChange={(event) => changeRow(rowIndex, { video_id: event.target.value })} />
          {editable && <button type="button" aria-label={`Delete result ${rowIndex + 1}`} onClick={() => {
            setAnswers(answers.filter((_, index) => index !== rowIndex)); setDirty(true);
          }}>×</button>}</header>
        {query.kind === "qa" && <textarea aria-label={`Answer ${rowIndex + 1}`} value={String(row.answer ?? "")}
          disabled={!editable} placeholder="Answer" onChange={(event) => changeRow(rowIndex, { answer: event.target.value })} />}
        <div className="team-result-frames">{frames.map((frameId, frameIndex) =>
          <figure key={frameIndex}>
            {row.video_id && frameId != null
              ? <img src={`/thumb/${String(row.video_id)}/${frameId}?w=160`} alt={`Frame ${frameId}`} loading="lazy" />
              : <span>No frame selected</span>}
            <figcaption><label>{query.kind === "trake" ? `E${frameIndex + 1}` : "Frame"}<input type="number" min="0"
              aria-label={`${query.kind === "trake" ? `E${frameIndex + 1}` : "Frame"} result ${rowIndex + 1}`}
              disabled={!editable} value={frameId ?? ""} onChange={(event) => {
                const value = event.target.value === "" ? null : Number(event.target.value);
                if (query.kind === "trake") {
                  const next = [...frames]; next[frameIndex] = value; changeRow(rowIndex, { frame_ids: next });
                } else changeRow(rowIndex, { frame_id: value });
              }} /></label></figcaption>
          </figure>)}</div>
      </article>;
    })}</div>
    {editable && <footer><button type="button" disabled={answers.length >= 100 || busy} onClick={addRow}>+ Add result</button>
      <button type="button" className="save" disabled={!dirty || busy} onClick={save}>{busy ? "Saving…" : "Save answer"}</button></footer>}
    {error && <p className="team-results-error" role="alert">{error}</p>}
  </section>;
}

export default function App() {
  const [visualQuery, setVisualQuery] = useState("");
  const [audioQuery, setAudioQuery] = useState("");
  const [ocrQuery, setOcrQuery] = useState("");
  const [temporalEvents, setTemporalEvents] = useState(["", ""]);
  const [temporalMaxGap, setTemporalMaxGap] = useState(120);
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [index, setIndex] = useState<IndexName>("flat");
  const [profiles] = useState<IndexProfile[]>([
    { name: "l2026", label: "PE-Core + NaFlex · 2026", is_default: true, count: 268254, video_count: 873, indexes: ["flat"] },
  ]);
  const [profile, setProfile] = useState("l2026");
  const profilesReady = true;
  const [nprobe, setNprobe] = useState(64);
  const [ef, setEf] = useState(64);
  const [queryImage, setQueryImage] = useState<QueryImage | null>(null);
  const [imageWeight, setImageWeight] = useState(50);
  const [audioMethod, setAudioMethod] = useState<AudioMethod>("semantic");
  const [peWeight, setPeWeight] = useState(75);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [temporalResponse, setTemporalResponse] = useState<TemporalSearchResponse | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [openResult, setOpenResult] = useState<SearchResult | null>(null);
  const [classicViewMode, setClassicViewMode] = useState<ResultsViewMode>("ranked");
  const [temporalViewMode, setTemporalViewMode] = useState<ResultsViewMode>("grouped");
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState("");
  const [error, setError] = useState("");
  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [login, setLogin] = useState("");
  const [rounds, setRounds] = useState<RoundSummary[]>([]);
  const [activeRound, setActiveRound] = useState<RoundSnapshot | null>(null);
  const [selectedQueryId, setSelectedQueryId] = useState<number | null>(null);
  const [sessionError, setSessionError] = useState("");
  const [tab, setTab] = useState<AppTab>("search");
  const requestRef = useRef<{ id: number; controller: AbortController } | null>(null);
  const pageRequestRef = useRef<AbortController | null>(null);
  const roundEtagRef = useRef("");
  const queryImageRef = useRef<QueryImage | null>(null);

  const groups = useMemo(() => groupResultsByVideo(response?.results ?? []), [response]);
  const currentQuery = activeRound?.queries.find((query) => query.id === selectedQueryId)
    ?? activeRound?.queries[0] ?? null;

  function replaceQueryImage(next: QueryImage | null) {
    if (queryImageRef.current?.kind === "upload") URL.revokeObjectURL(queryImageRef.current.preview);
    queryImageRef.current = next;
    setQueryImage(next);
  }

  function clearSearchSession() {
    requestRef.current?.controller.abort();
    pageRequestRef.current?.abort();
    replaceQueryImage(null);
    setVisualQuery(""); setAudioQuery(""); setOcrQuery(""); setTemporalEvents(["", ""]);
    setResponse(null); setTemporalResponse(null); setSelected(null); setOpenResult(null);
    setError(""); setMoreError(""); setCompareResults([]); setLoading(false); setLoadingMore(false);
  }

  function selectCollabQuery(id: number) {
    setSelectedQueryId(id);
  }

  function useUploadedImage(file: File) {
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 6 * 1024 * 1024) {
      setError("Query image must not exceed 6 MiB.");
      return;
    }
    setError("");
    replaceQueryImage({ kind: "upload", file, preview: URL.createObjectURL(file) });
  }

  async function executeSearch(activeImage: QueryImage | null = queryImageRef.current, requestedMode: SearchMode = mode) {
    const visualText = visualQuery.trim();
    const audioText = audioQuery.trim();
    const ocrText = ocrQuery.trim();
    const usesImage = requestedMode === "visual" || requestedMode === "hybrid" ? activeImage : null;
    const effectiveMode = requestedMode === "hybrid" && !visualText && !audioText && !usesImage && ocrText
      ? "ocr"
      : requestedMode;
    if (requestedMode === "temporal" && temporalEvents.some((item) => !item.trim())) {
      setError("Enter a description for every temporal event.");
      return;
    }
    if ((effectiveMode === "visual" && !visualText && !usesImage) || (effectiveMode === "audio" && !audioText) || (effectiveMode === "ocr" && !ocrText) || (effectiveMode === "hybrid" && !visualText && !audioText && !usesImage && !ocrText)) {
      setError("Enter a query for the selected search mode.");
      return;
    }
    requestRef.current?.controller.abort();
    pageRequestRef.current?.abort();
    const id = (requestRef.current?.id ?? 0) + 1;
    const controller = new AbortController();
    requestRef.current = { id, controller };
    setLoading(true);
    setError("");
    setMoreError("");
    setLoadingMore(false);
    try {
      if (requestedMode === "temporal") {
        const data = await searchTemporal({
          events: temporalEvents.map((item) => item.trim()),
          audio_text: audioText || undefined,
          audio_method: audioMethod,
          pe_weight: peWeight / 100,
          max_gap_seconds: temporalMaxGap,
          profile: profilesReady ? profile : undefined,
        }, controller.signal);
        if (requestRef.current?.id !== id) return;
        setTemporalResponse(data);
        const best = data.videos.flatMap((video) => video.sequences).sort((a, b) => a.rank - b.rank)[0];
        setSelected(best?.events[0] ? temporalSelection(best.events[0], best) : null);
        return;
      }
      const request = {
        mode: effectiveMode as ClassicSearchMode, index, nprobe, ef, profile: profilesReady ? profile : undefined, audio_method: audioMethod, pe_weight: peWeight / 100,
        visual_text: effectiveMode === "audio" || effectiveMode === "ocr" || !visualText ? undefined : visualText,
        audio_text: effectiveMode === "visual" || effectiveMode === "ocr" || !audioText ? undefined : audioText,
        ocr_text: (effectiveMode === "ocr" || effectiveMode === "hybrid") && ocrText ? ocrText : undefined,
      };
      const data = usesImage
        ? await searchVideosByImage({
            ...request,
            mode: effectiveMode as "visual" | "hybrid",
            image_weight: visualText ? imageWeight / 100 : 1,
            reference: usesImage.kind === "reference" ? {
              video_id: usesImage.result.video_id,
              frame_idx: usesImage.result.frame_idx,
            } : undefined,
          }, usesImage.kind === "upload" ? usesImage.file : null, controller.signal)
        : await searchVideos(request, controller.signal);
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

  function runSearch(event: FormEvent) {
    event.preventDefault();
    void executeSearch();
  }

  function rerankFrom(result: SearchResult) {
    const reference: QueryImage = { kind: "reference", result, preview: result.thumb };
    replaceQueryImage(reference);
    void executeSearch(reference);
  }

  async function loadNextPage() {
    if (mode === "temporal") {
      const cursor = temporalResponse?.next_cursor;
      const requestId = requestRef.current?.id;
      if (!cursor || !temporalResponse?.has_more || loadingMore || requestId === undefined) return;
      const controller = new AbortController();
      pageRequestRef.current?.abort();
      pageRequestRef.current = controller;
      setLoadingMore(true); setMoreError("");
      try {
        const data = await searchTemporalNextPage(cursor, controller.signal);
        if (requestRef.current?.id !== requestId || pageRequestRef.current !== controller) return;
        setTemporalResponse((current) => current?.next_cursor === cursor ? {
          ...data, videos: [...current.videos, ...data.videos],
        } : current);
      } catch (pageError) {
        if (controller.signal.aborted || requestRef.current?.id !== requestId) return;
        setMoreError(pageError instanceof Error ? pageError.message : "Unable to load more results");
      } finally {
        if (pageRequestRef.current === controller) setLoadingMore(false);
      }
      return;
    }
    const cursor = response?.next_cursor;
    const requestId = requestRef.current?.id;
    if (!cursor || !response?.has_more || loadingMore || requestId === undefined) return;
    const controller = new AbortController();
    pageRequestRef.current?.abort();
    pageRequestRef.current = controller;
    setLoadingMore(true);
    setMoreError("");
    try {
      const data = await searchNextPage(cursor, controller.signal);
      if (requestRef.current?.id !== requestId || pageRequestRef.current !== controller) return;
      setResponse((current) => {
        if (!current || current.next_cursor !== cursor) return current;
        const existingRanks = new Set(current.results.map((result) => result.rank));
        return {
          ...data,
          results: [...current.results, ...data.results.filter((result) => !existingRanks.has(result.rank))],
        };
      });
    } catch (pageError) {
      if (controller.signal.aborted || requestRef.current?.id !== requestId) return;
      setMoreError(pageError instanceof Error ? pageError.message : "Unable to load more results");
    } finally {
      if (pageRequestRef.current === controller) setLoadingMore(false);
    }
  }

  async function runCompare() {
    if (!visualQuery.trim()) {
      setError("Enter a visual query.");
      return;
    }
    setError("");
    try {
      setCompareResults(await compareIndexes(visualQuery.trim()));
    } catch (compareError) {
      setError(compareError instanceof Error ? compareError.message : "Comparison failed");
    }
  }

  useEffect(() => {
    let active = true;
    fetchSession().then(async (session) => {
      if (!active) return;
      setIsAdmin(session.is_admin);
      setLogin(session.login);
      const summaries = await fetchRounds();
      if (!active || !Array.isArray(summaries)) return;
      setRounds(summaries);
      const selectedRound = summaries.find((item) => item.active) ?? summaries[0];
      if (!selectedRound) return;
      const loaded = await fetchRound(selectedRound.id);
      if (!active || !loaded) return;
      roundEtagRef.current = loaded.etag;
      setActiveRound(loaded.round);
      setSelectedQueryId(loaded.round.queries[0]?.id ?? null);
    }).catch((sessionFetchError) => {
      if (!active) return;
      const message = sessionFetchError instanceof Error ? sessionFetchError.message : "Unable to verify your access";
      setSessionError(message === "Request failed" ? "Unable to verify your access" : message);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => () => {
    if (queryImageRef.current?.kind === "upload") {
      URL.revokeObjectURL(queryImageRef.current.preview);
    }
  }, []);

  useEffect(() => {
    if (!activeRound) return;
    const roundId = activeRound.id;
    const timer = window.setInterval(async () => {
      try {
        const summaries = await fetchRounds();
        if (!Array.isArray(summaries)) return;
        setRounds(summaries);
        const serverActive = summaries.find((item) => item.active);
        const nextRoundId = serverActive?.id ?? roundId;
        const loaded = await fetchRound(
          nextRoundId,
          nextRoundId === roundId ? roundEtagRef.current : undefined,
        );
        if (!loaded) return;
        roundEtagRef.current = loaded.etag;
        setActiveRound(loaded.round);
        if (nextRoundId !== roundId) setSelectedQueryId(loaded.round.queries[0]?.id ?? null);
      } catch { /* Keep the last usable snapshot; normal actions surface errors. */ }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeRound?.id]);

  async function selectRound(roundId: number) {
    try {
      const loaded = await fetchRound(roundId);
      if (!loaded) return;
      roundEtagRef.current = loaded.etag;
      clearSearchSession();
      setActiveRound(loaded.round);
      setSelectedQueryId(loaded.round.queries[0]?.id ?? null);
    } catch (roundError) {
      setError(roundError instanceof Error ? roundError.message : "Failed to load round");
    }
  }

  function loadRound(round: RoundSnapshot) {
    roundEtagRef.current = "";
    clearSearchSession();
    setActiveRound(round);
    setSelectedQueryId(round.queries[0]?.id ?? null);
    fetchRounds().then((items) => { if (Array.isArray(items)) setRounds(items); }).catch(() => undefined);
  }

  function updateCollabQuery(query: CollabQuery) {
    setActiveRound((round) => {
      if (!round) return round;
      const queries = round.queries.map((item) => item.id === query.id ? query : item);
      const counts = { todo: 0, in_progress: 0, ready: 0, reviewed: 0 };
      for (const item of queries) counts[item.status] += 1;
      return { ...round, queries, counts };
    });
    roundEtagRef.current = "";
  }

  useEffect(() => {
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setOpenResult(null);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const status = response
    ? `${response.results.length}/${response.total_results} keyframes loaded · ${groups.length} videos · ${response.latency_ms} ms`
    : "";
  const temporalStatus = temporalResponse
    ? `${temporalResponse.videos.length}/${temporalResponse.total_videos} videos · ${temporalResponse.total_sequences} sequences · ${temporalResponse.latency_ms} ms`
    : "";
  const searchStatus = mode === "temporal" ? temporalStatus : status;
  const currentViewMode = mode === "temporal" ? temporalViewMode : classicViewMode;

  const searchUi = <section className="search-page" hidden={tab !== "search"}>
      <SearchSidebar
        query={currentQuery}
        visualQuery={visualQuery}
        ocrQuery={ocrQuery}
        mode={mode}
        temporalEvents={temporalEvents}
        temporalMaxGap={temporalMaxGap}
        index={index}
        profile={profile}
        profiles={profiles}
        nprobe={nprobe}
        ef={ef}
        isAdmin={isAdmin}
        compareResults={compareResults}
        queryImage={queryImage}
        imageWeight={imageWeight}
        loading={loading}
        onOcrQuery={setOcrQuery}
        onMode={setMode}
        onTemporalEvents={setTemporalEvents}
        onTemporalMaxGap={setTemporalMaxGap}
        onIndex={setIndex}
        onProfile={(name) => {
          setProfile(name);
          const selected = profiles.find((item) => item.name === name);
          setIndex(selected?.indexes[0] ?? "flat");
          setResponse(null);
          setTemporalResponse(null);
        }}
        onNprobe={setNprobe}
        onEf={setEf}
        onCompare={runCompare}
        onQueryImage={useUploadedImage}
        onClearQueryImage={() => replaceQueryImage(null)}
        onImageWeight={setImageWeight}
      />
      <section className="search-main">
        <SearchComposer mode={mode} visualQuery={visualQuery} audioQuery={audioQuery}
          temporalEvents={temporalEvents} audioMethod={audioMethod} peWeight={peWeight} loading={loading}
          onVisualQuery={setVisualQuery} onAudioQuery={setAudioQuery}
          onTemporalEvents={setTemporalEvents} onAudioMethod={setAudioMethod} onPeWeight={setPeWeight}
          onSubmit={runSearch} />
        {mode === "temporal" ? <TemporalWorkspace
        response={temporalResponse} loading={loading} loadingMore={loadingMore}
        error={sessionError || error} moreError={moreError} onLoadMore={loadNextPage}
        viewMode={temporalViewMode} onViewMode={setTemporalViewMode}
        onOpen={(event: TemporalEventHit, sequence: TemporalSequence) => {
          const result = temporalSelection(event, sequence); setSelected(result); setOpenResult(result);
        }}
      /> : <ResultsWorkspace
        groups={groups}
        results={response?.results ?? []}
        selected={selected}
        loading={loading}
        loadingMore={loadingMore}
        hasMore={response?.has_more ?? false}
        searched={response !== null}
        error={sessionError || error}
        moreError={moreError}
        warning={response?.warning ?? ""}
        truncated={response?.truncated ?? false}
        profile={response?.profile}
        viewMode={classicViewMode} onViewMode={setClassicViewMode}
        onSelect={setSelected}
        onOpen={setOpenResult}
        onSimilar={rerankFrom}
        onLoadMore={loadNextPage}
      />}
      </section>
      <TranscriptPanel selected={selected} query={currentQuery} canAdd={Boolean(activeRound?.active && currentQuery)}
        onOpen={setOpenResult} onQueryChange={updateCollabQuery} />
  </section>;

  const tabs = <AppTabs tab={tab} onTab={setTab} />;
  const queryPicker = <div className="nav-query-current">
    <label className="nav-query-picker">Current query
      <select aria-label="Select current query" value={currentQuery?.id ?? ""}
        disabled={!activeRound?.queries.length} onChange={(event) => selectCollabQuery(Number(event.target.value))}>
        {!activeRound?.queries.length && <option value="">No queries yet</option>}
        {activeRound?.queries.map((query) => <option key={query.id} value={query.id}>{query.filename} · {query.kind.toUpperCase()}</option>)}
      </select>
    </label>
    {currentQuery && <div className="nav-query-meta">
      <span className={`query-status status-${currentQuery.status}`}>{({ todo: "To do", in_progress: "In progress", ready: "Ready", reviewed: "Reviewed" })[currentQuery.status]}</span>
      <span>Assignee: <b>{currentQuery.assignee?.split("@")[0] ?? "Unassigned"}</b></span>
      <span className={currentQuery.reviewed_by ? "reviewed" : "unreviewed"}>
        {currentQuery.reviewed_by ? `Reviewed by ${currentQuery.reviewed_by.split("@")[0]}` : "Not reviewed"}
      </span>
    </div>}
  </div>;

  if (tab === "clips") {
    return (
      <div className="app-shell clip-mode">
        <header className="team-toolbar">
          <div className="team-brand"><span className="eyebrow">AIC HCMC</span><strong>Clip export</strong></div>
          {tabs}
          {queryPicker}
        </header>
        <ClipStudio profile={profilesReady ? profile : undefined} isAdmin={isAdmin} />
      </div>
    );
  }

  return (
    <div className="app-shell workspace-mode">
      <header className="team-toolbar app-header">
        <div className="team-brand"><span className="eyebrow">AIC HCMC</span><strong>{tab === "queries" ? "Query management" : tab === "agent" ? "Retrieval agent" : "Search workspace"}</strong></div>
        {tabs}
        {tab === "search" && searchStatus && <div className="nav-search-status" aria-label="Search result status">{searchStatus}</div>}
        {tab === "search" && <div className="view-toggle" role="group" aria-label="Results view">
          <button type="button" className={currentViewMode === "grouped" ? "active" : ""} aria-pressed={currentViewMode === "grouped"}
            aria-label="Group by video" title="Group by video"
            onClick={() => (mode === "temporal" ? setTemporalViewMode : setClassicViewMode)("grouped")}>Grouped</button>
          <button type="button" className={currentViewMode === "ranked" ? "active" : ""} aria-pressed={currentViewMode === "ranked"}
            aria-label="Ranked list" title="Ranked list"
            onClick={() => (mode === "temporal" ? setTemporalViewMode : setClassicViewMode)("ranked")}>Ranked</button>
        </div>}
        {queryPicker}
      </header>
      {searchUi}
      <section className="queries-page" hidden={tab !== "queries"}>
        {login && <TeamToolbar rounds={rounds} round={activeRound} isAdmin={isAdmin}
          onSelectRound={selectRound} onRoundLoaded={loadRound} onError={setError} />}
        {activeRound && login ? <CollaborationPanels round={activeRound} login={login} selectedId={selectedQueryId}
          isAdmin={isAdmin}
          selectedResult={selected} openResult={openResult} onCloseVideo={() => setOpenResult(null)}
          searchResults={response?.results ?? []}
          onSelect={selectCollabQuery} onQueryChange={updateCollabQuery} />
          : <div className="empty-state query-empty"><h3>No query round yet</h3><p>Upload a round to start assigning and managing answers.</p></div>}
      </section>
      <section className="agent-page" hidden={tab !== "agent"}>
        <AgentPanel />
      </section>
      {openResult && !(activeRound && login) && <VideoDialog result={openResult} onClose={() => setOpenResult(null)} startPaused />}
    </div>
  );
}
