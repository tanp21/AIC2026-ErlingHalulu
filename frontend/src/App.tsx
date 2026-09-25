import { FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { compareIndexes, fetchKeyframeContext, fetchSession, refineQuery, searchNextPage, searchTemporalMultimodal, searchTemporalNextPage, searchVideos, searchVideosByImage } from "./api";
import { cachedCheck, forgetChecks, lookupCheck, runCheck } from "./checkApi";
import { CheckPanel, verdictTier } from "./checkView";
import { AgentPanel } from "./agentPanel";
import { fetchRound, fetchRounds } from "./collabApi";
import { CollaborationPanels, TeamToolbar } from "./collab";
import { AppTabs, ClipStudio } from "./clips";
import type { AppTab } from "./clips";
import { formatTimestamp, getRelativeScoreTier, groupResultsByVideo } from "./results";
import { LoadingOverlay } from "./loadingOverlay";
import { DresPanel } from "./dresPanel";
import { DresApiError, dresFrameTimestamp, dresPrepare, dresSetContext, dresStatus as fetchDresStatus, dresSubmit } from "./dresApi";
import { VideoDialog, type TrakeSequence } from "./videoDialog";
import { TemporalSidebarFields, temporalSelection, TemporalWorkspace } from "./temporal";
import type { ResultsViewMode } from "./temporal";
import { QueryBox } from "./queryBox";
import { ObjectCanvas, emptyObjectQuery } from "./objectCanvas";
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
  EventMode,
  ObjectQuery,
  VideoGroup,
  DresCandidate,
  DresStatus,
  QueryKind,
  RefineResponse,
  CheckConfig,
  CheckRequest,
  CheckState,
} from "./types";

type QueryImage =
  | { kind: "upload"; file: File; preview: string }
  | { kind: "reference"; result: SearchResult; preview: string };

interface SearchSidebarProps {
  visualQuery: string;
  ocrQuery: string;
  mode: SearchMode;
  temporalEvents: string[];
  temporalMaxGap: number;
  temporalImages: Array<QueryImage | null>;
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
  onTemporalEvents(events: string[]): void;
  onTemporalMaxGap(value: number): void;
  onTemporalImage(index: number, file: File): void;
  onClearTemporalImage(index: number): void;
  onIndex(value: IndexName): void;
  onProfile(value: string): void;
  onNprobe(value: number): void;
  onEf(value: number): void;
  onCompare(): void;
  onQueryImage(file: File): void;
  onClearQueryImage(): void;
  onImageWeight(value: number): void;
  composer: ReactNode;
}

function SearchSidebar(props: SearchSidebarProps) {
  const hasVisualIndex = props.mode === "hybrid" || props.mode === "temporal";
  const activeProfile = props.profiles.find((item) => item.name === props.profile) ?? props.profiles[0];
  const availableIndexes = activeProfile?.indexes ?? ["flat"];

  return (
    <aside className="search-sidebar" aria-label="Search controls">
      {props.composer}

      <div>
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

function RefineChip({ text, tooLong, muted, onApply }: { text: string; tooLong: boolean; muted?: boolean; onApply(): void }) {
  return <button type="button" className={`refine-chip${tooLong ? " refine-chip-warn" : ""}${muted ? " refine-chip-muted" : ""}`}
    title={tooLong ? "Longer than PE-Core's 72-token limit -- may be truncated" : undefined}
    onClick={onApply}>{text}</button>;
}

function RefinePanel({ text, kind, loading, error, refinement, onKind, onRefine, onDismiss,
  onApplyVisual, onApplyTranscript, onApplyOcr, onApplyEvents }: {
  text: string;
  kind: "kis" | "qa" | "trake";
  loading: boolean;
  error: string;
  refinement: RefineResponse | null;
  onKind(kind: "kis" | "qa" | "trake"): void;
  onRefine(): void;
  onDismiss(): void;
  onApplyVisual(text: string): void;
  onApplyTranscript(text: string): void;
  onApplyOcr(text: string): void;
  onApplyEvents(texts: string[]): void;
}) {
  return <div className="refine-panel">
    <div className="refine-panel-controls">
      <button type="button" className="refine-button" disabled={!text.trim() || loading} onClick={onRefine}>
        {loading ? "Refining…" : "✨ Refine with AI"}
      </button>
      <select aria-label="Refine as query kind" value={kind} onChange={(event) => onKind(event.target.value as typeof kind)}>
        <option value="kis">Moment (KIS)</option>
        <option value="qa">Question (QA)</option>
        <option value="trake">Sequence (TRAKE)</option>
      </select>
      {(refinement || error) && <button type="button" className="refine-dismiss" aria-label="Dismiss suggestions" onClick={onDismiss}>×</button>}
    </div>
    {error && <p className="refine-error" role="alert">{error}</p>}
    {refinement && <div className="refine-suggestions">
      {refinement.summary_vi && <p className="refine-summary">{refinement.summary_vi}</p>}
      {refinement.visual.some((candidate) => candidate.primary) && <div className="refine-group">
        <span>Visual (English) — best match</span>
        <div className="refine-chips">{refinement.visual.filter((candidate) => candidate.primary).map((candidate, index) => <RefineChip
          key={index} text={candidate.text} tooLong={(candidate.tokens ?? 0) > 72}
          onApply={() => onApplyVisual(candidate.text)} />)}</div>
      </div>}
      {refinement.visual.some((candidate) => !candidate.primary) && <div className="refine-group">
        <span>Other moments in this clip</span>
        <p className="refine-caveat">Less distinctive on its own — try the best match above first.</p>
        <div className="refine-chips">{refinement.visual.filter((candidate) => !candidate.primary).map((candidate, index) => <RefineChip
          key={index} text={candidate.text} tooLong={(candidate.tokens ?? 0) > 72} muted
          onApply={() => onApplyVisual(candidate.text)} />)}</div>
      </div>}
      {refinement.transcript.length > 0 && <div className="refine-group">
        <span>Transcript (Vietnamese)</span>
        <p className="refine-caveat">DeepSeek's guess at the wording, not a verified quote — may not match what's actually said.</p>
        <div className="refine-chips">{refinement.transcript.map((candidate, index) => <RefineChip
          key={index} text={candidate} tooLong={false} onApply={() => onApplyTranscript(candidate)} />)}</div>
      </div>}
      {refinement.ocr.length > 0 && <div className="refine-group">
        <span>On-screen text</span>
        <div className="refine-chips">{refinement.ocr.map((candidate, index) => <RefineChip
          key={index} text={candidate} tooLong={false} onApply={() => onApplyOcr(candidate)} />)}</div>
      </div>}
      {refinement.events.length > 1 && <div className="refine-group">
        <span>Events</span>
        <div className="refine-chips">{refinement.events.map((candidate, index) => <span key={index} className="refine-chip refine-chip-static">{candidate.text}</span>)}
          <button type="button" className="refine-chip refine-chip-apply-all"
            onClick={() => onApplyEvents(refinement.events.map((candidate) => candidate.text))}>Apply as temporal events</button></div>
      </div>}
      {refinement.lookups.length > 0 && <p className="refine-lookups">Consider searching the web for: {refinement.lookups.join("; ")}</p>}
    </div>}
  </div>;
}

function SearchComposer({ mode, temporalEvents, audioMethod, peWeight,
  queryImage, temporalMaxGap, temporalImages, loading, onVisualQuery, onAudioQuery, onOcrQuery, onTemporalEvents,
  temporalModes, onMode, onTemporalMode, onQueryImage, onClearQueryImage, onTemporalMaxGap, onTemporalImage,
  onClearTemporalImage, onRemoveTemporalEvent, onMoveTemporalEvent,
  objectQueries, onObjectQuery,
  onAudioMethod, onPeWeight, onSubmit, refinePanel }: {
  mode: SearchMode;
  visualQuery: string;
  audioQuery: string;
  ocrQuery: string;
  queryImage: QueryImage | null;
  temporalMaxGap: number;
  temporalImages: Array<QueryImage | null>;
  temporalModes: EventMode[];
  objectQueries: ObjectQuery[];
  temporalEvents: string[];
  audioMethod: AudioMethod;
  peWeight: number;
  loading: boolean;
  onVisualQuery(value: string): void;
  onAudioQuery(value: string): void;
  onOcrQuery(value: string): void;
  onMode(value: SearchMode): void;
  onTemporalMode(index: number, mode: EventMode): void;
  onObjectQuery(index: number, query: ObjectQuery): void;
  onQueryImage(file: File): void;
  onClearQueryImage(): void;
  onTemporalEvents(events: string[]): void;
  onTemporalMaxGap(value: number): void;
  onTemporalImage(index: number, file: File): void;
  onClearTemporalImage(index: number): void;
  onRemoveTemporalEvent(index: number): void;
  onMoveTemporalEvent(from: number, to: number): void;
  onAudioMethod(value: AudioMethod): void;
  onPeWeight(value: number): void;
  onSubmit(event: FormEvent): void;
  refinePanel?: ReactNode;
}) {
  const submitOnEnter = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };
  const firstMode = temporalModes[0] ?? "text";
  const firstLabel = firstMode === "audio" ? "Transcript query (Vietnamese)"
    : firstMode === "ocr" ? "OCR query" : "Visual query (English)";
  const peFooter = <>
    <span>PE-Core influence</span>
    <input type="range" min="0" max="100" aria-label="PE-Core influence" value={peWeight}
      onChange={(event) => onPeWeight(Number(event.target.value))} />
    <strong>{peWeight}%</strong>
  </>;
  return <section className="search-composer sidebar-composer" aria-label="Search query">
    <form onSubmit={onSubmit}>
      <div className={`composer-fields mode-${mode}`}>
        <div className="temporal-event-field temporal-first-event">
          <div className="event-modality-tabs first-event-tabs" role="group" aria-label="Event E1 modality">
            <button type="button" aria-label="Visual text for event E1" aria-pressed={firstMode === "text"}
              className={firstMode === "text" ? "active" : ""} onClick={() => onTemporalMode(0, "text")}>T</button>
            <button type="button" aria-label="Transcript for event E1" aria-pressed={firstMode === "audio"}
              className={firstMode === "audio" ? "active" : ""} onClick={() => onTemporalMode(0, "audio")}>♪</button>
            <button type="button" aria-label="OCR for event E1" aria-pressed={firstMode === "ocr"}
              className={firstMode === "ocr" ? "active" : ""} onClick={() => onTemporalMode(0, "ocr")}>OCR</button>
            <button type="button" aria-label="Image for event E1" aria-pressed={firstMode === "image"}
              className={firstMode === "image" ? "active" : ""} onClick={() => onTemporalMode(0, "image")}>▧</button>
            <button type="button" aria-label="Object for event E1" aria-pressed={firstMode === "object"}
              className={firstMode === "object" ? "active" : ""} onClick={() => onTemporalMode(0, "object")}>Obj</button>
          </div>
          {firstMode === "object" ? <ObjectCanvas title="Object event E1" query={objectQueries[0]}
            onChange={(query) => onObjectQuery(0, query)} /> : firstMode === "image" ? <div className="composer-query-box temporal-first-image">
            <div className="temporal-event-image">
              {queryImage && <img src={queryImage.preview} alt="Visual event E1 image" />}
              <label>{queryImage ? "Change image" : "Choose image"}<input aria-label="Visual event E1 image" type="file"
                accept="image/jpeg,image/png,image/webp" onChange={(event) => {
                  const file = event.target.files?.[0]; if (file) onQueryImage(file);
                  event.currentTarget.value = "";
                }} /></label>
              {queryImage && <button type="button" aria-label="Clear visual event E1 image" onClick={onClearQueryImage}>×</button>}
            </div>
          </div> : <QueryBox
            label={firstLabel}
            ariaLabel={firstLabel}
            value={temporalEvents[0] ?? ""}
            onChange={(value) => {
              onTemporalEvents(temporalEvents.map((event, index) => index === 0 ? value : event));
              if (firstMode === "audio") onAudioQuery(value);
              else if (firstMode === "ocr") onOcrQuery(value);
              else if (firstMode === "text") onVisualQuery(value);
            }}
            onKeyDown={submitOnEnter}
            placeholder={firstMode === "audio" ? "Words or speech to find…"
              : firstMode === "ocr" ? "Text shown on screen…" : "Describe the visible scene in English…"}
            autoFocus
            footer={refinePanel}
          />}
          {firstMode !== "object" && <ObjectCanvas title="Object position filter · E1" buttonLabel="Add object filter to E1" query={objectQueries[0]}
            onChange={(query) => onObjectQuery(0, query)} />}
          {mode === "temporal" && temporalEvents.length > 1 && <div className="event-actions temporal-primary-actions">
            <button type="button" aria-label="Move event E1 up" disabled>↑</button>
            <button type="button" aria-label="Move event E1 down" onClick={() => onMoveTemporalEvent(0, 1)}>↓</button>
            <button type="button" aria-label="Remove event E1" onClick={() => onRemoveTemporalEvent(0)}>×</button>
          </div>}
        </div>
        {mode === "temporal" ? <TemporalSidebarFields events={temporalEvents.slice(1)} maxGap={temporalMaxGap}
            modes={temporalModes}
            objectQueries={objectQueries}
            images={[queryImage, ...temporalImages.slice(1)]}
            onMode={onTemporalMode}
            onEvents={(events) => {
              if (!events.length) {
                onVisualQuery(temporalEvents[0] ?? "");
                onTemporalEvents([temporalEvents[0] ?? ""]);
                onMode("hybrid");
              } else onTemporalEvents([temporalEvents[0] ?? "", ...events]);
            }}
            onMaxGap={onTemporalMaxGap} onImage={onTemporalImage} onClearImage={onClearTemporalImage}
            onObjectQuery={onObjectQuery}
            onRemoveEvent={onRemoveTemporalEvent} onMoveEvent={onMoveTemporalEvent} /> : <button type="button" className="add-event" aria-label="Add event"
            onClick={() => onMode("temporal")}>＋ Add event</button>}
        <div className="shared-search-controls">
          <div className="composer-query-footer shared-pe-influence">{peFooter}</div>
          <label className="field shared-audio-method"><span>Transcript search</span>
            <select aria-label="Audio search" value={audioMethod} onChange={(event) => onAudioMethod(event.target.value as AudioMethod)}>
              <option value="semantic">Semantic</option><option value="keyword">Keyword</option>
            </select>
          </label>
        </div>
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
  dresKind: QueryKind;
  dresAnswer: string;
  trakeEvents: number[];
  trakeVideoId: string | null;
  onSelect(): void;
  onOpen(): void;
  onSimilar(): void;
  onSubmitDres(expectedTimestampMs: number): Promise<{ message: string; completed?: boolean; uncertain?: boolean; timestamp_ms?: number; timestamp_mismatch?: boolean }>;
  onAddTrakeEvent(): void;
}

function KeyframeCard({ result, selected, tier, dresKind, dresAnswer, trakeEvents,
  trakeVideoId, onSelect, onOpen, onSimilar, onSubmitDres, onAddTrakeEvent }: KeyframeCardProps) {
  const time = formatTimestamp(result.time);
  const [timestampOverride, setTimestampOverride] = useState<number | null>(null);
  const [dresBusy, setDresBusy] = useState(false);
  const [dresLocked, setDresLocked] = useState(false);
  const [dresMessage, setDresMessage] = useState("");
  const expectedTimestampMs = timestampOverride ?? Math.max(0, Math.round(result.time * 1000));
  const trakeOtherVideo = Boolean(trakeEvents.length && trakeVideoId && trakeVideoId !== result.video_id);
  useEffect(() => {
    setTimestampOverride(null); setDresBusy(false); setDresLocked(false); setDresMessage("");
  }, [result.video_id, result.frame_idx]);

  async function submitFrame() {
    if (dresBusy || dresLocked) return;
    setDresBusy(true); setDresMessage("");
    try {
      const outcome = await onSubmitDres(expectedTimestampMs);
      if (outcome.timestamp_mismatch && outcome.timestamp_ms !== undefined) {
        setTimestampOverride(outcome.timestamp_ms);
        setDresMessage(`Exact time refreshed · submit again`);
      } else {
        setDresMessage(outcome.message);
        setDresLocked(Boolean(outcome.completed || outcome.uncertain));
      }
    } catch (error) {
      setDresMessage(error instanceof Error ? error.message : "DRES submission failed");
    } finally { setDresBusy(false); }
  }
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
      {dresKind === "trake" ? <button type="button" className="keyframe-dres-action" title="Add this frame to the TRAKE sequence"
        aria-label={`Add ${result.video_id} frame ${result.frame_idx} to TRAKE`}
        disabled={trakeOtherVideo || trakeEvents.length >= 100} onClick={onAddTrakeEvent}>＋</button>
        : <button type="button" className="keyframe-dres-action"
          title={dresKind === "qa" && !dresAnswer.trim() ? "Enter a Q&A answer above the results" : dresMessage || "Submit this frame to DRES"}
          aria-label={`Submit ${result.video_id} frame ${result.frame_idx} to DRES`}
          disabled={dresBusy || dresLocked || dresKind === "qa" && !dresAnswer.trim()}
          onClick={() => void submitFrame()}>{dresBusy ? "…" : dresLocked ? "✓" : "↑"}</button>}
      <footer>
        <div className="score-block" aria-label={`${result.source === "object" ? "Object confidence" : "Similarity"} level ${tier} of 5, raw score ${result.score.toFixed(4)}`}>
          <span className="score-bars" aria-hidden="true">
            {[1, 2, 3, 4, 5].map((level) => <i key={level} className={level <= tier ? "filled" : ""} />)}
          </span>
          <strong>{result.score.toFixed(4)}</strong>
        </div>
        <button type="button" className="similar-icon" onClick={onSimilar} aria-label={`Find frames similar to ${result.video_id} at ${time}`}>≈</button>
        <button type="button" className="play-icon" onClick={onOpen} aria-label={`Open video ${result.video_id} at ${time}`}>▶</button>
      </footer>
      {dresMessage && dresKind !== "trake" && <small className="keyframe-dres-message" role="status">{dresMessage}</small>}
      {!!result.matched_objects?.length && <div className="object-result-tags">
        {result.matched_objects.map((item, index) => <span key={`${item.label}-${index}`}>{item.label} {Math.round(item.confidence * 100)}%</span>)}
      </div>}
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
  dresKind: QueryKind;
  dresAnswer: string;
  trakeCandidate: DresCandidate | null;
  trakeMessage: string;
  onDresAnswer(answer: string): void;
  onSubmitDresFrame(result: SearchResult, expectedTimestampMs: number): Promise<{ message: string; completed?: boolean; uncertain?: boolean; timestamp_ms?: number; timestamp_mismatch?: boolean }>;
  onAddTrakeFrame(result: SearchResult): void;
  onOpenTrakeFrame(videoId: string, frameId: number): void;
  onMoveTrakeFrame(from: number, to: number): void;
  onRemoveTrakeFrame(index: number): void;
  onSubmitTrake(): void;
  check: CheckState;
  checkAvailable: boolean;
  onCheck(): void;
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

  function renderKeyframe(result: SearchResult) {
    const events = props.trakeCandidate?.frame_ids ?? [];
    return <KeyframeCard key={`${result.video_id}-${result.frame_idx}-${result.rank}`} result={result}
      selected={props.selected?.video_id === result.video_id && props.selected.frame_idx === result.frame_idx}
      tier={getRelativeScoreTier(result.score, min, max)} dresKind={props.dresKind} dresAnswer={props.dresAnswer}
      trakeEvents={events} trakeVideoId={props.trakeCandidate?.video_id ?? null}
      onSelect={() => props.onSelect(result)} onOpen={() => props.onOpen(result)} onSimilar={() => props.onSimilar(result)}
      onSubmitDres={(ms) => props.onSubmitDresFrame(result, ms)} onAddTrakeEvent={() => props.onAddTrakeFrame(result)} />;
  }

  return (
    <main ref={workspaceRef} className="results-workspace" tabIndex={0} onKeyDown={handleKeys} aria-label="Search results">
      {props.error && <div className="message error" role="alert">{props.error}</div>}
      {props.warning && <div className="message warning">{props.warning}</div>}

      {props.dresKind === "qa" && <div className="dres-result-tools dres-qa-tools">
        <label>Q&amp;A answer<input aria-label="DRES answer for retrieved frames" value={props.dresAnswer}
          onChange={(event) => props.onDresAnswer(event.target.value)} placeholder="Type the answer to submit" /></label>
        <span>Choose a frame below to submit with this answer.</span>
      </div>}
      {props.dresKind === "trake" && <section className="dres-result-tools dres-trake-tools" aria-label="TRAKE event sequence">
        <header><strong>TRAKE events</strong><span>{props.trakeCandidate?.frame_ids?.length ?? 0} selected</span></header>
        {!!props.trakeCandidate?.frame_ids?.length && <ol>{props.trakeCandidate.frame_ids.map((frameId, index) => <li key={`${props.trakeCandidate?.video_id}-${frameId}`}>
          <button type="button" className="dres-trake-open" onClick={() => props.onOpenTrakeFrame(props.trakeCandidate!.video_id, frameId)}>
            E{index + 1} · FRAME_ID {frameId}
          </button>
          <button type="button" aria-label={`Move TRAKE event E${index + 1} up`} disabled={index === 0} onClick={() => props.onMoveTrakeFrame(index, index - 1)}>↑</button>
          <button type="button" aria-label={`Move TRAKE event E${index + 1} down`} disabled={index === props.trakeCandidate!.frame_ids!.length - 1} onClick={() => props.onMoveTrakeFrame(index, index + 1)}>↓</button>
          <button type="button" aria-label={`Remove TRAKE event E${index + 1}`} onClick={() => props.onRemoveTrakeFrame(index)}>×</button>
        </li>)}</ol>}
        <button type="button" className="dres-trake-submit" disabled={!props.trakeCandidate?.frame_ids?.length}
          onClick={props.onSubmitTrake}>Submit TRAKE sequence</button>
        {props.trakeMessage && <small role="status">{props.trakeMessage}</small>}
      </section>}

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
              <div className="frame-strip">{group.results.map(renderKeyframe)}</div>
            </section>
          ))}</div>
        ) : (
          <div className="results-grid">
                {props.results.map(renderKeyframe)}
          </div>
        )}
        <KeyframeContext
          selected={props.selected} profile={props.profile} suppressOwnLoading={props.loading}
          check={props.check} checkAvailable={props.checkAvailable} onCheck={props.onCheck}
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

/**
 * What a frame of the strip already knows: the verdict of an answer whose shot covers it.
 *
 * The server keys a check by its shot, so one press answers for every keyframe of that shot.
 * Showing the mark here is what makes that visible -- and clicking one of those frames renders
 * the answer from the cache with no request at all.
 */
function checkMarkFor(check: CheckState, frameId: number): "good" | "mid" | "bad" | null {
  const result = check.status === "done" ? check.result : null;
  if (!result?.covers || result.covers.start_frame === null || result.covers.end_frame === null) return null;
  if (frameId < result.covers.start_frame || frameId > result.covers.end_frame) return null;
  return verdictTier(result.verdict.score);
}

function KeyframeContext({ selected, profile, suppressOwnLoading, check, checkAvailable, onCheck, onSelect, onOpen }: { selected: SearchResult | null; profile?: string; suppressOwnLoading?: boolean; check: CheckState; checkAvailable: boolean; onCheck(): void; onSelect(result: SearchResult): void; onOpen(result: SearchResult): void }) {
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
    <div><strong>{selected.video_id} <b>Selected frame {selected.frame_idx}</b></strong>
      {checkAvailable && <button type="button" className="check-button" disabled={check.status === "loading"}
        onClick={onCheck}>{check.status === "loading" ? "Checking…" : "✓ Check by AI"}</button>}
      <button type="button" onClick={() => onOpen(selected)}>▶ Play video</button></div>
    <div className={`results-body context-frames-body${showOwnLoading ? " is-loading" : ""}`}>
      <LoadingOverlay active={showOwnLoading} />
      <div ref={stripRef} className="context-frames">
      {!loading && !frames.length && <span className="context-loading">No neighbouring keyframes.</span>}
      {frames.map((frame) => {
        const active = frame.frame_idx === selected.frame_idx;
        const mark = checkMarkFor(check, frame.frame_idx);
        return <button key={frame.frame_idx} type="button" aria-current={active ? "true" : undefined}
          aria-label={`${active ? "Selected " : "Select "}frame ${frame.frame_idx}${mark ? " · checked" : ""}`}
          onClick={() => onSelect({ ...frame, rank: selected.rank, score: 0, source: "visual", scores: {}, context_only: true })} className={active ? "active" : ""}>
          <img src={`${frame.thumb}?w=160`} alt="" /><span>{formatTimestamp(frame.time)}</span><em>Frame {frame.frame_idx}</em>
          {mark && <u className={`check-badge check-${mark}`} aria-hidden="true">
            {check.result?.verdict.score.toFixed(0)}
          </u>}
          {active && <i aria-hidden="true">✓</i>}
        </button>;
      })}
      </div>
    </div>
  </section>;
}

function TranscriptPanel({ selected, check, keywords, onRecheck }: {
  selected: SearchResult | null;
  check: CheckState;
  /** The query's words, lit up inside the quotes the checker read its verdicts from. */
  keywords: string[];
  onRecheck(): void;
}) {
  return (
    <aside className="transcript-panel" aria-label="Transcript evidence">
      {!selected ? (
        <div className="transcript-empty"><span>RANK</span><p>Select a keyframe to see ranking and matched content.</p></div>
      ) : (
        <>
          <section className="selection-summary">
            <img src={selected.thumb} alt="Selected keyframe" />
            <div><strong>{selected.video_id}</strong><span>frame {selected.frame_idx} · {formatTimestamp(selected.time)}</span></div>
          </section>
          {/* The AI check leads: it is the thing the person just pressed a button for. */}
          <CheckPanel state={check} variant="panel" keywords={keywords} onRecheck={onRecheck} />
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
            {selected.scores.object_score !== undefined &&
              <div className="ranking-group object"><h3>Object</h3><dl>
                <div><dt>Confidence</dt><dd>{selected.scores.object_score.toFixed(4)}</dd></div>
                {selected.scores.object_rank != null && <div><dt>Object rank</dt><dd>#{selected.scores.object_rank}</dd></div>}
                {selected.object_delta_ms != null && <div><dt>Detection offset</dt><dd>{selected.object_delta_ms.toFixed(0)} ms</dd></div>}
              </dl>
              {!!selected.matched_objects?.length && <p>{selected.matched_objects.map((item) => `${item.label} ${Math.round(item.confidence * 100)}%`).join(" · ")}</p>}
              <small>Positions come from the nearest detection frame, within 0.25 seconds.</small>
              </div>}
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
        </>
      )}
    </aside>
  );
}

export default function App() {
  const [visualQuery, setVisualQuery] = useState("");
  const [audioQuery, setAudioQuery] = useState("");
  const [ocrQuery, setOcrQuery] = useState("");
  const [temporalEvents, setTemporalEvents] = useState(["", ""]);
  const [temporalMaxGap, setTemporalMaxGap] = useState(120);
  const [temporalImages, setTemporalImages] = useState<Array<QueryImage | null>>([null, null]);
  const [temporalModes, setTemporalModes] = useState<EventMode[]>(["text", "text"]);
  const [objectQueries, setObjectQueries] = useState<ObjectQuery[]>([emptyObjectQuery(), emptyObjectQuery()]);
  const temporalImagesRef = useRef(temporalImages);
  temporalImagesRef.current = temporalImages;
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [index, setIndex] = useState<IndexName>("flat");
  const [profiles] = useState<IndexProfile[]>([
    { name: "l2026-v2", label: "PE-Core + NaFlex · 2026 (L+M+S)", is_default: true, count: 590325, video_count: 1189, indexes: ["flat"] },
  ]);
  const [profile, setProfile] = useState("l2026-v2");
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
  const [dresCandidate, setDresCandidate] = useState<DresCandidate | null>(null);
  const dresCandidateRef = useRef<DresCandidate | null>(null);
  dresCandidateRef.current = dresCandidate;
  const [dresKind, setDresKind] = useState<QueryKind>("kis");
  const [dresQaAnswer, setDresQaAnswer] = useState("");
  const [trakeMessage, setTrakeMessage] = useState("");
  const [dresPickError, setDresPickError] = useState("");
  const [dresStatus, setDresStatus] = useState<DresStatus | null>(null);
  const [dresSettingsOpen, setDresSettingsOpen] = useState(false);
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
  const [refineKind, setRefineKind] = useState<"kis" | "qa" | "trake">("kis");
  const [checkConfig, setCheckConfig] = useState<CheckConfig | null>(null);
  const [checkState, setCheckState] = useState<CheckState>({ status: "idle", result: null, error: null, asked: null });
  // The query actually SUBMITTED to produce the results/selection on screen -- not
  // `temporalEvents[0]`, which is live-editing state. Switching the query box to a modality
  // that has nothing typed yet (e.g. tapping the "Transcript" tab before writing one) empties
  // `temporalEvents[0]` while the results grid still shows the PREVIOUS query's matches, so a
  // check must keep asking about the query that actually found them, not whatever the box
  // currently holds. Kept as separate channels, not one flattened sentence: a visual
  // description, spoken words to find and on-screen text to find mean different things to the
  // checker (`agent.checker.CheckQuery`), exactly as they do to search itself.
  const [checkQuery, setCheckQuery] = useState<
    { visualText: string; audioText: string; ocrText: string; kind: QueryKind } | null>(null);
  const checkRequestRef = useRef<AbortController | null>(null);
  /** The query's own words, for lighting them up inside the quotes a check read its verdicts from. */
  const checkKeywords = useMemo(
    () => [checkQuery?.visualText, checkQuery?.audioText, checkQuery?.ocrText]
      .filter((text): text is string => !!text).join(" ").split(/\s+/).filter((word) => word.length > 1),
    [checkQuery?.visualText, checkQuery?.audioText, checkQuery?.ocrText]);
  const [refinement, setRefinement] = useState<RefineResponse | null>(null);
  const [refineLoading, setRefineLoading] = useState(false);
  const [refineError, setRefineError] = useState("");
  const requestRef = useRef<{ id: number; controller: AbortController } | null>(null);
  const refineRequestRef = useRef<AbortController | null>(null);
  const pageRequestRef = useRef<AbortController | null>(null);
  const roundEtagRef = useRef("");
  const queryImageRef = useRef<QueryImage | null>(null);

  useEffect(() => {
    let active = true;
    const refreshDresStatus = () => fetchDresStatus().then((status) => {
      if (active) {
        setDresStatus(status);
      }
    }).catch(() => { if (active) setDresStatus(null); });
    void refreshDresStatus();
    const timer = window.setInterval(refreshDresStatus, 30_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  function openDresSettings() {
    setTab("search");
    setDresSettingsOpen(true);
  }

  async function openDresFrame(videoId: string, frameId: number) {
    try {
    const time = await dresFrameTimestamp(videoId, frameId) / 1000;
    const found = response?.results.find((result) => result.video_id === videoId);
    const base = found ?? selected;
    const result: SearchResult = base ? { ...base, video_id: videoId, frame_idx: frameId, time,
      thumb: `/thumb/${encodeURIComponent(videoId)}/${frameId}`,
      frame: `/frame/${encodeURIComponent(videoId)}/${frameId}`,
      video: `/video/${encodeURIComponent(videoId)}`, context_only: true } : {
      rank: 0, score: 0, video_id: videoId, frame_idx: frameId, time,
      thumb: `/thumb/${encodeURIComponent(videoId)}/${frameId}`,
      frame: `/frame/${encodeURIComponent(videoId)}/${frameId}`,
      video: `/video/${encodeURIComponent(videoId)}`, source: "visual", scores: {}, context_only: true,
    };
    setOpenResult(result);
    } catch (reason) {
      setDresPickError(reason instanceof Error ? reason.message : "Could not resolve this event frame.");
    }
  }

  async function selectDresKind(kind: QueryKind) {
    const status = await fetchDresStatus();
    if (status.connected) {
      await dresSetContext({ expected_revision: status.context?.revision ?? 0, kind,
        label: `${kind.toUpperCase()} task`,
        event_count: null });
      setDresStatus(await fetchDresStatus());
    }
    if (dresKind !== kind) { setDresCandidate(null); setDresQaAnswer(""); setTrakeMessage(""); }
    setDresKind(kind);
  }

  async function submitDresCandidate(candidate: DresCandidate, expectedTimestampMs?: number) {
    let status = dresStatus;
    if (status?.connected && (!status.context_valid || status.context?.kind !== dresKind
      || dresKind === "trake" && status.context?.event_count !== null)) {
      try {
        await selectDresKind(dresKind);
        status = await fetchDresStatus();
      } catch (error) {
        return { message: error instanceof Error ? error.message : "Could not select DRES task.", completed: false };
      }
    }
    if (!status?.connected || !status.evaluation_id || !status.context_valid) {
      return { message: "DRES is not ready. Check DRES settings.", completed: false };
    }
    let prepared;
    try { prepared = await dresPrepare(candidate); }
    catch (error) { return { message: error instanceof Error ? error.message : "Could not prepare DRES payload.", completed: false }; }
    if (expectedTimestampMs !== undefined && prepared.timestamp_ms == null) {
      return { message: "DRES did not provide an exact frame timestamp; submission stopped.", completed: false };
    }
    if (expectedTimestampMs !== undefined && prepared.timestamp_ms !== expectedTimestampMs) {
      return { message: "Exact timestamp updated", timestamp_ms: prepared.timestamp_ms ?? undefined, timestamp_mismatch: true, completed: false };
    }
    try {
      const attempt = await dresSubmit(candidate, prepared);
      void fetchDresStatus().then(setDresStatus).catch(() => undefined);
      return { message: `${attempt.status.toUpperCase()} · HTTP ${attempt.http_status ?? "unknown"} · ${attempt.server_message ?? "No server message"}`, completed: true, timestamp_ms: prepared.timestamp_ms ?? undefined };
    } catch (error) {
      const uncertain = !(error instanceof DresApiError) || error.category === "uncertain";
      return { message: `${uncertain ? "UNCERTAIN" : "NOT SENT / REJECTED"} · ${error instanceof Error ? error.message : "DRES submission failed"}${uncertain ? ". Check DRES before any retry." : ""}`, uncertain, completed: false, timestamp_ms: prepared.timestamp_ms ?? undefined };
    }
  }

  function addTrakeEvent(videoId: string, frameId: number): string {
    const latest = dresCandidateRef.current;
    if (latest && latest.video_id !== videoId)
      return `This sequence uses ${latest.video_id}; remove its events before switching videos.`;
    const frames = latest?.frame_ids ?? [];
    if (frames.includes(frameId)) return `FRAME_ID ${frameId} is already in the sequence.`;
    if (frames.length >= 100) return "TRAKE supports at most 100 events.";
    const next = { video_id: videoId, frame_ids: [...frames, frameId] };
    dresCandidateRef.current = next;
    setDresCandidate(next);
    return `Added E${next.frame_ids.length} · FRAME_ID ${frameId}`;
  }

  async function addSearchResultTrake(result: SearchResult) {
    setTrakeMessage("");
    // The event is the frame the result points at; the shown time needs the frame map, the id does not.
    const message = addTrakeEvent(result.video_id, result.frame_idx);
    setTrakeMessage(message);
    try {
      const timestampMs = await dresFrameTimestamp(result.video_id, result.frame_idx);
      setTrakeMessage(`${message} · ${formatTimestamp(timestampMs / 1000)}`);
    } catch { /* an unverified time only costs the suffix */ }
  }

  function moveTrakeFrame(from: number, to: number) {
    setDresCandidate((current) => {
      if (!current?.frame_ids || to < 0 || to >= current.frame_ids.length) return current;
      const frame_ids = [...current.frame_ids]; [frame_ids[from], frame_ids[to]] = [frame_ids[to], frame_ids[from]];
      return { ...current, frame_ids };
    });
  }

  function removeTrakeFrame(index: number) {
    setDresCandidate((current) => {
      if (!current?.frame_ids) return current;
      const frame_ids = current.frame_ids.filter((_, i) => i !== index);
      return frame_ids.length ? { ...current, frame_ids } : null;
    });
  }

  async function submitTrakeSequence() {
    if (!dresCandidate?.frame_ids?.length) return { message: "Add a TRAKE event first.", completed: false };
    const outcome = await submitDresCandidate(dresCandidate);
    setTrakeMessage(outcome.message);
    return outcome;
  }

  async function submitPlayerFrame(videoId: string, frameId: number, _time: number, answer: string, kind: QueryKind) {
    let status = dresStatus;
    if (status?.context?.kind !== kind || !status.context_valid) {
      await selectDresKind(kind);
      status = await fetchDresStatus();
      setDresStatus(status);
    }
    const context = status?.context_valid ? status.context : null;
    if (!status?.connected || !status.evaluation_id || !context) {
      return { message: "DRES is not ready. Set the task and evaluation in the DRES panel first.", completed: false };
    }
    let candidate: DresCandidate;
    if (context.kind === "trake") {
      // The event is the source frame the player stopped on, the same number the button names.
      const message = addTrakeEvent(videoId, frameId);
      setTrakeMessage(message);
      return { message, completed: false };
    } else {
      candidate = { video_id: videoId, frame_idx: frameId,
        frame_source: "original_frame",
        ...(context.kind === "qa" ? { answer } : {}) };
    }
    return submitDresCandidate(candidate);
  }

  const dresLabel = !dresStatus
    ? "DRES · Status unavailable"
    : !dresStatus.configured || !dresStatus.credentials_configured
      ? "DRES · Setup required"
      : !dresStatus.connected
        ? "DRES · Disconnected"
        : !dresStatus.evaluation_id
          ? dresStatus.evaluations.length > 0 ? "DRES · Choose evaluation" : "DRES · No active evaluation"
          : "DRES · Connected";
  const dresTone = dresStatus?.connected && dresStatus.evaluation_id ? "ready"
    : dresStatus?.connected ? "warning" : "offline";
  const dresNavKind = dresKind;
  const dresNavControls = <div className="nav-dres-controls">
    <select aria-label="DRES task type" value={dresNavKind}
      onChange={(event) => { void selectDresKind(event.target.value as QueryKind).catch((reason) =>
        setError(reason instanceof Error ? reason.message : "Could not update DRES task type.")); }}>
      <option value="kis">KIS</option><option value="qa">Q&amp;A</option><option value="trake">TRAKE</option>
    </select>
    <button type="button" className={`nav-dres-status ${dresTone}`} onClick={openDresSettings} title={`DRES settings · ${dresLabel}`}>{dresLabel}</button>
  </div>;
  // One sequence, both video dialogs: the standalone player and the one inside a round. The events
  // live here, so the player can list, reorder and renumber them while the video plays. It is passed
  // to both regardless of the selected task: a player whose picker opens in TRAKE mode without it
  // counts zero events, so the button reads "Add E1" forever and no sequence is ever listed. Which
  // task the picker is in comes from the DRES context as well as this selector, so gating the events
  // on the selector alone leaves that door open. The player renders the sequence only for TRAKE.
  const trakeSequence: TrakeSequence = {
    videoId: dresCandidate?.video_id ?? null, frameIds: dresCandidate?.frame_ids ?? [],
    onRemove: removeTrakeFrame, onMove: moveTrakeFrame, onSubmit: submitTrakeSequence,
    onOpen: (frameId) => { if (dresCandidate) void openDresFrame(dresCandidate.video_id, frameId); },
  };

  const groups = useMemo(() => groupResultsByVideo(response?.results ?? []), [response]);
  const currentQuery = activeRound?.queries.find((query) => query.id === selectedQueryId)
    ?? activeRound?.queries[0] ?? null;

  function pickDresFrame(videoId: string, frameId: number) {
    setDresPickError("");
    if (dresKind === "trake") {
      if (dresCandidate && dresCandidate.video_id !== videoId) {
        setDresPickError("TRAKE frames must belong to the same video. Clear the candidate to start another video.");
        return;
      }
      const frames = dresCandidate?.frame_ids ?? [];
      if (frames.includes(frameId)) { setDresPickError("This FRAME_ID is already selected."); return; }
      setDresCandidate({ video_id: videoId, frame_ids: [...frames, frameId] });
    } else if (dresKind) {
      setDresCandidate({ video_id: videoId, frame_idx: frameId });
    }
  }

  function replaceQueryImage(next: QueryImage | null) {
    if (queryImageRef.current?.kind === "upload") URL.revokeObjectURL(queryImageRef.current.preview);
    queryImageRef.current = next;
    setQueryImage(next);
  }

  function setTemporalImage(index: number, file: File) {
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 6 * 1024 * 1024) {
      setError("Choose a JPEG, PNG, or WebP image no larger than 6 MiB.");
      return;
    }
    setTemporalImages((current) => {
      const next = [...current];
      const old = next[index];
      if (old?.kind === "upload") URL.revokeObjectURL(old.preview);
      next[index] = { kind: "upload", file, preview: URL.createObjectURL(file) };
      return next;
    });
  }

  function clearTemporalImage(index: number) {
    setTemporalImages((current) => {
      const next = [...current];
      if (next[index]?.kind === "upload") URL.revokeObjectURL(next[index]!.preview);
      next[index] = null;
      return next;
    });
  }

  function setTemporalMode(index: number, nextMode: EventMode) {
    setTemporalModes((current) => {
      const next = [...current];
      while (next.length <= index) next.push("text");
      next[index] = nextMode;
      return next;
    });
    setObjectQueries((current) => {
      const next = [...current];
      while (next.length <= index) next.push(emptyObjectQuery());
      return next;
    });
    // Event E1's textarea is one shared field for whichever modality tab is active
    // (see executeSearch, which reads temporalEvents[0] alone). Switching tabs must
    // restore that modality's own text instead of leaving the previous tab's text
    // showing under a new label -- otherwise a visual query and a transcript query
    // look like the same box being overwritten by each other.
    if (index === 0 && (nextMode === "text" || nextMode === "audio" || nextMode === "ocr")) {
      const restored = nextMode === "audio" ? audioQuery : nextMode === "ocr" ? ocrQuery : visualQuery;
      setTemporalEvents((events) => [restored, ...events.slice(1)]);
    }
  }

  function setObjectQuery(index: number, query: ObjectQuery) {
    setObjectQueries((current) => {
      const next = [...current];
      while (next.length <= index) next.push(emptyObjectQuery());
      next[index] = query;
      return next;
    });
  }

  function moveTemporalEvent(from: number, to: number) {
    if (from === to || from < 0 || to < 0 || from >= temporalEvents.length || to >= temporalEvents.length) return;
    const nextEvents = [...temporalEvents];
    const nextModes = [...temporalModes];
    const nextObjects = [...objectQueries];
    const nextImages = temporalEvents.map((_, index) => index === 0 ? queryImageRef.current : temporalImages[index] ?? null);
    [nextEvents[from], nextEvents[to]] = [nextEvents[to], nextEvents[from]];
    [nextModes[from], nextModes[to]] = [nextModes[to] ?? "text", nextModes[from] ?? "text"];
    [nextObjects[from], nextObjects[to]] = [nextObjects[to] ?? emptyObjectQuery(), nextObjects[from] ?? emptyObjectQuery()];
    [nextImages[from], nextImages[to]] = [nextImages[to], nextImages[from]];
    queryImageRef.current = nextImages[0] ?? null;
    setQueryImage(nextImages[0] ?? null);
    setTemporalImages(nextImages.map((image, index) => index === 0 ? null : image));
    setTemporalEvents(nextEvents);
    setTemporalModes(nextModes);
    setObjectQueries(nextObjects);
  }

  function removeTemporalEvent(index: number) {
    const nextEvents = temporalEvents.filter((_, eventIndex) => eventIndex !== index);
    if (index === 0) {
      replaceQueryImage(temporalImages[1] ?? null);
      setTemporalImages((current) => {
        const next = [null, ...current.slice(2)];
        while (next.length < 2) next.push(null);
        return next;
      });
    } else {
      setTemporalImages((current) => {
        const next = [...current];
        if (next[index]?.kind === "upload") URL.revokeObjectURL(next[index]!.preview);
        next.splice(index, 1);
        while (next.length < 2) next.push(null);
        return next;
      });
    }
    setTemporalEvents(nextEvents.length < 2 ? [nextEvents[0] ?? "", ""] : nextEvents);
    setMode(nextEvents.length < 2 ? "hybrid" : "temporal");
    setTemporalModes((current) => {
      const next = current.filter((_, eventIndex) => eventIndex !== index);
      while (next.length < 2) next.push("text");
      return next;
    });
    setObjectQueries((current) => {
      const next = current.filter((_, eventIndex) => eventIndex !== index);
      while (next.length < 2) next.push(emptyObjectQuery());
      return next;
    });
  }

  function clearSearchSession() {
    requestRef.current?.controller.abort();
    pageRequestRef.current?.abort();
    refineRequestRef.current?.abort();
    replaceQueryImage(null);
    temporalImages.forEach((image, index) => { if (index > 0 && image?.kind === "upload") URL.revokeObjectURL(image.preview); });
    setTemporalImages([null, null]);
    setVisualQuery(""); setAudioQuery(""); setOcrQuery(""); setTemporalEvents(["", ""]);
    setObjectQueries([emptyObjectQuery(), emptyObjectQuery()]);
    setResponse(null); setTemporalResponse(null); setSelected(null); setOpenResult(null);
    setError(""); setMoreError(""); setCompareResults([]); setLoading(false); setLoadingMore(false);
    setRefinement(null); setRefineError(""); setRefineLoading(false);
    checkRequestRef.current?.abort();
    setCheckState({ status: "idle", result: null, error: null, asked: null });
    setCheckQuery(null);
    forgetChecks();
  }

  async function runRefine() {
    const text = (temporalEvents[0] ?? "").trim();
    if (!text) return;
    refineRequestRef.current?.abort();
    const controller = new AbortController();
    refineRequestRef.current = controller;
    setRefineLoading(true);
    setRefineError("");
    try {
      const data = await refineQuery({ text, kind: refineKind }, controller.signal);
      if (refineRequestRef.current !== controller) return;
      setRefinement(data);
    } catch (refineErr) {
      if (refineRequestRef.current !== controller || (refineErr instanceof DOMException && refineErr.name === "AbortError")) return;
      setRefineError(refineErr instanceof Error ? refineErr.message : "Refine failed");
    } finally {
      if (refineRequestRef.current === controller) setRefineLoading(false);
    }
  }

  function dismissRefinement() {
    refineRequestRef.current?.abort();
    setRefinement(null);
    setRefineError("");
  }

  /** What "Check by AI" is checking against: the query that produced the current results. */
  function checkRequestFor(target: { video_id: string; frame_idx?: number; time?: number },
                           force = false): CheckRequest | null {
    if (!checkQuery || !(checkQuery.visualText || checkQuery.audioText || checkQuery.ocrText)) return null;
    return {
      visual_text: checkQuery.visualText || undefined, audio_text: checkQuery.audioText || undefined,
      ocr_text: checkQuery.ocrText || undefined, kind: checkQuery.kind, ...target, profile, force,
    };
  }

  /**
   * Check one moment, cache first.
   *
   * An answer already held for this frame's shot is shown without a request at all -- that is
   * the whole point of the server keying a check by its shot -- so stepping along the filmstrip
   * costs nothing after the first press. Only `force` re-reads one exact frame.
   */
  async function runCheckFor(target: { video_id: string; frame_idx?: number; time?: number },
                             force = false) {
    const request = checkRequestFor(target, force);
    if (!request) {
      setCheckState({ status: "error", result: null, asked: target,
        error: "Type a query first — the check is against what you are searching for." });
      return;
    }
    if (!force) {
      const held = cachedCheck(request);
      if (held) {
        setCheckState({ status: "done", result: { ...held, cached: true }, error: null, asked: target });
        return;
      }
    }
    checkRequestRef.current?.abort();
    const controller = new AbortController();
    checkRequestRef.current = controller;
    setCheckState({ status: "loading", result: null, error: null, asked: target });
    try {
      const result = await runCheck(request, controller.signal);
      if (checkRequestRef.current !== controller) return;
      setCheckState({ status: "done", result, error: null, asked: target });
    } catch (checkErr) {
      if (checkRequestRef.current !== controller
          || (checkErr instanceof DOMException && checkErr.name === "AbortError")) return;
      setCheckState({ status: "error", result: null, asked: target,
        error: checkErr instanceof Error ? checkErr.message : "The AI check failed" });
    }
  }

  // Selecting another frame shows whatever is already known about it -- the shot's answer from
  // this browser, or the server's if another teammate (or an earlier session) paid for it --
  // and otherwise clears the panel rather than leaving the previous frame's verdict up.
  useEffect(() => {
    if (!selected || !checkConfig?.available) {
      setCheckState({ status: "idle", result: null, error: null, asked: null });
      return;
    }
    const target = { video_id: selected.video_id, frame_idx: selected.frame_idx };
    const request = checkRequestFor(target);
    if (!request) {
      setCheckState({ status: "idle", result: null, error: null, asked: null });
      return;
    }
    const held = cachedCheck(request);
    if (held) {
      setCheckState({ status: "done", result: { ...held, cached: true }, error: null, asked: target });
      return;
    }
    setCheckState({ status: "idle", result: null, error: null, asked: null });
    const controller = new AbortController();
    lookupCheck(request, controller.signal).then((found) => {
      if (controller.signal.aborted || found.status !== "hit" || !found.result) return;
      setCheckState({ status: "done", result: { ...found.result, cached: true }, error: null, asked: target });
    }).catch(() => undefined);
    return () => controller.abort();
    // `checkQuery` is in here on purpose: a check belongs to the query that produced the
    // results, so a new search (which replaces it) must drop the verdict rather than leave a
    // stale one up -- editing the query box without re-searching must NOT do that.
  }, [selected?.video_id, selected?.frame_idx, checkConfig?.available, checkQuery, profile]);

  function applyRefinedVisual(text: string) {
    setTemporalModes((modes) => ["text", ...modes.slice(1)]);
    setTemporalEvents((events) => [text, ...events.slice(1)]);
    setVisualQuery(text);
  }

  function applyRefinedTranscript(text: string) {
    setTemporalModes((modes) => ["audio", ...modes.slice(1)]);
    setTemporalEvents((events) => [text, ...events.slice(1)]);
    setAudioQuery(text);
  }

  function applyRefinedOcr(text: string) {
    setTemporalModes((modes) => ["ocr", ...modes.slice(1)]);
    setTemporalEvents((events) => [text, ...events.slice(1)]);
    setOcrQuery(text);
  }

  function applyRefinedEvents(events: string[]) {
    if (events.length < 2) return;
    setTemporalEvents(events);
    setTemporalModes(events.map(() => "text" as EventMode));
    setObjectQueries(events.map(() => emptyObjectQuery()));
    setTemporalImages(events.map(() => null));
    setMode("temporal");
  }

  function selectCollabQuery(id: number) {
    setSelectedQueryId(id);
  }

  function switchSearchMode(nextMode: SearchMode) {
    if (nextMode === "temporal") {
      setTemporalEvents((events) => events.length < 2
        ? [visualQuery || events[0] || "", ""]
        : [visualQuery || events[0] || "", ...events.slice(1)]);
      setTemporalModes((modes) => [queryImage ? "image" : modes[0] ?? "text", modes[1] ?? "text", ...modes.slice(2)]);
    }
    else if (mode === "temporal") setVisualQuery(temporalEvents[0] || "");
    setMode(nextMode);
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
    const firstText = (temporalEvents[0] ?? "").trim();
    const firstModality = temporalModes[0] ?? "text";
    const visualText = firstModality === "text" ? firstText : "";
    const audioText = firstModality === "audio" ? firstText : "";
    const ocrText = firstModality === "ocr" ? firstText : "";
    const usesImage = firstModality === "image" ? activeImage : null;
    const effectiveMode: SearchMode = requestedMode === "temporal" ? "temporal"
      : firstModality === "audio" ? "audio" : firstModality === "ocr" ? "ocr"
      : firstModality === "object" ? "object" : "visual";
    if (requestedMode === "temporal" && temporalEvents.some((item, index) => temporalModes[index] === "image"
      ? !(index === 0 ? activeImage : temporalImages[index])
      : temporalModes[index] === "object" ? !objectQueries[index]?.clauses.length : !item.trim())) {
      setError("Complete every temporal event before searching.");
      return;
    }
    if ((effectiveMode === "visual" && !visualText && !usesImage) || (effectiveMode === "audio" && !audioText)
        || (effectiveMode === "ocr" && !ocrText) || (effectiveMode === "object" && !objectQueries[0].clauses.length)) {
      setError("Complete the query in event E1.");
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
        const events = temporalEvents.map((item, index) => {
          const modality = temporalModes[index] ?? "text";
          const image = index === 0 ? activeImage : temporalImages[index];
          const object_filter = modality !== "object" && objectQueries[index]?.clauses.length ? objectQueries[index] : undefined;
          if (modality === "image" && image) return image.kind === "upload"
            ? { image: image.file, object_filter }
            : { reference: { video_id: image.result.video_id, frame_idx: image.result.frame_idx }, object_filter };
          if (modality === "audio") return { audio: item.trim(), object_filter };
          if (modality === "ocr") return { ocr: item.trim(), object_filter };
          if (modality === "object") return { object: objectQueries[index] };
          return { text: item.trim(), object_filter };
        });
        if (events.some((event) => ("text" in event && !event.text) || ("audio" in event && !event.audio)
          || ("ocr" in event && !event.ocr) || ("image" in event && !event.image)
          || ("object" in event && !event.object?.clauses.length))) {
          setError("Complete every temporal event before searching.");
          setLoading(false);
          return;
        }
        const data = await searchTemporalMultimodal({
          events,
          audio_text: undefined,
          audio_method: audioMethod,
          pe_weight: peWeight / 100,
          max_gap_seconds: temporalMaxGap,
          profile: profilesReady ? profile : undefined,
        }, controller.signal);
        if (requestRef.current?.id !== id) return;
        setTemporalResponse(data);
        const best = data.videos.flatMap((video) => video.sequences).sort((a, b) => a.rank - b.rank)[0];
        setSelected(best?.events[0] ? temporalSelection(best.events[0], best) : null);
        // Group the readable events (image/object events carry no text a check could use) by
        // channel, exactly as a real search would keep visual/audio/ocr separate -- that is
        // what these results were actually searched with.
        const byModality = (wanted: EventMode) => temporalEvents
          .filter((_, eventIndex) => (temporalModes[eventIndex] ?? "text") === wanted)
          .map((item) => item.trim()).filter(Boolean).join(". ");
        const temporalVisual = byModality("text"), temporalAudio = byModality("audio"), temporalOcr = byModality("ocr");
        setCheckQuery(temporalVisual || temporalAudio || temporalOcr
          ? { visualText: temporalVisual, audioText: temporalAudio, ocrText: temporalOcr, kind: refineKind }
          : null);
        return;
      }
      const request = {
        mode: effectiveMode as ClassicSearchMode, index: effectiveMode === "object" ? "flat" as const : index,
        nprobe, ef, profile: profilesReady ? profile : undefined, audio_method: audioMethod, pe_weight: peWeight / 100,
        visual_text: visualText || undefined,
        audio_text: audioText || undefined,
        ocr_text: ocrText || undefined,
        object_query: objectQueries[0].clauses.length ? objectQueries[0] : undefined,
      };
      const data = usesImage
        ? await searchVideosByImage({
            ...request,
            mode: "visual",
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
      setCheckQuery(visualText || audioText || ocrText
        ? { visualText, audioText, ocrText, kind: refineKind } : null);
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
      setCheckConfig(session.check ?? null);
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
    temporalImagesRef.current.slice(1).forEach((image) => {
      if (image?.kind === "upload") URL.revokeObjectURL(image.preview);
    });
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
        visualQuery={visualQuery}
        ocrQuery={ocrQuery}
        mode={mode}
        temporalEvents={temporalEvents}
        temporalMaxGap={temporalMaxGap}
        temporalImages={temporalImages}
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
        onTemporalEvents={setTemporalEvents}
        onTemporalMaxGap={setTemporalMaxGap}
        onTemporalImage={setTemporalImage}
        onClearTemporalImage={clearTemporalImage}
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
        composer={<SearchComposer mode={mode} visualQuery={visualQuery} audioQuery={audioQuery}
          ocrQuery={ocrQuery} queryImage={queryImage} temporalEvents={temporalEvents} temporalMaxGap={temporalMaxGap}
          temporalImages={temporalImages}
          temporalModes={temporalModes}
          objectQueries={objectQueries}
          audioMethod={audioMethod} peWeight={peWeight} loading={loading}
          onVisualQuery={setVisualQuery} onAudioQuery={setAudioQuery} onOcrQuery={setOcrQuery}
          onTemporalEvents={setTemporalEvents} onTemporalMaxGap={setTemporalMaxGap}
          onTemporalImage={setTemporalImage} onClearTemporalImage={clearTemporalImage}
          onTemporalMode={setTemporalMode} onQueryImage={useUploadedImage}
          onObjectQuery={setObjectQuery}
          onClearQueryImage={() => replaceQueryImage(null)} onRemoveTemporalEvent={removeTemporalEvent}
          onMoveTemporalEvent={moveTemporalEvent}
          onMode={switchSearchMode}
          onAudioMethod={setAudioMethod} onPeWeight={setPeWeight} onSubmit={runSearch}
          refinePanel={<RefinePanel text={temporalEvents[0] ?? ""} kind={refineKind} loading={refineLoading}
            error={refineError} refinement={refinement} onKind={setRefineKind} onRefine={() => void runRefine()}
            onDismiss={dismissRefinement} onApplyVisual={applyRefinedVisual} onApplyTranscript={applyRefinedTranscript}
            onApplyOcr={applyRefinedOcr} onApplyEvents={applyRefinedEvents} />} />}
      />
      <section className="search-main">
        {mode === "temporal" ? <TemporalWorkspace
        response={temporalResponse} loading={loading} loadingMore={loadingMore}
        error={sessionError || error} moreError={moreError} onLoadMore={loadNextPage}
        viewMode={temporalViewMode} onViewMode={setTemporalViewMode}
        onOpen={(event: TemporalEventHit, sequence: TemporalSequence) => {
          const result = temporalSelection(event, sequence); setSelected(result); setOpenResult(result);
        }}
        onDresSequence={dresKind === "trake" ? (sequence) => {
          const frames = sequence.events.map((event) => event.frame_idx);
          if (!frames.length || sequence.events.some((event) => event.video_id !== sequence.video_id)
              || new Set(frames).size !== frames.length) {
            setDresPickError("This temporal sequence does not have distinct frames from one video.");
            return;
          }
          setDresPickError("");
          setDresCandidate({ video_id: sequence.video_id, frame_ids: frames });
        } : undefined}
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
        dresKind={dresNavKind}
        dresAnswer={dresQaAnswer}
        trakeCandidate={dresCandidate}
        trakeMessage={trakeMessage}
        onDresAnswer={setDresQaAnswer}
        onSubmitDresFrame={(result, expectedMs) => submitDresCandidate({ video_id: result.video_id, frame_idx: result.frame_idx,
          ...(dresNavKind === "qa" ? { answer: dresQaAnswer } : {}) }, expectedMs)}
        onAddTrakeFrame={addSearchResultTrake}
        onOpenTrakeFrame={(videoId, frameId) => { void openDresFrame(videoId, frameId); }}
        onMoveTrakeFrame={moveTrakeFrame}
        onRemoveTrakeFrame={removeTrakeFrame}
        onSubmitTrake={() => { void submitTrakeSequence(); }}
        check={checkState}
        checkAvailable={checkConfig?.available ?? false}
        onCheck={() => selected && void runCheckFor({ video_id: selected.video_id, frame_idx: selected.frame_idx })}
      />}
      </section>
      <TranscriptPanel selected={selected} check={checkState} keywords={checkKeywords}
        onRecheck={() => selected && void runCheckFor(
          { video_id: selected.video_id, frame_idx: selected.frame_idx }, true)} />
      <DresPanel selected={selected}
          candidate={dresCandidate} onCandidate={setDresCandidate} onPick={pickDresFrame}
          onKind={(kind) => { if (kind) void selectDresKind(kind); }} selectedKind={dresKind} externalStatus={dresStatus}
          isAdmin={isAdmin} pickError={dresPickError} onStatusChange={setDresStatus}
          settingsOpen={dresSettingsOpen} onSettingsClose={() => setDresSettingsOpen(false)} onOpenFrame={openDresFrame} showSubmission={false} />
  </section>;

  const tabs = <AppTabs tab={tab} onTab={setTab} />;
  if (tab === "clips") {
    return (
      <div className="app-shell clip-mode">
        <header className="team-toolbar">
          <div className="team-brand"><span className="eyebrow">AIC HCMC</span><strong>Clip export</strong></div>
          {tabs}
          {dresNavControls}
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
        {dresNavControls}
      </header>
      {searchUi}
      <section className="queries-page" hidden={tab !== "queries"}>
        {login && <TeamToolbar rounds={rounds} round={activeRound} isAdmin={isAdmin}
          onSelectRound={selectRound} onRoundLoaded={loadRound} onError={setError} />}
        {activeRound && login ? <CollaborationPanels round={activeRound} login={login} selectedId={selectedQueryId}
          isAdmin={isAdmin}
          selectedResult={selected} openResult={openResult} onCloseVideo={() => setOpenResult(null)}
          onDresPick={dresStatus?.context_valid && dresKind ? pickDresFrame : undefined}
          onDresSubmit={submitPlayerFrame}
          dresKind={dresStatus?.context_valid ? dresStatus.context?.kind ?? null : dresKind}
          trakeSequence={trakeSequence}
          searchResults={response?.results ?? []}
          onSelect={selectCollabQuery} onQueryChange={updateCollabQuery} />
          : <div className="empty-state query-empty"><h3>No query round yet</h3><p>Upload a round to start assigning and managing answers.</p></div>}
      </section>
      <section className="agent-page" hidden={tab !== "agent"}>
        <AgentPanel />
      </section>
      {openResult && !(activeRound && login) && <VideoDialog result={openResult} onClose={() => setOpenResult(null)} startPaused
        originalVideo profile={profile}
        check={checkConfig?.available
          ? { state: checkState, keywords: checkKeywords,
              run: (target, force) => void runCheckFor(target, force) }
          : undefined}
        picker={{ actions: [],
          submit: { label: dresKind === "trake" ? `Add TRAKE event ${(dresCandidate?.frame_ids?.length ?? 0) + 1}`
            : dresKind === "qa" ? "Submit Q&A frame" : "Submit selected frame",
            kind: dresKind, needsAnswer: dresKind === "qa", onSubmit: submitPlayerFrame },
          trakeSequence }} />}
    </div>
  );
}
