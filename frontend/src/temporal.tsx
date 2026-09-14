import { useEffect, useMemo, useRef } from "react";

import { formatTimestamp, getRelativeScoreTier } from "./results";
import { LoadingOverlay } from "./loadingOverlay";
import { QueryBox } from "./queryBox";
import type { AudioMethod, SearchResult, TemporalEventHit, TemporalSearchResponse, TemporalSequence } from "./types";

interface TemporalPrimaryFieldsProps {
  events: string[];
  audioQuery: string;
  audioMethod: AudioMethod;
  peWeight: number;
  onEvents(events: string[]): void;
  onAudioQuery(value: string): void;
  onAudioMethod(value: AudioMethod): void;
  onPeWeight(value: number): void;
}

export function TemporalPrimaryFields(props: TemporalPrimaryFieldsProps) {
  return <div className="temporal-primary">
    <QueryBox
      label="Visual event E1 (English)"
      ariaLabel="Visual event E1"
      value={props.events[0] ?? ""}
      onChange={(value) => props.onEvents(props.events.map((event, index) => index === 0 ? value : event))}
      maxLength={2000}
      footer={<>
        <span>PE-Core influence</span>
        <input type="range" min="0" max="100" aria-label="PE-Core influence" value={props.peWeight}
          onChange={(event) => props.onPeWeight(Number(event.target.value))} />
        <strong>{props.peWeight}%</strong>
      </>}
    />
    <QueryBox
      label="Transcript (Vietnamese) · optional"
      ariaLabel="Transcript query (Vietnamese)"
      value={props.audioQuery}
      onChange={props.onAudioQuery}
      maxLength={2000}
      footer={<>
        <span>Audio search</span>
        <select aria-label="Audio search" value={props.audioMethod} onChange={(event) => props.onAudioMethod(event.target.value as AudioMethod)}>
          <option value="semantic">Semantic</option>
          <option value="keyword">Keyword</option>
        </select>
      </>}
    />
  </div>;
}

interface TemporalSidebarFieldsProps {
  events: string[];
  maxGap: number;
  onEvents(events: string[]): void;
  onMaxGap(value: number): void;
}

export function TemporalSidebarFields(props: TemporalSidebarFieldsProps) {
  function update(index: number, value: string) {
    props.onEvents(props.events.map((event, current) => current === index ? value : event));
  }
  function move(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= props.events.length) return;
    const events = [...props.events];
    [events[index], events[target]] = [events[target], events[index]];
    props.onEvents(events);
  }
  return <section className="temporal-sidebar" aria-label="Temporal event sequence">
    <header><div><span className="eyebrow">Next event</span><strong>E2–E{props.events.length}</strong></div>
      <span>{props.events.length} events</span></header>
    <div className="temporal-sidebar-list">
      {props.events.slice(1).map((event, offset) => {
        const index = offset + 1;
        return <div className="temporal-event-field" key={index}>
        <label className="field query-field">
          <span><b>E{index + 1}</b> Visual query (English)</span>
          <textarea aria-label={`Visual event E${index + 1}`} value={event}
            onChange={(change) => update(index, change.target.value)} maxLength={2000} rows={2} />
        </label>
        <div className="event-actions">
          <button type="button" aria-label={`Move event E${index + 1} up`} disabled={index === 1}
            onClick={() => move(index, -1)}>↑</button>
          <button type="button" aria-label={`Move event E${index + 1} down`} disabled={index === props.events.length - 1}
            onClick={() => move(index, 1)}>↓</button>
          <button type="button" aria-label={`Remove event E${index + 1}`} disabled={props.events.length <= 2}
            onClick={() => props.onEvents(props.events.filter((_, current) => current !== index))}>×</button>
        </div>
      </div>})}
      <button className="add-event" type="button" aria-label="Add event" disabled={props.events.length >= 6}
        onClick={() => props.onEvents([...props.events, ""])}>＋ Add event</button>
    </div>
    <label className="field temporal-sidebar-gap">
      <span>Max gap between events · seconds</span>
      <input aria-label="Temporal max gap seconds" type="number" min="1" max="600" step="1"
        value={props.maxGap} onChange={(event) => props.onMaxGap(Number(event.target.value))} />
    </label>
  </section>;
}

export function temporalSelection(event: TemporalEventHit, sequence: TemporalSequence): SearchResult {
  const evidence = sequence.audio_evidence;
  return {
    ...event,
    rank: sequence.rank,
    source: sequence.source,
    score: sequence.score,
    transcript_snippet: evidence?.transcript_snippet,
    transcript_start: evidence?.transcript_start,
    transcript_end: evidence?.transcript_end,
    scores: {
      ...event.scores,
      visual_score: event.score,
      sequence_visual_score: sequence.visual_score,
      audio_score: evidence?.score,
      audio_rank: evidence?.audio_rank,
      bm25_rank: evidence?.bm25_rank,
      vector_rank: evidence?.vector_rank,
    },
  };
}

export type ResultsViewMode = "grouped" | "ranked";

interface TemporalWorkspaceProps {
  response: TemporalSearchResponse | null;
  loading: boolean;
  loadingMore: boolean;
  error: string;
  moreError: string;
  viewMode: ResultsViewMode;
  onViewMode(mode: ResultsViewMode): void;
  onOpen(event: TemporalEventHit, sequence: TemporalSequence): void;
  onLoadMore(): void;
}

export function TemporalWorkspace(props: TemporalWorkspaceProps) {
  const workspaceRef = useRef<HTMLElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const sequences = useMemo(() => props.response?.videos.flatMap((video) => video.sequences) ?? [], [props.response]);
  const rankedSequences = useMemo(() => [...sequences].sort((a, b) => a.rank - b.rank), [sequences]);
  const scores = sequences.map((sequence) => sequence.score);
  const min = scores.length ? Math.min(...scores) : 0;
  const max = scores.length ? Math.max(...scores) : 0;

  function renderSequence(sequence: TemporalSequence, showVideoId: boolean) {
    const tier = getRelativeScoreTier(sequence.score, min, max);
    return <article className={`temporal-sequence tier-${tier}`} data-testid="temporal-sequence" key={`${sequence.video_id}-${sequence.rank}`}>
      <header>
        <strong>{showVideoId && <span className="row-rank">{String(sequence.rank + 1).padStart(2, "0")}</span>}
          {showVideoId ? sequence.video_id : `${formatTimestamp(sequence.start_time)}–${formatTimestamp(sequence.end_time)}`}</strong>
        <span>{showVideoId && <>{formatTimestamp(sequence.start_time)}–{formatTimestamp(sequence.end_time)} · </>}
          <b>{sequence.score.toFixed(4)}</b>{sequence.source === "both" && <> · visual {sequence.visual_score.toFixed(4)}</>}</span>
      </header>
      <div className="temporal-timeline">{sequence.events.map((event, index) => <div className="temporal-event" key={`${event.frame_idx}-${index}`}>
        {index > 0 && <span className="temporal-arrow" aria-hidden="true">→</span>}
        <button type="button" aria-label={`Open event E${index + 1} ${event.video_id} at ${formatTimestamp(event.time)}`}
          onClick={() => props.onOpen(event, sequence)}>
          <img src={event.thumb} alt="" loading="lazy" />
          <span><b>E{index + 1}</b><time>{formatTimestamp(event.time)}</time><em>{event.score.toFixed(4)}</em></span>
        </button>
      </div>)}</div>
    </article>;
  }

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !props.response?.has_more || props.loadingMore || props.moreError) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries.some((entry) => entry.isIntersecting)) props.onLoadMore(); },
      { root: workspaceRef.current, rootMargin: "0px 0px 400px 0px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [props.response?.has_more, props.loadingMore, props.moreError, props.onLoadMore]);

  return <main ref={workspaceRef} className="results-workspace temporal-workspace" aria-label="Temporal search results">
    {props.error && <div className="message error" role="alert">{props.error}</div>}
    {props.response?.warning && <div className="message warning">{props.response.warning}</div>}
    {!props.response && !props.loading && <div className="empty-state"><span>⇢</span><h3>Find an ordered visual sequence</h3><p>Describe two or more events in chronological order.</p></div>}
    {props.response && !props.loading && !props.response.videos.length && <div className="empty-state"><span>∅</span><h3>No sequences found</h3><p>Broaden an event or increase the maximum gap.</p></div>}
    <div className={`results-body${props.loading ? " is-loading" : ""}`}>
      <LoadingOverlay active={props.loading} sticky />
      {props.viewMode === "grouped" ? (
        <div className="temporal-videos">{props.response?.videos.map((video, videoIndex) =>
          <section className="temporal-video" key={video.video_id} data-testid="temporal-video">
            <header><div><span className="row-rank">{String(videoIndex + 1).padStart(2, "0")}</span><h3>{video.video_id}</h3></div>
              <div className="row-stats"><span>{video.sequences.length} sequence{video.sequences.length === 1 ? "" : "s"}</span><strong>best {video.best_score.toFixed(4)}</strong></div></header>
            <div className="temporal-sequences">{video.sequences.map((sequence) => renderSequence(sequence, false))}</div>
          </section>)}</div>
      ) : (
        <div className="temporal-sequences temporal-sequences-flat">{rankedSequences.map((sequence) => renderSequence(sequence, true))}</div>
      )}
    </div>
    {props.response && props.response.videos.length > 0 && <div ref={sentinelRef} className="load-more-sentinel">
      {props.loadingMore && <span className="loading-indicator" role="status">Loading more</span>}
      {props.moreError && <><span role="alert">{props.moreError}</span><button type="button" onClick={props.onLoadMore}>Retry</button></>}
      {!props.response.has_more && !props.loadingMore && !props.moreError && <span>{props.response.truncated ? "Reached the 1,000-sequence search depth." : "All results loaded."}</span>}
    </div>}
  </main>;
}
