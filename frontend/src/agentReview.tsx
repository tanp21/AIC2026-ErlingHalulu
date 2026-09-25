import { Fragment, useEffect, useMemo, useState } from "react";

import { fetchKeyframeContext } from "./api";
import { captionsAround, loadVideoEvidence, ocrAround, speechAround } from "./agentEvidence";
import type { VideoEvidence } from "./agentEvidence";
import { Lit, STATUS_MARK } from "./checkView";
import { highlight } from "./highlight";
import { formatTimestamp } from "./results";
import { VideoDialog } from "./videoDialog";
import type {
  AgentCheckStatus, AgentLiveCandidate, AgentLiveSequence, AgentProgress, AgentReviewRequest,
  AgentReviewState, AgentRunResponse, AgentSummary, CaptionWindow, SearchResult,
} from "./types";

const CHANNEL_LABELS: Record<string, string> = {
  visual: "V", similar: "S", audio: "A", ocr: "O", ocr_exact: "O", caption: "C", temporal: "T",
};
const STAGE_LABELS: Record<string, string> = {
  planning: "Planning the searches", searching: "Searching", rewriting: "Rewriting the searches",
  awaiting_review: "Waiting for your review", finalizing: "Finishing", done: "Done",
};

type Judgement = { kind: "accept" | "reject" | "score" | null; score?: number };
type Frame = { video_id: string; frame_idx: number; time: number };
type Slot = { frame_idx: number; time: number } | null;

function judgementOf(review: AgentReviewState, id: string): Judgement {
  if (review.decisions[id]) return { kind: review.decisions[id] };
  const score = review.scores?.[id];
  return score === undefined ? { kind: null } : { kind: "score", score };
}

function asSearchResult(frame: Frame): SearchResult {
  return {
    rank: 0, score: 0, video_id: frame.video_id, frame_idx: frame.frame_idx, time: frame.time,
    thumb: `/thumb/${frame.video_id}/${frame.frame_idx}?w=320`, frame: `/frame/${frame.video_id}/${frame.frame_idx}`,
    video: `/video/${frame.video_id}`, source: "visual", scores: {},
  };
}

/** How many of the query's details the words around a candidate say in full, if they were read. */
function supportedCount(summary: AgentSummary | null | undefined): number {
  return summary ? new Set(summary.checks.filter((check) => check.status === "supported").map((check) => check.detail)).size : 0;
}

/** How many details the words around a candidate say differently from the query. */
function conflictCount(summary: AgentSummary | null | undefined): number {
  return summary ? new Set(summary.checks.filter((check) => check.status === "contradicted").map((check) => check.detail)).size : 0;
}

/** How many more they say in part: right subject, but a count, a kind or an action is missing. */
function partialCount(summary: AgentSummary | null | undefined): number {
  return summary ? new Set(summary.checks.filter((check) => check.status === "partial").map((check) => check.detail)).size : 0;
}

/**
 * The workspace of a run whose judge is the person using it: no model looks at a frame, so
 * everything a model would have been told is here to be read instead -- the query broken
 * into details, the words said and written around each candidate with the query's words lit
 * up, the channels that found it -- and everything a model would have decided is a button.
 */
export function AgentReviewWorkspace({ run, onReview }: {
  run: AgentRunResponse;
  onReview(request: AgentReviewRequest): void;
}) {
  const progress: AgentProgress | null | undefined = run.progress;
  const review: AgentReviewState = run.review ?? { decisions: {}, answer: null, stopped: false, decided: false };
  const candidates = progress?.candidates ?? [];
  const keywords = progress?.plan?.keywords ?? { vi: [], en: [] };
  const events = progress?.plan?.events ?? [];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hint, setHint] = useState("");
  const [answer, setAnswer] = useState("");
  const [hideRejected, setHideRejected] = useState(false);
  const [onlyMatching, setOnlyMatching] = useState(false);
  const [groupByVideo, setGroupByVideo] = useState(false);
  const [order, setOrder] = useState<"evidence" | "retrieval">("evidence");
  const [dialog, setDialog] = useState<Frame | null>(null);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [slotVideo, setSlotVideo] = useState<string | null>(null);

  const busy = progress ? progress.stage !== "awaiting_review" : true;
  const eventCount = run.kind === "trake" ? Math.max(events.length, review.sequence?.frames.length ?? 0, 2) : 0;

  // The server's copy of the path wins whenever it changes, so a second tab or a reload agrees.
  const savedPath = JSON.stringify(review.sequence ?? null);
  useEffect(() => {
    if (!review.sequence) return;
    setSlotVideo(review.sequence.video_id);
    setSlots(review.sequence.frames.map((frame, index) => ({ frame_idx: frame, time: review.sequence?.times?.[index] ?? 0 })));
  }, [savedPath]);

  const detailCount = progress?.details?.length ?? 0;
  const common = progress?.common_details ?? [];
  const reading = Boolean(progress?.summarizing);

  // The list arrives in EVIDENCE order (agent/orchestrator.py `_by_evidence`): the searches only find
  // candidates and any of them can be wrong, so what leads is what the words around a candidate and
  // in its video support, then candidates the words say nothing about. Accepted ones lead and
  // rejected ones trail. "Retrieval order" puts the middle group back in the order the
  // searches ranked it, to compare.
  const ordered = useMemo(() => {
    if (order === "evidence") return candidates;
    const group = (candidate: AgentLiveCandidate) => {
      const judged = judgementOf(review, candidate.id);
      const value = judged.kind === "accept" ? 10 : judged.kind === "reject" ? 0 : judged.kind === "score" ? judged.score ?? 5 : null;
      return value === null || (value > 3 && value < 6) ? 1 : value >= 6 ? 0 : 2;
    };
    return candidates
      .map((candidate, position) => ({ candidate, position, group: group(candidate) }))
      .sort((a, b) => a.group - b.group || (a.group === 1
        ? (a.candidate.agent_rank ?? a.position) - (b.candidate.agent_rank ?? b.position) : a.position - b.position))
      .map((entry) => entry.candidate);
  }, [candidates, review, order]);

  const visible = useMemo(() => {
    let shown = ordered.filter((candidate) => {
      const judged = judgementOf(review, candidate.id);
      if (hideRejected && (judged.kind === "reject" || (judged.kind === "score" && (judged.score ?? 5) <= 3))) return false;
      if (onlyMatching && supportedCount(candidate.summary) === 0) return false;
      return true;
    });
    if (groupByVideo) {
      const order = new Map<string, number>();
      shown.forEach((candidate) => { if (!order.has(candidate.video_id)) order.set(candidate.video_id, order.size); });
      shown = [...shown].sort((a, b) => (order.get(a.video_id) ?? 0) - (order.get(b.video_id) ?? 0));
    }
    return shown;
  }, [ordered, review, hideRejected, onlyMatching, groupByVideo]);

  const selected = candidates.find((candidate) => candidate.id === selectedId) ?? visible[0] ?? null;

  function decide(candidate: AgentLiveCandidate, choice: "accept" | "reject" | "clear" | "score5") {
    const current = judgementOf(review, candidate.id);
    if (choice === "score5") {
      onReview(current.kind === "score" && current.score === 5
        ? { candidate_id: candidate.id, decision: "clear" } : { candidate_id: candidate.id, score: 5 });
    } else if (choice === "clear" || current.kind === choice) {
      onReview({ candidate_id: candidate.id, decision: "clear" });
    } else {
      onReview({ candidate_id: candidate.id, decision: choice });
    }
  }

  function move(step: number) {
    if (!visible.length) return;
    const at = visible.findIndex((candidate) => candidate.id === selected?.id);
    setSelectedId(visible[Math.min(visible.length - 1, Math.max(0, at + step))].id);
  }

  function assign(index: number, frame: Frame) {
    const sameVideo = slotVideo === null || slotVideo === frame.video_id;
    const next: Slot[] = Array.from({ length: eventCount }, (_, slot) => (sameVideo ? slots[slot] : null) ?? null);
    next[index] = { frame_idx: frame.frame_idx, time: frame.time };
    setSlotVideo(frame.video_id);
    setSlots(next);
    if (next.every(Boolean)) {
      onReview({ sequence: {
        video_id: frame.video_id, frames: next.map((slot) => slot!.frame_idx), times: next.map((slot) => slot!.time),
      } });
    }
  }

  function useServerPath(path: AgentLiveSequence) {
    setSlotVideo(path.video_id);
    const next = path.frames.slice(0, eventCount).map((frame, index) => ({ frame_idx: frame, time: path.times?.[index] ?? 0 }));
    setSlots(next);
    if (next.length === eventCount) {
      onReview({ sequence: { video_id: path.video_id, frames: path.frames.slice(0, eventCount), times: next.map((slot) => slot.time) } });
    }
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (dialog || event.metaKey || event.ctrlKey || event.altKey) return;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (!selected) return;
      const key = event.key.toLowerCase();
      const actions: Record<string, () => void> = {
        j: () => move(1), arrowdown: () => move(1), k: () => move(-1), arrowup: () => move(-1),
        a: () => decide(selected, "accept"), r: () => decide(selected, "reject"), "5": () => decide(selected, "score5"),
        x: () => decide(selected, "clear"),
        n: () => onReview({ command: { type: "neighbours", cid: selected.id } }),
        v: () => onReview({ command: { type: "scope", video_id: selected.video_id } }),
        enter: () => setDialog({ video_id: selected.video_id, frame_idx: selected.frame_idx, time: selected.time }),
      };
      const action = actions[key];
      if (!action) return;
      event.preventDefault();
      action();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const accepted = candidates.filter((candidate) => review.decisions[candidate.id] === "accept").length;
  const stage = progress ? STAGE_LABELS[progress.stage] ?? progress.stage : "Starting";
  const suggestions = progress?.suggested_answers ?? [];

  return (
    <section className="review-workspace" aria-label="Human review">
      <div className="review-brief">
        <div className="review-brief-head">
          <span aria-live="polite">
            <b>{stage}</b> · {Math.round(progress?.elapsed ?? run.elapsed)}s
            {" · "}<span className="agent-judge">judged by you (human review)</span>
            {progress?.total_candidates !== undefined && <> · {progress.total_candidates} candidates found, {candidates.length} shown</>}
            {progress?.summarizing && <> · <em>reading the words around each candidate…</em></>}
            {(review.pending_commands ?? 0) > 0 && <> · <em>{review.pending_commands} search request queued</em></>}
          </span>
          <span className="review-brief-actions">
            <button type="button" className="open-video" disabled={review.stopped}
              onClick={() => onReview({ stop: true, order: ordered.map((candidate) => candidate.id) })}>Stop</button>
            <button type="button" className="search-button" disabled={review.finished}
              onClick={() => onReview({ finish: true, order: ordered.map((candidate) => candidate.id) })}>
              {review.finished ? "Submitting…" : `Submit ${accepted ? `(${accepted} accepted)` : "current ranking"}`}
            </button>
          </span>
        </div>
        <p className="review-query">{run.query}</p>
        {progress?.plan && <p className="review-scenes">
          {progress.plan.scenes.map((scene, index) => <span key={index}>{scene}
            {index === progress.plan?.target_scene && progress.plan.scenes.length > 1 && <em> (frame to submit)</em>}</span>)}
        </p>}
        {progress?.plan?.question_vi && <p><b>Question:</b> {progress.plan.question_vi}</p>}
        {(progress?.details?.length ?? 0) > 0 && <ol className="review-details" aria-label="Query details">
          {progress?.details?.map((detail, index) => <li key={index}>{detail}</li>)}
        </ol>}
        {(progress?.lookups?.length ?? 0) > 0 && <details className="review-lookups">
          <summary>Web notes ({progress?.lookups?.length})</summary>
          {progress?.lookups?.map((lookup, index) => <p key={index}>
            <b>{lookup.question}</b> {lookup.answer ?? lookup.error}
          </p>)}
        </details>}
        <form className="review-more" onSubmit={(event) => {
          event.preventDefault();
          onReview({ command: { type: "more", hint: hint.trim() } });
          setHint("");
        }}>
          <input aria-label="Hint for another search round" maxLength={300} value={hint}
            placeholder="Tell the agent what it got wrong, then search again (optional)"
            onChange={(event) => setHint(event.target.value)} />
          <button type="submit" className="open-video" disabled={busy}>Search more</button>
        </form>
        {run.kind === "qa" && <div className="review-answer">
          <form onSubmit={(event) => {
            event.preventDefault();
            if (answer.trim()) onReview({ answer: answer.trim() });
          }}>
            <input aria-label="Reviewer answer" maxLength={100} value={answer}
              placeholder={review.answer ?? "Your answer (exact string)"} onChange={(event) => setAnswer(event.target.value)} />
            <button type="submit" className="open-video" disabled={!answer.trim()}>Use this answer</button>
          </form>
          {review.answer && <span className="agent-answer-set">Using: {review.answer}</span>}
          {suggestions.map((suggestion, index) => <button key={index} type="button" className="review-chip"
            title={suggestion.quote} onClick={() => { setAnswer(suggestion.answer); onReview({ answer: suggestion.answer }); }}>
            {suggestion.answer} <small>{suggestion.source}</small>
          </button>)}
        </div>}
        {run.kind === "trake" && <SequenceBar events={events} count={eventCount} slots={slots} video={slotVideo}
          paths={progress?.sequences ?? []} onUsePath={useServerPath}
          onClear={() => { setSlots([]); setSlotVideo(null); }} />}
      </div>

      <div className="review-filters" role="group" aria-label="Filters">
        <label><input type="checkbox" checked={hideRejected} onChange={(event) => setHideRejected(event.target.checked)} /> Hide rejected</label>
        <label><input type="checkbox" checked={onlyMatching} onChange={(event) => setOnlyMatching(event.target.checked)} /> Only with a matching detail</label>
        <label><input type="checkbox" checked={groupByVideo} onChange={(event) => setGroupByVideo(event.target.checked)} /> Group by video</label>
        <label>Order <select aria-label="Candidate order" value={order} onChange={(event) => setOrder(event.target.value as "evidence" | "retrieval")}>
          <option value="evidence">Best evidence first</option>
          <option value="retrieval">Retrieval order (the searches can be wrong)</option>
        </select>{reading && order === "evidence" && <em className="review-reading"> · ordered by evidence once the checklist is read</em>}</label>
        <span className="review-keys">J/K move · A accept · R reject · 5 wrong moment · X clear · N neighbours · V this video · Enter play</span>
      </div>

      {candidates.length === 0
        ? <div className="message">The agent is still planning and searching…</div>
        : <div className="review-body">
          <div className="review-grid" aria-label="Candidates">
            {visible.map((candidate, index) => {
              const showHeader = groupByVideo && (index === 0 || visible[index - 1].video_id !== candidate.video_id);
              return <Fragment key={candidate.id}>
                {showHeader && <h3 className="review-video-head">{candidate.video_id}</h3>}
                <ReviewCard candidate={candidate} rank={index + 1} agentRank={candidate.agent_rank ?? candidates.indexOf(candidate) + 1}
                  detailCount={detailCount} judgement={judgementOf(review, candidate.id)}
                  selected={candidate.id === selected?.id} onSelect={() => setSelectedId(candidate.id)}
                  onDecide={(choice) => decide(candidate, choice)} />
              </Fragment>;
            })}
            {visible.length === 0 && <div className="message">Every candidate is filtered out.</div>}
          </div>
          {selected && <Inspector key={selected.id} candidate={selected} keywords={keywords}
            details={progress?.details ?? []} common={common} summarizing={Boolean(progress?.summarizing)}
            judgement={judgementOf(review, selected.id)} events={run.kind === "trake" ? eventCount : 0}
            onDecide={(choice) => decide(selected, choice)}
            onCommand={(command) => onReview({ command })}
            onAssign={assign} onOpen={(frame) => setDialog(frame)} />}
        </div>}

      {dialog && <VideoDialog result={asSearchResult(dialog)} initialFrameId={dialog.frame_idx} startPaused
        onClose={() => setDialog(null)}
        picker={{ actions: [
          { label: "Add frame", onPick: (frame, time) => onReview({ command: { type: "add_frame", video_id: dialog.video_id, frame_idx: frame, time } }) },
          { label: "Accept frame", onPick: (frame, time) => onReview({ command: { type: "add_frame", video_id: dialog.video_id, frame_idx: frame, time, accept: true } }) },
          ...Array.from({ length: eventCount }, (_, index) => ({
            label: `Set as E${index + 1}`,
            onPick: (frame: number, time: number) => assign(index, { video_id: dialog.video_id, frame_idx: frame, time }),
          })),
        ] }} />}
    </section>
  );
}

function ReviewCard({ candidate, rank, agentRank, detailCount, judgement, selected, onSelect, onDecide }: {
  candidate: AgentLiveCandidate;
  rank: number;
  agentRank: number;
  detailCount: number;
  judgement: Judgement;
  selected: boolean;
  onSelect(): void;
  onDecide(choice: "accept" | "reject" | "score5"): void;
}) {
  const channels = candidate.channels ?? [];
  const matched = supportedCount(candidate.summary);
  const close = partialCount(candidate.summary);
  const against = conflictCount(candidate.summary);
  const state = judgement.kind === "score" ? "maybe" : judgement.kind ?? "";
  return (
    <article className={`review-card${selected ? " selected" : ""}${state ? ` ${state}` : ""}${candidate.pinned ? " pinned" : ""}`}
      aria-label={`Candidate ${candidate.id}`} aria-current={selected ? "true" : undefined}>
      <button type="button" className="review-card-open" onClick={onSelect}
        aria-label={`Inspect ${candidate.id}: ${candidate.video_id} at ${formatTimestamp(candidate.time)}`}>
        <img src={`/thumb/${candidate.video_id}/${candidate.frame_idx}?w=320`} alt="" loading="lazy" />
        <span className="review-card-meta" title={rank === agentRank ? undefined : `Retrieval alone put this ${agentRank}`}>#{rank} {candidate.id} · {candidate.video_id} · {formatTimestamp(candidate.time)}</span>
      </button>
      <div className="review-badges">
        {channels.slice(0, 5).map((channel) => <span key={channel.source} className={`badge-${channel.kind}`}
          title={`${channel.source} · rank ${channel.rank}`}>{CHANNEL_LABELS[channel.kind] ?? channel.kind[0].toUpperCase()}{channel.rank}</span>)}
        {candidate.pinned && <span className="badge-pinned" title="Added by you">＋</span>}
      </div>
      {detailCount > 0 && candidate.summary && <div className="review-match" title={`${candidate.summary.summary}${candidate.evidence
        ? ` — this moment ${Math.round(100 * candidate.evidence.share)}%, its video ${Math.round(100 * (candidate.evidence.video ?? 0))}%` : ""}`}>
        <span className="review-match-bar"><i style={{ width: `${Math.round(100 * (candidate.evidence?.share ?? 0))}%` }} /></span>
        <small>{matched}/{detailCount} details</small>{close > 0 && <small className="review-close"> · {close} close</small>}
        {against > 0 && <small className="review-against" title="Shown, but it does not lower the candidate: captions get colours and counts wrong"> · {against} conflict</small>}
      </div>}
      {judgement.kind === "score" && <div className="review-state">Wrong moment ({judgement.score}/10)</div>}
      <div className="review-card-buttons">
        <button type="button" aria-pressed={judgement.kind === "accept"} aria-label={`Accept ${candidate.id}`}
          className="agent-accept" onClick={() => onDecide("accept")}>✓</button>
        <button type="button" aria-pressed={judgement.kind === "score"} aria-label={`Right video, wrong moment ${candidate.id}`}
          className="agent-maybe" onClick={() => onDecide("score5")}>~</button>
        <button type="button" aria-pressed={judgement.kind === "reject"} aria-label={`Reject ${candidate.id}`}
          className="agent-reject" onClick={() => onDecide("reject")}>✗</button>
      </div>
    </article>
  );
}

function Inspector({ candidate, keywords, details, common, summarizing, judgement, events, onDecide, onCommand, onAssign, onOpen }: {
  candidate: AgentLiveCandidate;
  keywords: { vi: string[]; en: string[] };
  details: string[];
  common: boolean[];
  summarizing: boolean;
  judgement: Judgement;
  events: number;
  onDecide(choice: "accept" | "reject" | "clear" | "score5"): void;
  onCommand(command: NonNullable<AgentReviewRequest["command"]>): void;
  onAssign(index: number, frame: Frame): void;
  onOpen(frame: Frame): void;
}) {
  const [viewed, setViewed] = useState<Frame>({ video_id: candidate.video_id, frame_idx: candidate.frame_idx, time: candidate.time });
  const [neighbours, setNeighbours] = useState<SearchResult[]>([]);
  const [evidence, setEvidence] = useState<VideoEvidence | null>(null);

  useEffect(() => {
    let active = true;
    setEvidence(null);
    loadVideoEvidence(candidate.video_id).then((loaded) => { if (active) setEvidence(loaded); });
    return () => { active = false; };
  }, [candidate.video_id]);

  useEffect(() => {
    let active = true;
    fetchKeyframeContext(candidate.video_id, candidate.frame_idx, undefined, 5)
      .then((data) => { if (active) setNeighbours(data.frames); })
      .catch(() => { if (active) setNeighbours([]); });
    return () => { active = false; };
  }, [candidate.video_id, candidate.frame_idx]);

  const speech = evidence ? speechAround(evidence.cues, viewed.time) : [];
  const ocr = evidence ? ocrAround(evidence.ocr, viewed.time) : [];
  const captions = evidence ? captionsAround(evidence.captions, viewed.time) : [];
  const both = useMemo(() => [...keywords.vi, ...keywords.en], [keywords]);
  const checks = candidate.summary?.checks ?? [];

  return (
    <aside className="review-inspector" aria-label={`Inspector for ${candidate.id}`}>
      <header>
        <strong>{candidate.id} · {candidate.video_id}</strong>
        <span>{formatTimestamp(viewed.time)} · frame {viewed.frame_idx}</span>
      </header>
      <img className="review-frame" src={`/thumb/${viewed.video_id}/${viewed.frame_idx}?w=640`}
        alt={`Frame ${viewed.frame_idx} of ${viewed.video_id}`} />
      {neighbours.length > 0 && <div className="review-neighbours" aria-label="Neighbouring keyframes">
        {neighbours.map((frame) => <span key={frame.frame_idx} className={frame.frame_idx === viewed.frame_idx ? "active" : ""}>
          <button type="button" aria-label={`View frame ${frame.frame_idx}`}
            onClick={() => setViewed({ video_id: frame.video_id, frame_idx: frame.frame_idx, time: frame.time })}>
            <img src={`/thumb/${frame.video_id}/${frame.frame_idx}?w=160`} alt="" loading="lazy" />
            <small>{formatTimestamp(frame.time)}</small>
          </button>
          {frame.frame_idx !== candidate.frame_idx && <button type="button" className="review-add"
            aria-label={`Add frame ${frame.frame_idx} to candidates`} title="Add to candidates"
            onClick={() => onCommand({ type: "add_frame", video_id: frame.video_id, frame_idx: frame.frame_idx, time: frame.time })}>＋</button>}
        </span>)}
      </div>}
      <div className="review-actions">
        <button type="button" aria-pressed={judgement.kind === "accept"} className="agent-accept" onClick={() => onDecide("accept")}>✓ Accept</button>
        <button type="button" aria-pressed={judgement.kind === "score"} className="agent-maybe" onClick={() => onDecide("score5")}>~ Wrong moment</button>
        <button type="button" aria-pressed={judgement.kind === "reject"} className="agent-reject" onClick={() => onDecide("reject")}>✗ Reject</button>
        <button type="button" onClick={() => onOpen(viewed)}>▶ Play</button>
        <button type="button" onClick={() => onCommand({ type: "neighbours", cid: candidate.id })}>Neighbours</button>
        <button type="button" onClick={() => onCommand({ type: "scope", video_id: candidate.video_id })}>Search this video</button>
      </div>
      {events > 0 && <div className="review-assign" aria-label="Assign to TRAKE event">
        {Array.from({ length: events }, (_, index) => <button key={index} type="button" aria-label={`Set frame ${viewed.frame_idx} as E${index + 1}`}
          onClick={() => onAssign(index, viewed)}>→ E{index + 1}</button>)}
      </div>}

      <section aria-label="Query details checked">
        <h4>Details {summarizing && !candidate.summary && <small>reading…</small>}</h4>
        {details.length === 0
          ? <p className="review-empty">No checklist for this run.</p>
          : <ul className="review-checks">
            {details.map((detail, index) => {
              const found = checks.filter((check) => check.detail === index);
              const status: AgentCheckStatus = (["supported", "partial", "contradicted"] as AgentCheckStatus[])
                .find((wanted) => found.some((check) => check.status === wanted)) ?? "not_mentioned";
              const proof = status === "not_mentioned" ? undefined : found.find((check) => check.status === status && check.quote);
              return <li key={index} className={`check-${status}${common[index] ? " common" : ""}`}>
                <b aria-label={status.replace("_", " ")}>{STATUS_MARK[status]}</b> {detail}
                {common[index] && <small className="review-common" title="Nearly every candidate satisfies this, so it does not tell them apart"> common</small>}
                {proof?.note && <em className="review-note">{proof.note}</em>}
                {proof && <q>{proof.quote}{proof.source && <small> {proof.source}{proof.at != null && ` ${formatTimestamp(proof.at)}`}</small>}</q>}
              </li>;
            })}
          </ul>}
        {candidate.summary?.summary && <p className="review-summary">{candidate.summary.summary}</p>}
      </section>

      <section aria-label="Speech around this moment">
        <h4>Speech</h4>
        {!evidence ? <p className="review-empty">Loading…</p>
          : speech.length === 0 ? <p className="review-empty">{evidence.missing.includes("transcript") ? "No transcript for this video." : "Nothing said nearby."}</p>
            : <ol className="review-speech">{speech.map((cue) => <li key={`${cue.start}-${cue.text}`} className={cue.active ? "active" : ""}>
              <time>{formatTimestamp(cue.start)}</time><span><Lit text={cue.text} keywords={keywords.vi} /></span></li>)}</ol>}
      </section>

      <section aria-label="Text on screen">
        <h4>On screen <small>this shot and the ones around it</small></h4>
        {!evidence ? <p className="review-empty">Loading…</p>
          : ocr.length === 0 ? <p className="review-empty">{evidence.missing.includes("ocr") ? "OCR is not available for this video." : "No text on screen."}</p>
            : <ul className="review-ocr">{ocr.map((shot) => <li key={shot.shot_idx} className={shot.current ? "active" : ""}>
              <time>{formatTimestamp(shot.start)}</time><span>{shot.text.trim()
                ? <Lit text={shot.text} keywords={keywords.vi} /> : <em>no text in this shot</em>}</span></li>)}</ul>}
      </section>

      <section aria-label="Shot captions">
        <h4>Captions <small>shots around this frame, most relevant first · automatic, can be wrong</small></h4>
        {!evidence ? <p className="review-empty">Loading…</p>
          : captions.length === 0 ? <p className="review-empty">{evidence.missing.includes("captions") ? "Captions could not be loaded." : "No caption for this video."}</p>
            : <CaptionList captions={captions} keywords={both} time={viewed.time} />}
      </section>
    </aside>
  );
}

function SequenceBar({ events, count, slots, video, paths, onUsePath, onClear }: {
  events: string[];
  count: number;
  slots: Slot[];
  video: string | null;
  paths: AgentLiveSequence[];
  onUsePath(path: AgentLiveSequence): void;
  onClear(): void;
}) {
  const filled = slots.filter(Boolean) as Array<{ frame_idx: number; time: number }>;
  const ordered = filled.every((slot, index) => index === 0 || slot.time >= filled[index - 1].time);
  return (
    <div className="review-sequence" aria-label="TRAKE path">
      <div className="review-slots">
        {Array.from({ length: count }, (_, index) => {
          const slot = slots[index];
          return <div key={index} className={`review-slot${slot ? " filled" : ""}`}>
            <b>E{index + 1}</b>
            {slot && video ? <img src={`/thumb/${video}/${slot.frame_idx}?w=160`} alt={`E${index + 1}: frame ${slot.frame_idx}`} /> : <span className="review-slot-empty">empty</span>}
            <small>{events[index] ?? ""}</small>
          </div>;
        })}
      </div>
      <div className="review-slot-tools">
        {video && <span>{video}</span>}
        {!ordered && <span className="review-warning">Frames are not in time order</span>}
        {paths.slice(0, 4).map((path, index) => <button key={index} type="button" className="review-chip"
          onClick={() => onUsePath(path)}>Use {path.video_id} {path.frames.join(" → ")}</button>)}
        {slots.length > 0 && <button type="button" className="review-chip" onClick={onClear}>Clear</button>}
      </div>
    </div>
  );
}

/** Sentences of `text`, cut where a full stop is followed by a capital or a quote. */
function sentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+(?=["“A-ZÀ-Ỹ])/).map((sentence) => sentence.trim()).filter(Boolean);
}

/**
 * A caption sentence's relevance: how many different words of the query it holds. One word
 * ("white", "four") is in every intro shot; a sentence with a woman, a white apron and a vase
 * is the one the query describes.
 */
const STRONG = 2;
/** How many shots a reviewer sees opened at once; the rest are one click away. */
const OPEN_WINDOWS = 4;

function scoredSentences(text: string, keywords: string[]): Array<{ sentence: string; score: number }> {
  return sentences(text).map((sentence) => ({ sentence, score: highlight(sentence, keywords).matched.length }));
}

/**
 * The shots around the frame, those that mention what the query is about first and open:
 * the rest (an intro, a logo) fold away behind one button instead of pushing them down.
 */
function CaptionList({ captions, keywords, time }: { captions: Array<CaptionWindow & { current: boolean }>; keywords: string[]; time: number }) {
  const [all, setAll] = useState(false);
  const rows = useMemo(() => captions.map((caption) => {
    const scored = scoredSentences(caption.text, keywords);
    return { caption, matching: scored.filter((entry) => entry.score >= STRONG).map((entry) => entry.sentence),
      best: Math.max(0, ...scored.map((entry) => entry.score)) };
  }), [captions, keywords]);
  // The shot holding the frame, then the ones that say most about the query, nearest first.
  // Time order would lead with an intro that mentions "white" and "flower" in passing.
  const ranked = useMemo(() => [...rows].sort((a, b) =>
    Number(b.caption.current) - Number(a.caption.current) || b.best - a.best
    || Math.abs((a.caption.start + a.caption.end) / 2 - time) - Math.abs((b.caption.start + b.caption.end) / 2 - time)), [rows, time]);
  const shown = (all ? ranked : ranked.filter((row) => row.caption.current || row.best >= 1).slice(0, OPEN_WINDOWS));
  const hidden = rows.length - shown.length;
  return <>
    <ol className="review-captions">{shown.map(({ caption, matching }) =>
      <CaptionWindowView key={`${caption.shot_idx}-${caption.window_idx}`} caption={caption} matching={matching} keywords={keywords} />)}</ol>
    {hidden > 0 || all
      ? <button type="button" className="review-more-text" onClick={() => setAll(!all)}>
        {all ? "Hide shots that match nothing" : `Show ${hidden} other shot${hidden === 1 ? "" : "s"} around this frame`}</button>
      : null}
  </>;
}

/**
 * One caption window: a caption is ~3,000 characters of scene description, so it opens on
 * the sentences that mention what the query is about and keeps the rest a click away.
 */
function CaptionWindowView({ caption, matching, keywords }: {
  caption: CaptionWindow & { current: boolean };
  matching: string[];
  keywords: string[];
}) {
  const [full, setFull] = useState(false);
  const shown = full ? caption.text : matching.length ? matching.slice(0, 4).join(" ") : caption.text.slice(0, 200);
  return <li className={caption.current ? "active" : ""}>
    <time>{formatTimestamp(caption.start)}–{formatTimestamp(caption.end)}{caption.current && <em> this shot</em>}</time>
    <p className="review-caption"><Lit text={shown} keywords={keywords} />{!full && shown.length < caption.text.length && "…"}</p>
    {caption.text.length > shown.length || full
      ? <button type="button" className="review-more-text" onClick={() => setFull(!full)}>
        {full ? "Show matching sentences" : `Show full caption (${matching.length} sentence${matching.length === 1 ? "" : "s"} match)`}</button>
      : null}
  </li>;
}
