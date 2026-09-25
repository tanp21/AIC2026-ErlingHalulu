import { useEffect, useState } from "react";

import { fetchKeyframeContext } from "./api";
import { STATUS_MARK, verdictTier } from "./checkView";
import { formatTimestamp } from "./results";
import type {
  AgentCandidate, AgentRunLookup, AgentRunResult, AgentSequence, AgentSubmitOutcome, AgentSubmitRequest,
  AgentVerdict, QueryKind,
} from "./types";

/** A keyframe, as the result view passes it around. */
export type AgentFrame = { video_id: string; frame_idx: number; time: number };

/** How many answers the list offers; the export carries the rest. */
const LISTED = 20;
const LISTED_PATHS = 6;
/** Keyframes either side of the one being inspected, to step to the exact moment. */
const NEIGHBOUR_RADIUS = 4;

const JUDGE_NAMES: Record<string, string> = { qwen: "Qwen", gemini: "Gemini", human: "you", deepseek: "DeepSeek" };
export function judgeLabel(judge: string | null | undefined): string {
  if (!judge) return "the judge";
  return JUDGE_NAMES[judge] ?? judge.charAt(0).toUpperCase() + judge.slice(1);
}

export function frameUrl(frame: AgentFrame) { return `/frame/${frame.video_id}/${frame.frame_idx}`; }
export function thumbUrl(frame: AgentFrame, width = 320) { return `/thumb/${frame.video_id}/${frame.frame_idx}?w=${width}`; }

/**
 * A finished run, laid out for checking: the answers on the left, the one being checked on the
 * right at full size, with the keyframes around it, what the judge saw and why, and a button that
 * submits exactly what is on screen to DRES.
 */
export function AgentResultView({ result, onOpen, onSubmit, downloading, onDownload }: {
  result: AgentRunResult;
  onOpen(frame: AgentFrame, score: number): void;
  onSubmit?(request: AgentSubmitRequest): Promise<AgentSubmitOutcome>;
  downloading: boolean;
  onDownload(): void;
}) {
  const [selected, setSelected] = useState(0);
  const [answer, setAnswer] = useState(result.answer ?? "");
  useEffect(() => { setSelected(0); setAnswer(result.answer ?? ""); }, [result]);
  const trake = result.kind === "trake";
  const candidates = result.ranking.slice(0, LISTED);
  const paths = result.sequences.slice(0, LISTED_PATHS);
  const count = trake ? paths.length : candidates.length;

  return (
    <section className="agent-result" aria-label="Agent result">
      {result.error && <div className="message error" role="alert">{result.error}</div>}
      <div className="agent-result-head">
        <span>{result.kind.toUpperCase()} · {Math.round(result.elapsed)}s · judged by {judgeLabel(result.judge)}
          {" · "}{trake ? `${result.sequences.length} paths` : `${result.ranking.length} candidates`}</span>
        <button type="button" className="open-video" onClick={onDownload} disabled={downloading}>
          {downloading ? "Exporting…" : "Download submission ZIP"}
        </button>
      </div>
      <AgentStrategy strategy={result.strategy ?? []} lookups={result.lookups ?? []} />
      {result.kind === "qa" && <div className="agent-answer-box">
        <label className="field"><span>Answer to submit</span>
          <input aria-label="Answer to submit" maxLength={100} value={answer}
            placeholder="The agent found no answer — type one"
            onChange={(event) => setAnswer(event.target.value)} />
        </label>
        {result.answer_evidence && <p className="agent-answer-evidence">{result.answer_evidence}</p>}
      </div>}
      {count === 0
        ? <div className="message warning">The agent found no {trake ? "path" : "candidate"} for this query.</div>
        : <div className="agent-check-layout">
          <ol className="agent-answer-list" aria-label={trake ? "Paths found" : "Answers found"}>
            {trake
              ? paths.map((path, index) => <li key={`${path.video_id}-${path.frames.join(".")}-${index}`}>
                <PathRow path={path} rank={index + 1} selected={index === selected} onSelect={() => setSelected(index)} />
              </li>)
              : candidates.map((candidate, index) => <li key={candidate.id}>
                <CandidateRow candidate={candidate} rank={index + 1} selected={index === selected}
                  onSelect={() => setSelected(index)} />
              </li>)}
          </ol>
          {trake
            ? <PathInspector key={selected} path={paths[Math.min(selected, paths.length - 1)]}
              rank={Math.min(selected, paths.length - 1) + 1} events={result.events ?? []}
              judge={result.judge} onOpen={onOpen} onSubmit={onSubmit} />
            : <AnswerInspector key={candidates[Math.min(selected, candidates.length - 1)].id}
              candidate={candidates[Math.min(selected, candidates.length - 1)]}
              rank={Math.min(selected, candidates.length - 1) + 1} kind={result.kind}
              answer={answer} judge={result.judge} onOpen={onOpen} onSubmit={onSubmit} />}
        </div>}
      {result.notes && <p className="agent-notes">{result.notes}</p>}
    </section>
  );
}

/** What the run decided to do, and what the web said. */
export function AgentStrategy({ strategy, lookups }: { strategy: string[]; lookups: AgentRunLookup[] }) {
  if (!strategy.length && !lookups.length) return null;
  return (
    <details className="agent-strategy">
      <summary>How the agent searched{lookups.length > 0 && ` · ${lookups.length} web lookup${lookups.length === 1 ? "" : "s"}`}</summary>
      {strategy.length > 0 && <ul className="agent-strategy-list">{strategy.map((line) => <li key={line}>{line}</li>)}</ul>}
      {lookups.length > 0 && <dl className="agent-lookups">
        {lookups.map((lookup, index) => <div key={`${lookup.question}-${index}`}>
          <dt>{lookup.question}</dt>
          <dd>{lookup.answer ?? <span className="agent-lookup-error">{lookup.error ?? "no answer"}</span>}</dd>
        </div>)}
      </dl>}
    </details>
  );
}

function VerdictBadge({ verdict }: { verdict: AgentVerdict | null | undefined }) {
  if (!verdict) return <span className="agent-score none">not judged</span>;
  return <span className={`agent-score ${verdictTier(verdict.score)}`}>{verdict.score.toFixed(0)}/10</span>;
}

function CandidateRow({ candidate, rank, selected, onSelect }: {
  candidate: AgentCandidate; rank: number; selected: boolean; onSelect(): void;
}) {
  return (
    <button type="button" className={`agent-answer-row${selected ? " selected" : ""}`} aria-pressed={selected}
      aria-label={`Answer ${rank}: ${candidate.video_id} at ${formatTimestamp(candidate.time)}`} onClick={onSelect}>
      <img src={thumbUrl(candidate, 240)} alt="" loading="lazy" />
      <span className="agent-answer-row-text">
        <b>#{rank}</b> {candidate.video_id} · {formatTimestamp(candidate.time)}
        <VerdictBadge verdict={candidate.verdict} />
      </span>
    </button>
  );
}

function PathRow({ path, rank, selected, onSelect }: {
  path: AgentSequence; rank: number; selected: boolean; onSelect(): void;
}) {
  const scores = path.event_scores;
  const mean = scores?.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
  return (
    <button type="button" className={`agent-answer-row path${selected ? " selected" : ""}`} aria-pressed={selected}
      aria-label={`Path ${rank}: ${path.video_id}`} onClick={onSelect}>
      <span className="agent-path-thumbs">
        {path.frames.map((frame, index) => <img key={`${frame}-${index}`} alt=""
          src={thumbUrl({ video_id: path.video_id, frame_idx: frame, time: 0 }, 160)} loading="lazy" />)}
      </span>
      <span className="agent-answer-row-text">
        <b>#{rank}</b> {path.video_id} · {path.frames.length} events
        {mean !== null && <span className={`agent-score ${verdictTier(mean)}`}>{mean.toFixed(1)}/10</span>}
        {path.refined && <em> refined</em>}
      </span>
    </button>
  );
}

/** The keyframes around one frame, to step to the exact moment before submitting it. */
function NeighbourStrip({ anchor, chosen, onPick, label }: {
  anchor: AgentFrame; chosen: number; onPick(frame: AgentFrame): void; label: string;
}) {
  const [frames, setFrames] = useState<AgentFrame[] | null>(null);
  useEffect(() => {
    let active = true;
    setFrames(null);
    fetchKeyframeContext(anchor.video_id, anchor.frame_idx, undefined, NEIGHBOUR_RADIUS)
      .then((context) => { if (active) setFrames(context.frames.map((frame) => ({
        video_id: frame.video_id, frame_idx: frame.frame_idx, time: frame.time,
      }))); })
      .catch(() => { if (active) setFrames([]); });
    return () => { active = false; };
  }, [anchor.video_id, anchor.frame_idx]);
  if (frames === null) return <div className="agent-neighbours loading" aria-label={label}>Loading nearby keyframes…</div>;
  if (!frames.length) return null;
  return (
    <div className="agent-neighbours" role="group" aria-label={label}>
      {frames.map((frame) => <button key={frame.frame_idx} type="button" aria-pressed={frame.frame_idx === chosen}
        className={frame.frame_idx === chosen ? "chosen" : frame.frame_idx === anchor.frame_idx ? "agent-pick" : ""}
        title={`${formatTimestamp(frame.time)} · frame ${frame.frame_idx}${frame.frame_idx === anchor.frame_idx ? " (the agent's pick)" : ""}`}
        aria-label={`Use frame ${frame.frame_idx} at ${formatTimestamp(frame.time)}`}
        onClick={() => onPick(frame)}>
        <img src={thumbUrl(frame, 160)} alt="" loading="lazy" />
        <span>{formatTimestamp(frame.time)}</span>
      </button>)}
    </div>
  );
}

function SubmitButton({ label, disabled, request, onSubmit }: {
  label: string; disabled?: boolean; request: AgentSubmitRequest; onSubmit?(request: AgentSubmitRequest): Promise<AgentSubmitOutcome>;
}) {
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<AgentSubmitOutcome | null>(null);
  const signature = JSON.stringify(request);
  useEffect(() => { setOutcome(null); }, [signature]);
  if (!onSubmit) return null;
  const done = Boolean(outcome?.completed || outcome?.uncertain);
  return (
    <div className="agent-submit">
      <button type="button" className="search-button agent-submit-button" disabled={disabled || busy || done}
        onClick={async () => {
          setBusy(true);
          try { setOutcome(await onSubmit(request)); }
          catch (error) { setOutcome({ message: error instanceof Error ? error.message : "Submission failed" }); }
          finally { setBusy(false); }
        }}>{busy ? "Submitting…" : done ? "Submitted" : label}</button>
      {outcome && <p className={`agent-submit-outcome${outcome.completed ? " ok" : outcome.uncertain ? " uncertain" : " failed"}`}
        role="status">{outcome.message}</p>}
    </div>
  );
}

const SOURCE_WORDS: Record<string, string> = { vision: "frames", speech: "speech", ocr: "on screen", caption: "caption" };

function Checks({ verdict }: { verdict: AgentVerdict }) {
  if (!verdict.checks?.length) return null;
  return (
    <ul className="agent-checks" aria-label="Query details">
      {verdict.checks.map((check, index) => <li key={`${check.text}-${index}`} className={`status-${check.status}`}>
        <span className="agent-check-mark" aria-label={check.status}>{STATUS_MARK[check.status] ?? "·"}</span>
        <span><b>{check.text}</b>{check.note && <> — {check.note}</>}
          {check.source && <em> ({SOURCE_WORDS[check.source] ?? check.source})</em>}</span>
      </li>)}
    </ul>
  );
}

/** One KIS/Q&A answer at full size, with everything needed to be sure of it before submitting. */
function AnswerInspector({ candidate, rank, kind, answer, judge, onOpen, onSubmit }: {
  candidate: AgentCandidate; rank: number; kind: QueryKind; answer: string; judge?: string | null;
  onOpen(frame: AgentFrame, score: number): void;
  onSubmit?(request: AgentSubmitRequest): Promise<AgentSubmitOutcome>;
}) {
  const [frame, setFrame] = useState<AgentFrame>(candidate);
  const moved = frame.frame_idx !== candidate.frame_idx;
  const verdict = candidate.verdict;
  const { speech, on_screen: onScreen, caption } = candidate.context ?? {};
  const request: AgentSubmitRequest = {
    kind, video_id: frame.video_id, frame_idx: frame.frame_idx, ...(kind === "qa" ? { answer: answer.trim() } : {}),
  };
  return (
    <article className="agent-inspector" aria-label={`Answer ${rank} in detail`}>
      <header className="agent-inspector-head">
        <b>#{rank} · {frame.video_id} · {formatTimestamp(frame.time)}</b>
        <span>frame {frame.frame_idx}{moved && " · moved from the agent's pick"}</span>
      </header>
      <img className="agent-inspector-frame" src={frameUrl(frame)} alt={`${frame.video_id} frame ${frame.frame_idx}`} />
      <NeighbourStrip anchor={candidate} chosen={frame.frame_idx} onPick={setFrame} label="Nearby keyframes" />
      {verdict && <div className={`agent-verdict-block ${verdictTier(verdict.score)}`}>
        <b>{verdict.score.toFixed(0)}/10</b> <span>by {judgeLabel(verdict.by ?? judge)}
          {verdict.frames_seen ? `, from ${verdict.frames_seen} keyframes around the moment` : ""}</span>
        <p>{verdict.reason}</p>
        {verdict.answer && <p className="agent-verdict-answer">Read here: <b>{verdict.answer}</b></p>}
      </div>}
      {verdict && <Checks verdict={verdict} />}
      {(speech || onScreen || caption) && <dl className="agent-context">
        {speech && <><dt>Speech</dt><dd>{speech}</dd></>}
        {onScreen && <><dt>On screen</dt><dd>{onScreen}</dd></>}
        {caption && <><dt>Caption</dt><dd>{caption}</dd></>}
      </dl>}
      <div className="agent-inspector-actions">
        <button type="button" className="open-video" onClick={() => onOpen(frame, verdict?.score ?? candidate.fused)}>▶ Play video here</button>
        <SubmitButton label={kind === "qa" ? "Submit frame and answer to DRES" : "Submit this frame to DRES"}
          disabled={kind === "qa" && !answer.trim()} request={request} onSubmit={onSubmit} />
      </div>
    </article>
  );
}

/** One TRAKE path, event by event, each at a size that can be checked, and adjustable before submitting. */
function PathInspector({ path, rank, events, judge, onOpen, onSubmit }: {
  path: AgentSequence; rank: number; events: string[]; judge?: string | null;
  onOpen(frame: AgentFrame, score: number): void;
  onSubmit?(request: AgentSubmitRequest): Promise<AgentSubmitOutcome>;
}) {
  const initial = path.frames.map((frame_idx, index) => ({
    video_id: path.video_id, frame_idx, time: path.times?.[index] ?? 0,
  }));
  const [frames, setFrames] = useState<AgentFrame[]>(initial);
  const [open, setOpen] = useState<number | null>(null);
  // Frame numbers rise with time inside one video, and unlike `times` they are always there.
  const inOrder = frames.every((frame, index) => index === 0 || frame.frame_idx > frames[index - 1].frame_idx);
  return (
    <article className="agent-inspector" aria-label={`Path ${rank} in detail`}>
      <header className="agent-inspector-head">
        <b>#{rank} · {path.video_id} · {frames.length} events</b>
        <span>{path.event_scores ? `scored by ${judgeLabel(judge)}` : "not checked event by event"}</span>
      </header>
      <ol className="agent-events">
        {frames.map((frame, index) => {
          const score = path.event_scores?.[index];
          return <li key={index} className="agent-event">
            <div className="agent-event-head">
              <b>E{index + 1}</b> <span>{events[index] ?? ""}</span>
              {score !== undefined && <span className={`agent-score ${verdictTier(score)}`}>{score.toFixed(0)}/10</span>}
            </div>
            <button type="button" className="agent-event-frame" onClick={() => setOpen(open === index ? null : index)}
              aria-label={`E${index + 1}: frame ${frame.frame_idx} at ${formatTimestamp(frame.time)} — show nearby keyframes`}>
              <img src={frameUrl(frame)} alt="" loading="lazy" />
              <span>{formatTimestamp(frame.time)} · frame {frame.frame_idx}
                {frame.frame_idx !== path.frames[index] && " · moved"}</span>
            </button>
            <div className="agent-event-actions">
              <button type="button" className="open-video" onClick={() => onOpen(frame, score ?? 0)}>▶ Play</button>
              <button type="button" className="open-video" onClick={() => setOpen(open === index ? null : index)}>
                {open === index ? "Hide nearby" : "Adjust"}
              </button>
            </div>
            {open === index && <NeighbourStrip anchor={initial[index]} chosen={frame.frame_idx}
              label={`Nearby keyframes for E${index + 1}`}
              onPick={(picked) => setFrames((current) => current.map((item, i) => i === index ? picked : item))} />}
          </li>;
        })}
      </ol>
      {!inOrder && <p className="message warning">The events are not in time order; TRAKE expects E1 before E2 before E3.</p>}
      <div className="agent-inspector-actions">
        <SubmitButton label="Submit this sequence to DRES" disabled={!inOrder}
          request={{ kind: "trake", video_id: path.video_id, frame_ids: frames.map((frame) => frame.frame_idx) }}
          onSubmit={onSubmit} />
      </div>
    </article>
  );
}
