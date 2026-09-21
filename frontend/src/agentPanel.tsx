import { FormEvent, useEffect, useState } from "react";

import { downloadAgentSubmission, fetchAgentRun, reviewAgentRun, startAgentRun } from "./agentApi";
import { formatTimestamp } from "./results";
import { VideoDialog } from "./videoDialog";
import type {
  AgentCandidate, AgentLiveCandidate, AgentMode, AgentProgress, AgentReviewRequest,
  AgentReviewState, AgentRunResponse, AgentRunResult, AgentStage, AgentVerdict, QueryKind, SearchResult,
} from "./types";

// While a run is live a teammate is judging alongside Gemini, so the view refreshes
// faster than a finished result needs to.
const POLL_MS = 1000;
const SHOWN = 20;

const STAGE_LABELS: Record<AgentStage, string> = {
  planning: "Planning the searches",
  searching: "Searching",
  judging: "Gemini is judging candidates",
  second_look: "Gemini is double-checking the best candidate",
  rewriting: "Rewriting the searches",
  agent_loop: "Searching further (deep loop)",
  reading_answer: "Gemini is reading the answer",
  checking_paths: "Gemini is checking TRAKE paths",
  finalizing: "Finishing",
  done: "Done",
};

type Frame = { video_id: string; frame_idx: number; time: number };

function asSearchResult(candidate: Frame, score: number, rank: number): SearchResult {
  return {
    rank,
    score,
    video_id: candidate.video_id,
    frame_idx: candidate.frame_idx,
    time: candidate.time,
    thumb: `/thumb/${candidate.video_id}/${candidate.frame_idx}?w=320`,
    frame: `/frame/${candidate.video_id}/${candidate.frame_idx}`,
    video: `/video/${candidate.video_id}`,
    source: "visual",
    scores: {},
  };
}

export function AgentPanel() {
  const [deepseekKey, setDeepseekKey] = useState("");
  const [geminiKey, setGeminiKey] = useState("");
  const [text, setText] = useState("");
  const [kind, setKind] = useState<QueryKind>("kis");
  const [mode, setMode] = useState<AgentMode>("fast");
  const [run, setRun] = useState<AgentRunResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [openCandidate, setOpenCandidate] = useState<SearchResult | null>(null);

  const runId = run?.run_id;
  const running = run?.status === "running";
  useEffect(() => {
    if (!runId || !running) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      fetchAgentRun(runId)
        .then((next) => { if (!cancelled) setRun(next); })
        .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "Request failed"); });
    }, POLL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [runId, running]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    // Neither key is checked here. Whether a key is needed depends on the
    // deployment: its vision backend decides on the Gemini side, and whether it
    // holds a key of its own decides on both -- and a copy of either rule in the
    // client is how this panel came to demand a Gemini key that the server had
    // stopped requiring. The server answers 422 naming the key that is missing.
    if (!text.trim()) { setError("Enter a query."); return; }
    if (!text.trim()) { setError("Enter a query."); return; }
    setBusy(true);
    setError("");
    setRun(null);
    try {
      setRun(await startAgentRun({
        text: text.trim(), kind, mode,
        deepseek_api_key: deepseekKey.trim(), gemini_api_key: geminiKey.trim(),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  async function review(request: AgentReviewRequest) {
    if (!run) return;
    setError("");
    try {
      setRun(await reviewAgentRun(run.run_id, request));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    }
  }

  async function download() {
    if (!run) return;
    setDownloading(true);
    setError("");
    try {
      const { blob, filename } = await downloadAgentSubmission(run.run_id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setDownloading(false);
    }
  }

  function updateKey(setter: (value: string) => void, value: string) {
    setter(value);
    setRun(null);
  }

  return (
    <div className="agent-panel">
      <form className="agent-form" onSubmit={submit}>
        <div className="agent-key-grid">
          <label className="field"><span>DeepSeek API key</span>
            <input type="password" autoComplete="off" aria-label="DeepSeek API key"
              value={deepseekKey} onChange={(event) => updateKey(setDeepseekKey, event.target.value)} />
          </label>
          <label className="field"><span>Gemini API key</span>
            <input type="password" autoComplete="off" aria-label="Gemini API key"
              value={geminiKey} onChange={(event) => updateKey(setGeminiKey, event.target.value)} />
          </label>
        </div>
        <p className="agent-key-note">Keys are used only for this run, are never stored, and are cleared when you reload the page. Leave a field empty to use the deployment's own key, if it has one.</p>
        <label className="field"><span>Query</span>
          <textarea rows={3} maxLength={4000} aria-label="Agent query" value={text}
            placeholder="Vietnamese KIS/Q&A/TRAKE query. TRAKE events go on E1:, E2:, ... lines."
            onChange={(event) => setText(event.target.value)} />
        </label>
        <div className="agent-form-footer">
          <label className="field"><span>Kind</span>
            <select aria-label="Query kind" value={kind} onChange={(event) => setKind(event.target.value as QueryKind)}>
              <option value="kis">KIS</option>
              <option value="qa">Q&amp;A</option>
              <option value="trake">TRAKE</option>
            </select>
          </label>
          <label className="field"><span>Mode</span>
            <select aria-label="Agent mode" value={mode} onChange={(event) => setMode(event.target.value as AgentMode)}>
              <option value="fast">Fast (~20 s)</option>
              <option value="auto">Auto (fast, deep if unsure)</option>
              <option value="deep">Deep (minutes)</option>
            </select>
          </label>
          <button type="submit" className="search-button" disabled={busy || running}>
            {busy ? "Starting…" : running ? "Running…" : "Run agent"}
          </button>
        </div>
      </form>
      {error && <div className="message error" role="alert">{error}</div>}
      {run && (running
        ? <AgentLiveView run={run} onReview={review} onOpen={setOpenCandidate} />
        : <AgentRunView run={run} downloading={downloading} onDownload={download} onOpen={setOpenCandidate} />)}
      {openCandidate && <VideoDialog result={openCandidate} onClose={() => setOpenCandidate(null)} startPaused />}
    </div>
  );
}

/**
 * The run as Gemini sees it, while Gemini sees it: the brief it is given, and every
 * candidate with the same speech, on-screen text and caption context. A teammate can
 * decide first; the agent applies the decision and stops waiting on Gemini.
 */
function AgentLiveView({ run, onReview, onOpen }: {
  run: AgentRunResponse;
  onReview(request: AgentReviewRequest): void;
  onOpen(result: SearchResult): void;
}) {
  const [answer, setAnswer] = useState("");
  const progress: AgentProgress | null | undefined = run.progress;
  const review: AgentReviewState = run.review ?? { decisions: {}, answer: null, stopped: false, decided: false };
  const judging = progress?.candidates.filter((candidate) => candidate.judging).length ?? 0;
  const stage = progress ? STAGE_LABELS[progress.stage] ?? progress.stage : "Starting";

  return (
    <section className="agent-live" aria-label="Live review">
      <div className="agent-live-head">
        <span aria-live="polite">
          <b>{stage}</b> · {Math.round(progress?.elapsed ?? run.elapsed)}s
          {judging > 0 && <> · {judging} with Gemini</>}
          {review.decided && <> · <em>your decision is being applied</em></>}
        </span>
        <button type="button" className="open-video" disabled={review.stopped}
          onClick={() => onReview({ stop: true })}>
          {review.stopped ? "Stopping…" : "Stop and use the current ranking"}
        </button>
      </div>
      <p className="agent-live-note">
        Accept a candidate as soon as you are sure: the agent stops waiting for Gemini and
        puts it first. Reject one to push it to the bottom. Your decisions always override Gemini.
      </p>
      <div className="agent-brief">
        <div className="agent-brief-title">What Gemini is told</div>
        <p className="agent-brief-query">{run.query}</p>
        {progress?.plan && <ol className="agent-brief-scenes">
          {progress.plan.scenes.map((scene, index) => <li key={index}>
            {scene}{index === progress.plan?.target_scene && progress.plan.scenes.length > 1 && <em> (frame to submit)</em>}
          </li>)}
        </ol>}
        {progress?.plan?.question_vi && <p><b>Question:</b> {progress.plan.question_vi}</p>}
        {progress?.plan && progress.plan.events.length > 0 && <p><b>Events:</b> {progress.plan.events.join(" → ")}</p>}
      </div>
      {run.kind === "qa" && <form className="agent-answer-form" onSubmit={(event) => {
        event.preventDefault();
        if (answer.trim()) onReview({ answer: answer.trim() });
      }}>
        <label className="field"><span>Your answer</span>
          <input aria-label="Reviewer answer" maxLength={100} value={answer}
            placeholder={review.answer ?? "Type the answer if you can read it first"}
            onChange={(event) => setAnswer(event.target.value)} />
        </label>
        <button type="submit" className="open-video" disabled={!answer.trim()}>Use this answer</button>
        {review.answer && <span className="agent-answer-set">Using: {review.answer}</span>}
      </form>}
      {progress && progress.sequences.length > 0 && <ul className="agent-sequences">
        {progress.sequences.slice(0, 4).map((sequence) => <li key={`${sequence.video_id}-${sequence.frames.join(".")}`}>
          {sequence.video_id}: {sequence.frames.join(" → ")}
          {sequence.event_scores && <> · scores {sequence.event_scores.join(", ")}</>}
        </li>)}
      </ul>}
      {!progress || progress.candidates.length === 0
        ? <div className="message">The agent is still planning and searching…</div>
        : <div className="agent-candidates">
          {progress.candidates.map((candidate, index) =>
            <LiveCandidateCard key={candidate.id} candidate={candidate} rank={index + 1}
              decision={review.decisions[candidate.id]} onReview={onReview} onOpen={onOpen} />)}
        </div>}
    </section>
  );
}

function LiveCandidateCard({ candidate, rank, decision, onReview, onOpen }: {
  candidate: AgentLiveCandidate;
  rank: number;
  decision: string | undefined;
  onReview(request: AgentReviewRequest): void;
  onOpen(result: SearchResult): void;
}) {
  const { speech, on_screen: onScreen, caption } = candidate.context;
  const decide = (choice: "accept" | "reject") =>
    onReview({ candidate_id: candidate.id, decision: decision === choice ? "clear" : choice });
  return (
    <article className={`agent-candidate${candidate.judging ? " judging" : ""}${decision ? ` ${decision}ed` : ""}`}>
      <button type="button" className="agent-candidate-open"
        onClick={() => onOpen(asSearchResult(candidate, candidate.verdict?.score ?? 0, rank))}>
        <img src={`/thumb/${candidate.video_id}/${candidate.frame_idx}?w=320`}
          alt={`Candidate ${rank}: ${candidate.video_id} frame ${candidate.frame_idx}`} loading="lazy" />
        <span>#{rank} {candidate.id} · {candidate.video_id} · {formatTimestamp(candidate.time)}</span>
      </button>
      {candidate.judging && <div className="agent-judging">Gemini is judging…</div>}
      {candidate.verdict && <Verdict verdict={candidate.verdict} />}
      {(speech || onScreen || caption) && <dl className="agent-context">
        {speech && <><dt>Speech</dt><dd>{speech}</dd></>}
        {onScreen && <><dt>On screen</dt><dd>{onScreen}</dd></>}
        {caption && <><dt>Caption</dt><dd>{caption}</dd></>}
      </dl>}
      <div className="agent-review-buttons">
        <button type="button" aria-pressed={decision === "accept"} aria-label={`Accept ${candidate.id}`}
          className="agent-accept" onClick={() => decide("accept")}>✓ Accept</button>
        <button type="button" aria-pressed={decision === "reject"} aria-label={`Reject ${candidate.id}`}
          className="agent-reject" onClick={() => decide("reject")}>✗ Reject</button>
      </div>
    </article>
  );
}

function Verdict({ verdict }: { verdict: AgentVerdict }) {
  const tier = verdict.score >= 6 ? "good" : verdict.score <= 3 ? "bad" : "mid";
  return (
    <div className={`agent-verdict ${tier}`}>
      <b>{verdict.score.toFixed(0)}/10</b>
      {verdict.by === "human" && <span className="agent-by-human">Reviewer</span>} {verdict.reason}
    </div>
  );
}

function AgentRunView({ run, downloading, onDownload, onOpen }: {
  run: AgentRunResponse;
  downloading: boolean;
  onDownload(): void;
  onOpen(result: SearchResult): void;
}) {
  if (run.status === "error") {
    return <div className="message error" role="alert">{run.error ?? "The agent run failed."}</div>;
  }
  const result: AgentRunResult | null = run.result;
  if (!result) return <div className="message error" role="alert">The run finished without a result.</div>;
  const candidates = result.ranking.slice(0, SHOWN);
  return (
    <section className="agent-result">
      {result.error && <div className="message error" role="alert">{result.error}</div>}
      <div className="agent-result-head">
        <span>{result.kind.toUpperCase()} · {Math.round(result.elapsed)}s · {result.ranking.length} candidates</span>
        <button type="button" className="open-video" onClick={onDownload} disabled={downloading}>
          {downloading ? "Exporting…" : "Download submission ZIP"}
        </button>
      </div>
      {result.answer && <p className="agent-answer"><b>Answer:</b> {result.answer}
        {result.answer_evidence && <em> — {result.answer_evidence}</em>}</p>}
      {result.sequences.length > 0 && <ul className="agent-sequences">
        {result.sequences.slice(0, 3).map((sequence) => <li key={`${sequence.video_id}-${sequence.frames.join(".")}`}>
          {sequence.video_id}: {sequence.frames.join(" → ")}{sequence.refined ? " (refined)" : ""}
        </li>)}
      </ul>}
      {candidates.length === 0
        ? <div className="message warning">The agent found no candidates for this query.</div>
        : <div className="agent-candidates">
          {candidates.map((candidate, index) =>
            <AgentCandidateCard key={candidate.id} candidate={candidate} rank={index + 1} onOpen={onOpen} />)}
        </div>}
      {result.notes && <p className="agent-notes">{result.notes}</p>}
    </section>
  );
}

function AgentCandidateCard({ candidate, rank, onOpen }: {
  candidate: AgentCandidate;
  rank: number;
  onOpen(result: SearchResult): void;
}) {
  return (
    <article className="agent-candidate">
      <button type="button" className="agent-candidate-open"
        onClick={() => onOpen(asSearchResult(candidate, candidate.verdict?.score ?? candidate.fused, rank))}>
        <img src={`/thumb/${candidate.video_id}/${candidate.frame_idx}?w=320`} alt={`Candidate ${rank}: ${candidate.video_id} frame ${candidate.frame_idx}`} loading="lazy" />
        <span>#{rank} {candidate.video_id} · {formatTimestamp(candidate.time)}</span>
      </button>
      {candidate.verdict && <Verdict verdict={candidate.verdict} />}
    </article>
  );
}
