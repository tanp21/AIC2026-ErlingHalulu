import { FormEvent, useEffect, useRef, useState } from "react";

import {
  AgentApiError, downloadAgentSubmission, fetchActiveAgentRun, fetchAgentRun, reviewAgentRun,
  startAgentRun,
} from "./agentApi";
import { AgentFrame, AgentResultView, AgentStrategy, judgeLabel, thumbUrl } from "./agentResult";
import { AgentReviewWorkspace } from "./agentReview";
import { verdictTier } from "./checkView";
import { formatTimestamp } from "./results";
import { VideoDialog } from "./videoDialog";
import type {
  AgentLiveCandidate, AgentMode, AgentProgress, AgentReviewRequest, AgentRunResponse, AgentStage,
  AgentSubmitOutcome, AgentSubmitRequest, AgentVerdict, QueryKind, SearchResult,
} from "./types";

const POLL_MS = 1000;

function stageLabel(stage: AgentStage, judge: string): string {
  const labels: Record<AgentStage, string> = {
    planning: "Planning the searches and starting web research",
    searching: "Searching every channel",
    judging: `${judge} is judging candidates`,
    second_look: `${judge} is double-checking the best candidate`,
    rewriting: "Rewriting the searches with what was learnt",
    agent_loop: "Searching further (deep loop)",
    reading_answer: "Reading the answer at the best-judged moments",
    checking_paths: `${judge} is checking the TRAKE paths event by event`,
    finalizing: "Finishing",
    awaiting_review: "Waiting for your review",
    done: "Done",
  };
  return labels[stage] ?? stage;
}

function asSearchResult(frame: AgentFrame, score: number, rank = 1): SearchResult {
  return {
    rank,
    score,
    video_id: frame.video_id,
    frame_idx: frame.frame_idx,
    time: frame.time,
    thumb: `/thumb/${frame.video_id}/${frame.frame_idx}?w=320`,
    frame: `/frame/${frame.video_id}/${frame.frame_idx}`,
    video: `/video/${frame.video_id}`,
    source: "visual",
    scores: {},
  };
}

/**
 * The Retrieval agent tab. A run is fully automatic -- it plans, searches every channel, looks
 * things up on the web, and has every candidate judged -- and what it finds is laid out for
 * checking and submitting: `onSubmit` sends exactly the frame (or sequence, or answer) on screen
 * to DRES.
 */
export function AgentPanel({ onSubmit }: { onSubmit?(request: AgentSubmitRequest): Promise<AgentSubmitOutcome> } = {}) {
  const [deepseekKey, setDeepseekKey] = useState("");
  const [geminiKey, setGeminiKey] = useState("");
  const [text, setText] = useState("");
  const [kind, setKind] = useState<QueryKind>("kis");
  const [mode, setMode] = useState<AgentMode>("fast");
  // Deep mode plans with thinking by default; a hard query that went wrong in fast mode can ask for it too.
  const [thinking, setThinking] = useState(false);
  const queryRef = useRef<HTMLTextAreaElement>(null);
  const [run, setRun] = useState<AgentRunResponse | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [openFrame, setOpenFrame] = useState<SearchResult | null>(null);

  const runId = run?.run_id;
  const running = run?.status === "running";
  useEffect(() => {
    if (!runId || !running) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      fetchAgentRun(runId)
        .then((next) => {
          if (cancelled) return;
          setRun(next);
          if (next.status !== "running") setNotice("");
        })
        .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "Request failed"); });
    }, POLL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [runId, running]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    // Neither key is checked here: whether one is needed depends on the deployment (which
    // model judges, whether it holds keys of its own), and the server answers 422 naming the
    // key that is missing.
    if (!text.trim()) { setError("Enter a query."); return; }
    setBusy(true);
    setError("");
    setNotice("");
    setRun(null);
    try {
      setRun(await startAgentRun({
        text: text.trim(), kind, mode, judge: "model", plan_thinking: thinking,
        deepseek_api_key: deepseekKey.trim(), gemini_api_key: geminiKey.trim(),
      }));
    } catch (err) {
      // One live run per user: show the run that holds the slot instead of dead-ending.
      if (err instanceof AgentApiError && err.status === 409) {
        const active = await fetchActiveAgentRun().catch(() => null);
        if (active) {
          setRun(active);
          setNotice("You already have a run in progress — it is shown below. Stop it to run this query.");
          return;
        }
      }
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

  /** Back to the form with this run's query, to reword what the agent should focus on. */
  function editQuery() {
    if (!run) return;
    setText(run.query);
    setKind(run.kind);
    queryRef.current?.scrollIntoView({ block: "center" });
    queryRef.current?.focus();
  }

  function updateKey(setter: (value: string) => void, value: string) {
    setter(value);
    setRun(null);
  }

  // A run someone started with a person as the judge (an older client, another tab) still gets its workspace.
  const humanLive = running && run?.review?.judge === "human";
  const finished = run && !running;
  return (
    <div className={`agent-panel${humanLive || finished ? " wide" : ""}`}>
      {!humanLive && <form className="agent-form" onSubmit={submit}>
        <label className="field"><span>Query</span>
          <textarea ref={queryRef} rows={3} maxLength={4000} aria-label="Agent query" value={text}
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
            <select aria-label="Agent mode" value={mode} onChange={(event) => {
              const next = event.target.value as AgentMode;
              setMode(next);
              setThinking(next === "deep");
            }}>
              <option value="fast">Fast (about 1–2 min)</option>
              <option value="auto">Auto (fast, then deeper if unsure)</option>
              <option value="deep">Deep (several minutes)</option>
            </select>
          </label>
          <label className="agent-thinking" title="DeepSeek reasons before planning: 10-30 s more, for a query whose plan went wrong">
            <input type="checkbox" aria-label="Think harder when planning" checked={thinking}
              onChange={(event) => setThinking(event.target.checked)} />
            <span>Think harder when planning</span>
          </label>
          <button type="submit" className="search-button" disabled={busy || running}>
            {busy ? "Starting…" : running ? "Running…" : "Run agent"}
          </button>
        </div>
        <details className="agent-keys">
          <summary>API keys (optional)</summary>
          <div className="agent-key-grid">
            <label className="field"><span>DeepSeek API key</span>
              <input type="password" autoComplete="off" aria-label="DeepSeek API key"
                value={deepseekKey} onChange={(event) => updateKey(setDeepseekKey, event.target.value)} />
            </label>
            <label className="field"><span>Gemini API key (web search)</span>
              <input type="password" autoComplete="off" aria-label="Gemini API key"
                value={geminiKey} onChange={(event) => updateKey(setGeminiKey, event.target.value)} />
            </label>
          </div>
          <p className="agent-key-note">Candidates are judged by the team's shared model on the server. Keys pasted
            here are used only for this run, never stored, and cleared on reload; leave a field empty to use the
            deployment's own key, if it has one.</p>
        </details>
      </form>}
      {error && <div className="message error" role="alert">{error}</div>}
      {notice && <div className="message" role="status">{notice}</div>}
      {finished && <div className="agent-rerun">
        <span>Not what you were looking for? Reword the query to point the agent at the part that matters.</span>
        <button type="button" className="open-video" onClick={editQuery}>Edit query and run again</button>
      </div>}
      {run && (running
        ? (humanLive
          ? <AgentReviewWorkspace run={run} onReview={review} />
          : <AgentLiveView run={run} onStop={() => review({ stop: true })}
            onOpen={(frame) => setOpenFrame(asSearchResult(frame, 0))} />)
        : run.status === "error" || !run.result
          ? <div className="message error" role="alert">{run.error ?? "The run finished without a result."}</div>
          : <AgentResultView result={run.result} downloading={downloading} onDownload={download}
            onOpen={(frame, score) => setOpenFrame(asSearchResult(frame, score))} onSubmit={onSubmit} />)}
      {openFrame && <VideoDialog result={openFrame} onClose={() => setOpenFrame(null)} startPaused />}
    </div>
  );
}

/** The run while it runs: what it planned, what the web said, and each candidate as it is judged. */
function AgentLiveView({ run, onStop, onOpen }: {
  run: AgentRunResponse;
  onStop(): void;
  onOpen(frame: AgentFrame): void;
}) {
  const progress: AgentProgress | null | undefined = run.progress;
  const judge = judgeLabel(progress?.judge);
  const judging = progress?.candidates.filter((candidate) => candidate.judging).length ?? 0;
  const stopped = run.review?.stopped ?? false;
  const stage = progress ? stageLabel(progress.stage, judge) : "Starting";
  return (
    <section className="agent-live" aria-label="Agent progress">
      <div className="agent-live-head">
        <span aria-live="polite">
          <b>{stage}</b> · {Math.round(progress?.elapsed ?? run.elapsed)}s
          {judging > 0 && <> · {judging} with {judge}</>}
        </span>
        <button type="button" className="open-video" disabled={stopped} onClick={onStop}>
          {stopped ? "Stopping…" : "Stop and use the current ranking"}
        </button>
      </div>
      <div className="agent-brief">
        <div className="agent-brief-title">The query</div>
        <p className="agent-brief-query">{run.query}</p>
        {progress?.plan && <ol className="agent-brief-scenes">
          {progress.plan.scenes.map((scene, index) => <li key={index}>
            {scene}{index === progress.plan?.target_scene && progress.plan.scenes.length > 1 && <em> (frame to submit)</em>}
          </li>)}
        </ol>}
        {progress?.plan?.question_vi && <p><b>Question:</b> {progress.plan.question_vi}</p>}
        {!!progress?.events?.length && <p><b>Events:</b> {progress.events.join(" → ")}</p>}
      </div>
      <AgentStrategy strategy={progress?.strategy ?? []} lookups={progress?.lookups ?? []} />
      {!progress || progress.candidates.length === 0
        ? <div className="message">The agent is still planning and searching…</div>
        : <div className="agent-candidates">
          {progress.candidates.map((candidate, index) =>
            <LiveCandidateCard key={candidate.id} candidate={candidate} rank={index + 1} judge={judge} onOpen={onOpen} />)}
        </div>}
    </section>
  );
}

function LiveCandidateCard({ candidate, rank, judge, onOpen }: {
  candidate: AgentLiveCandidate;
  rank: number;
  judge: string;
  onOpen(frame: AgentFrame): void;
}) {
  return (
    <article className={`agent-candidate${candidate.judging ? " judging" : ""}`}>
      <button type="button" className="agent-candidate-open" onClick={() => onOpen(candidate)}>
        <img src={thumbUrl(candidate)} alt={`Candidate ${rank}: ${candidate.video_id} frame ${candidate.frame_idx}`} loading="lazy" />
        <span>#{rank} {candidate.video_id} · {formatTimestamp(candidate.time)}</span>
      </button>
      {candidate.judging && <div className="agent-judging">{judge} is judging…</div>}
      {candidate.verdict && <Verdict verdict={candidate.verdict} />}
    </article>
  );
}

function Verdict({ verdict }: { verdict: AgentVerdict }) {
  return (
    <div className={`agent-verdict ${verdictTier(verdict.score)}`}>
      <b>{verdict.score.toFixed(0)}/10</b>
      {verdict.by === "human" && <span className="agent-by-human">Reviewer</span>} {verdict.reason}
    </div>
  );
}
