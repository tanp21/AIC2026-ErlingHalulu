import { FormEvent, useEffect, useState } from "react";

import { downloadAgentSubmission, fetchAgentRun, startAgentRun } from "./agentApi";
import { formatTimestamp } from "./results";
import { VideoDialog } from "./videoDialog";
import type { AgentCandidate, AgentMode, AgentRunResponse, AgentRunResult, QueryKind, SearchResult } from "./types";

const POLL_MS = 2000;
const SHOWN = 20;

function asSearchResult(candidate: AgentCandidate, rank: number): SearchResult {
  return {
    rank,
    score: candidate.verdict?.score ?? candidate.fused,
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
    if (!deepseekKey.trim() || !geminiKey.trim()) { setError("Both API keys are required."); return; }
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
        <p className="agent-key-note">Keys are used only for this run, are never stored, and are cleared when you reload the page.</p>
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
      {run && <AgentRunView run={run} downloading={downloading} onDownload={download} onOpen={setOpenCandidate} />}
      {openCandidate && <VideoDialog result={openCandidate} onClose={() => setOpenCandidate(null)} startPaused />}
    </div>
  );
}

function AgentRunView({ run, downloading, onDownload, onOpen }: {
  run: AgentRunResponse;
  downloading: boolean;
  onDownload(): void;
  onOpen(result: SearchResult): void;
}) {
  if (run.status === "running") {
    return <div className="message" aria-live="polite">The agent is working… {Math.round(run.elapsed)}s elapsed.</div>;
  }
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
  const verdict = candidate.verdict;
  const tier = !verdict ? "" : verdict.score >= 6 ? "good" : verdict.score <= 3 ? "bad" : "mid";
  return (
    <article className="agent-candidate">
      <button type="button" className="agent-candidate-open" onClick={() => onOpen(asSearchResult(candidate, rank))}>
        <img src={`/thumb/${candidate.video_id}/${candidate.frame_idx}?w=320`} alt={`Candidate ${rank}: ${candidate.video_id} frame ${candidate.frame_idx}`} loading="lazy" />
        <span>#{rank} {candidate.video_id} · {formatTimestamp(candidate.time)}</span>
      </button>
      {verdict && <div className={`agent-verdict ${tier}`}>
        <b>{verdict.score.toFixed(0)}/10</b> {verdict.reason}
      </div>}
    </article>
  );
}
