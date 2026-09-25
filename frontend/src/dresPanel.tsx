import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { DresApiError, dresConnect, dresHistory, dresPrepare, dresRefresh, dresRelogin, dresReset, dresSelectEvaluation, dresSetContext, dresStatus, dresSubmit } from "./dresApi";
import type { DresAttempt, DresCandidate, DresPrepared, DresStatus, QueryKind, SearchResult } from "./types";

function errorText(error: unknown): string { return error instanceof Error ? error.message : "DRES request failed"; }

export function DresPanel({ selected, candidate, onCandidate, onPick, onKind, selectedKind, externalStatus, isAdmin, pickError, onStatusChange, suggestedTask, settingsOpen = false, onSettingsClose, onOpenFrame, showSubmission = true }: {
  selected: SearchResult | null;
  candidate: DresCandidate | null;
  onCandidate(candidate: DresCandidate | null): void;
  onPick(videoId: string, frameId: number): void;
  onKind(kind: QueryKind | null): void;
  selectedKind?: QueryKind;
  externalStatus?: DresStatus | null;
  isAdmin: boolean;
  pickError: string;
  onStatusChange?(status: DresStatus): void;
  suggestedTask?: { kind: QueryKind; label: string; event_count: number | null } | null;
  settingsOpen?: boolean;
  onSettingsClose?(): void;
  onOpenFrame?(videoId: string, frameId: number): void;
  showSubmission?: boolean;
}) {
  const [status, setStatus] = useState<DresStatus | null>(null);
  const [history, setHistory] = useState<DresAttempt[]>([]);
  const [kind, setKind] = useState<QueryKind>("kis");
  const [label, setLabel] = useState("");
  const [answer, setAnswer] = useState("");
  const [prepared, setPrepared] = useState<DresPrepared | null>(null);
  const [lastAttempt, setLastAttempt] = useState<DresAttempt | null>(null);
  const [localUncertain, setLocalUncertain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const submitting = useRef(false);
  const syncedTask = useRef<string | null>(null);
  const syncStatus = externalStatus ?? status;

  async function reload() {
    const [nextStatus, attempts] = await Promise.all([dresStatus(), dresHistory()]);
    setStatus(nextStatus);
    onStatusChange?.(nextStatus);
    setHistory(attempts);
  }

  useEffect(() => {
    let active = true;
    Promise.all([dresStatus(), dresHistory()]).then(([nextStatus, attempts]) => {
      if (!active) return;
      setStatus(nextStatus);
      onStatusChange?.(nextStatus);
      setHistory(attempts);
    }).catch((reason) => { if (active) setError(errorText(reason)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (externalStatus) setStatus(externalStatus);
  }, [externalStatus]);

  useEffect(() => {
    if (status?.context) {
      setKind(status.context.kind);
      setLabel(status.context.label);
    } else if (suggestedTask) {
      setKind(suggestedTask.kind);
      setLabel(suggestedTask.label);
    }
  }, [status?.context?.revision, suggestedTask?.kind, suggestedTask?.label, suggestedTask?.event_count]);

  useEffect(() => {
    if (!syncStatus?.connected || !syncStatus.evaluations.length) {
      syncedTask.current = null;
      return;
    }
    if (busy) return;
    if (selectedKind || !suggestedTask && syncStatus.context_valid) return;
    const targetKind = suggestedTask?.kind ?? selectedKind ?? kind;
    const targetLabel = suggestedTask?.label ?? `${targetKind.toUpperCase()} task`;
    const targetCount = targetKind === "trake" ? suggestedTask?.event_count ?? null : null;
    const matches = syncStatus.context_valid && syncStatus.evaluation_id && syncStatus.context?.kind === targetKind
      && syncStatus.context.event_count === targetCount && (suggestedTask ? syncStatus.context.label === targetLabel : true);
    if (matches) return;
    const key = `${targetKind}:${targetLabel}:${targetCount ?? ""}:${syncStatus.context?.revision ?? 0}:${syncStatus.evaluation_id ?? ""}:${syncStatus.evaluation_manual ?? false}:${syncStatus.evaluations.map((item) => item.id).join(",")}`;
    if (syncedTask.current === key) return;
    syncedTask.current = key;
    setBusy(true);
    setError("");
    dresSetContext({ expected_revision: syncStatus.context?.revision ?? 0,
      kind: targetKind, label: targetLabel, event_count: targetKind === "trake" ? targetCount : null })
      .then(async () => {
        onCandidate(null);
        setAnswer("");
        await reload();
      })
      .catch((reason) => setError(errorText(reason)))
      .finally(() => setBusy(false));
  }, [suggestedTask?.kind, suggestedTask?.label, suggestedTask?.event_count, selectedKind, syncStatus, busy, kind]);

  useEffect(() => { setPrepared(null); setLastAttempt(null); setLocalUncertain(false); }, [candidate, answer, status?.evaluation_id, status?.context?.revision]);

  async function action(fn: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await fn(); await reload(); }
    catch (reason) {
      setError(errorText(reason));
      try { await reload(); } catch { /* Keep the first, relevant error. */ }
    }
    finally { setBusy(false); }
  }

  function submissionCandidate(): DresCandidate | null {
    if (!candidate) return null;
    if (status?.context?.kind === "qa") return { ...candidate, answer };
    return candidate;
  }

  async function submit(retryUncertain = false, candidateOverride?: DresCandidate, preparedOverride?: DresPrepared) {
    const value = candidateOverride ?? submissionCandidate();
    const payloadPreview = preparedOverride ?? prepared;
    if (!value || !payloadPreview || submitting.current) return;
    submitting.current = true;
    setBusy(true); setError("");
    try {
      const attempt = await dresSubmit(value, payloadPreview, retryUncertain);
      setLastAttempt(attempt); setLocalUncertain(false);
      try { await reload(); } catch { /* Submission receipt is already known. */ }
      if (attempt.status !== "uncertain" && attempt.status !== "not_sent") setPrepared(null);
    } catch (reason) {
      setError(errorText(reason));
      if (reason instanceof DresApiError && reason.category === "not_sent") return;
      setLocalUncertain(true);
      // A lost browser response may follow a successful POST. Inspect the local ledger,
      // never resend automatically.
      try {
        const attempts = await dresHistory();
        setHistory(attempts);
        const match = attempts.find((item) => item.evaluation_id === payloadPreview.evaluation_id && item.fingerprint === payloadPreview.fingerprint);
        if (match) {
          setLastAttempt(match); setLocalUncertain(false);
          if (match.status !== "uncertain" && match.status !== "not_sent") setPrepared(null);
        }
      } catch { /* Keep the preview and require an explicit operator decision. */ }
    } finally { submitting.current = false; setBusy(false); }
  }

  async function submitSelectedFrame() {
    if (busy) return;
    const value: DresCandidate | null = context?.kind === "trake"
      ? candidate?.frame_ids?.length ? candidate : null
      : candidate?.frame_idx !== undefined ? { ...candidate, ...(context?.kind === "qa" ? { answer } : {}) }
        : selected ? { video_id: selected.video_id, frame_idx: selected.frame_idx,
          ...(context?.kind === "qa" ? { answer } : {}) } : null;
    if (!value) return;
    setBusy(true); setError("");
    try {
      const next = await dresPrepare(value);
      if (next.context_revision !== status?.context?.revision || next.evaluation_id !== status?.evaluation_id) {
        await reload();
        setError("The current task or evaluation changed. Review it and select the frame again.");
        return;
      }
      setPrepared(next);
      await submit(false, value, next);
    } catch (reason) {
      setError(errorText(reason));
    } finally { setBusy(false); }
  }

  const context = status?.context_valid ? status.context : null;
  const frames = candidate?.frame_ids ?? [];
  const candidateReady = context?.kind === "trake"
    ? Boolean(candidate?.frame_ids?.length)
    : candidate?.frame_idx !== undefined || Boolean(selected);
  const duplicate = prepared && history.some((item) => item.evaluation_id === prepared.evaluation_id
    && item.fingerprint === prepared.fingerprint && item.status !== "not_sent");
  const canRetryUncertain = Boolean(prepared && !busy && (localUncertain || duplicate && history.some((item) =>
    item.evaluation_id === prepared.evaluation_id && item.fingerprint === prepared.fingerprint && item.status === "uncertain")));

  return <>
  {showSubmission && <section className="dres-panel dres-submit-panel" aria-label="DRES submission">
    <header><strong>Submit to DRES</strong><span className={status?.connected ? "dres-inline-state ready" : "dres-inline-state"}>{status?.connected ? "Connected" : "Disconnected"}</span></header>
    <label>Task<select aria-label="DRES task type" value={kind} disabled={busy} onChange={(event) => {
      const nextKind = event.target.value as QueryKind; setKind(nextKind); onKind(nextKind);
      void action(async () => {
        const taskLabel = suggestedTask?.label ?? (label || `${nextKind.toUpperCase()} task`);
        await dresSetContext({ expected_revision: status?.context?.revision ?? 0, kind: nextKind,
          label: taskLabel, event_count: nextKind === "trake" ? suggestedTask?.event_count ?? null : null });
        onCandidate(null); setAnswer("");
      });
    }}><option value="kis">KIS</option><option value="qa">Q&amp;A</option><option value="trake">TRAKE</option></select></label>
    {context?.kind === "qa" && <label>Answer<input aria-label="DRES Q&A answer" value={answer} onChange={(event) => setAnswer(event.target.value)} /></label>}
    {context?.kind === "trake" && selected && <button type="button" disabled={busy} onClick={() => onPick(selected.video_id, selected.frame_idx)}>Add selected frame as event</button>}
    {context && <div className="dres-context"><small>{context.kind.toUpperCase()} · {context.kind === "trake" ? `${frames.length} events selected` : "single frame"}</small></div>}
    {context?.kind === "trake" && candidate?.frame_ids && <ol className="dres-frame-list">{frames.map((frame, index) => <li key={`${frame}-${index}`}>
      <button type="button" className="dres-event-open" onClick={() => onOpenFrame?.(candidate.video_id, frame)}>E{index + 1} · FRAME_ID {frame}</button>
      <button type="button" aria-label={`Move frame ${frame} up`} disabled={index === 0 || busy} onClick={() => {
        const next = [...frames]; [next[index - 1], next[index]] = [next[index], next[index - 1]];
        onCandidate({ video_id: candidate.video_id, frame_ids: next });
      }}>↑</button>
      <button type="button" aria-label={`Remove frame ${frame}`} disabled={busy} onClick={() => onCandidate({ video_id: candidate.video_id, frame_ids: frames.filter((_, i) => i !== index) })}>×</button>
    </li>)}</ol>}
    <button type="button" className="dres-submit" disabled={busy || !status?.evaluation_id || !context || !candidateReady
      || duplicate || localUncertain || context.kind === "qa" && !answer.trim()}
      onClick={submitSelectedFrame}>{busy ? "Submitting…" : context?.kind === "trake" ? "Submit event sequence" : "Submit selected frame"}</button>
    {lastAttempt && <p className={`dres-outcome ${lastAttempt.status}`} role="status">{lastAttempt.status.toUpperCase()} · HTTP {lastAttempt.http_status ?? "?"}</p>}
    {pickError && <p className="dres-error" role="alert">{pickError}</p>}
    {error && <p className="dres-error" role="alert">{error}</p>}
    {localUncertain && <p className="dres-warning">Submission uncertain. Check DRES before retrying.</p>}
    {canRetryUncertain && <button type="button" disabled={busy} onClick={() => submit(true)}>Retry after checking DRES</button>}
    {duplicate && <p className="dres-warning">This exact payload was already attempted.</p>}
  </section>}
  {settingsOpen && createPortal(<div className="dres-settings-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onSettingsClose?.()}>
    <section className="dres-settings-modal" role="dialog" aria-modal="true" aria-label="DRES settings">
    <header><strong>DRES settings</strong><button type="button" aria-label="Close DRES settings" onClick={onSettingsClose}>×</button></header>
    <p className="dres-note">Credentials stay on the server. Evaluations refresh before every submission.</p>
    <div className="dres-status">
      <span>{status?.configured ? status.host : "DRES URL not configured"}</span>
      <b>{status?.connected ? "Session ready" : "Disconnected"}</b>
      <span>{status?.evaluation_id ? `Evaluation ${status.evaluation_id}` : status?.evaluations.length ? "No unique evaluation match" : "No active evaluation"}</span>
    </div>
    {status?.connected && status.evaluations.length === 0 &&
      <p className="dres-warning">DRES returned no ACTIVE SYNCHRONOUS evaluation. Ask the owner to refresh after the round starts.</p>}
    {status?.connected && status.evaluations.length > 0 && !status.evaluation_id &&
      <p className="dres-warning">{status.evaluations.length > 1 ? "Several ACTIVE SYNCHRONOUS evaluations exist." : "The active evaluation name does not match this task type."} Choose the matching evaluation below.</p>}
    {status?.evaluation_id && !status.context_valid &&
      <p className="dres-warning">Setting the selected DRES task…</p>}
    {isAdmin && <div className="dres-controls">
      <button type="button" disabled={busy || !status?.configured || !status.credentials_configured} onClick={() => action(dresConnect)}>Connect</button>
      <button type="button" disabled={busy || !status?.credentials_configured} onClick={() => action(dresRelogin)}>Re-login</button>
      <button type="button" disabled={busy || !status?.connected} onClick={() => action(dresRefresh)}>Refresh evaluations</button>
      <button type="button" disabled={busy || !status?.connected} onClick={() => action(dresReset)}>Reset session</button>
    </div>}
    {status?.connected && status.evaluations.length > 0 && (!status.evaluation_id || status.evaluations.length > 1) && <label className="dres-evaluation-fallback">
      Evaluation fallback<select aria-label="DRES evaluation fallback"
        value={status.evaluation_manual ? status.evaluation_id ?? "" : "__auto__"} disabled={busy}
        onChange={(event) => { const id = event.target.value; void action(() => dresSelectEvaluation(id === "__auto__" ? null : id)); }}>
        <option value="__auto__">{status.evaluation_id ? `Automatic match · ${status.evaluation_id}` : "Automatic match unavailable · choose evaluation"}</option>
        {status.evaluations.map((item) => <option key={item.id} value={item.id}>{item.name || item.id} · {item.id}</option>)}
      </select>
    </label>}
    {status?.evaluation_manual && status.evaluation_id && <small>Fallback evaluation selected: {status.evaluation_id}</small>}
    <div className="dres-context"><strong>Active task</strong>
      <small>{context ? `${context.label} · ${context.kind.toUpperCase()}` : "Choose KIS, Q&A or TRAKE in the navigation bar."}</small>
    </div>
    {error && <p className="dres-error" role="alert">{error}</p>}
    {history.length > 0 && <details><summary>Recent attempts ({history.length})</summary>
      <ul>{history.map((item) => <li key={item.id}>#{item.id} {item.task_type.toUpperCase()} · {item.video_id} · {item.status}</li>)}</ul>
    </details>}
    </section>
  </div>, document.body)}
  </>;
}
