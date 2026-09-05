import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { searchVideosByImage } from "./api";
import { activateRound, CollabApiError, downloadExport, fetchExportPreview, patchQuery, reviewQuery, uploadRound } from "./collabApi";
import type { AnswerDraft, CollabQuery, RoundSnapshot, RoundSummary, SearchResult } from "./types";
import { VideoDialog } from "./videoDialog";
import { formatTimestamp } from "./results";

const STATUS_LABEL = {
  todo: "Chưa làm", in_progress: "Đang làm", ready: "Sẵn sàng", reviewed: "Đã duyệt",
};

function shortLogin(login: string | null) {
  return login?.split("@")[0] ?? "Chưa nhận";
}

function emptyAnswer(query: CollabQuery): AnswerDraft {
  if (query.kind === "qa") return { video_id: "", frame_id: null, answer: "" };
  if (query.kind === "trake") return { video_id: "", frame_ids: Array(query.event_count ?? 0).fill(null) };
  return { video_id: "", frame_id: null };
}

function answerReference(query: CollabQuery, answers: AnswerDraft[]) {
  const answer = answers.find((item) => item.video_id && (query.kind === "trake"
    ? (item as { frame_ids: Array<number | null> }).frame_ids.some((frame) => frame !== null)
    : (item as { frame_id: number | null }).frame_id !== null));
  if (!answer) return null;
  const frame = query.kind === "trake"
    ? (answer as { frame_ids: Array<number | null> }).frame_ids.find((item) => item !== null)
    : (answer as { frame_id: number | null }).frame_id;
  return frame === null || frame === undefined ? null : { video_id: answer.video_id, frame_idx: frame };
}

function filledAnswers(query: CollabQuery, answers: AnswerDraft[], results: SearchResult[], qaAnswer: string) {
  const existing = new Set(answers.flatMap((answer) => query.kind === "trake"
    ? (answer as { frame_ids: Array<number | null> }).frame_ids.filter((frame): frame is number => frame !== null)
      .map((frame) => `${answer.video_id}:${frame}`)
    : [`${answer.video_id}:${(answer as { frame_id: number | null }).frame_id}`]));
  const filled = [...answers];
  for (const result of results) {
    if (filled.length >= 100) break;
    const key = `${result.video_id}:${result.frame_idx}`;
    if (existing.has(key)) continue;
    existing.add(key);
    if (query.kind === "trake") {
      const frames = Array(query.event_count ?? 0).fill(null); frames[0] = result.frame_idx;
      filled.push({ video_id: result.video_id, frame_ids: frames, auto_filled: true } as AnswerDraft);
    } else if (query.kind === "qa") {
      filled.push({ video_id: result.video_id, frame_id: result.frame_idx, answer: qaAnswer.trim(), auto_filled: true });
    } else filled.push({ video_id: result.video_id, frame_id: result.frame_idx, auto_filled: true });
  }
  return filled;
}

export function applyExactFrame(
  query: CollabQuery,
  answers: AnswerDraft[],
  videoId: string,
  frameId: number,
  rowIndex: number,
  eventIndex: number,
): { answers: AnswerDraft[]; rowIndex: number } {
  const rows = structuredClone(answers);
  if (rowIndex >= rows.length) throw new Error("Dòng đáp án không còn tồn tại");
  if (query.kind !== "trake") {
    if (rowIndex < 0) {
      if (rows.length >= 100) throw new Error("Tối đa 100 dòng đáp án");
      rows.push(query.kind === "qa"
        ? { video_id: videoId, frame_id: frameId, answer: "" }
        : { video_id: videoId, frame_id: frameId });
      return { answers: rows, rowIndex: rows.length - 1 };
    }
    const old = rows[rowIndex];
    rows[rowIndex] = { ...old, video_id: videoId, frame_id: frameId } as AnswerDraft;
    return { answers: rows, rowIndex };
  }

  const old = rows[rowIndex] as { video_id: string; frame_ids: Array<number | null> } | undefined;
  if (!old || (old.video_id && old.video_id !== videoId)) {
    if (rows.length >= 100) throw new Error("Tối đa 100 dòng đáp án");
    const frameIds = Array(query.event_count ?? 0).fill(null);
    frameIds[eventIndex] = frameId;
    rows.push({ video_id: videoId, frame_ids: frameIds });
    return { answers: rows, rowIndex: rows.length - 1 };
  }
  old.video_id = videoId;
  old.frame_ids[eventIndex] = frameId;
  return { answers: rows, rowIndex };
}

export function QuickAddResult({ query, result, onChange }: {
  query: CollabQuery | null;
  result: SearchResult | null;
  onChange(query: CollabQuery): void;
}) {
  const [qaAnswer, setQaAnswer] = useState("");
  const [eventIndex, setEventIndex] = useState(0);
  const [rowIndex, setRowIndex] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setQaAnswer(""); setEventIndex(0); setRowIndex(-1); setMessage(""); setError("");
  }, [query?.id, result?.video_id, result?.frame_idx]);

  if (!query || !result || !query.round_id) return null;
  const matchingRows = query.kind === "trake"
    ? query.answers.map((answer, index) => ({ answer, index })).filter(({ answer }) => !answer.video_id || answer.video_id === result.video_id)
    : [];
  const needsNewRow = query.kind !== "trake" || rowIndex < 0;
  const atLimit = needsNewRow && query.answers.length >= 100;
  const disabled = busy || atLimit || (query.kind === "qa" && !qaAnswer.trim());

  async function add() {
    if (!query || !result || disabled) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const applied = applyExactFrame(query, query.answers, result.video_id, result.frame_idx, rowIndex, eventIndex);
      if (query.kind === "qa") {
        applied.answers[applied.rowIndex] = { ...applied.answers[applied.rowIndex], answer: qaAnswer.trim() } as AnswerDraft;
      }
      const updated = await patchQuery(query.id, { expected_version: query.version, answers: applied.answers });
      onChange(updated);
      setQaAnswer("");
      setMessage(`Đã lưu frame ${result.frame_idx} vào ${query.filename}.`);
    } catch (caught) {
      const apiError = caught as CollabApiError;
      if (apiError.current) onChange(apiError.current);
      setError(apiError.current ? "Query vừa được cập nhật trên server. Hãy kiểm tra và thử lại." : apiError.message);
    } finally { setBusy(false); }
  }

  return <section className="quick-add" aria-label="Thêm nhanh vào đáp án">
    <header><span>Query đang làm</span><strong>{query.filename}</strong></header>
    {query.kind === "qa" && <label>Câu trả lời
      <input aria-label="Câu trả lời thêm nhanh" maxLength={100} value={qaAnswer}
        onChange={(event) => setQaAnswer(event.target.value)} placeholder="Nhập câu trả lời QA…" />
    </label>}
    {query.kind === "trake" && <div className="quick-add-options">
      <label>Event<select aria-label="Event thêm nhanh" value={eventIndex} onChange={(event) => setEventIndex(Number(event.target.value))}>
        {Array.from({ length: query.event_count ?? 0 }, (_, index) => <option key={index} value={index}>E{index + 1}</option>)}
      </select></label>
      <label>Dòng<select aria-label="Dòng thêm nhanh" value={rowIndex} onChange={(event) => setRowIndex(Number(event.target.value))}>
        <option value={-1}>Dòng mới</option>
        {matchingRows.map(({ index }) => <option key={index} value={index}>#{index + 1}</option>)}
      </select></label>
    </div>}
    <button type="button" disabled={disabled} onClick={add}>{busy ? "Đang lưu…" : `Thêm frame ${result.frame_idx}`}</button>
    {atLimit && <small className="quick-add-error">Đáp án đã đủ 100 dòng.</small>}
    {error && <small className="quick-add-error" role="alert">{error}</small>}
    {message && <small className="quick-add-success" role="status">{message}</small>}
  </section>;
}

export function TeamToolbar({ rounds, round, isAdmin, onSelectRound, onRoundLoaded, onError }: {
  rounds: RoundSummary[]; round: RoundSnapshot | null; isAdmin: boolean;
  onSelectRound(id: number): void; onRoundLoaded(round: RoundSnapshot): void; onError(message: string): void;
}) {
  const [uploading, setUploading] = useState(false);
  const [roundName, setRoundName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [trakeRound, setTrakeRound] = useState<RoundSnapshot | null>(null);
  const [eventCounts, setEventCounts] = useState<Record<number, string>>({});
  const [setupError, setSetupError] = useState("");

  async function submitUpload() {
    if (!roundName.trim() || !file) return;
    try {
      const imported = await uploadRound(roundName.trim(), file);
      const trake = imported.queries.filter((query) => query.kind === "trake" && query.event_count === null);
      if (!trake.length) { finishUpload(imported); return; }
      setTrakeRound(imported);
      setEventCounts(Object.fromEntries(trake.map((query) => [query.id, ""])));
    }
    catch (error) { onError(error instanceof Error ? error.message : "Upload thất bại"); }
  }

  function finishUpload(imported: RoundSnapshot) {
    onRoundLoaded(imported); setUploading(false); setFile(null); setRoundName("");
    setTrakeRound(null); setEventCounts({}); setSetupError("");
  }

  async function saveTrakeCounts() {
    if (!trakeRound) return;
    let updatedRound = trakeRound;
    try {
      for (const query of updatedRound.queries.filter((item) => item.kind === "trake" && item.event_count === null)) {
        const eventCount = Number(eventCounts[query.id]);
        const updated = await patchQuery(query.id, { expected_version: query.version, event_count: eventCount });
        updatedRound = { ...updatedRound, queries: updatedRound.queries.map((item) => item.id === updated.id ? updated : item) };
      }
      finishUpload(updatedRound);
    } catch (error) {
      setTrakeRound(updatedRound);
      setSetupError(error instanceof Error ? error.message : "Không thể lưu số event");
    }
  }

  async function exportZip() {
    if (!round) return;
    try {
      const preview = await fetchExportPreview(round.id);
      const warningCount = preview.missing.length + preview.invalid.length + preview.unreviewed.length;
      if (warningCount && !window.confirm(
        `Còn ${preview.missing.length} câu thiếu, ${preview.invalid.length} câu sai và ${preview.unreviewed.length} câu chưa duyệt. Vẫn xuất?`
      )) return;
      const { blob, filename } = await downloadExport(round.id, warningCount > 0);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) { onError(error instanceof Error ? error.message : "Không thể xuất ZIP"); }
  }

  const trakeCountsReady = trakeRound?.queries.filter((query) => query.kind === "trake" && query.event_count === null)
    .every((query) => Number.isInteger(Number(eventCounts[query.id])) && Number(eventCounts[query.id]) >= 2 && Number(eventCounts[query.id]) <= 100);

  return <section className="team-toolbar query-toolbar" aria-label="Quản lý vòng thi">
    <label>Vòng<select aria-label="Chọn vòng" value={round?.id ?? ""}
      onChange={(event) => onSelectRound(Number(event.target.value))}>
      {!rounds.length && <option value="">Chưa có vòng</option>}
      {rounds.map((item) => <option value={item.id} key={item.id}>{item.name}{item.active ? " · active" : ""}</option>)}
    </select></label>
    {round && <div className="progress-chips">
      <span>Chưa làm {round.counts.todo}</span><span>Đang làm {round.counts.in_progress}</span>
      <span>Sẵn sàng {round.counts.ready}</span><span>Đã duyệt {round.counts.reviewed}</span>
    </div>}
    <div className="toolbar-actions">
      {isAdmin && round && !round.active && <button onClick={async () => {
        try { onRoundLoaded(await activateRound(round.id)); } catch (error) { onError(String(error)); }
      }}>Activate vòng</button>}
      {isAdmin && <button onClick={() => setUploading(!uploading)}>Upload vòng</button>}
      <button disabled={!round} onClick={exportZip}>Xuất ZIP</button>
    </div>
    {uploading && <div className="upload-popover">
      {trakeRound ? <>
        <strong>Thiết lập event TRAKE</strong><p>Nhập số event cho từng câu trước khi bắt đầu vòng.</p>
        {trakeRound.queries.filter((query) => query.kind === "trake" && query.event_count === null).map((query) => <label key={query.id}>
          <b>{query.filename}</b><span>{query.prompt}</span>
          <input aria-label={`Số event ${query.filename}`} type="number" min="2" max="100" value={eventCounts[query.id] ?? ""}
            onChange={(event) => setEventCounts((counts) => ({ ...counts, [query.id]: event.target.value }))} />
        </label>)}
        {setupError && <p className="quick-add-error" role="alert">{setupError}</p>}
        <button disabled={!trakeCountsReady}
          onClick={saveTrakeCounts}>Lưu số event & mở vòng</button>
      </> : <>
      <label>Tên vòng<input value={roundName} onChange={(event) => setRoundName(event.target.value)} /></label>
      <label>Query ZIP<input type="file" accept=".zip,application/zip" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>
      <button disabled={!roundName.trim() || !file} onClick={submitUpload}>Import</button>
      </>}
    </div>}
  </section>;
}

interface Props {
  round: RoundSnapshot;
  login: string;
  isAdmin?: boolean;
  selectedId: number | null;
  selectedResult: SearchResult | null;
  searchResults?: SearchResult[];
  openResult?: SearchResult | null;
  onCloseVideo?(): void;
  onSelect(id: number): void;
  onQueryChange(query: CollabQuery): void;
}

export function CollaborationPanels(props: Props) {
  const [mineOnly, setMineOnly] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [filterText, setFilterText] = useState("");
  const selected = props.round.queries.find((query) => query.id === props.selectedId)
    ?? props.round.queries[0] ?? null;
  const visible = useMemo(() => props.round.queries.filter((query) => (
    (!mineOnly || query.assignee === props.login)
    && (statusFilter === "all" || query.status === statusFilter)
    && (kindFilter === "all" || query.kind === kindFilter)
    && (!filterText || `${query.filename} ${query.prompt}`.toLowerCase().includes(filterText.toLowerCase()))
  )), [props.round.queries, mineOnly, statusFilter, kindFilter, filterText, props.login]);

  return (
    <>
      <aside className="query-panel" aria-label="Danh sách query">
        <header><span className="eyebrow">Team queue</span><h2>{props.round.name}</h2></header>
        <input aria-label="Tìm query" placeholder="Tìm mã hoặc nội dung…" value={filterText}
          onChange={(event) => setFilterText(event.target.value)} />
        <div className="query-filters">
          <button className={mineOnly ? "active" : ""} onClick={() => setMineOnly(!mineOnly)}>Của tôi</button>
          <select aria-label="Lọc trạng thái" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">Mọi trạng thái</option><option value="todo">Chưa làm</option>
            <option value="in_progress">Đang làm</option><option value="ready">Sẵn sàng</option>
            <option value="reviewed">Đã duyệt</option>
          </select>
          <select aria-label="Lọc loại query" value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
            <option value="all">Mọi loại</option><option value="kis">KIS</option>
            <option value="qa">QA</option><option value="trake">TRAKE</option>
          </select>
        </div>
        <div className="query-list">
          {visible.map((query) => <button key={query.id} className={query.id === selected?.id ? "active" : ""}
            onClick={() => props.onSelect(query.id)}>
            <span><strong>{query.filename}</strong><i className={`query-kind kind-${query.kind}`}>{query.kind}</i></span>
            <span><em className={`query-status status-${query.status}`}>{STATUS_LABEL[query.status]}</em>
              <small title={query.assignee ?? ""}>{shortLogin(query.assignee)}</small></span>
          </button>)}
        </div>
      </aside>
      <AnswerEditor key={selected?.id ?? "none"} query={selected} login={props.login}
        selectedResult={props.selectedResult} searchResults={props.searchResults} openResult={props.openResult} onCloseVideo={props.onCloseVideo}
        active={props.round.active} isAdmin={props.isAdmin ?? false} onChange={props.onQueryChange} />
    </>
  );
}

function AnswerEditor({ query, login, selectedResult, searchResults = [], openResult, onCloseVideo, active, isAdmin, onChange }: {
  query: CollabQuery | null; login: string; selectedResult: SearchResult | null;
  searchResults?: SearchResult[];
  openResult?: SearchResult | null; onCloseVideo?(): void; active: boolean;
  isAdmin: boolean;
  onChange(query: CollabQuery): void;
}) {
  const [current, setCurrent] = useState(query);
  const [answers, setAnswers] = useState<AnswerDraft[]>(() => structuredClone(query?.answers ?? []));
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState("");
  const [remoteChanged, setRemoteChanged] = useState(false);
  const [conflictQuery, setConflictQuery] = useState<CollabQuery | null>(null);
  const [eventSlot, setEventSlot] = useState(0);
  const [busy, setBusy] = useState(false);
  const [qaAutoAnswer, setQaAutoAnswer] = useState("");
  const [eventCount, setEventCount] = useState("");
  const [answerFrame, setAnswerFrame] = useState<{
    videoId: string; frameId: number; rowIndex: number; eventIndex: number;
  } | null>(null);

  useEffect(() => {
    if (!query) return;
    if (dirty && query.version !== current?.version) { setRemoteChanged(true); setConflictQuery(query); }
    else if (!dirty) {
      setCurrent(query);
      setAnswers(structuredClone(query.answers));
      setRemoteChanged(false);
      setConflictQuery(null);
    }
  }, [query]);

  if (!current) return <aside className="answer-panel"><p>Chưa có query trong vòng này.</p></aside>;
  const editorQuery = current;
  const trakeReady = current.kind !== "trake" || current.event_count !== null;

  function accept(updated: CollabQuery) {
    setCurrent(updated); setAnswers(structuredClone(updated.answers)); setDirty(false);
    setRemoteChanged(false); setError(""); onChange(updated);
    setConflictQuery(null);
  }

  async function mutate(action: () => Promise<CollabQuery>) {
    setBusy(true); setError("");
    try { accept(await action()); }
    catch (caught) {
      const apiError = caught as CollabApiError;
      setError(apiError.message);
      if (apiError.current) { setRemoteChanged(true); setConflictQuery(apiError.current); }
    } finally { setBusy(false); }
  }

  function updateAnswer(index: number, answer: AnswerDraft) {
    setAnswers((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...answer, auto_filled: false } as AnswerDraft : row));
    setDirty(true);
  }

  function addSelected() {
    if (!selectedResult || !current || answers.length >= 100) return;
    if (current.kind === "trake") {
      const rows = structuredClone(answers);
      if (!rows.length) rows.push(emptyAnswer(current));
      const row = rows[0] as { video_id: string; frame_ids: Array<number | null> };
      row.video_id = selectedResult.video_id;
      row.frame_ids[eventSlot] = selectedResult.frame_idx;
      setAnswers(rows);
    } else {
      const row = current.kind === "qa"
        ? { video_id: selectedResult.video_id, frame_id: selectedResult.frame_idx, answer: "" }
        : { video_id: selectedResult.video_id, frame_id: selectedResult.frame_idx };
      setAnswers((rows) => [...rows, row]);
    }
    setDirty(true);
  }

  function fillTo100() {
    if (!current || (current.kind === "qa" && !qaAutoAnswer.trim())) return;
    const reference = answerReference(current, answers);
    void mutate(async () => {
      const results = reference
        ? (await searchVideosByImage({ mode: "visual", index: "flat", nprobe: 0, ef: 64, profile: "l2026", image_weight: 1, reference }, null, new AbortController().signal)).results
        : searchResults;
      return patchQuery(current.id, { expected_version: current.version, answers: filledAnswers(current, answers, results, qaAutoAnswer) });
    });
  }

  function removeAutoFilled() {
    setAnswers((rows) => rows.filter((row) => !row.auto_filled));
    setDirty(true);
  }

  function move(index: number, direction: number) {
    const target = index + direction;
    if (target < 0 || target >= answers.length) return;
    const rows = [...answers];
    [rows[index], rows[target]] = [rows[target], rows[index]];
    setAnswers(rows); setDirty(true);
  }

  function captureFrame(frameId: number, rowIndex: number, targetEvent: number) {
    const videoId = answerFrame?.videoId ?? openResult?.video_id;
    if (!videoId) return rowIndex;
    const updated = applyExactFrame(editorQuery, answers, videoId, frameId, rowIndex, targetEvent);
    setAnswers(updated.answers);
    setDirty(true);
    return updated.rowIndex;
  }

  return <><aside className="answer-panel" aria-label="Đáp án query">
    <header><span className="eyebrow">{current.kind.toUpperCase()} answer</span><h2>{current.filename}</h2></header>
    <section className="query-prompt"><p>{current.prompt}</p></section>
    {current.kind === "trake" && !trakeReady && <section className="message warning">
      {isAdmin ? <label>Số event TRAKE
        <input aria-label="Số event TRAKE" type="number" min="2" max="100" value={eventCount}
          onChange={(event) => setEventCount(event.target.value)} />
        <button disabled={busy || !eventCount} onClick={() => mutate(() => patchQuery(current.id, {
          expected_version: current.version, event_count: Number(eventCount),
        }))}>Lưu số event</button>
      </label> : "Chờ admin đặt số event cho câu TRAKE này."}
    </section>}
    {selectedResult && <section className="selected-evidence">
      <img src={selectedResult.thumb} alt="Keyframe đang chọn" />
      <div><strong>{selectedResult.video_id} · frame {selectedResult.frame_idx}</strong>
        <small className="fusion-evidence">{selectedResult.context_only ? "Neighbouring context frame · not ranked"
          : selectedResult.source === "both"
          ? "Passage used in hybrid fusion"
          : selectedResult.source === "transcript" ? "Passage used for audio rank" : "No audio passage used for this frame rank"}
          {selectedResult.source === "both" && selectedResult.transcript_start !== undefined && selectedResult.transcript_end !== undefined
            ? ` · ${formatTimestamp(selectedResult.transcript_start)}–${formatTimestamp(selectedResult.transcript_end)}` : ""}
        </small>
        <p>{selectedResult.transcript_snippet ?? "Không có transcript cho keyframe này."}</p>
        <dl className="selected-verification">
          {!selectedResult.context_only && <div><dt>{selectedResult.source === "both" ? "Hybrid" : "Result"}</dt><dd>#{selectedResult.rank + 1}</dd></div>}
          {selectedResult.scores.pe_core_rank !== undefined && <div><dt>PE-Core</dt><dd>#{selectedResult.scores.pe_core_rank}</dd></div>}
          {selectedResult.scores.naflex_rank !== undefined && <div><dt>NaFlex</dt><dd>#{selectedResult.scores.naflex_rank}</dd></div>}
          {selectedResult.scores.audio_rank !== undefined && <div><dt>Audio</dt><dd>#{selectedResult.scores.audio_rank}</dd></div>}
          {selectedResult.scores.keyword_rank != null && <div><dt>Keyword</dt><dd>#{selectedResult.scores.keyword_rank}</dd></div>}
          {selectedResult.scores.semantic_rank != null && <div><dt>Semantic</dt><dd>#{selectedResult.scores.semantic_rank}</dd></div>}
        </dl>
        {selectedResult.matched_keywords?.length ? <small className="selected-keywords">Match: {selectedResult.matched_keywords.join(", ")}</small> : null}
      </div>
    </section>}
    <div className="ownership-row">
      <span title={current.assignee ?? ""}>Phụ trách: <strong>{shortLogin(current.assignee)}</strong></span>
      {active && <button disabled={busy || dirty} onClick={() => mutate(() => patchQuery(current.id, {
        expected_version: current.version, assignee: current.assignee === login ? null : login,
      }))}>{current.assignee === login ? "Bỏ nhận câu" : "Nhận câu"}</button>}
    </div>
    {!active && <div className="message warning">Vòng lịch sử · chỉ đọc</div>}
    {remoteChanged && <div className="message warning">Có bản cập nhật mới trên server. Bản đang sửa vẫn được giữ.
      {conflictQuery && <div className="conflict-actions">
        <button onClick={() => { setCurrent(conflictQuery); setRemoteChanged(false); setConflictQuery(null); setError(""); }}>Giữ nháp và dùng phiên bản server</button>
        <button onClick={() => accept(conflictQuery)}>Bỏ nháp và tải bản server</button>
      </div>}
    </div>}
    {error && <div className="message error" role="alert">{error}</div>}
    {active && trakeReady && selectedResult && answers.length < 100 && <div className="add-result-row">
      {current.kind === "trake" && <select aria-label="TRAKE event" value={eventSlot}
        onChange={(event) => setEventSlot(Number(event.target.value))}>
        {Array.from({ length: current.event_count ?? 0 }, (_, index) => <option key={index} value={index}>E{index + 1}</option>)}
      </select>}
      <button onClick={addSelected}>Thêm frame {selectedResult.frame_idx}</button>
    </div>}
    {active && trakeReady && (answers.length < 100 || answers.some((row) => row.auto_filled)) && <div className="add-result-row auto-fill-row">
      {answers.length < 100 && <>
        {current.kind === "qa" && <input aria-label="QA answer for auto fill" placeholder="Câu trả lời dùng chung" value={qaAutoAnswer} onChange={(event) => setQaAutoAnswer(event.target.value)} />}
        <button disabled={busy || (!answerReference(current, answers) && !searchResults.length) || (current.kind === "qa" && !qaAutoAnswer.trim())} onClick={fillTo100}>
          {busy ? "Đang tìm và lưu…" : answerReference(current, answers) ? "Tìm tương tự & fill đủ 100" : "Fill đủ 100 từ kết quả"}
        </button>
      </>}
      {answers.some((row) => row.auto_filled) && <button disabled={busy} onClick={removeAutoFilled}>Xóa kết quả đã fill</button>}
    </div>}
    <div className="answer-rows">
      {answers.map((answer, index) => <AnswerRow key={index} query={current} answer={answer} index={index}
        onUpdate={(row) => updateAnswer(index, row)} onDelete={() => { setAnswers(answers.filter((_, i) => i !== index)); setDirty(true); }}
        onMove={move} onOpenFrame={(videoId, frameId, rowIndex, eventIndex) =>
          setAnswerFrame({ videoId, frameId, rowIndex, eventIndex })} editable={active} />)}
      {active && trakeReady && answers.length < 100 && <button className="add-answer" onClick={() => { setAnswers([...answers, emptyAnswer(current)]); setDirty(true); }}>+ Thêm dòng dự đoán</button>}
    </div>
    <footer className="answer-actions">
      <span>{answers.length}/100 dòng · {STATUS_LABEL[current.status]}</span>
      {active && <button disabled={!dirty || busy || !trakeReady} onClick={() => mutate(() => patchQuery(current.id, {
        expected_version: current.version, answers,
      }))}>Lưu đáp án</button>}
      {active && (current.status === "ready" || current.status === "reviewed") && <button disabled={dirty || busy}
        onClick={() => mutate(() => reviewQuery(current.id, current.version, current.status !== "reviewed"))}>
        {current.status === "reviewed" ? "Bỏ duyệt" : "Đánh dấu đã duyệt"}
      </button>}
    </footer>
  </aside>
    {(answerFrame || openResult) && createPortal(<VideoDialog result={answerFrame ? {
      rank: 0, score: 0, video_id: answerFrame.videoId, frame_idx: answerFrame.frameId, time: 0,
      thumb: `/thumb/${answerFrame.videoId}/${answerFrame.frameId}`, frame: `/frame/${answerFrame.videoId}/${answerFrame.frameId}`,
      video: `/video/${answerFrame.videoId}`, source: "visual", scores: {},
    } : openResult!} initialFrameId={answerFrame?.frameId} initialRowIndex={answerFrame?.rowIndex}
      initialEventIndex={answerFrame?.eventIndex} startPaused
      onClose={() => answerFrame ? setAnswerFrame(null) : onCloseVideo?.()} capture={{
      query: editorQuery, answers, active, onCapture: captureFrame,
    }} />, document.body)}
  </>;
}

function AnswerRow({ query, answer, index, onUpdate, onDelete, onMove, onOpenFrame, editable }: {
  query: CollabQuery; answer: AnswerDraft; index: number;
  onUpdate(answer: AnswerDraft): void; onDelete(): void; onMove(index: number, direction: number): void;
  onOpenFrame(videoId: string, frameId: number, rowIndex: number, eventIndex: number): void; editable: boolean;
}) {
  const row = answer as unknown as Record<string, unknown>;
  return <div className={`answer-row${answer.auto_filled ? " auto-filled" : ""}`}>
    <span className="answer-rank">#{index + 1}</span>
    <label>Video<input aria-label={`Video dòng ${index + 1}`} value={String(row.video_id ?? "")}
      disabled={!editable}
      onChange={(event) => onUpdate({ ...answer, video_id: event.target.value } as AnswerDraft)} /></label>
    {query.kind === "trake" ? (row.frame_ids as Array<number | null>).map((frame, frameIndex) =>
      <label key={frameIndex}>E{frameIndex + 1}<input type="number" min="0" aria-label={`E${frameIndex + 1} dòng ${index + 1}`}
        disabled={!editable}
        value={frame ?? ""} onChange={(event) => {
          const frames = [...(row.frame_ids as Array<number | null>)];
          frames[frameIndex] = event.target.value === "" ? null : Number(event.target.value);
          onUpdate({ ...answer, frame_ids: frames } as AnswerDraft);
        }} /></label>) : <label>Frame<input type="number" min="0" aria-label={`Frame dòng ${index + 1}`}
          disabled={!editable}
          value={(row.frame_id as number | null) ?? ""} onChange={(event) => onUpdate({ ...answer,
            frame_id: event.target.value === "" ? null : Number(event.target.value),
          } as AnswerDraft)} /></label>}
    {query.kind === "qa" && <label className="answer-text">Answer<input maxLength={100} aria-label={`Answer dòng ${index + 1}`}
      disabled={!editable}
      value={String(row.answer ?? "")} onChange={(event) => onUpdate({ ...answer, answer: event.target.value } as AnswerDraft)} /></label>}
    <div className="row-actions"><button disabled={!editable} aria-label={`Đưa dòng ${index + 1} lên`} onClick={() => onMove(index, -1)}>↑</button>
      <button disabled={!editable} aria-label={`Đưa dòng ${index + 1} xuống`} onClick={() => onMove(index, 1)}>↓</button>
      <button disabled={!editable} aria-label={`Xóa dòng ${index + 1}`} onClick={onDelete}>×</button></div>
    <div className={`answer-previews${query.kind === "trake" ? " trake" : ""}`}>
      {query.kind === "trake" ? (row.frame_ids as Array<number | null>).map((frame, frameIndex) =>
        <FramePreview key={frameIndex} videoId={String(row.video_id ?? "")} frameId={frame}
          label={`E${frameIndex + 1}`} rowIndex={index}
          onOpen={(videoId, frameId) => onOpenFrame(videoId, frameId, index, frameIndex)} />)
        : <FramePreview videoId={String(row.video_id ?? "")} frameId={row.frame_id as number | null}
          rowIndex={index} onOpen={(videoId, frameId) => onOpenFrame(videoId, frameId, index, 0)} />}
    </div>
  </div>;
}

function FramePreview({ videoId, frameId, label, rowIndex, onOpen }: {
  videoId: string; frameId: number | null; label?: string; rowIndex: number;
  onOpen(videoId: string, frameId: number): void;
}) {
  const [shown, setShown] = useState({ videoId, frameId });
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  useEffect(() => {
    const timer = window.setTimeout(() => setShown({ videoId, frameId }), 400);
    return () => window.clearTimeout(timer);
  }, [videoId, frameId]);
  const valid = /^[A-Z]\d{2}_V\d{3}$/.test(shown.videoId)
    && Number.isInteger(shown.frameId) && (shown.frameId ?? -1) >= 0;
  const url = valid ? `/thumb/${shown.videoId}/${shown.frameId}?w=160` : null;
  const caption = label ? `${label} · ${shown.frameId ?? "—"}` : `Frame ${shown.frameId ?? "—"}`;
  const alt = `Preview ${label ? `${label} ` : ""}frame ${shown.frameId} dòng ${rowIndex + 1}`;
  const fallback = failedUrl === url ? "Không có ảnh" : "Chưa chọn frame";
  const fallbackLabel = `${fallback} ${label ? `${label} ` : ""}frame ${shown.frameId ?? "trống"} dòng ${rowIndex + 1}`;
  return <figure className="answer-preview">
    {url && failedUrl !== url
      ? <button type="button" aria-label={`Mở video ${shown.videoId} tại frame ${shown.frameId}`}
        onClick={() => onOpen(shown.videoId, shown.frameId!)}>
        <img src={url} alt={alt} loading="lazy" decoding="async" onError={() => setFailedUrl(url)} />
      </button>
      : <div className="answer-preview-empty" role="img" aria-label={fallbackLabel}>{fallback}</div>}
    <figcaption>{caption}</figcaption>
  </figure>;
}
