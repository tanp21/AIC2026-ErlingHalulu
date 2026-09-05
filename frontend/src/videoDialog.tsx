import { useEffect, useRef, useState } from "react";

import { fetchTranscript } from "./api";
import { fetchFrameMap, frameTime, resolveFrameId } from "./frameMap";
import type { FrameMap } from "./frameMap";
import { formatTimestamp } from "./results";
import type { AnswerDraft, CollabQuery, SearchResult, TranscriptCue } from "./types";

export interface ExactFrameCapture {
  query: CollabQuery;
  answers: AnswerDraft[];
  active: boolean;
  onCapture(frameId: number, rowIndex: number, eventIndex: number): number;
}

export function VideoDialog({ result, onClose, capture, initialFrameId, initialRowIndex, initialEventIndex, startPaused = false }: {
  result: SearchResult;
  onClose(): void;
  capture?: ExactFrameCapture;
  initialFrameId?: number;
  initialRowIndex?: number;
  initialEventIndex?: number;
  startPaused?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const seekingRef = useRef(false);
  const seekReadyRef = useRef(false);
  const requestedFrameRef = useRef<number | null>(null);
  const presentedFrameRef = useRef<number | null>(null);
  const answersRef = useRef(capture?.answers);
  const ownCaptureRef = useRef(false);
  const followedCueRef = useRef<HTMLLIElement>(null);
  const transcriptRef = useRef<HTMLElement>(null);
  const [mapping, setMapping] = useState<FrameMap | null>(null);
  const [liveFrameId, setLiveFrameId] = useState<number | null>(null);
  const [paused, setPaused] = useState(startPaused);
  const [seeking, setSeeking] = useState(false);
  const [rowIndex, setRowIndex] = useState(initialRowIndex
    ?? (capture?.query.kind === "trake" && capture.answers.length ? 0 : -1));
  const [eventIndex, setEventIndex] = useState(initialEventIndex ?? 0);
  const [message, setMessage] = useState("");
  const [messageIsError, setMessageIsError] = useState(false);
  const [cues, setCues] = useState<TranscriptCue[]>([]);
  const [transcriptLoading, setTranscriptLoading] = useState(true);
  const [transcriptError, setTranscriptError] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(result.time);
  const [metadataVideoId, setMetadataVideoId] = useState<string | null>(null);
  const [followTranscript, setFollowTranscript] = useState(true);

  useEffect(() => {
    const video = videoRef.current;
    return () => {
      if (!video) return;
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, []);

  useEffect(() => {
    if (answersRef.current === capture?.answers) return;
    answersRef.current = capture?.answers;
    if (ownCaptureRef.current) { ownCaptureRef.current = false; return; }
    setRowIndex(capture?.query.kind === "trake" && capture.answers.length ? 0 : -1);
  }, [capture?.answers, capture?.query.kind]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== "function") return;
    let active = true;
    let callbackId: number | undefined;
    fetchFrameMap(result.video_id).then((loaded) => {
      if (!active) return;
      setMapping(loaded);
      const update: VideoFrameRequestCallback = (_now, metadata) => {
        if (!active) return;
        const frameId = resolveFrameId(loaded, metadata.mediaTime);
        if (!seekingRef.current) setLiveFrameId(frameId);
        else {
          presentedFrameRef.current = frameId;
          if (seekReadyRef.current && frameId !== null
            && (requestedFrameRef.current === null || requestedFrameRef.current === frameId)) {
            setLiveFrameId(frameId);
            seekingRef.current = false;
            setSeeking(false);
          }
        }
        callbackId = video.requestVideoFrameCallback(update);
      };
      callbackId = video.requestVideoFrameCallback(update);
    }).catch(() => { setMapping(null); setLiveFrameId(null); });
    return () => {
      active = false;
      if (callbackId !== undefined) video.cancelVideoFrameCallback(callbackId);
    };
  }, [result.video_id, initialFrameId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || initialFrameId === undefined || metadataVideoId !== result.video_id
      || !mapping || mapping.videoId !== result.video_id
      || initialFrameId < 0 || initialFrameId >= mapping.frameCount) return;
    const targetTime = frameTime(mapping, initialFrameId);
    seekingRef.current = true;
    seekReadyRef.current = false;
    requestedFrameRef.current = initialFrameId;
    presentedFrameRef.current = null;
    setSeeking(true);
    setPlaybackTime(targetTime);
    video.currentTime = targetTime;
  }, [initialFrameId, mapping, metadataVideoId, result.video_id]);

  useEffect(() => {
    let active = true;
    setTranscriptLoading(true);
    setTranscriptError(false);
    setCues([]);
    fetchTranscript(result.video_id).then((response) => {
      if (!Array.isArray(response.cues)) throw new Error("Invalid transcript");
      if (active) setCues(response.cues);
    }).catch(() => {
      if (active) setTranscriptError(true);
    }).finally(() => {
      if (active) setTranscriptLoading(false);
    });
    return () => { active = false; };
  }, [result.video_id]);

  function step(delta: number) {
    const video = videoRef.current;
    if (!video || !mapping || liveFrameId === null) return;
    const target = Math.max(0, Math.min(mapping.frameCount - 1, liveFrameId + delta));
    if (target === liveFrameId) return;
    seekingRef.current = true;
    seekReadyRef.current = false;
    requestedFrameRef.current = target;
    presentedFrameRef.current = null;
    setSeeking(true);
    video.currentTime = frameTime(mapping, target);
    setMessage("");
  }

  function useFrame() {
    if (!capture || liveFrameId === null || !paused) return;
    try {
      ownCaptureRef.current = true;
      const actualRow = capture.onCapture(liveFrameId, rowIndex, eventIndex);
      if (capture.query.kind === "trake") setRowIndex(actualRow);
      setMessageIsError(false);
      setMessage(`Đã đưa frame ${liveFrameId} vào dòng #${actualRow + 1}${capture.query.kind === "trake" ? ` · E${eventIndex + 1}` : ""}`);
    } catch (error) {
      ownCaptureRef.current = false;
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Không thể dùng frame này");
    }
  }

  const canCapture = Boolean(capture?.active && paused && !seeking && liveFrameId !== null);
  const activeCue = cues.findIndex((cue) => cue.start <= playbackTime && playbackTime < cue.end);
  const nextCue = cues.findIndex((cue) => cue.start > playbackTime);
  const followedCue = activeCue >= 0 ? activeCue : nextCue >= 0 ? nextCue : Math.max(0, cues.length - 1);
  const evidenceLabel = result.context_only ? "Neighbouring context frame · not ranked"
    : result.source === "both" ? "Passage used in hybrid fusion"
    : result.source === "transcript" ? "Passage used for audio rank" : "No audio passage used for this rank";
  useEffect(() => {
    const panel = transcriptRef.current;
    const cue = followedCueRef.current;
    if (followTranscript && cues.length && panel && cue) {
      panel.scrollTo({ top: cue.offsetTop - panel.clientHeight / 2 + cue.clientHeight / 2 });
    }
  }, [followTranscript, followedCue, cues.length]);
  return (
    <div className="dialog-backdrop" role="dialog" aria-modal="true" aria-label={`Video ${result.video_id}`}
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
      onKeyDown={(event) => {
        if (event.key !== "Tab") return;
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
          "button:not(:disabled), select:not(:disabled), video, [tabindex]:not([tabindex='-1'])"
        ));
        const edge = event.shiftKey ? controls[0] : controls.at(-1);
        if (document.activeElement !== edge) return;
        event.preventDefault();
        (event.shiftKey ? controls.at(-1) : controls[0])?.focus();
      }}>
      <div className="video-dialog">
        <header><div><strong>{result.video_id}</strong><span>{formatTimestamp(playbackTime)} · {liveFrameId === null ? "Exact frame unavailable" : `frame ${liveFrameId}`}</span></div>
          <button autoFocus onClick={onClose} aria-label="Close video">×</button></header>
        <div className="video-dialog-body">
          <div className="video-pane">
        <video ref={videoRef} src={result.video} controls preload="metadata" autoPlay={!startPaused} aria-label="Video player"
          onPlay={() => { setPaused(false); setMessage(""); }} onPause={() => setPaused(true)}
          onTimeUpdate={() => setPlaybackTime(videoRef.current?.currentTime ?? 0)}
          onSeeking={() => {
            if (!seekingRef.current) {
              seekingRef.current = true;
              seekReadyRef.current = false;
              requestedFrameRef.current = null;
              presentedFrameRef.current = null;
            }
            setSeeking(true);
          }}
          onSeeked={() => {
            setPlaybackTime(videoRef.current?.currentTime ?? 0);
            seekReadyRef.current = true;
            const frameId = presentedFrameRef.current;
            if (frameId !== null && (requestedFrameRef.current === null || requestedFrameRef.current === frameId)) {
              setLiveFrameId(frameId);
              seekingRef.current = false;
              setSeeking(false);
            }
          }}
          onLoadedData={() => setPaused(videoRef.current?.paused ?? false)}
          onLoadedMetadata={() => {
            setMetadataVideoId(result.video_id);
            if (videoRef.current && initialFrameId === undefined) {
              videoRef.current.currentTime = result.time; setPlaybackTime(result.time);
            }
          }} />
        {capture && <section className="exact-frame-controls" aria-label="Chọn frame chính xác">
          <div className="frame-stepper">
            <button disabled={!paused || seeking || liveFrameId === null} aria-label="Lùi 10 frame" onClick={() => step(-10)}>−10</button>
            <button disabled={!paused || seeking || liveFrameId === null} aria-label="Lùi 1 frame" onClick={() => step(-1)}>−1</button>
            <strong>{liveFrameId === null ? "Chưa xác định frame" : `Frame ${liveFrameId}`}</strong>
            <button disabled={!paused || seeking || liveFrameId === null} aria-label="Tiến 1 frame" onClick={() => step(1)}>+1</button>
            <button disabled={!paused || seeking || liveFrameId === null} aria-label="Tiến 10 frame" onClick={() => step(10)}>+10</button>
          </div>
          <div className="capture-destination">
            <label>Dòng đáp án<select disabled={!capture.active} aria-label="Dòng đáp án" value={rowIndex} onChange={(event) => setRowIndex(Number(event.target.value))}>
              <option value={-1}>{capture.query.kind === "trake" ? "Dòng mới" : "Thêm prediction mới"}</option>
              {capture.answers.map((answer, index) => <option key={index} value={index}>Dòng #{index + 1} · {answer.video_id || "trống"}</option>)}
            </select></label>
            {capture.query.kind === "trake" && <label>Event<select disabled={!capture.active} aria-label="TRAKE event trong player" value={eventIndex}
              onChange={(event) => setEventIndex(Number(event.target.value))}>
              {Array.from({ length: capture.query.event_count ?? 0 }, (_, index) => <option key={index} value={index}>E{index + 1}</option>)}
            </select></label>}
            <button className="use-exact-frame" disabled={!canCapture} onClick={useFrame}>
              {liveFrameId === null ? "Dùng frame này" : `Dùng frame ${liveFrameId}`}
            </button>
          </div>
          {!capture.active && <p className="message warning">Vòng lịch sử · không thể ghi frame</p>}
          {message && <p className={`capture-message${messageIsError ? " error" : ""}`} role={messageIsError ? "alert" : "status"}>{message}</p>}
        </section>}
          </div>
          <aside ref={transcriptRef} className="video-transcript" aria-label="Transcript video" tabIndex={0}
            onWheel={() => setFollowTranscript(false)} onPointerDown={() => setFollowTranscript(false)}
            onKeyDown={(event) => {
              if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ", "Spacebar"].includes(event.key)) {
                setFollowTranscript(false);
              }
            }}>
            <header><span className="eyebrow">TRANSCRIPT</span><strong>Lời thoại đang phát</strong>
              {!followTranscript && <button type="button" onClick={() => setFollowTranscript(true)}>Theo video</button>}
            </header>
            <section className="video-ranking-evidence" aria-label="Ranking evidence">
              <strong>{evidenceLabel}</strong>
              {result.transcript_snippet && <p>{result.transcript_snippet}</p>}
              {result.source === "both" && result.transcript_start !== undefined && result.transcript_end !== undefined &&
                <small>{formatTimestamp(result.transcript_start)}–{formatTimestamp(result.transcript_end)}</small>}
              {!result.context_only && <small>{result.source === "both" ? "Hybrid" : "Result"} #{result.rank + 1}
                {result.scores.pe_core_rank !== undefined ? ` · PE-Core #${result.scores.pe_core_rank}` : ""}
                {result.scores.audio_rank !== undefined ? ` · Audio #${result.scores.audio_rank}` : ""}
                {result.scores.keyword_rank != null ? ` · Keyword #${result.scores.keyword_rank}` : ""}
                {result.scores.semantic_rank != null ? ` · Semantic #${result.scores.semantic_rank}` : ""}
              </small>}
              {result.matched_keywords?.length ? <small>Match: {result.matched_keywords.join(", ")}</small> : null}
            </section>
            {transcriptLoading && <p className="transcript-state">Đang tải transcript…</p>}
            {!transcriptLoading && transcriptError && <p className="transcript-state">Không tải được transcript.</p>}
            {!transcriptLoading && !transcriptError && cues.length === 0 && <p className="transcript-state">Video này không có transcript.</p>}
            {cues.length > 0 && <ol>
              {cues.map((cue, index) => <li key={`${cue.start}-${cue.text}`}
                ref={index === followedCue ? followedCueRef : undefined}
                className={index === activeCue ? "active" : ""}>
                <time>{formatTimestamp(cue.start)}</time><span>{cue.text}</span>
              </li>)}
            </ol>}
          </aside>
        </div>
      </div>
    </div>
  );
}
