import { useEffect, useRef, useState } from "react";

import { fetchKeyframeContext, fetchTranscript } from "./api";
import { dresOfficialKeyframes } from "./dresApi";
import type { OfficialKeyframe } from "./dresApi";
import { fetchFrameMapWithRetry, frameIdAt, frameTime, frameTimestampMs } from "./frameMap";
import type { FrameMap } from "./frameMap";
import { CheckPanel } from "./checkView";
import { formatTimestamp } from "./results";
import type { AnswerDraft, CheckState, CollabQuery, SearchResult, TranscriptCue } from "./types";

export interface ExactFrameCapture {
  query: CollabQuery;
  answers: AnswerDraft[];
  active: boolean;
  onCapture(frameId: number, rowIndex: number, eventIndex: number): number;
}

/**
 * Exact-frame stepping for a caller that is not the collaboration workspace: whatever the
 * buttons do with a frame (the agent's review view adds it to the candidates, or assigns
 * it to a TRAKE event) is theirs, the dialog only finds the frame and its time.
 */
/** The player's half of "Check by AI": what to check, how, and what came back. */
export interface CheckControl {
  state: CheckState;
  /** The query's words, lit inside the quotes a verdict was read from. */
  keywords: string[];
  /**
   * `frame_idx` when the player could resolve an exact one, `time` otherwise -- while a video
   * is playing `resolveFrameId` answers null off an exact PTS, and the server snaps a time to
   * the nearest indexed keyframe anyway.
   */
  run(target: { video_id: string; frame_idx?: number; time?: number }, force?: boolean): void;
}

/** The TRAKE events staged so far, and the operations the player offers on them. */
export interface TrakeSequence {
  videoId: string | null;
  frameIds: number[];
  onRemove(index: number): void;
  onMove(from: number, to: number): void;
  onOpen(frameId: number): void;
  onSubmit(): Promise<{ message: string; uncertain?: boolean; completed?: boolean }>;
}

export interface FramePicker {
  actions: Array<{ label: string; onPick(frameId: number, time: number): void }>;
  submit?: {
    label: string;
    kind?: "kis" | "qa" | "trake";
    needsAnswer?: boolean;
    onSubmit(videoId: string, frameId: number, time: number, answer: string, kind: "kis" | "qa" | "trake"): Promise<{ message: string; uncertain?: boolean; completed?: boolean }>;
  };
  /**
   * Passed by every host whose picker is in TRAKE mode. Without it the submit button falls
   * back to `Add E1` forever: the label counts these events, and an absent sequence counts 0.
   */
  trakeSequence?: TrakeSequence;
}

function VideoKeyframes({ videoId, profile, currentTime, initialFrameId, showFrameId, officialFrames, onSeek }: {
  videoId: string;
  profile?: string;
  currentTime: number;
  initialFrameId: number;
  showFrameId: boolean;
  officialFrames: OfficialKeyframe[] | null;
  onSeek(frameId: number, time?: number): void;
}) {
  const [frames, setFrames] = useState<SearchResult[]>([]);
  const [unavailable, setUnavailable] = useState(false);
  const [followFrames, setFollowFrames] = useState(true);
  const stripRef = useRef<HTMLDivElement>(null);
  const nearest = frames.length ? frames.reduce((best, frame) =>
    Math.abs(frame.time - currentTime) < Math.abs(best.time - currentTime) ? frame : best, frames[0]) : undefined;

  useEffect(() => {
    let active = true;
    if (showFrameId) {
      if (!officialFrames) {
        setFrames([]);
        setUnavailable(false);
      } else {
        setFrames(officialFrames.map((frame, rank) => ({
          rank, score: 0, video_id: videoId, frame_idx: frame.source_frame_idx,
          time: frame.timestamp_ms / 1000,
          thumb: `/thumb/${encodeURIComponent(videoId)}/${frame.source_frame_idx}`,
          frame: `/frame/${encodeURIComponent(videoId)}/${frame.source_frame_idx}`,
          video: `/video/${encodeURIComponent(videoId)}`, source: "visual", scores: {},
          context_only: true,
        })));
        setUnavailable(!officialFrames.length);
      }
      return () => { active = false; };
    }
    setFrames([]);
    setUnavailable(false);
    void fetchKeyframeContext(videoId, initialFrameId, profile, 10, true).then((data) => {
      if (!active) return;
      if (!Array.isArray(data.frames) || !data.frames.length) { setUnavailable(true); return; }
      setFrames(data.frames);
    }).catch(() => { if (active) setUnavailable(true); });
    return () => { active = false; };
  }, [videoId, profile, initialFrameId, showFrameId, officialFrames]);

  useEffect(() => {
    if (!followFrames) return;
    if (!frames.length) return;
    const strip = stripRef.current;
    const nearestIndex = frames.reduce((best, frame, index) =>
      Math.abs(frame.time - currentTime) < Math.abs(frames[best].time - currentTime) ? index : best, 0);
    const button = strip?.children[nearestIndex] as HTMLElement | undefined;
    if (strip && button && typeof strip.scrollTo === "function") {
      strip.scrollTo({ left: button.offsetLeft - strip.offsetLeft - strip.clientWidth / 2 + button.clientWidth / 2 });
    }
  }, [frames, currentTime, followFrames]);

  if (unavailable) return null;
  return <section className="video-keyframes" aria-label={showFrameId ? "Official TRAKE keyframes" : "Keyframes in this video"}>
    <div className="video-keyframe-head"><strong>{frames.length ? `${frames.length} ${showFrameId ? "official TRAKE" : "video"} keyframes` : "Keyframes"}</strong>
      {followFrames ? <span>Following video</span> : <button type="button" onClick={() => setFollowFrames(true)}>Follow video</button>}</div>
    <div ref={stripRef} className="video-keyframe-strip" tabIndex={0} aria-label="Video keyframe thumbnails"
      onWheel={() => setFollowFrames(false)} onPointerDown={() => setFollowFrames(false)}
      onKeyDown={(event) => { if (["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"].includes(event.key)) setFollowFrames(false); }}>
      {frames.map((frame, index) => <button key={`${frame.frame_idx}-${index}`} type="button"
        aria-label={showFrameId ? `Show keyframe ${frame.frame_idx} at ${formatTimestamp(frame.time)}` : `Go to ${formatTimestamp(frame.time)}`}
        aria-current={frame.frame_idx === nearest?.frame_idx ? "true" : undefined}
        onClick={() => onSeek(frame.frame_idx, showFrameId ? frame.time : undefined)}>
        <img src={`${frame.thumb}?w=160`} alt="" loading="lazy" />
        <span>{showFrameId ? `${formatTimestamp(frame.time)} · #${frame.frame_idx}` : formatTimestamp(frame.time)}</span>
      </button>)}
      {!frames.length && <span className="video-keyframe-loading">Loading keyframes…</span>}
    </div>
  </section>;
}

export function VideoDialog({ result, onClose, capture, picker, profile, originalVideo = false, initialFrameId, initialRowIndex, initialEventIndex, startPaused = false, check }: {
  result: SearchResult;
  onClose(): void;
  capture?: ExactFrameCapture;
  picker?: FramePicker;
  profile?: string;
  originalVideo?: boolean;
  initialFrameId?: number;
  initialRowIndex?: number;
  initialEventIndex?: number;
  startPaused?: boolean;
  /**
   * "Check by AI" for whatever is on screen right now. The answer renders in the LEFT pane,
   * under the video, so the person can keep watching while they read it -- the right pane is
   * the transcript and it follows playback.
   */
  check?: CheckControl;
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
  const [submitAnswer, setSubmitAnswer] = useState("");
  const [submitKind, setSubmitKind] = useState<"kis" | "qa" | "trake">(picker?.submit?.kind ?? "kis");
  const activeKind = picker?.submit?.kind ?? capture?.query.kind;
  const showFrameId = activeKind ? activeKind === "trake" : Boolean(picker?.actions.length);
  const [officialKeyframes, setOfficialKeyframes] = useState<{ videoId: string; frames: OfficialKeyframe[] } | null>(null);
  const officialFrames = officialKeyframes?.videoId === result.video_id ? officialKeyframes.frames : null;
  /**
   * The frame under the playhead, and the one number every control here names: the stepper, the
   * submit button and the payload. The decoder answers for the frame it presented; when it has not
   * named one -- while a seek settles, or right after the video opens and no callback has fired yet
   * -- the playhead time resolves to the nearest frame, so a seek that lands between two PTS ticks
   * can never leave the controls dead. A seek clears the decoder's answer, so it cannot name a frame
   * the player has already left.
   */
  const playheadFrame = mapping && mapping.videoId === result.video_id
    ? (seeking ? frameIdAt(mapping, playbackTime) : liveFrameId ?? frameIdAt(mapping, playbackTime))
    : null;
  const [submitBusy, setSubmitBusy] = useState(false);
  const [submitFinished, setSubmitFinished] = useState(false);
  const [submitUncertain, setSubmitUncertain] = useState(false);

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
    fetchFrameMapWithRetry(result.video_id).then((loaded) => {
      if (!active) return;
      setMapping(loaded);
      const update: VideoFrameRequestCallback = (_now, metadata) => {
        if (!active) return;
        // The nearest frame answers for a mediaTime between two PTS ticks: an exact match is not
        // required to name what is on screen, and requiring it left the controls stuck.
        const frameId = frameIdAt(loaded, metadata.mediaTime);
        if (!seekingRef.current) setLiveFrameId(frameId);
        else if (frameId !== null) {
          presentedFrameRef.current = frameId;
          if (seekReadyRef.current && (requestedFrameRef.current === null || requestedFrameRef.current === frameId)) {
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

  useEffect(() => { if (picker?.submit?.kind) setSubmitKind(picker.submit.kind); }, [picker?.submit?.kind]);
  useEffect(() => { setSubmitFinished(false); setSubmitUncertain(false); }, [picker?.submit?.kind, result.video_id]);

  useEffect(() => {
    if (!showFrameId) return;
    let active = true;
    void dresOfficialKeyframes(result.video_id)
      .then((frames) => { if (active) setOfficialKeyframes({ videoId: result.video_id, frames }); })
      .catch(() => { if (active) setOfficialKeyframes({ videoId: result.video_id, frames: [] }); });
    return () => { active = false; };
  }, [result.video_id, showFrameId]);

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

  function seekToFrame(target: number) {
    const video = videoRef.current;
    if (!video || !mapping || target < 0 || target >= mapping.frameCount) return;
    video.pause();
    setPaused(true);
    seekingRef.current = true;
    seekReadyRef.current = false;
    requestedFrameRef.current = target;
    presentedFrameRef.current = null;
    setLiveFrameId(null);
    setSeeking(true);
    const time = frameTime(mapping, target);
    video.currentTime = time;
    setPlaybackTime(time);
    setMessage("");
  }

  function seekToTime(time: number) {
    const video = videoRef.current;
    if (!video || !Number.isFinite(time) || time < 0) return;
    video.pause();
    setPaused(true);
    seekingRef.current = true;
    seekReadyRef.current = false;
    requestedFrameRef.current = null;
    presentedFrameRef.current = null;
    setLiveFrameId(null);
    setSeeking(true);
    video.currentTime = time;
    setPlaybackTime(time);
    setMessage("");
  }

  function step(delta: number) {
    if (!mapping || playheadFrame === null) return;
    const target = Math.max(0, Math.min(mapping.frameCount - 1, playheadFrame + delta));
    if (target !== playheadFrame) seekToFrame(target);
  }

  function useFrame() {
    if (!capture || playheadFrame === null || !paused) return;
    try {
      ownCaptureRef.current = true;
      const actualRow = capture.onCapture(playheadFrame, rowIndex, eventIndex);
      if (capture.query.kind === "trake") setRowIndex(actualRow);
      setMessageIsError(false);
      setMessage(`Placed frame ${playheadFrame} into row #${actualRow + 1}${capture.query.kind === "trake" ? ` · E${eventIndex + 1}` : ""}`);
    } catch (error) {
      ownCaptureRef.current = false;
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Could not use this frame");
    }
  }

  const canCapture = Boolean(capture?.active && paused && !seeking && playheadFrame !== null);
  const canPick = submitKind === "trake"
    ? Boolean(picker?.submit && playheadFrame !== null)
    : Boolean(picker && paused && !seeking && playheadFrame !== null);
  /**
   * Why the submit button is dimmed, in one line. An empty Q&A answer and a still-playing video both
   * gate it silently, and the button on its own does not say which one is holding it -- the answer
   * field sits next to it, unlabelled enough that people try to submit without filling it.
   */
  const submitHint = !picker?.submit || submitBusy || submitFinished || submitUncertain ? ""
    : submitKind === "qa" && !submitAnswer.trim() ? "Type the answer to submit it."
    : submitKind !== "trake" && !paused ? "Pause the video on the frame you want to submit."
    : submitKind !== "trake" && (seeking || playheadFrame === null) ? "Waiting for the exact frame…"
    : "";
  async function submitLiveFrame() {
    const action = picker?.submit;
    if (!action || !canPick || submitBusy || submitFinished || submitUncertain) return;
    // The frame the button names is the frame it sends: same value, same render.
    const frame = playheadFrame;
    if (frame === null || !mapping) return;
    setSubmitBusy(true); setMessage("");
    try {
      const outcome = await action.onSubmit(result.video_id, frame, frameTime(mapping, frame), submitAnswer, submitKind);
      setMessage(outcome.message); setMessageIsError(outcome.uncertain ?? false);
      setSubmitUncertain(Boolean(outcome.uncertain)); setSubmitFinished(outcome.completed ?? !outcome.uncertain);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "DRES submission failed; check status before retrying.");
      setSubmitUncertain(true);
    } finally { setSubmitBusy(false); }
  }
  async function submitTrakeSequence() {
    const sequence = picker?.trakeSequence;
    if (!sequence?.frameIds.length || submitBusy || submitFinished || submitUncertain) return;
    setSubmitBusy(true); setMessage("");
    try {
      const outcome = await sequence.onSubmit();
      setMessage(outcome.message); setMessageIsError(outcome.uncertain ?? false);
      setSubmitUncertain(Boolean(outcome.uncertain)); setSubmitFinished(Boolean(outcome.completed));
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "DRES submission failed; check status before retrying.");
      setSubmitUncertain(true);
    } finally { setSubmitBusy(false); }
  }
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
        <header><div><strong>{result.video_id}</strong><span>{formatTimestamp(playbackTime)}{showFrameId
          ? ` · ${playheadFrame === null ? "Exact frame unavailable" : `frame ${playheadFrame}`}` : ""}</span></div>
          <button autoFocus onClick={onClose} aria-label="Close video">×</button></header>
        <div className="video-dialog-body">
          <div className="video-pane">
        <video ref={videoRef} src={originalVideo ? `/video/${encodeURIComponent(result.video_id)}?original=1` : result.video}
          controls preload="metadata" autoPlay={!startPaused} aria-label="Video player"
          onPlay={() => { setPaused(false); setMessage(""); }}
          // `timeupdate` only fires a few times a second; re-read the playhead when it stops so the
          // frame the button names is the frame the video is stopped on.
          onPause={() => { setPaused(true); setPlaybackTime(videoRef.current?.currentTime ?? playbackTime); }}
          onTimeUpdate={() => setPlaybackTime(videoRef.current?.currentTime ?? 0)}
          onSeeking={() => {
            if (!seekingRef.current) {
              seekingRef.current = true;
              seekReadyRef.current = false;
              requestedFrameRef.current = null;
              presentedFrameRef.current = null;
              // The decoder's frame is the one the player just left, not the one the seek is bound for.
              setLiveFrameId(null);
            }
            setSeeking(true);
          }}
          onSeeked={() => {
            setPlaybackTime(videoRef.current?.currentTime ?? 0);
            seekReadyRef.current = true;
            const frameId = presentedFrameRef.current;
            if (frameId !== null && (requestedFrameRef.current === null || requestedFrameRef.current === frameId))
              setLiveFrameId(frameId);
            // The seek is over whether or not a callback has named the frame it landed on: the
            // playhead time answers for it, so the controls must not wait on the decoder.
            seekingRef.current = false;
            setSeeking(false);
          }}
          onLoadedData={() => setPaused(videoRef.current?.paused ?? false)}
          onLoadedMetadata={() => {
            setMetadataVideoId(result.video_id);
            if (videoRef.current && initialFrameId === undefined) {
              videoRef.current.currentTime = result.time; setPlaybackTime(result.time);
            }
          }} />
        {check && <section className="video-check" aria-label="AI check">
          <div className="video-check-actions">
            <button type="button" className="check-button" disabled={check.state.status === "loading"}
              onClick={() => check.run(liveFrameId !== null && !seeking
                ? { video_id: result.video_id, frame_idx: liveFrameId }
                : { video_id: result.video_id, time: playbackTime })}>
              {check.state.status === "loading" ? "Checking…" : "✓ Check this moment"}
            </button>
            <small>{liveFrameId !== null && !seeking
              ? `frame ${liveFrameId}` : formatTimestamp(playbackTime)} · ±5 min around it</small>
          </div>
          <CheckPanel state={check.state} variant="inline" keywords={check.keywords}
            onSeek={(time) => { if (videoRef.current) videoRef.current.currentTime = time; }}
            onRecheck={() => check.run(liveFrameId !== null && !seeking
              ? { video_id: result.video_id, frame_idx: liveFrameId }
              : { video_id: result.video_id, time: playbackTime }, true)} />
        </section>}
        {mapping && <VideoKeyframes key={`${result.video_id}-${profile ?? "default"}`} videoId={result.video_id} profile={profile} currentTime={playbackTime}
          initialFrameId={result.frame_idx} showFrameId={showFrameId} officialFrames={officialFrames}
          onSeek={(frameId, time) => time === undefined ? seekToFrame(frameId) : seekToTime(time)} />}
        {(capture || picker) && <section className="exact-frame-controls" aria-label="Select exact frame">
          {showFrameId && <div className="frame-stepper">
            <button disabled={!paused || seeking || playheadFrame === null} aria-label="Back 10 frames" onClick={() => step(-10)}>−10</button>
            <button disabled={!paused || seeking || playheadFrame === null} aria-label="Back 1 frame" onClick={() => step(-1)}>−1</button>
            <strong>{playheadFrame === null ? "Frame not determined" : `Source frame ${playheadFrame}`}</strong>
            <button disabled={!paused || seeking || playheadFrame === null} aria-label="Forward 1 frame" onClick={() => step(1)}>+1</button>
            <button disabled={!paused || seeking || playheadFrame === null} aria-label="Forward 10 frames" onClick={() => step(10)}>+10</button>
          </div>}
          {capture && !picker?.submit && <div className="capture-destination">
            <label>Answer row<select disabled={!capture.active} aria-label="Answer row" value={rowIndex} onChange={(event) => setRowIndex(Number(event.target.value))}>
              <option value={-1}>{capture.query.kind === "trake" ? "New row" : "Add new prediction"}</option>
              {capture.answers.map((answer, index) => <option key={index} value={index}>Row #{index + 1} · {answer.video_id || "empty"}</option>)}
            </select></label>
            {capture.query.kind === "trake" && <label>Event<select disabled={!capture.active} aria-label="TRAKE event in player" value={eventIndex}
              onChange={(event) => setEventIndex(Number(event.target.value))}>
              {Array.from({ length: capture.query.event_count ?? 0 }, (_, index) => <option key={index} value={index}>E{index + 1}</option>)}
            </select></label>}
            <button className="use-exact-frame" disabled={!canCapture} onClick={useFrame}>
              {playheadFrame === null ? "Use this frame" : `Use frame ${playheadFrame}`}
            </button>
          </div>}
          {picker?.submit && <div className="capture-destination dres-player-submit">
            {submitKind === "qa" && <label>Answer<input aria-label="DRES answer in player" value={submitAnswer}
              onChange={(event) => setSubmitAnswer(event.target.value)} /></label>}
            <button type="button" className="use-exact-frame dres-submit" disabled={!canPick || submitBusy || submitFinished || submitUncertain || submitKind === "qa" && !submitAnswer.trim()
              || submitKind === "trake" && Boolean(picker.trakeSequence?.frameIds.length && picker.trakeSequence.videoId !== result.video_id)}
              onClick={submitLiveFrame}>{submitBusy ? "Working…" : submitUncertain ? "Submission uncertain · check DRES" : submitFinished ? "Submission sent"
                : submitKind === "trake" && playheadFrame !== null && mapping
                  ? `Add E${(picker.trakeSequence?.frameIds.length ?? 0) + 1} · #${playheadFrame} · ${formatTimestamp(frameTime(mapping, playheadFrame))}`
                : submitKind !== "trake" && playheadFrame !== null && mapping
                  ? `Submit ${formatTimestamp(frameTimestampMs(mapping, playheadFrame) / 1000)} (${frameTimestampMs(mapping, playheadFrame)} ms)`
                  : picker.submit.label}</button>
          </div>}
          {submitHint && <small className="dres-submit-hint" role="status">{submitHint}</small>}
          {submitKind === "trake" && picker?.trakeSequence && <div className="dres-player-events">
            <span>{picker.trakeSequence.frameIds.length} TRAKE events selected</span>
            {picker.trakeSequence.videoId && picker.trakeSequence.videoId !== result.video_id &&
              <small>Sequence uses {picker.trakeSequence.videoId}. Remove its events to switch videos.</small>}
            {officialFrames?.length === 0 && <small>No organizer keyframe strip for this video.</small>}
            {!!picker.trakeSequence.frameIds.length && <ol>{picker.trakeSequence.frameIds.map((frameId, index) => <li key={`${frameId}-${index}`}>
              <button type="button" onClick={() => {
                if (picker.trakeSequence?.videoId === result.video_id) seekToFrame(frameId);
                else picker.trakeSequence?.onOpen(frameId);
              }}>E{index + 1} · #{frameId}</button>
              <button type="button" aria-label={`Move TRAKE event E${index + 1} up`} disabled={index === 0}
                onClick={() => picker.trakeSequence?.onMove(index, index - 1)}>↑</button>
              <button type="button" aria-label={`Move TRAKE event E${index + 1} down`} disabled={index === picker.trakeSequence!.frameIds.length - 1}
                onClick={() => picker.trakeSequence?.onMove(index, index + 1)}>↓</button>
              <button type="button" aria-label={`Remove TRAKE event E${index + 1}`}
                onClick={() => picker.trakeSequence?.onRemove(index)}>×</button>
            </li>)}</ol>}
            <button type="button" className="dres-trake-submit" disabled={!picker.trakeSequence.frameIds.length || submitBusy || submitFinished || submitUncertain}
              onClick={() => void submitTrakeSequence()}>Submit TRAKE sequence ({picker.trakeSequence.frameIds.length})</button>
          </div>}
          {capture && !picker?.submit && !capture.active && <p className="message warning">Historical round · cannot record frame</p>}
          {message && <p className={`capture-message${messageIsError ? " error" : ""}`} role={messageIsError ? "alert" : "status"}>{message}</p>}
        </section>}
          </div>
          <aside ref={transcriptRef} className="video-transcript" aria-label="Video transcript" tabIndex={0}
            onWheel={() => setFollowTranscript(false)} onPointerDown={() => setFollowTranscript(false)}
            onKeyDown={(event) => {
              if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ", "Spacebar"].includes(event.key)) {
                setFollowTranscript(false);
              }
            }}>
            <header><span className="eyebrow">TRANSCRIPT</span><strong>Currently playing</strong>
              {!followTranscript && <button type="button" onClick={() => setFollowTranscript(true)}>Follow video</button>}
            </header>
            {!picker && <section className="video-ranking-evidence" aria-label="Ranking evidence">
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
            </section>}
            {transcriptLoading && <p className="transcript-state">Loading transcript…</p>}
            {!transcriptLoading && transcriptError && <p className="transcript-state">Failed to load transcript.</p>}
            {!transcriptLoading && !transcriptError && cues.length === 0 && <p className="transcript-state">This video has no transcript.</p>}
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
