import { useEffect, useMemo, useRef, useState } from "react";

import { fetchCaptions, fetchTopicSegments, fetchTranscript } from "./api";
import { copyKeyframe, downloadClip, fetchObjectDetections, fetchShots, fetchVideoList, updateShotLabels } from "./clipApi";
import { fetchFrameMap, frameTime, resolveFrameId } from "./frameMap";
import type { FrameMap } from "./frameMap";
import { formatTimestamp } from "./results";
import type { CaptionWindow, KeyframeSource, ObjectDetectionFrame, Shot, ShotKeyframe, ShotLabelReason, TopicSegment, TranscriptCue, VideoSummary } from "./types";

export type AppTab = "search" | "queries" | "agent" | "clips";

const TAB_LABELS: Record<AppTab, string> = {
  search: "Search", queries: "Query management", agent: "Retrieval agent", clips: "Clip Studio",
};

export function AppTabs({ tab, onTab }: { tab: AppTab; onTab(next: AppTab): void }) {
  return <nav className="app-tabs" aria-label="Work mode">
    {(["search", "queries", "agent", "clips"] as AppTab[]).map((name) => <button key={name}
      className={name === tab ? "active" : ""} aria-current={name === tab ? "page" : undefined}
      aria-label={name === "search" ? "Search tab" : undefined}
      onClick={() => onTab(name)}>{TAB_LABELS[name]}</button>)}
  </nav>;
}

/** Index of the shot containing `frameId`, or -1 when it falls in a gap. */
export function shotAt(shots: Shot[], frameId: number | null): number {
  if (frameId === null) return -1;
  let low = 0;
  let high = shots.length - 1;
  let candidate = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (shots[middle].startFrame <= frameId) { candidate = middle; low = middle + 1; }
    else high = middle - 1;
  }
  return candidate >= 0 && frameId <= shots[candidate].endFrame ? candidate : -1;
}

export function cuesForShot(cues: TranscriptCue[], shot: Shot | null): TranscriptCue[] {
  return shot ? cues.filter((cue) => cue.start < shot.end && cue.end > shot.start) : [];
}

/**
 * The caption window holding `time`, by the second. A shot over 20 s is split into
 * overlapping windows, so the deepest one wins — the same rule `caption_for_frame`
 * uses on the server.
 */
export function captionAt(windows: CaptionWindow[], time: number): CaptionWindow | null {
  if (!windows.length) return null;
  const ordered = [...windows].sort((a, b) => a.start - b.start);
  const covering = ordered.filter((window) => window.start <= time && time <= window.end);
  if (covering.length) {
    return covering.reduce((best, window) =>
      Math.min(time - window.start, window.end - time) > Math.min(time - best.start, best.end - time)
        ? window : best);
  }
  return ordered.reduce((best, window) =>
    Math.abs((window.start + window.end) / 2 - time) < Math.abs((best.start + best.end) / 2 - time)
      ? window : best);
}

/** Nearest detector sample, held for at most 1.5 s between extracted frames. */
export function detectionsAt(frames: ObjectDetectionFrame[], time: number): ObjectDetectionFrame | null {
  if (!frames.length || !Number.isFinite(time)) return null;
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (frames[middle].time < time) low = middle + 1;
    else high = middle;
  }
  const after = frames[Math.min(low, frames.length - 1)];
  const before = frames[Math.max(0, low - 1)];
  const nearest = Math.abs(after.time - time) < Math.abs(before.time - time) ? after : before;
  return Math.abs(nearest.time - time) <= 1.5 ? nearest : null;
}

export function containedVideoRect(width: number, height: number, videoWidth: number, videoHeight: number) {
  if (![width, height, videoWidth, videoHeight].every(Number.isFinite)
      || width <= 0 || height <= 0 || videoWidth <= 0 || videoHeight <= 0) return null;
  const scale = Math.min(width / videoWidth, height / videoHeight);
  const contentWidth = videoWidth * scale;
  const contentHeight = videoHeight * scale;
  return { left: (width - contentWidth) / 2, top: (height - contentHeight) / 2,
    width: contentWidth, height: contentHeight };
}

/** The caption of the shot under the playhead, in full, with a copy button. */
function ShotCaption({ ariaLabel, shot, caption, loading, error, empty }: {
  ariaLabel: string;
  shot: Shot | null;
  caption: CaptionWindow | null;
  loading: boolean;
  error: boolean;
  /** The video itself was never captioned (2025 `K*`, or a 2026 shot pass that has not reached it). */
  empty: boolean;
}) {
  const [copy, setCopy] = useState<{ text: string; state: "copied" | "error" } | null>(null);
  const copyState = copy && copy.text === caption?.text ? copy.state : "idle";

  async function copyCaption() {
    if (!caption) return;
    try {
      if (!navigator.clipboard?.writeText) throw new Error();
      await navigator.clipboard.writeText(caption.text);
      setCopy({ text: caption.text, state: "copied" });
    } catch {
      setCopy({ text: caption.text, state: "error" });
    }
  }

  return <section className="shot-caption" aria-label={ariaLabel}>
    <header><span className="eyebrow">Caption</span>
      <div className="shot-caption-meta">
        {caption && <strong>Shot #{shot?.index}{caption.window_count && caption.window_count > 1
          ? ` · window ${caption.window_idx + 1}/${caption.window_count}` : ""}</strong>}
        <button disabled={!caption} onClick={() => void copyCaption()}>
          {copyState === "copied" ? "Copied" : copyState === "error" ? "Copy failed" : "Copy"}
        </button>
      </div>
    </header>
    {loading && <p className="clip-state">Loading caption…</p>}
    {!loading && error && <p className="clip-state error">Failed to load caption.</p>}
    {!loading && !error && empty && <p className="clip-state">This video has no captions.</p>}
    {!loading && !error && !empty && !shot && <p className="clip-state">No shot under the playhead.</p>}
    {!loading && !error && !empty && shot && !caption && <p className="clip-state">This shot has no caption.</p>}
    {!loading && !error && caption && <p className="shot-caption-text">{caption.text}</p>}
  </section>;
}

export function ClipStudio({ profile, isAdmin = false }: { profile?: string; isAdmin?: boolean }) {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [group, setGroup] = useState("all");
  const [videoId, setVideoId] = useState<string | null>(null);
  const [shots, setShots] = useState<Shot[]>([]);
  const [shotsError, setShotsError] = useState("");
  const [shotsLoading, setShotsLoading] = useState(false);
  const [listError, setListError] = useState("");
  const [sourcePreference, setSourcePreference] = useState(profile);
  const [keyframeSource, setKeyframeSource] = useState<string | null>(null);
  const [keyframeSources, setKeyframeSources] = useState<KeyframeSource[]>([]);
  const [labelRevision, setLabelRevision] = useState(0);
  const [labelFilter, setLabelFilter] = useState("all");
  const [labelReason, setLabelReason] = useState<ShotLabelReason>("presenter");
  const [savingLabel, setSavingLabel] = useState(false);
  const [mapping, setMapping] = useState<FrameMap | null>(null);
  const [liveFrameId, setLiveFrameId] = useState<number | null>(null);
  const [paused, setPaused] = useState(true);
  const [seeking, setSeeking] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [copyingFrame, setCopyingFrame] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [messageIsError, setMessageIsError] = useState(false);
  const [cues, setCues] = useState<TranscriptCue[]>([]);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptError, setTranscriptError] = useState(false);
  const [captions, setCaptions] = useState<CaptionWindow[]>([]);
  const [captionLoading, setCaptionLoading] = useState(false);
  const [captionError, setCaptionError] = useState(false);
  const [topicSegments, setTopicSegments] = useState<TopicSegment[]>([]);
  const [topicLoading, setTopicLoading] = useState(false);
  const [topicError, setTopicError] = useState(false);
  const [objectFrames, setObjectFrames] = useState<ObjectDetectionFrame[]>([]);
  const [objectLoading, setObjectLoading] = useState(false);
  const [objectError, setObjectError] = useState(false);
  const [showObjectOverlay, setShowObjectOverlay] = useState(true);
  const [objectConfidence, setObjectConfidence] = useState(0.35);
  const [videoContentRect, setVideoContentRect] = useState<ReturnType<typeof containedVideoRect>>(null);
  const [videoFullscreen, setVideoFullscreen] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [followTranscript, setFollowTranscript] = useState(true);
  const [transcriptCopyState, setTranscriptCopyState] = useState<"idle" | "copied" | "error">("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoStageRef = useRef<HTMLDivElement>(null);
  const seekingRef = useRef(false);
  const activeShotRef = useRef<HTMLLIElement>(null);
  const followedCueRef = useRef<HTMLLIElement>(null);
  const transcriptRef = useRef<HTMLElement>(null);

  useEffect(() => { setSourcePreference(profile); }, [profile]);

  useEffect(() => {
    fetchVideoList()
      .then(setVideos)
      .catch((error) => setListError(error instanceof Error ? error.message : "Failed to load video list"));
  }, []);

  useEffect(() => {
    if (!videoId) return;
    let active = true;
    setShots([]);
    setShotsError("");
    setShotsLoading(true);
    setLiveFrameId(null);
    setMessage("");
    fetchShots(videoId, sourcePreference).then((response) => {
      if (!active) return;
      setShots(response.shots);
      setLabelRevision(response.labelRevision ?? 0);
      setKeyframeSource(response.keyframeSource);
      setKeyframeSources(response.keyframeSources);
    }).catch((error) => {
      if (active) setShotsError(error instanceof Error ? error.message : "Failed to load shots");
    }).finally(() => {
      if (active) setShotsLoading(false);
    });
    return () => { active = false; };
  }, [videoId, sourcePreference]);

  useEffect(() => {
    if (!videoId) return;
    let active = true;
    setObjectFrames([]);
    setObjectLoading(true);
    setObjectError(false);
    fetchObjectDetections(videoId).then((response) => {
      if (active) setObjectFrames(response.frames ?? []);
    }).catch(() => {
      if (active) setObjectError(true);
    }).finally(() => {
      if (active) setObjectLoading(false);
    });
    return () => { active = false; };
  }, [videoId]);

  useEffect(() => {
    if (!videoId) return;
    let active = true;
    setCues([]);
    setTranscriptCopyState("idle");
    setTranscriptLoading(true);
    setTranscriptError(false);
    fetchTranscript(videoId).then((response) => {
      if (active) setCues(response.cues);
    }).catch(() => {
      if (active) setTranscriptError(true);
    }).finally(() => {
      if (active) setTranscriptLoading(false);
    });
    return () => { active = false; };
  }, [videoId]);

  useEffect(() => {
    if (!videoId) return;
    let active = true;
    setCaptions([]);
    setCaptionLoading(true);
    setCaptionError(false);
    fetchCaptions(videoId).then((response) => {
      if (active) setCaptions(response.captions ?? []);
    }).catch(() => {
      if (active) setCaptionError(true);
    }).finally(() => {
      if (active) setCaptionLoading(false);
    });
    return () => { active = false; };
  }, [videoId]);

  useEffect(() => {
    if (!videoId) return;
    let active = true;
    setTopicSegments([]);
    setTopicLoading(true);
    setTopicError(false);
    fetchTopicSegments(videoId).then((response) => {
      if (active) setTopicSegments(response.segments ?? []);
    }).catch(() => {
      if (active) setTopicError(true);
    }).finally(() => {
      if (active) setTopicLoading(false);
    });
    return () => { active = false; };
  }, [videoId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!videoId || !video || typeof video.requestVideoFrameCallback !== "function") return;
    let active = true;
    let callbackId: number | undefined;
    fetchFrameMap(videoId).then((loaded) => {
      if (!active) return;
      setMapping(loaded);
      const update: VideoFrameRequestCallback = (_now, metadata) => {
        if (!active) return;
        const frameId = resolveFrameId(loaded, metadata.mediaTime);
        if (frameId !== null) setLiveFrameId(frameId);
        if (seekingRef.current && frameId !== null) { seekingRef.current = false; setSeeking(false); }
        callbackId = video.requestVideoFrameCallback(update);
      };
      callbackId = video.requestVideoFrameCallback(update);
    }).catch(() => { setMapping(null); setLiveFrameId(null); });
    return () => {
      active = false;
      if (callbackId !== undefined) video.cancelVideoFrameCallback(callbackId);
    };
  }, [videoId]);

  useEffect(() => {
    const video = videoRef.current;
    const stage = videoStageRef.current;
    if (!videoId || !video || !stage) return;
    const update = () => {
      const bounds = stage.getBoundingClientRect();
      setVideoContentRect(containedVideoRect(bounds.width, bounds.height, video.videoWidth, video.videoHeight));
      setVideoFullscreen(document.fullscreenElement === stage);
    };
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(stage);
    window.addEventListener("resize", update);
    document.addEventListener("fullscreenchange", update);
    update();
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", update);
      document.removeEventListener("fullscreenchange", update);
    };
  }, [videoId]);

  const groups = useMemo(
    () => [...new Set(videos.map((video) => video.group))].sort(),
    [videos],
  );
  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return videos.filter((video) => (
      (group === "all" || video.group === group)
      && (!needle || video.video_id.toLowerCase().includes(needle))
    ));
  }, [videos, filter, group]);

  const currentIndex = shotAt(shots, liveFrameId);
  const currentShot = currentIndex >= 0 ? shots[currentIndex] : null;
  // Caption windows are measured on the source clock, so prefer the playhead frame
  // over `timeupdate`, which only ticks a few times a second.
  const playheadSeconds = liveFrameId !== null && mapping ? frameTime(mapping, liveFrameId) : playbackTime;
  const objectFrame = useMemo(() => detectionsAt(objectFrames, playheadSeconds), [objectFrames, playheadSeconds]);
  const visibleObjects = objectFrame?.detections.filter((detection) => detection.confidence >= objectConfidence) ?? [];
  const shotCaption = useMemo(() => captionAt(
    currentShot ? captions.filter((caption) => caption.shot_idx === currentShot.index) : [],
    playheadSeconds,
  ), [captions, currentShot, playheadSeconds]);
  const displayedShots = useMemo(() => shots.filter((shot) => (
    labelFilter === "all" || (labelFilter === "unlabelled" ? !shot.label : shot.label?.state === labelFilter)
  )), [shots, labelFilter]);
  const labelCounts = useMemo(() => ({
    suggested: shots.filter((shot) => shot.label?.state === "suggested").length,
    excluded: shots.filter((shot) => shot.label?.state === "excluded").length,
    included: shots.filter((shot) => shot.label?.state === "included").length,
  }), [shots]);
  const shotCues = useMemo(() => cuesForShot(cues, currentShot), [cues, currentShot]);
  const currentTopicSegmentIndex = topicSegments.findIndex(
    (segment) => segment.start <= playheadSeconds && playheadSeconds < segment.end,
  );
  const fullTranscript = useMemo(
    () => cues.map((cue) => cue.text).join(" "),
    [cues],
  );
  const activeCue = cues.findIndex((cue) => cue.start <= playbackTime && playbackTime < cue.end);
  const nextCue = cues.findIndex((cue) => cue.start > playbackTime);
  const followedCue = activeCue >= 0 ? activeCue : nextCue >= 0 ? nextCue : Math.max(0, cues.length - 1);

  useEffect(() => {
    activeShotRef.current?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [currentIndex]);

  useEffect(() => {
    const panel = transcriptRef.current;
    const cue = followedCueRef.current;
    if (followTranscript && cues.length && panel && cue) {
      panel.scrollTo?.({ top: cue.offsetTop - panel.clientHeight / 2 + cue.clientHeight / 2 });
    }
  }, [followTranscript, followedCue, cues.length]);

  useEffect(() => {
    if (currentShot?.label?.reason) setLabelReason(currentShot.label.reason);
  }, [currentShot?.index, currentShot?.label?.reason]);

  function seekFrame(frameId: number) {
    const video = videoRef.current;
    if (!video || !mapping) return;
    seekingRef.current = true;
    setSeeking(true);
    setLiveFrameId(frameId);
    const target = frameTime(mapping, frameId);
    video.currentTime = target;
    setPlaybackTime(target);
    setMessage("");
  }

  function seekToShot(shot: Shot) { seekFrame(shot.startFrame); }

  function stepShot(delta: number) {
    const base = currentIndex >= 0 ? currentIndex : 0;
    const target = shots[Math.max(0, Math.min(shots.length - 1, base + delta))];
    if (target) seekToShot(target);
  }

  function seekCue(cue: TranscriptCue) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = cue.start;
    setPlaybackTime(cue.start);
    setFollowTranscript(true);
  }

  function seekTopicSegment(segment: TopicSegment) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = segment.start;
    setPlaybackTime(segment.start);
  }

  async function exportCurrentShot() {
    if (!videoId || !currentShot) return;
    setExporting(true);
    setMessage("");
    try {
      const filename = await downloadClip(videoId, currentShot.index);
      setMessageIsError(false);
      setMessage(`Downloaded ${filename}`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Failed to export clip");
    } finally {
      setExporting(false);
    }
  }

  async function copyFrame(keyframe: ShotKeyframe) {
    setCopyingFrame(keyframe.frameId);
    setMessage("");
    try {
      await copyKeyframe(keyframe.frame);
      setMessageIsError(false);
      setMessage(`Copied keyframe ${keyframe.frameId}`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Failed to copy keyframe");
    } finally {
      setCopyingFrame(null);
    }
  }

  async function copyTranscript() {
    try {
      if (!navigator.clipboard?.writeText) throw new Error();
      await navigator.clipboard.writeText(fullTranscript);
      setTranscriptCopyState("copied");
    } catch {
      setTranscriptCopyState("error");
    }
  }

  async function reviewShot(state: "excluded" | "included" | null) {
    if (!videoId || !currentShot || !isAdmin) return;
    setSavingLabel(true);
    setMessage("");
    try {
      const response = await updateShotLabels(videoId, labelRevision, [{
        shot_index: currentShot.index, state, reason: state ? labelReason : undefined,
      }]);
      setLabelRevision(response.labelRevision);
      setShots((items) => items.map((shot) => ({
        ...shot, label: response.labels[String(shot.index)] ?? null,
      })));
      setMessageIsError(false);
      setMessage(state === "excluded" ? "Marked as excluded from index"
        : state === "included" ? "Confirmed kept in index" : "Label removed");
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Failed to save shot label");
    } finally {
      setSavingLabel(false);
    }
  }

  function nextSuggestion() {
    if (!shots.length) return;
    for (let offset = 1; offset <= shots.length; offset += 1) {
      const candidate = shots[(Math.max(currentIndex, -1) + offset) % shots.length];
      if (candidate.label?.state === "suggested") { seekToShot(candidate); return; }
    }
  }

  async function reviewAllSuggestions(state: "excluded" | "included") {
    if (!videoId || !isAdmin) return;
    const candidates = shots.filter((shot) => shot.label?.state === "suggested");
    if (!candidates.length || !window.confirm(
      `${state === "excluded" ? "Exclude" : "Keep"} ${candidates.length} suggestions for ${videoId}?`,
    )) return;
    setSavingLabel(true);
    try {
      const response = await updateShotLabels(videoId, labelRevision, candidates.map((shot) => ({
        shot_index: shot.index, state, reason: shot.label!.reason,
      })));
      setLabelRevision(response.labelRevision);
      setShots((items) => items.map((shot) => ({
        ...shot, label: response.labels[String(shot.index)] ?? null,
      })));
      setMessageIsError(false);
      setMessage(`Reviewed ${candidates.length} suggestions`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Failed to save shot label");
    } finally {
      setSavingLabel(false);
    }
  }

  const canExport = Boolean(videoId && currentShot && paused && !seeking && !exporting);

  async function toggleVideoFullscreen() {
    const stage = videoStageRef.current;
    if (!stage) return;
    try {
      if (document.fullscreenElement === stage) await document.exitFullscreen();
      else await stage.requestFullscreen();
    } catch {
      setMessageIsError(true);
      setMessage("Fullscreen is unavailable in this browser.");
    }
  }

  return <section className="clip-studio">
    <aside className="clip-video-list" aria-label="Video list">
      <header>
        <span className="eyebrow">Video · {visible.length}</span>
        <select value={group} aria-label="Filter by group"
          onChange={(event) => setGroup(event.target.value)}>
          <option value="all">All groups</option>
          {groups.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
        <input value={filter} aria-label="Filter videos" placeholder="K01_V001"
          onChange={(event) => setFilter(event.target.value)} />
      </header>
      {listError && <p className="clip-state error">{listError}</p>}
      {!listError && !videos.length && <p className="clip-state">Loading list…</p>}
      <ol>
        {visible.map((video) => <li key={video.video_id}>
          <button className={video.video_id === videoId ? "active" : ""}
            aria-current={video.video_id === videoId ? "true" : undefined}
            onClick={() => setVideoId(video.video_id)}>
            <span className={video.indexed ? "" : "unindexed"}
              title={video.indexed ? undefined : "Not in search index"}>
              {video.video_id}
            </span>
            {!video.has_shots && <em title="TransNetV2 not run yet">no shots</em>}
          </button>
        </li>)}
      </ol>
    </aside>

    <main className="clip-dashboard">
      {!videoId && <p className="clip-state clip-empty">Select a video to begin.</p>}
      {videoId && <>
        <section className="clip-player">
          <div ref={videoStageRef} className="clip-video-stage">
            <video ref={videoRef} key={videoId} src={`/video/${videoId}`} controls controlsList="nofullscreen" preload="metadata"
              aria-label={`Video ${videoId}`}
              onPlay={() => { setPaused(false); setMessage(""); }}
              onPause={() => setPaused(true)}
              onTimeUpdate={() => setPlaybackTime(videoRef.current?.currentTime ?? 0)}
              onSeeking={() => { seekingRef.current = true; setSeeking(true); }}
              onSeeked={() => { seekingRef.current = false; setSeeking(false); setPlaybackTime(videoRef.current?.currentTime ?? 0); }}
              onLoadedMetadata={() => {
                setPaused(videoRef.current?.paused ?? true);
                const stage = videoStageRef.current;
                const video = videoRef.current;
                if (stage && video) {
                  const bounds = stage.getBoundingClientRect();
                  setVideoContentRect(containedVideoRect(bounds.width, bounds.height, video.videoWidth, video.videoHeight));
                }
              }}
              onLoadedData={() => setPaused(videoRef.current?.paused ?? true)} />
            {showObjectOverlay && objectFrame && visibleObjects.length > 0 && <svg
              className="clip-object-overlay" viewBox="0 0 1 1" preserveAspectRatio="none"
              style={videoContentRect ? {
                left: videoContentRect.left, top: videoContentRect.top,
                width: videoContentRect.width, height: videoContentRect.height,
              } : undefined}
              role="img" aria-label={`Object detections near frame ${objectFrame.frame_idx}`}>
              {visibleObjects.map((detection, index) => {
                const [x0, y0, x1, y1] = detection.box;
                return <g key={`${detection.label}-${index}`}>
                  <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} strokeWidth=".003" />
                  <text x={x0} y={Math.max(0.025, y0 - 0.008)} fontSize=".028" strokeWidth=".004">
                    {detection.label} {Math.round(detection.confidence * 100)}%
                  </text>
                </g>;
              })}
            </svg>}
          </div>
          <div className="clip-actions">
            <div className="clip-readout">
              <strong>{videoId}</strong>
              <span>{liveFrameId === null ? "Frame not determined" : `frame ${liveFrameId}`}
                {currentShot ? ` · shot #${currentShot.index}` : shots.length ? " · outside any shot" : ""}</span>
            </div>
            <div className="shot-stepper">
              <button disabled={!shots.length} aria-label="Previous shot" onClick={() => stepShot(-1)}>‹ Shot</button>
              <button disabled={!shots.length} aria-label="Next shot" onClick={() => stepShot(1)}>Shot ›</button>
            </div>
            <button type="button" className="clip-fullscreen-button" aria-label={videoFullscreen ? "Exit video fullscreen" : "Fullscreen video"}
              onClick={() => void toggleVideoFullscreen()}>{videoFullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
            <label className="clip-object-toggle"><input type="checkbox" checked={showObjectOverlay}
              onChange={(event) => setShowObjectOverlay(event.target.checked)} /> Object detection</label>
            {showObjectOverlay && <label className="clip-object-confidence">Confidence {Math.round(objectConfidence * 100)}%
              <input aria-label="Object overlay confidence" type="range" min="25" max="95" step="5"
                value={Math.round(objectConfidence * 100)}
                onChange={(event) => setObjectConfidence(Number(event.target.value) / 100)} />
            </label>}
            <button className="export-clip" disabled={!canExport} onClick={exportCurrentShot}>
              {exporting ? "Exporting clip…"
                : currentShot
                  ? `Export shot #${currentShot.index} · ${formatTimestamp(currentShot.start)}–${formatTimestamp(currentShot.end)}`
                  : "Export clip"}
            </button>
          </div>
          {!paused && <p className="clip-state">Pause the video to export the shot you're viewing.</p>}
          {objectLoading && <p className="clip-state">Loading object detections…</p>}
          {objectError && <p className="clip-state error" role="status">Object detections are unavailable for this video.</p>}
          {message && <p className={`clip-state${messageIsError ? " error" : ""}`}
            role={messageIsError ? "alert" : "status"}>{message}</p>}
        </section>

        <section className="shot-overview" aria-label="Shot timeline">
          <header>
            <div><span className="eyebrow">Shot timeline</span><strong>{shots.length} shot</strong>
              <small>{labelCounts.suggested} suggested · {labelCounts.excluded} excluded · {labelCounts.included} kept</small></div>
            <div className="shot-overview-filters">
              <label>Status<select aria-label="Filter label status" value={labelFilter}
                onChange={(event) => setLabelFilter(event.target.value)}>
                <option value="all">All</option><option value="suggested">Suggested</option>
                <option value="excluded">Excluded</option><option value="included">Kept</option>
                <option value="unlabelled">Unlabelled</option>
              </select></label>
              <label>Keyframe source<select aria-label="Keyframe source" value={keyframeSource ?? ""}
                disabled={!keyframeSources.length}
                onChange={(event) => setSourcePreference(event.target.value)}>
                {!keyframeSources.length && <option value="">No keyframe</option>}
                {keyframeSources.map((source) => <option key={source.name} value={source.name}>{source.label}</option>)}
              </select></label>
              {isAdmin && <button disabled={!labelCounts.suggested} onClick={nextSuggestion}>Next suggestion</button>}
              {isAdmin && <button disabled={!labelCounts.suggested || savingLabel}
                onClick={() => void reviewAllSuggestions("excluded")}>Exclude all suggestions</button>}
              {isAdmin && <button disabled={!labelCounts.suggested || savingLabel}
                onClick={() => void reviewAllSuggestions("included")}>Keep all suggestions</button>}
            </div>
          </header>
          {shotsLoading && <p className="clip-state">Loading shots…</p>}
          {shotsError && <p className="clip-state error">{shotsError}</p>}
          <ol className="shot-timeline">
            {displayedShots.map((shot) => <li key={shot.index}
              ref={shot.index === currentShot?.index ? activeShotRef : undefined}
              className={`${shot.index === currentShot?.index ? "active " : ""}${shot.label?.state ?? ""}`}>
              <button onClick={() => seekToShot(shot)} aria-label={`Shot #${shot.index}`}>
                <strong>#{shot.index}</strong>
                <time>{formatTimestamp(shot.start)}</time>
                <span>{shot.label ? `${shot.label.state} · ${shot.label.reason}` : `${shot.keyframes.length} KF`}</span>
              </button>
            </li>)}
          </ol>
        </section>

        <section className="topic-segments" aria-label="DeepSeek topic segments">
          <header>
            <div><span className="eyebrow">Topic segments · DeepSeek pilot</span>
              <strong>{topicSegments.length} segment{topicSegments.length === 1 ? "" : "s"}</strong></div>
          </header>
          {topicLoading && <p className="clip-state">Loading topic segments…</p>}
          {!topicLoading && topicError && <p className="clip-state error">Failed to load topic segments.</p>}
          {!topicLoading && !topicError && !topicSegments.length
            && <p className="clip-state">No topic segments for this video.</p>}
          <ol className="topic-segment-list">
            {topicSegments.map((segment, index) => <li key={`${segment.start}-${index}`}
              className={index === currentTopicSegmentIndex ? "active" : ""}>
              <button onClick={() => seekTopicSegment(segment)}>
                <time>{formatTimestamp(segment.start)}–{formatTimestamp(segment.end)}</time>
                <span>{segment.text}</span>
              </button>
            </li>)}
          </ol>
        </section>

        {currentShot && <section className="shot-detail" aria-label={`Shot detail ${currentShot.index}`}>
          <header>
            <div><span className="eyebrow">Selected shot</span><h2>Shot #{currentShot.index}</h2>
              {currentShot.label && <span className={`shot-label ${currentShot.label.state}`}>
                {currentShot.label.state} · {currentShot.label.reason}
                {currentShot.label.confidence !== undefined ? ` · ${Math.round(currentShot.label.confidence * 100)}%` : ""}
              </span>}</div>
            <dl>
              <div><dt>Time</dt><dd>{formatTimestamp(currentShot.start)}–{formatTimestamp(currentShot.end)}</dd></div>
              <div><dt>Frame</dt><dd>{currentShot.startFrame}–{currentShot.endFrame}</dd></div>
              <div><dt>Keyframe</dt><dd>{currentShot.keyframes.length}</dd></div>
            </dl>
          </header>
          {isAdmin && <section className="shot-review" aria-label="Review shot label">
            <label>Reason<select aria-label="Shot exclusion reason" value={labelReason}
              onChange={(event) => setLabelReason(event.target.value as ShotLabelReason)}>
              <option value="intro">Intro</option><option value="preview">Preview</option>
              <option value="presenter">MC</option><option value="outro">Outro</option>
              <option value="other">Other</option>
            </select></label>
            <button disabled={savingLabel} onClick={() => void reviewShot("excluded")}>Exclude from index</button>
            <button disabled={savingLabel} onClick={() => void reviewShot("included")}>Keep in index</button>
            {currentShot.label && <button disabled={savingLabel} onClick={() => void reviewShot(null)}>Remove label</button>}
          </section>}
          {!!currentShot.reviewFrames?.length && <section className="shot-review-frames" aria-label="Representative frames for review">
            {currentShot.reviewFrames.map((frame) => <button key={frame} onClick={() => seekFrame(frame)}
              aria-label={`View representative frame ${frame}`}><img src={`/thumb/${videoId}/${frame}?w=320`}
                loading="lazy" alt={`Representative frame ${frame}`} /><span>f{frame}</span></button>)}
          </section>}
          <div className="shot-data-grid">
            <section className="shot-keyframes" aria-label="Keyframes in shot">
              <header><strong>Keyframes</strong><span>Extracted directly from the video</span></header>
              {!currentShot.keyframes.length && <p className="clip-state">This shot has no keyframes in the selected source.</p>}
              <div className="shot-keyframe-grid">
                {currentShot.keyframes.map((keyframe) => <article key={keyframe.frameId}>
                  <button className="keyframe-seek" onClick={() => seekFrame(keyframe.frameId)}
                    aria-label={`View frame ${keyframe.frameId}`}>
                    <img src={`${keyframe.thumb}?w=320`} loading="lazy" alt={`Frame ${keyframe.frameId}`} />
                  </button>
                  <footer><div><strong>f{keyframe.frameId}</strong><time>{formatTimestamp(keyframe.time)}</time></div>
                    <button disabled={copyingFrame !== null} onClick={() => void copyFrame(keyframe)}
                      aria-label={`Copy keyframe ${keyframe.frameId}`}>
                      {copyingFrame === keyframe.frameId ? "Copying…" : "Copy image"}
                    </button></footer>
                </article>)}
              </div>
            </section>
            <section className="shot-transcript" aria-label="Transcript for shot">
              <header><strong>Transcript for shot</strong><span>{shotCues.length} segments</span></header>
              {transcriptLoading && <p className="clip-state">Loading transcript…</p>}
              {!transcriptLoading && transcriptError && <p className="clip-state">Failed to load transcript.</p>}
              {!transcriptLoading && !transcriptError && !shotCues.length && <p className="clip-state">This shot has no speech.</p>}
              <ol>{shotCues.map((cue) => <li key={`${cue.start}-${cue.text}`}>
                <button onClick={() => seekCue(cue)}><time>{formatTimestamp(cue.start)}</time><span>{cue.text}</span></button>
              </li>)}</ol>
            </section>
          </div>
          <ShotCaption ariaLabel="Shot caption" shot={currentShot} caption={shotCaption}
            loading={captionLoading} error={captionError} empty={!captions.length} />
        </section>}
        <section className="full-transcript" aria-label="Full transcript to copy">
          <header><div><span className="eyebrow">Full transcript</span><strong>{cues.length} segments</strong></div>
            <button disabled={!fullTranscript} onClick={() => void copyTranscript()}>
              {transcriptCopyState === "copied" ? "Copied" : transcriptCopyState === "error" ? "Copy failed" : "Copy all"}
            </button>
          </header>
          {transcriptLoading && <p className="clip-state">Loading transcript…</p>}
          {!transcriptLoading && transcriptError && <p className="clip-state">Failed to load transcript.</p>}
          {!transcriptLoading && !transcriptError && !cues.length && <p className="clip-state">This video has no transcript.</p>}
          {fullTranscript && <pre tabIndex={0}>{fullTranscript}</pre>}
        </section>
      </>}
    </main>

    <aside ref={transcriptRef} className="clip-transcript" aria-label="Full video transcript" tabIndex={0}
      onWheel={() => setFollowTranscript(false)} onPointerDown={() => setFollowTranscript(false)}>
      <header><span className="eyebrow">Transcript</span><strong>Full video</strong>
        {!followTranscript && <button onClick={() => setFollowTranscript(true)}>Follow video</button>}
      </header>
      {!videoId && <p className="clip-state">Select a video to view the transcript.</p>}
      {videoId && transcriptLoading && <p className="clip-state">Loading transcript…</p>}
      {videoId && !transcriptLoading && transcriptError && <p className="clip-state">Failed to load transcript.</p>}
      {videoId && !transcriptLoading && !transcriptError && !cues.length && <p className="clip-state">This video has no transcript.</p>}
      <ol>{cues.map((cue, index) => <li key={`${cue.start}-${cue.text}`}
        ref={index === followedCue ? followedCueRef : undefined}
        className={index === activeCue ? "active" : ""}>
        <button onClick={() => seekCue(cue)}><time>{formatTimestamp(cue.start)}</time><span>{cue.text}</span></button>
      </li>)}</ol>
    </aside>
  </section>;
}
