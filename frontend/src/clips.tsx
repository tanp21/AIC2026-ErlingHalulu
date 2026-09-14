import { useEffect, useMemo, useRef, useState } from "react";

import { fetchTranscript } from "./api";
import { copyKeyframe, downloadClip, fetchShots, fetchVideoList, updateShotLabels } from "./clipApi";
import { fetchFrameMap, frameTime, resolveFrameId } from "./frameMap";
import type { FrameMap } from "./frameMap";
import { formatTimestamp } from "./results";
import type { KeyframeSource, Shot, ShotKeyframe, ShotLabelReason, TranscriptCue, VideoSummary } from "./types";

export type AppTab = "search" | "queries" | "clips";

export function AppTabs({ tab, onTab }: { tab: AppTab; onTab(next: AppTab): void }) {
  return <nav className="app-tabs" aria-label="Work mode">
    {(["search", "queries", "clips"] as AppTab[]).map((name) => <button key={name}
      className={name === tab ? "active" : ""} aria-current={name === tab ? "page" : undefined}
      aria-label={name === "search" ? "Search tab" : undefined}
      onClick={() => onTab(name)}>{name === "search" ? "Search" : name === "queries" ? "Query management" : "Clip Studio"}</button>)}
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
  const [playbackTime, setPlaybackTime] = useState(0);
  const [followTranscript, setFollowTranscript] = useState(true);
  const [transcriptCopyState, setTranscriptCopyState] = useState<"idle" | "copied" | "error">("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
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
  const displayedShots = useMemo(() => shots.filter((shot) => (
    labelFilter === "all" || (labelFilter === "unlabelled" ? !shot.label : shot.label?.state === labelFilter)
  )), [shots, labelFilter]);
  const labelCounts = useMemo(() => ({
    suggested: shots.filter((shot) => shot.label?.state === "suggested").length,
    excluded: shots.filter((shot) => shot.label?.state === "excluded").length,
    included: shots.filter((shot) => shot.label?.state === "included").length,
  }), [shots]);
  const shotCues = useMemo(() => cuesForShot(cues, currentShot), [cues, currentShot]);
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
          <video ref={videoRef} key={videoId} src={`/video/${videoId}`} controls preload="metadata"
            aria-label={`Video ${videoId}`}
            onPlay={() => { setPaused(false); setMessage(""); }}
            onPause={() => setPaused(true)}
            onTimeUpdate={() => setPlaybackTime(videoRef.current?.currentTime ?? 0)}
            onSeeking={() => { seekingRef.current = true; setSeeking(true); }}
            onSeeked={() => { seekingRef.current = false; setSeeking(false); setPlaybackTime(videoRef.current?.currentTime ?? 0); }}
            onLoadedData={() => setPaused(videoRef.current?.paused ?? true)} />
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
            <button className="export-clip" disabled={!canExport} onClick={exportCurrentShot}>
              {exporting ? "Exporting clip…"
                : currentShot
                  ? `Export shot #${currentShot.index} · ${formatTimestamp(currentShot.start)}–${formatTimestamp(currentShot.end)}`
                  : "Export clip"}
            </button>
          </div>
          {!paused && <p className="clip-state">Pause the video to export the shot you're viewing.</p>}
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
