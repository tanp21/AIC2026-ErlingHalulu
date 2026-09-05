import { useEffect, useMemo, useRef, useState } from "react";

import { fetchTranscript } from "./api";
import { copyKeyframe, downloadClip, fetchShots, fetchVideoList, updateShotLabels } from "./clipApi";
import { fetchFrameMap, frameTime, resolveFrameId } from "./frameMap";
import type { FrameMap } from "./frameMap";
import { formatTimestamp } from "./results";
import type { KeyframeSource, Shot, ShotKeyframe, ShotLabelReason, TranscriptCue, VideoSummary } from "./types";

export type AppTab = "search" | "queries" | "clips";

export function AppTabs({ tab, onTab }: { tab: AppTab; onTab(next: AppTab): void }) {
  return <nav className="app-tabs" aria-label="Chế độ làm việc">
    {(["search", "queries", "clips"] as AppTab[]).map((name) => <button key={name}
      className={name === tab ? "active" : ""} aria-current={name === tab ? "page" : undefined}
      onClick={() => onTab(name)}>{name === "search" ? "Tìm kiếm" : name === "queries" ? "Quản lý truy vấn" : "Cắt clip"}</button>)}
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
      .catch((error) => setListError(error instanceof Error ? error.message : "Không tải được danh sách video"));
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
      if (active) setShotsError(error instanceof Error ? error.message : "Không tải được shot");
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
      setMessage(`Đã tải ${filename}`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Không cắt được clip");
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
      setMessage(`Đã copy keyframe ${keyframe.frameId}`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Không copy được keyframe");
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
      setMessage(state === "excluded" ? "Đã đánh dấu loại khỏi index"
        : state === "included" ? "Đã xác nhận giữ trong index" : "Đã xóa nhãn");
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Không lưu được nhãn shot");
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
      `${state === "excluded" ? "Loại" : "Giữ"} ${candidates.length} gợi ý của ${videoId}?`,
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
      setMessage(`Đã duyệt ${candidates.length} gợi ý`);
    } catch (error) {
      setMessageIsError(true);
      setMessage(error instanceof Error ? error.message : "Không lưu được nhãn shot");
    } finally {
      setSavingLabel(false);
    }
  }

  const canExport = Boolean(videoId && currentShot && paused && !seeking && !exporting);

  return <section className="clip-studio">
    <aside className="clip-video-list" aria-label="Danh sách video">
      <header>
        <span className="eyebrow">Video · {visible.length}</span>
        <select value={group} aria-label="Lọc theo nhóm"
          onChange={(event) => setGroup(event.target.value)}>
          <option value="all">Tất cả nhóm</option>
          {groups.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
        <input value={filter} aria-label="Lọc video" placeholder="K01_V001"
          onChange={(event) => setFilter(event.target.value)} />
      </header>
      {listError && <p className="clip-state error">{listError}</p>}
      {!listError && !videos.length && <p className="clip-state">Đang tải danh sách…</p>}
      <ol>
        {visible.map((video) => <li key={video.video_id}>
          <button className={video.video_id === videoId ? "active" : ""}
            aria-current={video.video_id === videoId ? "true" : undefined}
            onClick={() => setVideoId(video.video_id)}>
            <span className={video.indexed ? "" : "unindexed"}
              title={video.indexed ? undefined : "Không có trong index tìm kiếm"}>
              {video.video_id}
            </span>
            {!video.has_shots && <em title="Chưa chạy TransNetV2">chưa có shot</em>}
          </button>
        </li>)}
      </ol>
    </aside>

    <main className="clip-dashboard">
      {!videoId && <p className="clip-state clip-empty">Chọn một video để bắt đầu.</p>}
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
              <span>{liveFrameId === null ? "Chưa xác định frame" : `frame ${liveFrameId}`}
                {currentShot ? ` · shot #${currentShot.index}` : shots.length ? " · ngoài mọi shot" : ""}</span>
            </div>
            <div className="shot-stepper">
              <button disabled={!shots.length} aria-label="Shot trước" onClick={() => stepShot(-1)}>‹ Shot</button>
              <button disabled={!shots.length} aria-label="Shot sau" onClick={() => stepShot(1)}>Shot ›</button>
            </div>
            <button className="export-clip" disabled={!canExport} onClick={exportCurrentShot}>
              {exporting ? "Đang cắt clip…"
                : currentShot
                  ? `Export shot #${currentShot.index} · ${formatTimestamp(currentShot.start)}–${formatTimestamp(currentShot.end)}`
                  : "Export clip"}
            </button>
          </div>
          {!paused && <p className="clip-state">Tạm dừng video để cắt shot đang xem.</p>}
          {message && <p className={`clip-state${messageIsError ? " error" : ""}`}
            role={messageIsError ? "alert" : "status"}>{message}</p>}
        </section>

        <section className="shot-overview" aria-label="Timeline shot">
          <header>
            <div><span className="eyebrow">Shot timeline</span><strong>{shots.length} shot</strong>
              <small>{labelCounts.suggested} gợi ý · {labelCounts.excluded} loại · {labelCounts.included} giữ</small></div>
            <div className="shot-overview-filters">
              <label>Trạng thái<select aria-label="Lọc trạng thái nhãn" value={labelFilter}
                onChange={(event) => setLabelFilter(event.target.value)}>
                <option value="all">Tất cả</option><option value="suggested">Gợi ý</option>
                <option value="excluded">Đã loại</option><option value="included">Đã giữ</option>
                <option value="unlabelled">Chưa nhãn</option>
              </select></label>
              <label>Nguồn keyframe<select aria-label="Nguồn keyframe" value={keyframeSource ?? ""}
                disabled={!keyframeSources.length}
                onChange={(event) => setSourcePreference(event.target.value)}>
                {!keyframeSources.length && <option value="">Không có keyframe</option>}
                {keyframeSources.map((source) => <option key={source.name} value={source.name}>{source.label}</option>)}
              </select></label>
              {isAdmin && <button disabled={!labelCounts.suggested} onClick={nextSuggestion}>Gợi ý tiếp theo</button>}
              {isAdmin && <button disabled={!labelCounts.suggested || savingLabel}
                onClick={() => void reviewAllSuggestions("excluded")}>Loại mọi gợi ý</button>}
              {isAdmin && <button disabled={!labelCounts.suggested || savingLabel}
                onClick={() => void reviewAllSuggestions("included")}>Giữ mọi gợi ý</button>}
            </div>
          </header>
          {shotsLoading && <p className="clip-state">Đang tải shot…</p>}
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

        {currentShot && <section className="shot-detail" aria-label={`Chi tiết shot ${currentShot.index}`}>
          <header>
            <div><span className="eyebrow">Shot đang chọn</span><h2>Shot #{currentShot.index}</h2>
              {currentShot.label && <span className={`shot-label ${currentShot.label.state}`}>
                {currentShot.label.state} · {currentShot.label.reason}
                {currentShot.label.confidence !== undefined ? ` · ${Math.round(currentShot.label.confidence * 100)}%` : ""}
              </span>}</div>
            <dl>
              <div><dt>Thời gian</dt><dd>{formatTimestamp(currentShot.start)}–{formatTimestamp(currentShot.end)}</dd></div>
              <div><dt>Frame</dt><dd>{currentShot.startFrame}–{currentShot.endFrame}</dd></div>
              <div><dt>Keyframe</dt><dd>{currentShot.keyframes.length}</dd></div>
            </dl>
          </header>
          {isAdmin && <section className="shot-review" aria-label="Duyệt nhãn shot">
            <label>Lý do<select aria-label="Lý do loại shot" value={labelReason}
              onChange={(event) => setLabelReason(event.target.value as ShotLabelReason)}>
              <option value="intro">Intro</option><option value="preview">Điểm tin</option>
              <option value="presenter">MC</option><option value="outro">Outro</option>
              <option value="other">Khác</option>
            </select></label>
            <button disabled={savingLabel} onClick={() => void reviewShot("excluded")}>Loại khỏi index</button>
            <button disabled={savingLabel} onClick={() => void reviewShot("included")}>Giữ trong index</button>
            {currentShot.label && <button disabled={savingLabel} onClick={() => void reviewShot(null)}>Xóa nhãn</button>}
          </section>}
          {!!currentShot.reviewFrames?.length && <section className="shot-review-frames" aria-label="Frame đại diện để duyệt">
            {currentShot.reviewFrames.map((frame) => <button key={frame} onClick={() => seekFrame(frame)}
              aria-label={`Xem frame đại diện ${frame}`}><img src={`/thumb/${videoId}/${frame}?w=320`}
                loading="lazy" alt={`Frame đại diện ${frame}`} /><span>f{frame}</span></button>)}
          </section>}
          <div className="shot-data-grid">
            <section className="shot-keyframes" aria-label="Keyframe trong shot">
              <header><strong>Keyframes</strong><span>Trích trực tiếp từ video</span></header>
              {!currentShot.keyframes.length && <p className="clip-state">Shot này không có keyframe trong nguồn đã chọn.</p>}
              <div className="shot-keyframe-grid">
                {currentShot.keyframes.map((keyframe) => <article key={keyframe.frameId}>
                  <button className="keyframe-seek" onClick={() => seekFrame(keyframe.frameId)}
                    aria-label={`Xem frame ${keyframe.frameId}`}>
                    <img src={`${keyframe.thumb}?w=320`} loading="lazy" alt={`Frame ${keyframe.frameId}`} />
                  </button>
                  <footer><div><strong>f{keyframe.frameId}</strong><time>{formatTimestamp(keyframe.time)}</time></div>
                    <button disabled={copyingFrame !== null} onClick={() => void copyFrame(keyframe)}
                      aria-label={`Copy keyframe ${keyframe.frameId}`}>
                      {copyingFrame === keyframe.frameId ? "Đang copy…" : "Copy ảnh"}
                    </button></footer>
                </article>)}
              </div>
            </section>
            <section className="shot-transcript" aria-label="Transcript trong shot">
              <header><strong>Transcript theo shot</strong><span>{shotCues.length} đoạn</span></header>
              {transcriptLoading && <p className="clip-state">Đang tải transcript…</p>}
              {!transcriptLoading && transcriptError && <p className="clip-state">Không tải được transcript.</p>}
              {!transcriptLoading && !transcriptError && !shotCues.length && <p className="clip-state">Shot này không có lời thoại.</p>}
              <ol>{shotCues.map((cue) => <li key={`${cue.start}-${cue.text}`}>
                <button onClick={() => seekCue(cue)}><time>{formatTimestamp(cue.start)}</time><span>{cue.text}</span></button>
              </li>)}</ol>
            </section>
          </div>
        </section>}
        <section className="full-transcript" aria-label="Toàn bộ transcript để copy">
          <header><div><span className="eyebrow">Transcript đầy đủ</span><strong>{cues.length} đoạn</strong></div>
            <button disabled={!fullTranscript} onClick={() => void copyTranscript()}>
              {transcriptCopyState === "copied" ? "Đã copy" : transcriptCopyState === "error" ? "Copy thất bại" : "Copy toàn bộ"}
            </button>
          </header>
          {transcriptLoading && <p className="clip-state">Đang tải transcript…</p>}
          {!transcriptLoading && transcriptError && <p className="clip-state">Không tải được transcript.</p>}
          {!transcriptLoading && !transcriptError && !cues.length && <p className="clip-state">Video này không có transcript.</p>}
          {fullTranscript && <pre tabIndex={0}>{fullTranscript}</pre>}
        </section>
      </>}
    </main>

    <aside ref={transcriptRef} className="clip-transcript" aria-label="Transcript toàn video" tabIndex={0}
      onWheel={() => setFollowTranscript(false)} onPointerDown={() => setFollowTranscript(false)}>
      <header><span className="eyebrow">Transcript</span><strong>Toàn video</strong>
        {!followTranscript && <button onClick={() => setFollowTranscript(true)}>Theo video</button>}
      </header>
      {!videoId && <p className="clip-state">Chọn video để xem transcript.</p>}
      {videoId && transcriptLoading && <p className="clip-state">Đang tải transcript…</p>}
      {videoId && !transcriptLoading && transcriptError && <p className="clip-state">Không tải được transcript.</p>}
      {videoId && !transcriptLoading && !transcriptError && !cues.length && <p className="clip-state">Video này không có transcript.</p>}
      <ol>{cues.map((cue, index) => <li key={`${cue.start}-${cue.text}`}
        ref={index === followedCue ? followedCueRef : undefined}
        className={index === activeCue ? "active" : ""}>
        <button onClick={() => seekCue(cue)}><time>{formatTimestamp(cue.start)}</time><span>{cue.text}</span></button>
      </li>)}</ol>
    </aside>
  </section>;
}
