import { fetchCaptions, fetchOcr, fetchTranscript } from "./api";
import type { CaptionWindow, OcrShot, TranscriptCue } from "./types";

/** Everything written or said in one video: the words a reviewer reads a candidate against. */
export interface VideoEvidence {
  cues: TranscriptCue[];
  ocr: OcrShot[];
  captions: CaptionWindow[];
  /** Which sources this deployment could not give (an index that is not built, a video never captioned). */
  missing: Array<"transcript" | "ocr" | "captions">;
}

// One request per kind per video, however many of its moments are looked at: a query's
// candidates cluster in a handful of videos, and the reviewer moves between them fast.
const cache = new Map<string, Promise<VideoEvidence>>();

export function loadVideoEvidence(videoId: string): Promise<VideoEvidence> {
  let pending = cache.get(videoId);
  if (!pending) {
    pending = Promise.allSettled([fetchTranscript(videoId), fetchOcr(videoId), fetchCaptions(videoId)])
      .then(([transcript, ocr, captions]) => {
        const missing: VideoEvidence["missing"] = [];
        if (transcript.status !== "fulfilled" || !Array.isArray(transcript.value.cues)) missing.push("transcript");
        if (ocr.status !== "fulfilled" || !Array.isArray(ocr.value.shots)) missing.push("ocr");
        if (captions.status !== "fulfilled" || !Array.isArray(captions.value.captions)) missing.push("captions");
        return {
          cues: transcript.status === "fulfilled" && Array.isArray(transcript.value.cues) ? transcript.value.cues : [],
          ocr: ocr.status === "fulfilled" && Array.isArray(ocr.value.shots) ? ocr.value.shots : [],
          captions: captions.status === "fulfilled" && Array.isArray(captions.value.captions) ? captions.value.captions : [],
          missing,
        };
      });
    cache.set(videoId, pending);
  }
  return pending;
}

export function clearEvidenceCache(): void {
  cache.clear();
}

/** Cues overlapping `time ± seconds`; `active` marks the one being spoken at `time`. */
export function speechAround(cues: TranscriptCue[], time: number, seconds = 30): Array<TranscriptCue & { active: boolean }> {
  return cues
    .filter((cue) => cue.end >= time - seconds && cue.start <= time + seconds)
    .map((cue) => ({ ...cue, active: cue.start <= time && time < cue.end }));
}

/**
 * How much of the video around a moment a reviewer reads, matching the agent's checklist
 * (`BEFORE_SECONDS`/`AFTER_SECONDS` in agent/evidence.py). A query describes consecutive
 * scenes and a candidate is one frame of one of them, so what the query mentions first is
 * in the shots before it.
 */
export const SPAN_BEFORE = 45;
export const SPAN_AFTER = 25;

// The channel's own logo and clock, OCR'd on nearly every shot ("HTV Online", "HD 06:30:39 giây"):
// never what a query is about, and it buries the lines that are.
const FURNITURE_WORDS = new Set(["htv", "hit", "hiv", "hd", "online", "giay", "vtv", "vad"]);

export function isFurniture(text: string): boolean {
  const words = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").match(/[a-z]+/g) ?? [];
  return words.length > 0 && words.every((word) => FURNITURE_WORDS.has(word));
}

/** Shots with text overlapping the span around `time`, plus the one holding it, in time order. */
export function ocrAround(shots: OcrShot[], time: number, limit = 8): Array<OcrShot & { current: boolean }> {
  if (!shots.length) return [];
  const ordered = [...shots].sort((a, b) => a.start - b.start);
  let inside = ordered.findIndex((shot) => shot.start <= time && time <= shot.end);
  if (inside < 0) {
    inside = ordered.reduce((best, shot, index) =>
      Math.abs(shot.time - time) < Math.abs(ordered[best].time - time) ? index : best, 0);
  }
  return ordered
    .map((shot, index) => ({ ...shot, current: index === inside }))
    .filter((shot) => shot.current || (shot.text.trim() && !isFurniture(shot.text) && shot.end >= time - SPAN_BEFORE && shot.start <= time + SPAN_AFTER))
    .slice(0, limit);
}

/** Caption windows overlapping the span around `time` in time order; `current` holds `time`. */
export function captionsAround(captions: CaptionWindow[], time: number, limit = 14): Array<CaptionWindow & { current: boolean }> {
  if (!captions.length) return [];
  const ordered = [...captions].sort((a, b) => a.start - b.start);
  let inside = ordered.findIndex((caption) => caption.start <= time && time <= caption.end);
  if (inside < 0) {
    inside = ordered.reduce((best, caption, index) =>
      Math.abs((caption.start + caption.end) / 2 - time) < Math.abs((ordered[best].start + ordered[best].end) / 2 - time) ? index : best, 0);
  }
  return ordered
    .map((caption, index) => ({ ...caption, current: index === inside }))
    .filter((caption) => caption.current || (caption.end >= time - SPAN_BEFORE && caption.start <= time + SPAN_AFTER))
    .slice(0, limit);
}
