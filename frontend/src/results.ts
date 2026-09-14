import type { SearchResult, VideoGroup } from "./types";

export function groupResultsByVideo(results: SearchResult[]): VideoGroup[] {
  const groups = new Map<string, VideoGroup>();

  for (const result of results) {
    const group = groups.get(result.video_id);
    if (group) {
      group.results.push(result);
      group.maxScore = Math.max(group.maxScore, result.score);
      group.bestRank = Math.min(group.bestRank, result.rank);
    } else {
      groups.set(result.video_id, {
        videoId: result.video_id,
        maxScore: result.score,
        bestRank: result.rank,
        results: [result],
      });
    }
  }

  return [...groups.values()]
    .map((group) => ({
      ...group,
      results: [...group.results].sort(
        (a, b) => a.time - b.time || a.frame_idx - b.frame_idx || a.rank - b.rank,
      ),
    }))
    .sort(
      // bestRank is authoritative: the backend's rank can reflect tiering the raw
      // score doesn't capture (e.g. an exact OCR match outranking a higher-scored
      // fuzzy one), so re-deriving row order from maxScore would silently discard it.
      (a, b) => a.bestRank - b.bestRank || b.maxScore - a.maxScore || a.videoId.localeCompare(b.videoId),
    );
}

export function getRelativeScoreTier(score: number, min: number, max: number): 1 | 2 | 3 | 4 | 5 {
  if (max === min) return 3;
  const normalized = Math.max(0, Math.min(1, (score - min) / (max - min)));
  return Math.min(5, Math.floor(normalized * 5) + 1) as 1 | 2 | 3 | 4 | 5;
}

export function formatTimestamp(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;

  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${minutes}:${String(secs).padStart(2, "0")}`;
}
