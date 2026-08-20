export type SearchMode = "visual" | "audio" | "hybrid";
export type SearchSource = "visual" | "transcript" | "both";
export type IndexName = "flat" | "ivf" | "hnsw";

export interface ScoreDetails {
  visual_score?: number;
  audio_score?: number;
  visual_rank?: number;
  audio_rank?: number;
  bm25_rank?: number | null;
  vector_rank?: number | null;
}

export interface SearchResult {
  rank: number;
  score: number;
  video_id: string;
  frame_idx: number;
  pts?: number;
  time: number;
  thumb: string;
  frame: string;
  video: string;
  source: SearchSource;
  scores: ScoreDetails;
  transcript_snippet?: string;
  transcript_start?: number;
  transcript_end?: number;
}

export interface SearchRequest {
  text?: string;
  visual_text?: string;
  audio_text?: string;
  mode: SearchMode;
  index: IndexName;
  k: number;
  nprobe: number;
  ef: number;
}

export interface SearchResponse {
  results: SearchResult[];
  latency_ms: number;
  mode: SearchMode;
  index: string;
  index_info: { type: string; [key: string]: unknown };
  warning?: string;
  latencies?: { visual_ms: number; audio_ms: number };
}

export interface CompareResult {
  index: string;
  latency_ms: number;
  recall: number;
}

export interface Session {
  login: string;
  is_admin: boolean;
}

export interface VideoGroup {
  videoId: string;
  maxScore: number;
  bestRank: number;
  results: SearchResult[];
}
