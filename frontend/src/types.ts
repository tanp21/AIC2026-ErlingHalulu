export type SearchMode = "visual" | "audio" | "hybrid" | "temporal" | "ocr";
export type ClassicSearchMode = Exclude<SearchMode, "temporal">;
export type SearchSource = "visual" | "transcript" | "both" | "ocr";
export type IndexName = "flat" | "ivf" | "hnsw";
export type AudioMethod = "semantic" | "keyword";

export interface IndexProfile {
  name: string;
  label: string;
  is_default: boolean;
  count: number | null;
  video_count: number | null;
  indexes: IndexName[];
}

export interface ScoreDetails {
  visual_score?: number;
  audio_score?: number;
  visual_rank?: number;
  audio_rank?: number;
  bm25_rank?: number | null;
  vector_rank?: number | null;
  keyword_rank?: number | null;
  semantic_rank?: number | null;
  pe_core_rank?: number | null;
  naflex_rank?: number | null;
  sequence_visual_score?: number;
  ocr_score?: number;
  ocr_rank?: number;
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
  audio_method?: AudioMethod;
  matched_keywords?: string[];
  ocr_matched_keywords?: string[];
  context_only?: boolean;
  ocr_snippet?: string;
  ocr_shot_index?: number;
  ocr_start?: number;
  ocr_end?: number;
}

export interface SearchRequest {
  text?: string;
  visual_text?: string;
  audio_text?: string;
  ocr_text?: string;
  mode: ClassicSearchMode;
  index: IndexName;
  nprobe: number;
  ef: number;
  profile?: string;
  audio_method?: AudioMethod;
  pe_weight?: number;
}

export interface ImageReference {
  video_id: string;
  frame_idx: number;
}

export interface ImageSearchRequest extends SearchRequest {
  mode: "visual" | "hybrid";
  image_weight: number;
  reference?: ImageReference;
}

export interface SearchResponse {
  results: SearchResult[];
  latency_ms: number;
  mode: ClassicSearchMode;
  index: string;
  index_info: { type: string; [key: string]: unknown };
  profile?: string;
  audio_method?: AudioMethod;
  pe_weight?: number;
  warning?: string;
  latencies?: { visual_ms?: number; audio_ms?: number; ocr_ms?: number };
  next_cursor: string | null;
  has_more: boolean;
  total_results: number;
  truncated: boolean;
}

export interface TemporalEventHit extends SearchResult { event_index: number }
export interface TemporalAudioEvidence {
  video_id: string;
  time: number;
  score: number;
  audio_rank: number;
  transcript_snippet: string;
  transcript_start?: number;
  transcript_end?: number;
  bm25_rank?: number | null;
  vector_rank?: number | null;
}
export interface TemporalSequence {
  rank: number;
  video_id: string;
  score: number;
  visual_score: number;
  source: "visual" | "both";
  start_time: number;
  end_time: number;
  events: TemporalEventHit[];
  audio_evidence?: TemporalAudioEvidence;
}
export interface TemporalVideoGroup {
  video_id: string;
  best_score: number;
  sequences: TemporalSequence[];
}
export interface TemporalSearchRequest {
  events: string[];
  audio_text?: string;
  audio_method: AudioMethod;
  pe_weight: number;
  max_gap_seconds: number;
  profile?: string;
}
export interface TemporalSearchResponse {
  videos: TemporalVideoGroup[];
  total_sequences: number;
  total_videos: number;
  latency_ms: number;
  faiss_latency_ms: number;
  mode: "temporal";
  candidate_k: number;
  scoring_method: string;
  warning?: string;
  next_cursor: string | null;
  has_more: boolean;
  truncated: boolean;
  profile?: string;
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

export interface TranscriptCue { start: number; end: number; text: string }
export interface TranscriptResponse { videoId: string; cues: TranscriptCue[] }

export interface VideoSummary {
  video_id: string;
  group: string;
  year: string;
  /** Null for videos on disk that no embedding profile covers. */
  fps: number | null;
  /** False means the video is exportable but not reachable from search. */
  indexed: boolean;
  has_shots: boolean;
}

export interface Shot {
  index: number;
  startFrame: number;
  endFrame: number;
  start: number;
  end: number;
  keyframes: ShotKeyframe[];
  reviewFrames?: number[];
  label?: ShotLabel | null;
}
export type ShotLabelState = "suggested" | "excluded" | "included";
export type ShotLabelReason = "intro" | "preview" | "presenter" | "outro" | "other";
export interface ShotLabel {
  state: ShotLabelState;
  reason: ShotLabelReason;
  confidence?: number;
  reviewed_by?: string;
  reviewed_at?: string;
}
export interface ShotKeyframe {
  frameId: number;
  time: number;
  thumb: string;
  frame: string;
}
export interface KeyframeSource { name: string; label: string }
export interface ShotsResponse {
  videoId: string;
  fps: number;
  frameCount: number;
  labelRevision: number;
  keyframeSource: string | null;
  keyframeSources: KeyframeSource[];
  shots: Shot[];
}
export interface ShotLabelsResponse {
  videoId: string;
  labelRevision: number;
  labels: Record<string, ShotLabel>;
}

export interface VideoGroup {
  videoId: string;
  maxScore: number;
  bestRank: number;
  results: SearchResult[];
}

export type QueryKind = "kis" | "qa" | "trake";
export type QueryStatus = "todo" | "in_progress" | "ready" | "reviewed";

export interface KisAnswer { video_id: string; frame_id: number | null; auto_filled?: boolean }
export interface QaAnswer extends KisAnswer { answer: string }
export interface TrakeAnswer { video_id: string; frame_ids: Array<number | null>; auto_filled?: boolean }
export type AnswerDraft = KisAnswer | QaAnswer | TrakeAnswer;

export interface CollabQuery {
  id: number;
  round_id: number;
  filename: string;
  kind: QueryKind;
  prompt: string;
  event_count: number | null;
  assignee: string | null;
  answers: AnswerDraft[];
  version: number;
  reviewed_by: string | null;
  reviewed_at: string | null;
  updated_by: string | null;
  updated_at: string | null;
  status: QueryStatus;
}

export interface StatusCounts { todo: number; in_progress: number; ready: number; reviewed: number }
export interface RoundSummary {
  id: number;
  name: string;
  active: boolean;
  revision: number;
  created_by: string;
  created_at: string;
  counts: StatusCounts;
  total: number;
}
export interface RoundSnapshot extends RoundSummary { queries: CollabQuery[] }
export interface ExportPreview { missing: string[]; invalid: string[]; unreviewed: string[] }

export type AgentRunStatus = "running" | "done" | "error";
export type AgentMode = "fast" | "auto" | "deep";
export interface AgentRunRequest {
  text: string;
  kind: QueryKind;
  mode: AgentMode;
  deepseek_api_key: string;
  gemini_api_key: string;
  max_steps?: number;
  time_budget?: number;
  deepseek_model?: string;
  gemini_model?: string;
}
export interface AgentVerdict { score: number; reason: string; answer?: string | null; scene?: number | null }
export interface AgentEvidence {
  source: string;
  kind: string;
  rank: number;
  score: number;
  text?: string | null;
  start?: number | null;
  end?: number | null;
}
export interface AgentCandidate {
  id: string;
  video_id: string;
  frame_idx: number;
  time: number;
  fused: number;
  verdict?: AgentVerdict | null;
  evidence: AgentEvidence[];
}
export interface AgentSequence {
  video_id: string;
  frames: number[];
  times?: number[];
  score: number;
  refined?: boolean;
}
export interface AgentRunResult {
  query_id: string;
  kind: QueryKind;
  query: string;
  answer: string | null;
  answer_evidence: string | null;
  ranking: AgentCandidate[];
  sequences: AgentSequence[];
  notes: string;
  elapsed: number;
  error: string | null;
}
export interface AgentUsage {
  deepseek_calls: number;
  deepseek_input: number;
  deepseek_output: number;
  gemini_calls: number;
  gemini_input: number;
  gemini_output: number;
}
export interface AgentRunResponse {
  run_id: string;
  status: AgentRunStatus;
  query_id: string;
  kind: QueryKind;
  query: string;
  elapsed: number;
  result: AgentRunResult | null;
  usage: AgentUsage | null;
  error: string | null;
}
