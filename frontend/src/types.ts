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
/** On-screen text of one OCR'd shot (`GET /ocr/{video_id}`). */
export interface OcrShot { shot_idx: number; frame_idx: number; start: number; end: number; time: number; text: string }
export interface OcrResponse { videoId: string; shots: OcrShot[] }
/** One shot-caption window (`GET /captions/{video_id}`). */
export interface CaptionWindow {
  shot_idx: number;
  window_idx: number;
  /** How many windows the shot was split into; absent when the caption came from the search index only. */
  window_count?: number;
  frame_idx: number | null;
  start: number;
  end: number;
  time: number;
  text: string;
}
export interface CaptionsResponse { videoId: string; captions: CaptionWindow[] }

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
/** Who looks at the frames: a vision model, or the person running the agent. */
export type AgentJudge = "model" | "human";
export interface AgentRunRequest {
  text: string;
  kind: QueryKind;
  mode: AgentMode;
  judge?: AgentJudge;
  deepseek_api_key: string;
  gemini_api_key: string;
  max_steps?: number;
  time_budget?: number;
  deepseek_model?: string;
  gemini_model?: string;
}
export interface AgentVerdict {
  score: number;
  reason: string;
  answer?: string | null;
  scene?: number | null;
  by?: "gemini" | "human";
}
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
  // Judgements made by a local vision backend (AIC_AGENT_VISION=local). No token
  // counts: the local models do not bill.
  vlm_calls: number;
}
export type AgentStage =
  | "planning" | "searching" | "judging" | "second_look" | "rewriting" | "agent_loop"
  | "reading_answer" | "checking_paths" | "finalizing" | "awaiting_review" | "done";
export type AgentDecision = "accept" | "reject";
/** The retrieval context Gemini is shown beside a keyframe, trimmed exactly as it sees it. */
export interface AgentCandidateContext { speech?: string; on_screen?: string; caption?: string }
/** One channel that found a moment, and how high it ranked it there. */
export interface AgentChannel { kind: string; source: string; rank: number }
/**
 * `supported`: the words say the whole detail. `partial`: right subject, a part missing or
 * generic ("flowers" for a torch-ginger vase, no count). `contradicted`: a part is stated
 * differently. `not_mentioned` is not evidence against a candidate.
 */
export type AgentCheckStatus = "supported" | "partial" | "contradicted" | "not_mentioned";
/** One query detail checked against the words around a candidate (text only, by DeepSeek). */
/** `at` is the second the evidence starts at, so a reviewer can go and look. */
export interface AgentCheck { detail: number; status: AgentCheckStatus; source: "speech" | "ocr" | "caption" | null; at?: number | null; quote: string; /** why, in a few words */ note?: string }
export interface AgentSummary { checks: AgentCheck[]; summary: string }
export interface AgentLiveCandidate {
  id: string;
  video_id: string;
  frame_idx: number;
  time: number;
  /** Its image is with Gemini right now. */
  judging: boolean;
  verdict: AgentVerdict | null;
  context: AgentCandidateContext;
  sources: string[];
  // Only in a human-judged run:
  channels?: AgentChannel[];
  fused?: number;
  /** Asked for by hand (a neighbour or an exact frame), so it stays in view. */
  pinned?: boolean;
  summary?: AgentSummary | null;
  /**
   * What the checklist says about it as numbers: `share` (what the words around the moment itself
   * support, the bar's fill), `video` (what its whole video supports), `score` (their mix, what the
   * order uses) and `tier` (0 supported, 1 nothing either way). Only support counts: a contradiction
   * is shown but lowers nothing. Null when nothing has been read for it.
   */
  evidence?: { score: number; share: number; video?: number; tier: 0 | 1 } | null;
  /** Where retrieval alone put it, 1 = first. The list itself is in evidence order. */
  agent_rank?: number;
}
export interface AgentLivePlan {
  summary_vi: string;
  scenes: string[];
  target_scene: number;
  question_vi: string | null;
  events: string[];
  /** The words to light up in a transcript or OCR line (`vi`) and in a caption (`en`). */
  keywords?: { vi: string[]; en: string[] };
}
export interface AgentLiveSequence {
  video_id: string;
  frames: number[];
  event_scores: number[] | null;
  refined: boolean;
  source?: string;
  times?: number[];
}
export interface AgentLookup { question: string; answer: string | null; error: string | null; sources: string[] }
export interface AgentAnswerSuggestion { answer: string; source: string; quote: string; cid: string | null; confidence: number }
export interface AgentProgress {
  stage: AgentStage;
  step: number;
  elapsed: number;
  plan: AgentLivePlan | null;
  candidates: AgentLiveCandidate[];
  sequences: AgentLiveSequence[];
  answer: string | null;
  // Only in a human-judged run:
  lookups?: AgentLookup[];
  total_candidates?: number;
  details?: string[];
  /** How much each detail counts, and which are satisfied by nearly every candidate read. */
  detail_weights?: number[];
  common_details?: boolean[];
  summarizing?: boolean;
  suggested_answers?: AgentAnswerSuggestion[];
}
export interface AgentReviewState {
  decisions: Record<string, AgentDecision>;
  answer: string | null;
  stopped: boolean;
  decided: boolean;
  judge?: AgentJudge;
  scores?: Record<string, number>;
  sequence?: { video_id: string; frames: number[]; times?: number[] } | null;
  finished?: boolean;
  pending_commands?: number;
}
export type AgentCommand =
  | { type: "more"; hint?: string }
  | { type: "scope"; video_id: string }
  | { type: "neighbours"; cid: string }
  | { type: "add_frame"; video_id: string; frame_idx: number; time: number; accept?: boolean };
export interface AgentReviewRequest {
  candidate_id?: string;
  decision?: AgentDecision | "clear";
  /** 0-10, between accept (10) and reject (0); 5 means "right video, wrong moment". */
  score?: number;
  answer?: string;
  sequence?: { video_id: string; frames: number[]; times?: number[] };
  command?: AgentCommand;
  stop?: boolean;
  /** Submit: the person is done. */
  finish?: boolean;
  /** The candidate ids in the order on screen, sent with submit or stop so the export matches it. */
  order?: string[];
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
  progress?: AgentProgress | null;
  review?: AgentReviewState;
}
