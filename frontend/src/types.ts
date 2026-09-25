export type SearchMode = "visual" | "audio" | "hybrid" | "temporal" | "ocr" | "object";
export type ClassicSearchMode = Exclude<SearchMode, "temporal">;
export type SearchSource = "visual" | "transcript" | "both" | "ocr" | "object" | "mixed";
export type EventMode = "text" | "image" | "audio" | "ocr" | "object";
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
  object_score?: number;
  object_rank?: number;
}

export interface ObjectClause {
  label: string;
  region: [number, number, number, number];
  min_count: number;
}

export interface ObjectQuery {
  clauses: ObjectClause[];
  min_confidence: number;
}

export interface MatchedObject {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface ObjectDetectionFrame {
  frame_idx: number;
  time: number;
  detections: MatchedObject[];
}
export interface ObjectDetectionsResponse { video_id: string; frames: ObjectDetectionFrame[] }

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
  matched_objects?: MatchedObject[];
  object_delta_ms?: number;
}

export interface SearchRequest {
  text?: string;
  visual_text?: string;
  audio_text?: string;
  ocr_text?: string;
  object_query?: ObjectQuery;
  mode: ClassicSearchMode;
  index: IndexName;
  nprobe: number;
  ef: number;
  profile?: string;
  audio_method?: AudioMethod;
  pe_weight?: number;
}

export interface RefineRequest {
  text: string;
  kind: "kis" | "qa" | "trake";
}

export interface RefineCandidate {
  text: string;
  tokens: number | null;
  primary: boolean;
}

export interface RefineResponse {
  summary_vi: string;
  visual: RefineCandidate[];
  transcript: string[];
  ocr: string[];
  events: RefineCandidate[];
  lookups: string[];
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
  latencies?: { visual_ms?: number; audio_ms?: number; ocr_ms?: number; object_ms?: number };
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
  source: "visual" | "both" | "transcript" | "ocr" | "object" | "mixed";
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
  /**
   * Whether this deployment has an OpenRouter key, so the Search tab can hide "Check by AI"
   * instead of letting it fail when pressed. Never the key itself.
   */
  check?: CheckConfig;
}

export interface CheckConfig { available: boolean; model: string; window_seconds: number }

export interface TranscriptCue { start: number; end: number; text: string }
export interface TranscriptResponse { videoId: string; cues: TranscriptCue[] }
/** One DeepSeek topic/story segment (`GET /audio/topic-segments/{video_id}`) --
 * a Clip Studio visualization of the `topic` chunking pilot, not a production
 * search channel. */
export interface TopicSegment { start: number; end: number; text: string }
export interface TopicSegmentsResponse { videoId: string; segments: TopicSegment[] }
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
export interface DresContext {
  kind: QueryKind;
  label: string;
  event_count: number | null;
  revision: number;
  updated_by: string;
  updated_at: string;
}
export interface DresStatus {
  configured: boolean;
  credentials_configured: boolean;
  host: string;
  connected: boolean;
  evaluations: Array<{ id: string; name: string }>;
  evaluation_id: string | null;
  evaluation_manual?: boolean;
  context: DresContext | null;
  context_valid: boolean;
}
export interface DresCandidate {
  video_id: string;
  frame_idx?: number;
  frame_source?: "indexed_keyframe" | "original_frame";
  frame_ids?: number[];
  answer?: string;
}
export interface DresPrepared {
  evaluation_id: string;
  context_revision: number;
  context_label: string;
  kind: QueryKind;
  video_id: string;
  frame_idx: number | null;
  frame_ids: number[] | null;
  timestamp_ms: number | null;
  payload: { answerSets: Array<{ answers: Array<Record<string, string>> }> };
  preview: string;
  fingerprint: string;
}
export interface DresAttempt {
  id: number;
  evaluation_id: string;
  context_revision: number;
  context_label: string;
  task_type: QueryKind;
  payload: DresPrepared["payload"];
  fingerprint: string;
  video_id: string;
  frame_idx: number | null;
  frame_ids: number[] | null;
  actor: string;
  submitted_at: string;
  status: "pending" | "accepted" | "rejected" | "uncertain" | "not_sent";
  http_status: number | null;
  latency_ms: number | null;
  server_message: string | null;
}
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
  /** Plan with DeepSeek's thinking on; omitted, the deployment decides (deep mode thinks). */
  plan_thinking?: boolean;
  deepseek_api_key: string;
  gemini_api_key: string;
  max_steps?: number;
  time_budget?: number;
  deepseek_model?: string;
  gemini_model?: string;
}
/** One of the query's details as the moment judge saw it at this candidate (`agent/judge.py`). */
export interface AgentJudgeCheck {
  text: string;
  status: AgentCheckStatus;
  /** `vision` when it was read off the frames, else the words (speech, on-screen text). */
  source: string | null;
  note: string;
  quote?: string;
}
export interface AgentVerdict {
  score: number;
  reason: string;
  answer?: string | null;
  scene?: number | null;
  /** Who judged: "human", "gemini", "qwen", ... */
  by?: string;
  answer_confidence?: number | null;
  checks?: AgentJudgeCheck[] | null;
  frames_seen?: number | null;
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
  /** The speech, on-screen text and caption around the moment. */
  context?: AgentCandidateContext;
}
export interface AgentSequence {
  video_id: string;
  frames: number[];
  times?: number[];
  score: number;
  refined?: boolean;
  /** 0-10 per event from the judge, in order; null when the path was not checked. */
  event_scores?: number[] | null;
  source?: string;
}
/** A web lookup the run made, and what came back. */
export interface AgentRunLookup {
  question: string;
  answer?: string | null;
  error?: string | null;
  sources?: string[];
  about_the_query?: boolean;
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
  /** Which strategies the run chose, in Vietnamese. */
  strategy?: string[];
  /** Who judged the candidates: "qwen", "gemini", "human". */
  judge?: string | null;
  /** TRAKE: the events in order, as the query words them. */
  events?: string[];
  lookups?: AgentRunLookup[];
}
/** What the Retrieval agent tab asks the app to send to DRES. */
export interface AgentSubmitRequest {
  kind: QueryKind;
  video_id: string;
  frame_idx?: number;
  frame_ids?: number[];
  answer?: string;
}
export interface AgentSubmitOutcome { message: string; completed?: boolean; uncertain?: boolean }
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
  strategy?: string[];
  judge?: string | null;
  events?: string[];
  lookups?: AgentLookup[];
  // Only in a human-judged run:
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

// -- "Check by AI": one moment, checked on demand against the query (POST /api/check) ----------

/** One checkable part of the query, with how much knowing it is true would narrow things down. */
export interface CheckDetail {
  text: string;
  meaning_en: string;
  /** 0-1 from the model: is this true of nearly every candidate of this kind, or of one moment? */
  distinctiveness: number;
  /** `0.2 + 0.8 × distinctiveness` -- what it is worth in `support.share`. */
  weight: number;
}

/**
 * One detail checked against this moment. `source` is what answered: `vision` looked at the
 * pixels of this shot and the ones around it, `speech`/`ocr` read the words, and `at` is the
 * second that evidence starts at so the person can go and look.
 */
export interface CheckItem {
  detail: number;
  status: AgentCheckStatus;
  source: "vision" | "speech" | "ocr" | null;
  at?: number | null;
  quote: string;
  note?: string;
}

/** A Q&A answer read off this moment -- never recalled from what the model knows. */
export interface CheckAnswer {
  text: string;
  confidence: number;
  source: "frame" | "speech" | "ocr" | "unknown";
  quote?: string;
}

/** How much of the query the evidence around this moment says. A contradiction lowers nothing. */
export interface CheckSupport {
  share: number;
  supported: number;
  partial: number;
  contradicted: number;
  total: number;
}

/**
 * The frames one answer speaks for: its whole shot. Every keyframe in this range shares the
 * result, which is what makes stepping along the filmstrip free after the first press.
 */
export interface CheckCoverage {
  start_frame: number | null;
  end_frame: number | null;
  start: number;
  end: number;
}

export interface CheckResult {
  key: string;
  video_id: string;
  frame_idx: number;
  time: number;
  window_seconds: number;
  shot_index: number | null;
  covers: CheckCoverage | null;
  /** The model's own 0-10 for the moment as a whole, with its reason in Vietnamese. */
  verdict: { score: number; reason: string };
  details: CheckDetail[];
  checks: CheckItem[];
  support: CheckSupport;
  answer: CheckAnswer | null;
  frames_seen: number;
  model: string;
  by: string;
  checked_at: number;
  cached: boolean;
  latency_ms: number;
}

/**
 * The same three channels a real search request carries (`SearchRequest`), not one flattened
 * sentence -- a check has to know whether it is reading a scene description, words to find
 * spoken, or a string to find written on screen, and the server judges each differently. At
 * least one must be non-empty.
 */
export interface CheckRequest {
  visual_text?: string;
  audio_text?: string;
  ocr_text?: string;
  kind: QueryKind;
  question?: string;
  video_id: string;
  frame_idx?: number;
  time?: number;
  window_seconds?: number;
  profile?: string;
  force?: boolean;
}

/** What the two surfaces render: the same result, loading or failed. */
export interface CheckState {
  status: "idle" | "loading" | "done" | "error";
  result: CheckResult | null;
  error: string | null;
  /** The moment the person asked about, which may differ from `result.frame_idx` on a cache hit. */
  asked: { video_id: string; frame_idx?: number; time?: number } | null;
}

export interface CheckLookup { status: "hit" | "miss" | "running" | "unavailable"; result: CheckResult | null }
