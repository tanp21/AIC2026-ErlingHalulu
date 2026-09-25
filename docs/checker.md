# Checking one moment with AI (the Search tab)

`POST /api/check` answers one question: *is this frame the moment the query describes?* It is
the retrieval agent's own verification (`agent/evidence.py`) cut down to one candidate and one
press of a button, so a person browsing the Search tab's results grid can get the same kind of
answer the agent gives a candidate, without starting a run.

## Where it lives

- **The bottom filmstrip** (`KeyframeContext` in `frontend/src/App.tsx`, next to **▶ Play
  video**): a **✓ Check by AI** button checks the selected frame, and the answer renders in the
  right-hand inspector panel (`TranscriptPanel`).
- **The video player** (`frontend/src/videoDialog.tsx`): a **✓ Check this moment** button in the
  left pane, under the video, checks whatever frame is paused on (or the current playback time
  while it's still playing). The answer renders there too, so the person can keep watching.

Both surfaces share one component, `CheckPanel` (`frontend/src/checkView.tsx`): the model's own
0-10 verdict and reason, the query's checklist with each detail marked ✓ supported / ~ close / ✗
contradicted / · not mentioned (the same four verdicts and the same styling as the Retrieval
agent tab's review checklist — `frontend/src/agentReview.tsx` imports `Lit` and `STATUS_MARK`
from the same module), and, for a Q&A query, the answer read off the same frames.

## The model

`OPENROUTER_API_KEY` (one shared team key, `qwen/qwen3.7-flash` by default) via
`agent/openrouter.py`'s `OpenRouterClient`, which speaks the same `judge()`/`chat_json()`
contract as `DeepSeekClient` and `GeminiClient` — it drops into `agent/evidence.py` and
`agent/vlm.py` unchanged. `AIC_AGENT_VISION=openrouter` makes the Retrieval agent tab judge with
it too, which is what to set when the Gemini quota is gone; web lookups (Google Search grounding)
stay Gemini-only and are simply skipped without a Gemini key.

Qwen does both halves that Gemini and DeepSeek split in the agent: it writes the query's
checklist (`agent/evidence.query_details`, generalised to take any client with `chat_json`) *and*
judges the pixels, in `agent/checker.py`. One call returns the per-detail verdicts, the overall
score, and — for a Q&A query — the answer, because the images are already assembled and paid
for; the agent's own Q&A reading (`Inspector.answer`) is a separate call for exactly this reason,
which the checker doesn't need to pay twice for.

## What one check looks at

A ±5 minute window (`AIC_CHECK_WINDOW_SECONDS`, 60–600 s) around the frame: the shots either side
of it, walked outward the same way `agent/evidence._context_frames` does for the agent's
checklist (nearest shots keep every keyframe, the rest keep one, capped at ~14 frames a side so
one call stays under 30 images), plus the speech and on-screen text in the window. Wider than the
agent's own ±180 s, because a person pressing a button on one candidate is asking about a whole
segment, not a moment already narrowed down by retrieval.

## Caching — the point of the design

**A check is keyed by the shot it's in, not the frame.** Every keyframe of one shot maps to the
same cache entry, because the image set a check assembles is essentially the same for any of
them — so stepping along the filmstrip inside one shot is free after the first press, and a
teammate who checks a candidate someone else already checked gets that answer for free too (the
cache is not owner-scoped: what a check says about a shot is a fact about the video and the
query). **Re-check this exact frame** asks about one frame specifically when that's what's
wanted, at the cost of its own cache entry. The query's checklist is cached separately, per
query, so checking twenty candidates of one query costs one text call and twenty vision calls,
not forty.

Three stores, `backend/checker.py`:

| Store | Key | TTL |
| --- | --- | --- |
| `CheckStore` | query · video · shot (or a 10 s time bucket for a video with no shot boundaries) · window · model | `AIC_CHECK_TTL_SECONDS` (1 h default) |
| `DetailsStore` | query · kind · question | same |
| `EvidenceStore` (`agent/evidence.py`, LRU-bounded) | video, for transcript/OCR/captions | process lifetime |

`GET /api/check/lookup` answers `hit`/`miss`/`running` without ever calling a model — what a
reloaded page, or the browser-side cache in `frontend/src/checkApi.ts`, falls back to.

## Configuration

See `.env.secure.example`. The button is hidden, not broken, when `OPENROUTER_API_KEY` is unset:
`GET /session` reports `check: {available, model, window_seconds}`, never the key itself.
