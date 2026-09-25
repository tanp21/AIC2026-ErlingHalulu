# Retrieval agent

`python -m agent` takes a Vietnamese competition query (KIS, Q&A or TRAKE), plans
searches across every channel the server offers, checks candidates by looking at
them, and writes a ranked, submission-ready answer for a teammate to confirm.

The agent is an HTTP client of the search server. It runs as a CLI on a laptop, or
server-side from the **Retrieval agent** tab in the web UI (see *Server-hosted runs*).

## Fully automatic runs (the Retrieval agent tab)

The tab is for the hard queries -- Q&A that needs the web, TRAKE, and KIS that plain search
misses -- and a run needs nobody while it runs. Who does what:

| Model | Job |
| --- | --- |
| DeepSeek | plans the searches, rewrites them after the first round, splits the query into checkable details |
| **Qwen** (`qwen/qwen3.7-flash`, OpenRouter, the team's shared key) | **judges every candidate**, checks TRAKE paths, reads Q&A answers off the frames |
| Gemini | **web search only** (Google Search grounding); never sees a frame |

Judging moved from Gemini to Qwen for cost. It is on whenever the deployment has an OpenRouter
key -- `OPENROUTER_API_KEY`, or `QWEN_API_KEY`, the name it was handed round under -- and
`AIC_AGENT_VISION` is unset (`agent.config.default_vision`).

**A candidate is judged as a moment, not a keyframe** (`agent/judge.py`, `MomentJudge`). Qwen
gets one call per candidate holding its keyframe plus the keyframes of the shots around it
(±60 s, nearest shots densest, ~9-15 images), the speech and on-screen text of that window, the
organisers' query, the frame the plan wants (target scene) and, for Q&A, the question. It is the
Search tab's "Check by AI" (`agent/checker.py`) pointed at the agent's candidates. It answers
per detail of the query (✓/~/✗, from DeepSeek's checklist, `evidence.query_details`), an overall
0-10 and, for Q&A, the answer read off that moment. Measured live: a call takes 13-16 s and
eight in flight take the same wall time, so a round of 16 candidates is ~15-20 s on a healthy
host. A round has a 60 s deadline (a slow compute host serving the keyframes once stretched one
to 190 s); what misses it stays unjudged. A reply the parser cannot repair is asked once more.
Because the judge already sees the shots around a candidate, the falsifier is off in this mode,
and the budgets are longer (`MOMENT_DEFAULTS`: 90 s fast, 240 s auto, 300 s deep) so the rewrite
round still fits after two rounds of judging.

**Strategies, chosen per query** (each one is listed in the result, `AgentResult.strategy`):

| The query... | What the run does |
| --- | --- |
| cooks something (regex, or the plan's `topic: cooking`) | Gemini looks up the dish and recipe at t=0 ("Món ngon mỗi ngày" publishes them); the dish name becomes a transcript query in round 2; a Q&A also looks up the recipe of the top video's own dish |
| names a festival, custom, event, place or person (`web.TOPIC_PATTERNS`, or the plan's `topic`) | Gemini looks up which one it is and where (`TOPIC_LOOKUP_PROMPT`) at t=0; the names in its `TỪ KHÓA:` line are searched in round 2 as transcript queries and, when capitalised, as on-screen text (`_web_queries`) |
| leaves a name to the web (the plan's `lookups`, now 0-3 and asked for whenever the web can name what the corpus would say) | each question is looked up; its `TỪ KHÓA:` names feed round 2 the same way and its `ĐÁP ÁN:` joins the Q&A answers |
| is Q&A | the answer is read **after all judging is over**, at the best-judged moments of the top video: the judge's own readings of those moments vote with a dedicated high-resolution reading, non-answers ("không rõ", "Không xác định được", a sentence that opens with a denial) are no vote, a count keeps only its number, and the result is reconciled with a fact/recipe lookup made from the top video's narration |
| is TRAKE | each event is searched inside the leading videos, paths are joined in order and checked event by event by the judge |

**The result is laid out for checking and submitting** (`frontend/src/agentResult.tsx`): the
answers on the left; the selected one on the right at full resolution, with the keyframes around
it (click one to submit that frame instead), the judge's score, reason and per-detail checks, and
the speech/on-screen text/caption around it. A TRAKE path shows every event side by side, each
with its own frame, score and the query's wording of the event, and each adjustable to a nearby
keyframe. **Submit to DRES** sends exactly what is on screen through the app's DRES flow
(`submitAgentAnswer` in `App.tsx`: the DRES task is switched to the run's kind first); a Q&A
submits the answer in the box above, pre-filled with the agent's.

**When a run misses, steer it.** Under the result, **Edit query and run again** puts the query
back in the form to reword the part that matters. **Think harder when planning** plans with
DeepSeek's thinking on (`plan_thinking`, per run; on by default in deep mode, `--plan-thinking`
on the CLI): measured on the planner regressions, a plan then takes 5-50 s (usually 15-30 s)
instead of 2-3 s.

The live view shows the plan, the strategies, each web lookup as it lands and each candidate as
Qwen judges it; **Stop** ends a run early with its current ranking. The tab no longer starts a
run with a person as the judge; the API still accepts `judge: "human"` and such a run still opens
in the review workspace described below.

## How one query is answered

There are three modes. They share every step except what happens after round 1.

**`--mode fast`** (the default) takes about 15-20 s per query (median; TRAKE ~30 s):

```text
t=0   cooking query? ──► Gemini + Google Search: which "Món ngon mỗi ngày" dish/episode (background)
      DeepSeek plan, no thinking (~2 s) ──► plan lookups start in the background
      round 1: one /agent/search batch (~0.5-1 s): visual, caption, transcript, OCR, temporal
      Gemini verifies 15 candidates as 3 parallel calls of 5 (~5 s): each query's best
        hits round-robin, then the rest by fusion
      best ≥ 9 and the falsifier (~5 s) accepts it? ── yes ──► done
      otherwise ──► one DeepSeek rewrite (no thinking, sees verdicts + web notes)
                    + every scene searched inside the top 3 videos
                    + neighbours of a 5-8 match ──► batch ──► parallel verification
      Q&A: reading at the top 2 moments  ∥  a web answer (recipe for cooking shows,
           a news/fact search otherwise, none for counts) ──► reconciled
      TRAKE: no point verification; each event searched inside the 3-4 leading videos
             and joined in order ≤ 45 s apart; those paths and the fused server paths
             checked event by event in parallel; the best one's events re-picked in
             parallel (the unrefined path is kept as the next row)
```

**`--mode auto`** is fast mode, then the deep tool loop (up to 120 s in total) only for
a KIS/Q&A query whose best verdict is still under 7.

Where the time went before (deep mode, round 1 of 2026): the multi-turn DeepSeek loop
with thinking, sequential Gemini calls, and a few requests that stalled. Measured on the
live server: one Gemini judge call takes ~4 s whether it looks at 2 or 8 frames, a
DeepSeek call without thinking 1-3 s, a batch 0.3-4 s. Two tail fixes matter as much as
the parallelism:

- A Gemini call still pending after 8 s is **hedged**: a duplicate request races it and
  the first answer wins (`GeminiClient(hedge_after=...)`).
- About one thumbnail request in 50 stalls for ~30 s over the tailnet while an
  immediate retry takes 0.3 s, so images time out after 5 s and are retried
  (`agent/search_client.py`).

**`--mode deep`** is the original multi-step tool loop, for when there is time:

```text
query ──► DeepSeek plan ──► round 1: one /agent/search batch ──► Gemini verifies top 12
            (JSON)            visual (EN caption + paraphrases/scene)       │
                              temporal (multi-scene / TRAKE)              │ best ≥ 9 (and, for Q&A,
                              transcript (VI)                            │ an answer read)? ── yes ──► finalize
                              OCR (distinctive on-screen strings)         │ no
            Gemini + Google Search for names the query only describes     ▼
                                                              DeepSeek tool loop (≤ 6 steps, ≤ 300 s)
                                                              search · inspect · neighbours ·
                                                              read_transcript · web_search · finish
                                                                          │
finalize: ranking (verified matches first) · Q&A answer read from the frame/speech ·
          TRAKE: top 6 paths checked event by event, best 2 moved to each event's first keyframe
```

- **Round 1 is deterministic** after the plan: `agent/plan.py` turns the plan into one
  batch, target scene first, so a query with a clear match needs no agent turns at all.
- **Candidates are moments, not hits** (`agent/pool.py`). Keyframe hits within 4 s of
  each other merge; transcript passages and OCR shots back every moment inside their
  span. Ranking is reciprocal-rank fusion plus a video-level term, then Gemini verdicts
  override it: verified matches (≥ 6) first, contradicted ones (≤ 3) last.
- **What gets verified is not just the fused top.** Fusion favours moments that many
  weak hits agree on, and on round 2 it left a caption's rank-1 keyframe (the gold frame
  of p2-3) unverified at fused rank 32. `CandidatePool.to_verify` gives 60% of the
  verification slots round-robin to each query's best unverified hits, then fills the
  rest by fusion.
- **Gemini looks, DeepSeek decides.** Gemini scores labelled keyframes 0-10 against the
  original Vietnamese query, reads Q&A answers, and picks exact TRAKE frames. DeepSeek
  plans, rewrites queries and chooses tools. Neither model is the source of a Q&A answer
  on its own: answers are read from the video, and web search only resolves names or
  cross-checks. A loop that spends its budget without calling `finish` gets one last
  finish-only turn, so what it learned (a name found on the web, its own ranking) is not
  lost. For Q&A the final answer then comes from `finish`, or else from a dedicated
  reading of the top moment (full frame, neighbours, transcript, and web notes as hints).
- **An early stop must survive a falsifier.** Before round 1 is accepted, Gemini sees
  the top candidate with its neighbouring keyframes and tries to contradict each detail
  of the query. A rejection keeps the score (its first live run rejected two correct
  matches over details a few keyframes cannot show) but hands the run to the loop, with
  the objection attached for the human. On the first live run, 2 of 13 early stops were
  confident wrong matches. Deep mode only.
- **Q&A answers are read at up to 3 moments** of the top video and decided by majority
  (ties to confidence): one keyframe can hide what is being counted or read.
- **A caption says only what a camera shows.** A relationship no picture carries ("hai mẹ
  con", "hai vợ chồng") becomes the bare count, "two people", and a guessed gender or age
  is never written at all. Measured on one live query: "a mother and her child" and "a
  young man and an older woman" both put the right video 34th-66th, while "two people
  holding a phone together" put it first. The same query also showed the other half of
  the rule — the caption keeps every detail the query states or plainly implies, so a
  question about the number shown on a scale still says "digital scale, display showing a
  number".
- **Ambiguity becomes separate queries, not a blurrier one.** `readings_en` holds up to 2
  alternative *pictures* of whatever the Vietnamese leaves open ("cầm điện thoại trò
  chuyện" = two people walking with a phone held up, or two people sitting over a screen),
  each searched as its own query right after the target scene. Paraphrases are fused into
  one vector because they are the same picture; readings never are, since averaging two
  incompatible pictures matches neither. Each reading describes one frame, adds nothing
  the query does not mention, and may be a cautious version that drops what is uncertain.
- **Captions are a channel of their own.** Every 2026 shot has an offline CapRL caption
  (`docs/caprl-shot-captioning.md`); `kind: "caption"` searches them as text. A caption
  says what a keyframe embedding cannot: how many, in what order inside the shot, and
  what a slide or an exam paper actually says. A hit is a span, like a transcript
  passage, so it backs every moment inside its shot window and its text rides along to
  Gemini in the candidate label and into the Q&A reading (`/captions/{video_id}` returns
  a video's captions untruncated). The plan's `caption_en` puts 1-2 caption-style
  English sentences in round 1. A server with no caption index fails that sub-query
  alone, and `SearchClient` drops the channel for the rest of the run.
- **Web search is automatic where it pays** (`agent/web.py`). Cooking queries (15 of 55
  in 2026 rounds 1-2, mostly "Món ngon mỗi ngày") get a Google-Search lookup of the dish
  from the raw description before planning, so the rewrite round can search the
  transcript for the dish name. For cooking Q&A, a second lookup starts from the top
  video's opening narration (where the host names the dish) and asks for the published
  recipe. Q&A whose answer is spoken or world knowledge gets a news lookup, because ASR
  misspells names ("Đèo Bứa" for "Đèo Tà Pứa"). Web answers end in an `ĐÁP ÁN:` line;
  `reconcile_answer` keeps a confident video reading, lets the web decide an unsure
  one, and for spoken answers takes the web's spelling of the same name. `--no-web`
  turns these off.
- **TRAKE paths are fused, re-found inside the video, then judged per event.** Round 1
  runs a tight temporal query (45 s maximum gap between events: every TRAKE key so far
  spans under ~20 s) and a loose one (300 s). Paths are ranked by both, plus how
  strongly the other channels support their video, since the query's context sentence
  (an intro shot) is not an event. The server's DP picks videos well but can take one
  event from minutes away, so each event is also searched inside the leading videos and
  the hits are joined in order by a client-side DP (`pool.local_path`). On round 2 this
  turned p2-8 from a miss (right video, last event 5 min late) into a hit. Verification scores a frame against the whole
  query, where an intro shot the query mentions for context can outscore the events
  themselves. Each of the server's top paths is therefore sent to Gemini as E1..En and
  scored against its own events before the best ones are refined.

### A teammate judging alongside Gemini

*The tab no longer shows these controls: runs are fully automatic (above). The API and
`agent/review.py` still carry them.*

Gemini judges every candidate the agent submits. On a quiet network that is ~4 s a
call; on a congested venue network it can be far longer, and the run waits. So while a
UI-hosted run is in progress, the **Retrieval agent** tab shows what Gemini is being
shown, as it is shown:

- the brief: the query, the plan's scenes (with the scene whose frame will be
  submitted), the Q&A question or the TRAKE events;
- every candidate, marked while its image is with Gemini, with exactly the speech,
  on-screen text and caption snippets Gemini's label carries (`vision.label_context`
  builds both, so they cannot drift apart), and Gemini's verdict once it arrives.

The teammate can act on it (`POST /api/agent/runs/{id}/review`, `agent/review.py`):

| Action | Effect on the run |
| --- | --- |
| **Accept** a candidate | Becomes a 10/10 reviewer verdict, ranked first; the agent stops waiting on Gemini and finishes |
| **Reject** a candidate | Becomes 0/10, ranked last, and is never sent to Gemini; the run goes on |
| Click the same button again | Withdraws the decision; Gemini's own verdict comes back |
| **Type an answer** (Q&A) | Used as the answer; a Gemini reading still in flight is abandoned |
| **Stop** | Finishes now with the current ranking, skipping every remaining model call |

A reviewer's verdict is never overwritten by a model reply, even one that arrives
late. The agent checks for decisions every 0.25 s while it waits on a model call
(`_await`), and those calls run on their own executor so abandoning one does not block
on it. On a live run against the team server, Gemini started judging 15 candidates at
12.8 s, a simulated reviewer accepted one at 13.8 s, and the run finished at 13.8 s.

The CLI runs have no reviewer; `Agent.run(task, review=None, progress=None)` behaves
exactly as before.

### A person as the judge (`judge: "human"`)

The benchmark (`docs/reports/agent-benchmark-2026-09.md`) puts the right moment in the
pool for 93% of queries and finds the judge's *coverage*, not its accuracy, to be the
leak: Gemini sees 15 candidates. A person scanning a grid sees 80, can play the video, and
can fix a "right video, wrong moment" by stepping to the exact frame. So the tab's **Judge**
select can be **You (human review)**: DeepSeek still plans, searches and rewrites, but no
model ever sees a frame (`run_query` builds no vision backend, and `OfflineGemini.judge`
raises if anything tried). `AgentOptions.judge = "human"`, `_Run._human()`.

```text
plan ──► round 1 batch (TRAKE: + paths searched inside the leading videos)
      ──► awaiting_review ◄──────────────────────────────┐
             the person accepts / rejects / scores 5,    │ each request is a command the
             types a Q&A answer, assembles a TRAKE path, │ orchestrator runs on its own
             or asks for more:                           │ thread, then publishes again
               more(hint) · scope(video) · neighbours(candidate) · add_frame(exact frame)
      ──► finish (Submit) / stop / no activity for AIC_AGENT_HUMAN_TIMEOUT s ──► ranking + CSV
```

- **Accepting does not end the run** (it does when Gemini judges): the person may accept
  several moments and then Submit. A 0-10 **score** sits between accept (10) and reject
  (0); **5 means "right video, wrong moment"**, ranked above unjudged candidates.
- **More** is `_rewrite_round` with the person's note and their verdicts in the prompt
  (`HUMAN_REWRITE_NOTE`), and no judging afterwards; it can be repeated. **Search this
  video** runs each scene inside that one video; **Neighbours** and **Add frame** put
  exact keyframes in the pool and pin them in the view however low they fuse.
- **Evidence** is words only. The workspace fetches `/transcript`, `/ocr` and `/captions`
  once per video and cuts them to ±30 s around the frame being viewed, with the query's
  words highlighted (`frontend/src/highlight.ts`: accents and case folded like the
  `vi_folded` analyzer; phrases beat single words; stopwords never light). `GET
  /ocr/{video_id}` is the one route this added.
- **The checklist** (`agent/evidence.py`): DeepSeek splits the query once into 3-8 details
  and, for each, the words that would mention it (`terms_en` for the English captions,
  `terms_vi` for speech and OCR). For each candidate the sentences that contain those words
  are selected from the span **45 s before to 25 s after** the moment, top 3 per detail, and
  DeepSeek judges only those sentences, at temperature 0. Why a span and excerpts: a query
  describes consecutive scenes and a candidate is one frame, so on p2-30 (the "nghêu"
  query) the apron, the vase and the plate are in the shots *before* the frame that shows
  the clam, and inside a caption they sit past character 2,000 of ~3,900; handing DeepSeek
  the raw 22-window span (37k characters) made it answer "nothing found" for the gold moment.
- **Four verdicts, not two.** Per sentence the judge writes `why` first, then `exact`
  (every part the detail names is said), `partial` (right subject, a part missing or
  generic), `conflict` (a part is stated differently) or `unrelated`. In the UI:
  ✓ supported, **~ close** (amber, with the reason), ✗ contradicted, · not mentioned.
  Only ✓ is counted in "n/N details"; the card adds "· k close". The first version had one
  question, "does this support the detail?", and on a live query it ticked "4 con X" and
  "hoa riềng tía" for a candidate whose captions said *two shrimp* and *a vase of pink
  flowers*: nothing was invented, the quotes were verbatim, the judge was lenient (and the
  prompt had told it that X matches anything and colours may drift). The prompt now says: kind
  must be named (flowers ≠ ginger), counts must be equal, X is unknown only as a *name*, the
  action must apply to the right object (lifting a bowl's lid ≠ holding up two shrimp), and
  extra detail in the sentence is not a mismatch.
- **Each language is judged in its own language, by its own call.** Captions (English) go to
  `MATCH_CAPTION_SYSTEM`: an English rubric with worked examples, where each detail is its
  `meaning_en` plus `Same:` / `Different:` lists (the Vietnamese phrase is not sent), and the
  judge answers `why` in English (reasoning in the evidence's language) followed by `note_vi`,
  the Vietnamese sentence the reviewer reads. Speech and OCR (Vietnamese) go to `MATCH_SYSTEM`,
  a Vietnamese rubric that sees the detail as the query wrote it. The two calls run side by side
  and their verdicts are merged (✓ over ~ over ✗ per detail, nearest sentence within reach); if
  one fails the other's verdicts are kept, and only if both fail is there no summary. The first
  version sent everything to one Vietnamese prompt, so an English caption was translated and
  compared in one step. The count check in code applies to both languages.
- **The judge is told what each detail means.** With only the Vietnamese phrase it read
  "hoa riềng tía" as *purple* and marked a vase of red ginger "not riềng tía" (✗) while
  yellow-and-purple daisies were only "close" (~), and a man holding a plate was "close" to
  "cầm lên 2 con X". The details call now also returns `meaning_en` (a sentence saying what a
  caption must state, with the canonical English name: riềng tía = red / torch ginger, red to
  magenta), `same_en` (names that still count) and `different_en` (look-alikes that do not);
  the judge sees them under each detail. And the split is fixed: `partial` is only for a part
  that is *missing or generic* ("flowers", no count); a different kind or colour, or an action
  applied to a different object (holding the plate, lifting a lid), is `conflict`. On p2-30 the
  flower detail is now ✓ for the gold video and ✗ for most others.
- **A count is checked in code, not by the model.** A detail that names a quantity of two or
  more (digits, `hai/ba/bốn`, `two…ten`) can only be `exact` if the sentence states that
  number; otherwise it is downgraded to `partial` with the numbers in the note
  (`count_mismatch`). One exception, both in the code and in the prompt: a detail naming a
  graphic mark the broadcast draws onto the footage afterwards (a circle, arrow, box, highlight)
  is not counted at all — how many times the device is used is a count across the whole story,
  which no handful of frames can total. Asked about p2-6's "2 vòng tròn màu đỏ khoanh vùng vị trí",
  the state of the art was one instance of the mark visible in the sampled frames, and the judge
  answered `conflict` ("only one, not two") in three calls out of three, so the instruction now
  travels on **the detail's own line** (`MARKER_HINT`, next to the detail text the judge reads)
  rather than only in the system prompt's rules: measured 8/8 `exact` on the same frame.
- **Retrieval finds candidates; evidence orders them, and only support counts.** The searches (visual,
  audio, OCR, temporal) only propose candidates and any of them can be wrong: on p2-30 a frame of blue
  bowls and shrimp led the agent's order because three visual queries ranked it first. So the view's order
  is not retrieval's. The server (`_by_evidence`, `agent.evidence`) ranks the candidates nobody has settled
  by their checklist, in two tiers: **supported** (score > 0, in bands of `TIE_BAND` = 0.10, ties settled
  by own share before retrieval, see below) and **no evidence** (unread, or read and
  silent; retrieval order). Accepted candidates still lead and rejected ones still trail. Nothing moves
  while the checklist is still arriving. The snapshot carries `agent_rank`, `evidence`
  (`score`/`share`/`video`/`tier`), `detail_weights` and `common_details`; **Order → Retrieval order**
  shows the searches' ranking to compare; `Submit`/`Stop` send the screen's order (`order`) and the export
  follows it (in evidence order too when none arrives).
  `score = 0.5 × own + 0.5 × video`:
  - *own*: the weighted share of details the words around the moment say (✓ 1, ~ 0.25), each check
    scaled by `proximity` = 1/(1 + d/20 s) to its shot, so a frame is judged by its own shot and its
    neighbours, not by one half a minute away (this is also what separates moments of one video);
  - *video* (`video_coverage`): the share of details ANY read moment of that video says. A story query is
    spread over shots and a frame shows one of them, so no single moment says it all; the right video says
    most of it somewhere. This is what makes a story query's evidence strong instead of 0.16.
  - **A contradiction lowers nothing.** It is shown (✗, "n conflict") but not counted. Measured, not assumed:
    with contradictions demoting (tier below "no evidence"), the right moment's mean reciprocal rank over
    ten queries was 0.36 against retrieval's 0.50 (p2-14 fell from 1st to 12th: the caption says "a black
    strap" for the query's white string); with contradictions ignored, 0.53. Captions get colours, counts
    and identities wrong, so a conflict is as often the caption's mistake as the candidate's.
  - **Retrieval breaks a tie only after the candidate's own moment has had its say.** Video coverage is
    what lifts a story query's evidence off the floor, but on its own it can also lift the *wrong* video:
    reported on p2-6, where a duplicate broadcast of the same real event (L21_V018) put a candidate one
    place ahead of the gold video's (L22_V024) in the same `TIE_BAND` band. Coverage alone decided it --
    the gold video's coverage 0.698 against the duplicate's 0.755 -- while the gold candidate's own moment
    read *more* of the query. Retrieval, meanwhile, had no opinion to offer: the two videos are the same
    event, so its order is not evidence of which is right. So inside a band the key is
    `(score, own share, retrieval position)`: a candidate whose own moment supports more of the query
    leads, and only candidates tied on that too fall back to retrieval.
  Evaluation (a snapshot of the 80 shown per query, rank of the first moment inside 250 frames of an answer):
  36 runs of 21 distinct round-2 queries whose answer was among the 80 shown, in three collections (runs of
  the same query are not paired: the plan, hence retrieval, varies between runs):

  | collection | queries | MRR retrieval | MRR evidence (band .10) | hit@10 retrieval → evidence |
  | --- | ---: | ---: | ---: | --- |
  | tuned on | 10 | 0.496 | 0.546 | 9 → 9 |
  | held out | 10 | 0.605 | 0.679 | 10 → 11 |
  | 40 candidates read | 16 | 0.536 | 0.572 | 12 → 13 |

  The gain is real but small and uneven: it rescued p2-29 (15th → 1st) and p2-25 (17th → 3rd) and cost
  p2-14 (1st → 8th) and p2-22 (1st → 7th), where a wrong video's captions match a visual query as well as
  the right one's. Evidence and retrieval are complementary, and the checklist is a coarse signal, not a
  replacement. Reading 40 candidates instead of 20 covered 74% of the answer candidates against 59-75%
  and showed no gain in MRR; `SUMMARY_TOP` nevertheless became every candidate shown (80 in review mode,
  `PROGRESS_TOP_HUMAN`) after p2-6, where the gold candidate sat at rank 25 of the fused order with no
  checklist at all -- an ordering rule cannot reorder what it never read.

  **After the distinctiveness prior and the action/purpose fix**, re-run on 22 further queries (fresh
  runs, so not paired with the numbers above): MRR retrieval 0.480, evidence 0.488 -- close to a wash
  overall, smaller than the +0.05-0.07 above. p2-1 and p2-30 (the reported cases) both improved (2nd→1st,
  2nd→1st); p2-28-qa regressed (1st→5th), but all five top candidates were the same shot of the right
  video seconds apart with an equally strong checklist, which the frame-tolerance metric treats as a miss
  though a submission of any of them would likely score -- a measurement artifact more than a real loss.
  The prior does what it was built to do (weight a guessable detail down before any candidate is read);
  it does not, on this evidence, reliably move the aggregate further than the ordering change already did.

- **DeepSeek looks at the shots around the candidate, several frames of each (`checklist_vision`, on
  by default in human judge mode).** A caption is one model's guess about a frame, made once, offline,
  without the query in mind; a caption of the reported p2-1 candidate never independently confirmed the
  lion, and elsewhere a caption called three small items "two raw shrimp". `agent.evidence.
  vision_summarize` sends each candidate as a **set of images** (`VISION_MATCH_SYSTEM`): its own
  keyframe, then the keyframes of the shots around it -- nearest shots outward, `VISION_CONTEXT_FRAMES`
  = 4 keyframes of each, up to `VISION_CONTEXT_MAX` = 16 frames per side within `VISION_CONTEXT_SECONDS`
  = 180 s, each labelled `[C7] shot 3s later in the same story`. A person asked for exactly this on the reported p2-1
  candidate ("also consider the 2-3 shots right before and right after the frame: the man waking up and
  shaking the lion are in shots 4 and 5"), and both halves turned out to be needed and measurable. The
  *shots*: sampling by time stepped over shot 4 (the pumpkin-like object is carried away in the shot 3 s
  after the moment, and the waking in the shots 22 s and 36 s after it), so the context is chosen per
  shot from `GET /shots/{video_id}` instead -- a video's shots are the units a story's beats live in.
  The *several frames*: one frame per shot had the judge answer "unrelated" to "he wakes the animal up"
  in every run, where a shot's three keyframes had it answer `exact` in every run on the gold video and
  still `unrelated`/`partial` for the same detail on a different team's skit in another video, which is
  the property that matters. Each candidate is up to 25 images at 480 px, **one candidate per call**
  (`VISION_CHUNK = 1` -- mixing two candidates' stories into one judgement measured worse than the same
  images judged alone) and three calls in flight. The same DeepSeek client already used for planning is
  the judge (no extra model or key). Its answer only ever **upgrades** a check (never demotes what
  speech/OCR already supported) and only when it has an opinion; a detail the reply is silent on keeps
  its text-sourced verdict. A vision verdict counts at the candidate's own moment (the shots around it
  are that moment's story) and goes through the **same** `detail_weights`, so a criterion vision
  confirms still counts by its `distinctiveness` prior × empirical share, and the same numeric guard:
  a detail naming a count is capped at `partial` when the judge's own words state a different number
  (a live p2-30 run marked "places 4 X on a white plate" supported from a frame showing two clams).
  Details holding a placeholder (`4 con X`) are not resolved to one species in `meaning_en` -- the
  count, colour and action are what the checklist checks, while which species X is stays a question for
  the person; guessing one there made the gold video's own shot a "conflict". A detail worded as a
  resemblance ("một vật trông như trái bí đỏ") keeps that resemblance in `meaning_en` ("an object that
  LOOKS LIKE a pumpkin") and lists look-alikes in `same_en`, so a prop or decoration that resembles it
  is a match rather than a conflict -- the query said the object looks like a pumpkin, not that it is
  one. What this does **not** fix, measured rather than assumed: a detail naming a change in a *person*
  ("người đàn ông thức dậy") still comes back `unrelated` -- on the p2-1 gold the frames contain the
  performer lying back on the mat at 41.8 s and upright at 48 s, and the judge answers "no man is shown
  lying down then getting up" with the shot's first frame, with two frames of that shot, with the
  action's visible signature spelled out in `same_en`, and with a describe-then-judge pass. The two
  beats the person pointed at are not equally readable; one is ticked now, the other is left to them.
  This is still a hint: the person reviews the reasons and the frame themselves, and no model decides
  anything -- `checklist_vision` costs one more DeepSeek call per candidate (the run's checklist phase
  went from ~30 s to ~90 s for 80 candidates on the live server) and nothing else changes about who
  accepts, rejects or scores a candidate. A backend that cannot do this (no vision support, no key, a
  fetch failure) is caught and simply leaves the checklist as it was; `AIC_AGENT_CHECKLIST_VISION=0`
  turns it off.
  **A reply lands on the candidate whose images it was shown, not on the id it types back.** The
  schema asks the judge to name the candidate it is answering about, and it does get that wrong: on
  p2-6's gold frame a call whose images were labelled `[GOLD]` answered `{"id": "C1"}`, and the reply
  was thrown away whole — the candidate silently kept a text-only checklist. `VISION_CHUNK` is 1, so
  the label is the truth (`attribute`); an id is only what tells SEVERAL candidates of one call
  apart, which is why a wider chunk still uses it and still drops an id nobody asked about. The same
  call is also told, in its own prompt, exactly how many candidates it carries and which ids, after
  three calls out of eight were measured answering with a dozen *invented* candidates (C1, C10, …),
  running past the token cap and being dropped — and `judge(..., salvage=True)` now keeps the
  complete entries of a reply the cap cut off instead of discarding all of it. Measured together on
  that frame: 0/8 calls dropped, against 3/8 before, with the mark detail `exact` in 8/8.
- **Caption is suspended while vision is available.** A caption is one model's guess about a frame,
  made once offline, without the query in mind; asking a second model to re-describe what the first
  one already may have got wrong is not worth the call once a vision-capable judge can look at the
  same pixels directly. With `checklist_vision` on (the default), `summarize()` never submits the
  caption call at all -- only speech/OCR (`MATCH_SYSTEM`, real evidence a frame cannot give) and
  vision run. Turning `checklist_vision` off brings the caption channel straight back, unchanged.
- **A detail's weight has a prior, before any candidate is read.** `detail_weights`'s empirical half
  (below) only kicks in once `ENOUGH_TO_JUDGE` = 6 candidates have a checklist; before that every detail
  weighed the same, so a detail almost every candidate of the right *kind* of scene satisfies -- "a
  yellow animal" is guessable as the lion once the query plan has read the corpus as lion dance -- counted
  as much as one true of only the right moment ("carries the pumpkin away to hide it"). The details step
  now also asks DeepSeek for `distinctiveness` (0-1: would knowing this detail is true actually narrow
  down which candidate is the right one, or is it true of nearly every candidate of this kind), turned
  into a prior weight `0.2 + 0.8 × distinctiveness` and multiplied into the empirical factor
  (`_prior_weight`). On p2-1 ("một con vật màu vàng" 0.3, "mang bí đỏ đi giấu" 0.9, "đánh thức con vật"
  0.9) this alone moved the gold moment from 2nd to 1st before any empirical adjustment could apply.
- **An action's purpose is not itself checkable.** A detail can name an action together with why it
  happens or what follows ("carries X away, to hide it"; "wakes up because he cannot find it"). The purpose
  is usually not in the same shot as the action -- it happens off-screen or later -- so a caption that
  shows only the action ("running across the stage carrying the pumpkin") was scored `partial` for lacking
  the word "hide". Both judge prompts now say the purpose is a reason, not a separate checkable part: show
  the action and do not contradict the purpose, and it is `exact`.
- **What went wrong on p2-1, and the rules it added.** With evidence ordering the gold moment (retrieval's
  first candidate) fell to 18th: the checklist gave it two false ✗. Causes and fixes:
  - *The sentence that named the lion was never shown to the judge.* The query says "một con vật màu vàng";
    the terms were "animal, yellow", so the caption's "yellow lion dance costume lies sprawled" was not
    selected, while "the yellow flower is next to a pumpkin" (a different shot, +25 s) was. The details
    step now sees the search plan's English reading of the query and the corpus description, so "animal"
    is expanded to "lion / lion dance costume" (`_plan_reading`, `DETAILS_SYSTEM`).
  - *Sentences were ranked by how many terms they held.* Now by the rarest term they hold plus a quarter of
    each other's (`log(1 + N / (1 + df))`, `RARE_SHARE`), and 4 per detail instead of 3.
  - *A different thing that shares a word was called a conflict.* "A yellow flower" for "a yellow animal" is
    `unrelated`; `conflict` is only for the same element stated differently (both judge prompts, with
    worked examples).
  - *One detail held two objects* ("five people beside a yellow animal"), which no single sentence can
    support; the details step splits them.
  - *A partial is worth 0.25 now, and scores within `TIE_BAND` (0.10) are a tie* that own share settles
    first and retrieval order only after that, so a hair of evidence never reorders candidates. On a story
    query nearly every frame of the right video is "men gathered" and "a lion nearby" in part.
  On the live server, p2-1's gold moment is 1st again (its evidence 0.16 is weak: the query is a story
  and a frame shows one scene of it) and p2-30's are 1st and 2nd.
- **Common details are dimmed.** A detail nearly every read candidate satisfies ("a woman in an
  apron", "they talk") tells the moment apart from nothing, so once six candidates have a
  checklist it is marked *common* and weighs little in the score and the bar (weight = 1 − share of
  candidates supporting it, at least 0.15). The model was asked to flag such details itself and never did.
- **Captions are approximate, and that is a separate limit.** They are written by a captioning
  model from the frames: on L26_V433 a plate of three small items became "two raw shrimp", and
  logos come out as "MONGON monday". They never said the ginger flower or the four items the
  query wants, so a strict reading of them is the right one; the UI labels them "automatic, can
  be wrong". For Q&A the same words give **suggested answers** (chips, plus the web lookup's
  `ĐÁP ÁN`; on p2-30 it reads "nghêu" from the caption and the speech); a suggestion is only
  submitted, marked unconfirmed, if the person submits without typing one.
- **Captions come from disk.** `GET /captions/{video_id}/windows` reads
  `data/2026/captions/parts/...` (falling back to the index); `GET /captions/{video_id}`
  needs the Elasticsearch caption index, which a deployment may not have built (it is also
  what the agent probes to decide whether the caption *search channel* exists, so it must
  keep answering 503 there). The review view and the checklist use the disk route; before
  that they used the index route and every caption was silently missing.
- **Web lookups stay on Gemini** (Google Search) and only run if a Gemini key is present;
  no key means no lookup, not an error. Gemini never receives an image.
- **TRAKE**: slots E1..En are filled from the inspector (`→ E2` on the frame being
  viewed, or `Set as E2` in the player) or from a path the agent found; a complete path is
  sent as `sequence` and leads the exported rows, the agent's paths follow.
- Keyboard: `J`/`K` move, `A` accept, `R` reject, `5` wrong moment, `X` clear, `N`
  neighbours, `V` search this video, `Enter` play.
- Every action is logged with its second (`review_log` in the run JSON) so the person can
  be measured the way the agent is: `python -m agent eval <run> --answers ...`.

### Why this shape

Drawn from agents built for the same kind of competition:

- *Can Agents Win the Video Browser Showdown?* (arXiv 2609.07311). An autonomous
  agent matched experts on textual KIS. The target was outside the initial top-100 in
  46% of solved tasks and was recovered by rewriting (78% of the time after an empty
  look) and by searching inside candidate videos. Most errors came from committing too
  early, and a separate check rejected a third of wrong submissions. Hence the
  verify-before-commit loop, video-scoped search, and a human doing the final check.
- *Deep Video Discovery* (arXiv 2505.18079). A small set of search-centric tools at
  several granularities beat fixed workflows. It also named two failure traps: endless
  inspecting and repeating queries. Hence the step and time budgets and the
  duplicate-query guard.
- RAPID (AIC 2024), QUEST and MADTempo (AIC 2025). Parallel query variants, and
  resolving out-of-knowledge entities before searching. Hence the lookup step.
- The team's own benchmarks (`docs/reports/visual-holistic-paraphrase-benchmark-2026.md`,
  `visual-query-expansion-benchmark-2026.md`). Faithful paraphrases fused by *embedding
  mean* (caption weight 0.5) beat the single caption on dev and holdout. Rank fusion of
  the same paraphrases got worse with every one added, and shortened "aspect" captions
  lost outright. Hence one visual query per scene: the caption plus up to 4 faithful
  paraphrases, fused server-side.
- The deployed transcript index is built from cleaned transcripts that write numbers as
  digits. On the live server "130.000" found its video and "một trăm ba mươi ngàn" did
  not, so transcript queries keep digits (`AIC_AGENT_SPOKEN_NUMBERS=1` spells them out
  for a raw-ASR index).

## Server side: `POST /agent/search`

One request runs up to 12 sub-queries in parallel. A sub-query that fails reports its
own `error` instead of failing the batch.

```json
{"queries": [
  {"kind": "visual",   "id": "v0", "text": "red boat on a river", "paraphrases": ["a river with a red boat on it"],
                       "limit": 200, "video_ids": ["L22_V001"]},
  {"kind": "similar",  "id": "s0", "video_id": "L22_V001", "frame_idx": 1234},
  {"kind": "audio",    "id": "a0", "text": "cá nhám", "method": "keyword"},
  {"kind": "ocr",      "id": "o0", "text": "Nà Ní"},
  {"kind": "temporal", "id": "t0", "events": ["...", "..."], "audio_text": "...", "max_gap_seconds": 180}
]}
```

- `paraphrases` (up to 4) are encoded with `text` and fused into one query vector by
  `backend.subquery_fusion.combine_vectors` (the frozen `B*` weights), then searched once.
- `video_ids` scores *every* keyframe of those videos: it reconstructs their Flat index
  rows instead of filtering a corpus-wide top-k.
- Hits are compact: `video_id`, `frame_idx`, `time`, `score`, `rank`, plus `text`,
  `start` and `end` for transcript and OCR hits. Temporal hits are videos with their
  best 3 paths.
- It runs under the coordinator identity `<login>#agent`. The agent can therefore share
  a teammate's login without the two cancelling each other's searches (latest-wins
  answers 409 when one login overlaps itself). The agent serializes its own batches.

### Servers without `/agent/search`

With `AIC_AGENT_SEARCH_API=auto` (the default), the first 404 from `/agent/search`
switches the client to `agent/compat.py`. That module maps every sub-query onto the
public `/search`, `/search/image` and `/search/temporal` routes, paging `/search/next`
for video-scoped queries. The agent runs unchanged, with three costs:

- Sub-queries run one at a time. Each audio query pools its full 10k-result snapshot,
  about 5-10 s on the team server.
- Paraphrases are dropped: the public routes take one text, and rank-fusing paraphrases
  measured worse than the caption alone.
- Searches run under the caller's own login, so a teammate searching in the UI with
  that login can supersede them. The client retries 409s.

## Server-hosted runs (the UI tab)

The **Retrieval agent** tab posts the query, its kind, the `--mode` (default `fast`)
and the teammate's own `DEEPSEEK_API_KEY`/`GEMINI_API_KEY` to `POST /api/agent/runs`;
the server runs the same orchestrator in a bounded thread pool and the panel polls
`GET /api/agent/runs/{run_id}`. One run per user at a time, 1 h TTL, and the keys
live in memory for the run only — never written to disk or logs. The run's JSON,
`report.html` and `submission.zip` land under `AIC_AGENT_OUT_DIR`
(`/workspace/agent` in the container); `GET /api/agent/runs/{run_id}/submission.zip`
serves the archive. `GET /api/agent/runs/active` answers with that user's live run
(or `null`), which is what a reloaded page or a second tab shows instead of the
start route's "already in progress" refusal with nothing to look at. The orchestrator
reaches the server over loopback
(`AIC_AGENT_INTERNAL_URL`, default `http://127.0.0.1:8000`) under the caller's own
identity. `backend/agent_runner.py` implements this; see AGENTS.md for the wiring.

## Local vision

Every image the agent looks at goes out through one function —
`GeminiClient.judge` in `agent/llm.py`. `AIC_AGENT_VISION` chooses what sits
behind it: `gemini` (the default), `deepseek`, or `local`, which puts a model on
this machine behind the same call so the five jobs in `agent/vision.py`
(`verify`, `falsify`, `check_sequence`, `pick_frame`, `answer`) keep their
signatures and their schemas while the keyframes stop leaving the building. That
matters on competition day, when the venue's uplink is the bottleneck and the
image payload is the heaviest thing the agent sends.

`agent/vlm.py` holds the backends and `agent/vlm_models.py` the loaders; the
harness is `scripts/benchmarks/bench_agent_vision.py`.

**The models are downloaded, not fetched at run time.** The container is offline
and read-only, so `~/.cache/aic/vlm` is materialised on the dev machine and
mounted at `/models/vlm`:

```bash
uv run python -m scripts.vlm.fetch_models          # all candidates
uv run python -m scripts.vlm.fetch_models --list   # what is present
```

**A small model is prompted differently, and that is deliberate.** The prompts in
`agent/prompts.py` are long, in English, and left to Gemini's
`response_json_schema` to define the output shape — which means a local model
handed the same text has nothing telling it what to emit. Three consequences,
all handled inside `agent/vlm.py`:

- The schema is rendered back into words and appended to the prompt.
- `verify` and `sequence` fan out into **one call per frame** with a short prompt.
  Measured on Vintern-1B-v3_5: handed five frames and a five-element schema it
  invents the ones it cannot see and runs past its token budget, and a `reason`
  field it is not asked to write is what it spends the budget on. Neither compact
  prompt asks for one — `reason` is only ever logged, and `CandidatePool.ordered`
  ranks on `score` and `scene`. The prompt's *language* is per family
  (`AIC_VLM_PROMPT`): Vietnamese for the Vietnamese-tuned model, English for the
  Qwen VLMs.
- Ids are **stamped from the label, not read from the reply**. Asked about
  `[C1]`, Vintern answers `"L21_V024 t=810.6s"`; `Inspector.verify` drops any id
  it does not know, so trusting it would score nothing, silently.

Measured on the RTX 3060, warm, one frame per call, alongside the server's own
PE-Core and Qwen3-Embedding:

| Model | s/frame | Weights | Fits beside the server |
| --- | ---: | ---: | --- |
| `5CD-AI/Vintern-1B-v3_5` | 1.9 | 2.0 GB | yes |
| `Qwen/Qwen3-VL-2B-Instruct` | 4.3 | 4.1 GB | yes |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 4.6 | 7.3 GB | yes, but under 0.5 GB spare |

The `nvidia-smi` figure for the server's own models with the container warm is
**~4.2 GB**, so 4.2 + 7.3 is 11.5 GB against a 12.0 GB card: the 3B does load,
and ran a full 91-case sweep with the container up. It is still the wrong choice
— that margin is one allocation deep, and the 2B is both smaller and more
accurate (see `docs/reports/agent-vision-backends-2026.md`). A model that cannot
fit at all is what `AIC_VLM_URL` exists for: run it in another process or on
another host. `AIC_AGENT_VISION=gemini` remains the default and the fallback.

## Setup

```bash
uv sync --only-group server --only-group agent   # or add --group agent to your usual sync
```

Configuration comes from the environment, falling back to the repository `.env`
(gitignored). UI runs need none of this on a teammate's machine — the keys are pasted
into the tab and only the server needs the `AIC_AGENT_*` deployment settings. On a
split-development workstation, point `AIC_AGENT_URL` (and `AIC_AGENT_TRUSTED_USER`)
at your own app so runs use your local orchestrator; see
[Split Development](split-development.md):

| Variable | Default | |
| --- | --- | --- |
| `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` | required | |
| `AIC_AGENT_URL` | `http://127.0.0.1:8000` | The team server, e.g. its `https://<host>.<tailnet>.ts.net` name |
| `AIC_AGENT_USER`, `AIC_AGENT_PASSWORD` | unset | Basic Auth, only for a deployment that still uses it. Over Tailscale, `tailscale serve` identifies you and no credentials are needed |
| `AIC_AGENT_TRUSTED_USER` | unset | Direct-to-uvicorn development only: sent as `X-AIC-User` |
| `DEEPSEEK_MODEL` / `GEMINI_MODEL` | `deepseek-flash` / `gemini-3.8-flash` | |
| `DEEPSEEK_THINKING` | `enabled` | Thinking mode for deep-mode loop turns (fast mode never thinks) |
| `DEEPSEEK_PLAN_THINKING` | `disabled` | Planning takes 2-3 s without thinking and 12-28 s with it, and plan quality was comparable. Deep mode and a run with **Think harder when planning** think anyway |
| `GEMINI_THINKING_LEVEL` | `low` | |
| `AIC_AGENT_SEARCH_API` | `auto` | `batch` requires `/agent/search`; `compat` always uses the public routes |
| `AIC_AGENT_SPOKEN_NUMBERS` | `0` | `1` spells digits out in transcript queries (raw-ASR index only) |
| `AIC_AGENT_JUDGE_STYLE` | `moment` with OpenRouter, else `keyframe` | `moment`: one call per candidate over its keyframe and the shots around it (`agent/judge.py`); `keyframe`: five keyframes to a call (`Inspector.verify`) |
| `AIC_AGENT_VISION` | `openrouter` when an OpenRouter key exists, else `gemini` | Which model does the five vision jobs: `gemini`, `local` (a model on this machine), `deepseek`, or `openrouter` (one shared team key, default `qwen/qwen3.7-flash`, see `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` below). `local` makes `GEMINI_API_KEY` optional — the web lookup goes without — but it does not need to be. `deepseek` and `openrouter` are remote APIs like Gemini, so they solve nothing about a congested uplink; they exist because they may simply judge more accurately, or (`openrouter`) because the Gemini quota ran out |
| `OPENROUTER_API_KEY` (or `QWEN_API_KEY`), `OPENROUTER_MODEL` | required for `openrouter` / `qwen/qwen3.7-flash` | The team's one shared key — never a per-teammate paste. Also what `backend/checker.py`'s "Check by AI" (`docs/checker.md`) uses, whether or not the agent's own vision backend is `openrouter` |
| `AIC_VLM_URL` | unset | Runs the local model in another process or on another host (`POST /core/vision/judge`), the same split as `AIC_CORE_URL`. Only read when `AIC_AGENT_VISION=local` |
| `AIC_VLM_MODEL` | `deepseek-flash` / smallest candidate | The model the selected vision backend uses: a registry slug for `local`, a DeepSeek model id for `deepseek`. **Only `deepseek-flash` reads images** — `deepseek-v4-pro` answers a vision call with schema-shaped JSON whose every score is 0, which reads as a judge that works and scores badly. `bench_agent_vision` probes any remote judge with a solid-colour image before scoring it. The local default is deliberately the smallest, so a first run on a card that may already hold PE-Core and Qwen does not start by trying to fit the largest. See *Local vision* below |
| `AIC_VLM_DIR` | `~/.cache/aic/vlm` | Where the snapshots are materialised; `/models/vlm` in the container |
| `AIC_VLM_DEVICE` | `cuda` | `cpu` when the card is full; a 2B model on CPU is only viable in `--mode deep` |
| `AIC_VLM_MAX_NEW_TOKENS` | `128` | A one-object reply is ~40 tokens. This is what stops a rambling model from spending the time budget, and a reply cut off mid-JSON parses as nothing |
| `AIC_VLM_IMAGE_SIDE` | `448` | Long edge a frame is resized to before judging |
| `AIC_VLM_ANSWER_SIDE` | `1280` | The `answer` job has to *read* text, so it keeps more resolution |
| `AIC_AGENT_HUMAN_TIMEOUT` | `900` | Human judge only: seconds without a reviewer action after which the run finishes with its current ranking, so an abandoned tab does not hold one of the five run workers |
| `AIC_AGENT_CAPTIONS` | `1` | `0` stops the agent reading CapRL shot captions into candidate labels |
| `AIC_VLM_PROMPT` | per family | `vi` or `en` for the short local prompt. Defaults to `vi` for `InternVL` (Vietnamese-tuned; give it English and it answers in English) and `en` for the Qwen VLMs. Override it to measure the other side — both are scored in `benchmarks/visual/vlm_verify_2026.json` |

## Commands

```bash
python -m agent check                                    # server, DeepSeek and Gemini reachable?
python -m agent plan queries/SOTUYEN2-bo-de-thi.zip      # plans + round-1 queries only; no server needed
python -m agent ask "Nhiều con rùa nhỏ đang ăn rau cải" --kind kis
python -m agent run queries/SOTUYEN1-bo-de-thi.zip --answers queries/Round12026_answer.zip
python -m agent run queries/SOTUYEN1-bo-de-thi.zip --mode deep --only query-p1-9-qa   # hard case, minutes
python -m agent eval artifacts/agent/<run> --answers queries/Round12026_answer.zip --tolerance 250
```

`run` and `ask` write `artifacts/agent/<timestamp>/`:

- `<query>.json`: plan, ranking with verdicts and evidence, answer, and the full tool trace.
- `submission/<query>.csv` and `submission.zip`, in the organisers' format (at most 100
  rows, answers quoted when needed).
- `report.html`: the top 20 candidates per query with thumbnails loaded from the
  server, for the final human check.
- `usage.json`: token counts per model. With `--answers`, also `scores.json`.

Labelled data for evaluation: `queries/round1-2025.csv` (answers to
`queries/query-p1-groupA`), `queries/Round12026_answer.zip` (answers to
`SOTUYEN1-bo-de-thi.zip`), and the team's visual benchmark for later rounds, which
carries gold frames but no Q&A answer strings:

```bash
python -m agent run queries/SOTUYEN2-bo-de-thi.zip   --answers "benchmarks/visual/visual_embedding_benchmark_2026_en.json::Round 2 - 2026"
```
 The organisers do not publish a frame tolerance, so
`eval` counts a frame within `--tolerance` frames (default 250, about the 10 s the team's
visual benchmarks use) of any gold frame as a hit, or anywhere between the two frames
of an organiser key that marks a clip by its first and last frame. Use its
numbers to compare agent versions, not to predict the leaderboard.

## Results so far

Same 10 s tolerance as `benchmarks/visual/visual_embedding_benchmark_2026_en.json`. The
PE-Core row is that benchmark's single human-approved caption per case. Its case sets
differ slightly (it scores TRAKE events separately), so read the comparison as
direction, not exact margin.

| Round (queries) | Method | R@1 | R@5 | MRR | Video@1 | Time per query (median / p90) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Round 1 (25) | PE-Core caption | 22.2% | 37.0% | 0.31 | | |
| Round 1 (25) | deep v2 / v3 (2026-09-18, compat) | 56% / 52% | 76% / 72% | 0.63 / 0.61 | 84% / 72% | 29 s / ~170 s |
| Round 1 (25) | fast v6 / v7 (2026-09-19, batch) | 48% / 48% | 60% / 68% | 0.54 / 0.56 | 60% / 72% | 18-20 s / 28-40 s |
| Round 1 (25) | fast + readings (2026-09-21) | 52% | 64% | 0.58 | 64% | 18 s / 28 s |
| Round 2 (30) | PE-Core caption | 38.9% | 63.9% | 0.50 | | |
| Round 2 (30) | deep v3 (compat) | 66.7% | 73.3% | 0.70 | 86.7% | 99 s / ~250 s |
| Round 2 (30) | fast v6 | 56.7% | 73.3% | 0.64 | 76.7% | 19 s / 35 s |
| Round 2 (30) | fast + readings (2026-09-21) | 63.3% | 80.0% | 0.70 | 73.3% | 21 s / 34 s |

- The 2026-09-21 rows are the caption-blind numbers for the same code the caption channel
  ships in: the deployed server has no caption index yet, so those sub-queries were dropped.
  On round 2 they match deep mode (66.7% / 0.70) at a fifth of the time; on round 1, 7
  queries improved and 5 worsened against the previous run, most by one or two ranks.
- Fast mode's earlier gap to deep mode was about a tenth of R@1 for a 5x shorter run;
  `--mode auto` sits between them. R@5 is what matters most when a human checks the list.
- Round 2 was never used while tuning deep mode; fast mode's fixes (verification
  quota, local TRAKE paths, scene coherence) were found on both rounds.
- One query is 3-4 points, and LLM plans and verdicts vary between runs: the same
  fast code scored 40-56% R@1 on round 1 across runs. Treat a few points as noise.
- Round-1 retrieval is the same in both modes (the gold video is in a round-1 visual
  top 20 for 25 of 30 round-2 queries); what differs is how many candidates are looked
  at and re-searched.
- TRAKE: 2/2 on round 2 with the in-video local paths (previously 1/2). Round 1's p1-16
  (lion dance) still misses: the gold video reaches the top 4 only in some runs.
- Q&A answers: 2/4 on round 1 in the latest run. p1-17's pass name is fixed by the web
  lookup ("Đèo Bứa" in the ASR, "Đèo Tà Pứa" in the news) when the right video is on
  top; counting (p1-15: 10-11 read, 12 in the key) stays wrong even at Gemini's `high`
  thinking level, which also took 20-28 s. The team benchmark has no answer strings
  for round 2.
- Remaining failure types: right video but wrong frame in long lecture shots; confident
  look-alikes among same-genre videos (lion dances, cycling); VLM counting.

## Tuning

- Prompts: all in `agent/prompts.py`. Iterate with `python -m agent plan`, which is
  cheap and fast.
- **The planner prompt states rules, not cases.** It once carried the stories of the queries
  its rules came from ("hai mẹ con", the fish on a scale, "4 con X", the tug-of-war slide),
  and the planner copied their words into other plans -- a lecture query searched the speech
  for the very topic words an example had used. Those stories now live here, and the prompt
  says an example only illustrates a kind of case. `python -m scripts.benchmarks.check_planner
  --runs 3` replans each of those queries and checks the property its rule protects (no
  relationship words in a caption, the scale's display kept, the unknown "X" never named, one
  slide planned as one lecture scene, no phrase copied from the prompt, ...). Run it after any
  prompt edit: the first general rewrite lost "hai mẹ con" 3 times out of 3 until the rule
  listed the words a caption must not contain. 18/18 with the current prompt; 11/12 with
  thinking on (the miss was the check's own stale word list, since fixed).
- Budgets: `--mode`, `--time-budget` (default 20 s fast, 300 s deep), `--max-steps`
  (deep), and `AgentOptions` / `FAST_DEFAULTS` in `agent/orchestrator.py`
  (`verify_top`, `verify_chunk`, `accept_score`, `early_stop_score`, `trake_refine`).
  The budget is soft: a step already started finishes, and a Q&A answer is always read.
- Fusion weights: `KIND_WEIGHTS`, `POINT_WINDOW_SECONDS` and `VIDEO_WEIGHT` in
  `agent/pool.py`.

## Not done yet

- **Captions for the humans too**: the caption channel is agent-only so far. `/search`
  could take a caption query, and the UI show a shot's caption beside its keyframe.
- **Dense captions**: `build_caption_index --dense` plus
  `bench_caption_search --method hybrid` decides whether embedding 103k captions beats
  BM25 over them.
- **The caption channel's fusion weight**, now `KIND_WEIGHTS["caption"] = 1.0`. The
  offline comparison (`docs/reports/caption-retrieval-vs-pe-core-2026.md`) has captions
  losing to PE-Core as a *ranker* (MRR .295 vs .466) while retrieving **more** golds
  (100/105 vs 95/105) — a recall channel ranked badly, which is either worth less than
  the visual channel or worth more. The weight is a guess until measured.
- **Image exemplars** for out-of-knowledge entities (QUEST/MADTempo style): fetch a web
  image and search with `/search/image`.
- **Group priors**: past answer keys show L26 is cooking, L25 lectures, L24 lion dance
  and L23 cycling. A team-maintained group description could bias the plan.
- **Collaboration-workspace integration**: capture candidate frames into a round's
  answers from the agent panel instead of copying video/frame by hand.
