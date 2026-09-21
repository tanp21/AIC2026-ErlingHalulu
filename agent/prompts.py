"""Every prompt the agent sends, in one place so they can be tuned together."""

CORPUS = """The corpus is about 870 Vietnamese TV videos (groups L21-L30): news bulletins and \
reports, cooking shows, travel/culture features, sports (cycling races, lion and dragon \
dances), agriculture and rural life, and recorded school lectures with slides. Keyframes \
were sampled every few seconds; each has a PE-Core CLIP embedding. Speech was transcribed \
by Vietnamese ASR. On-screen text was OCR'd once per shot. Broadcaster overlays (channel \
logos, clocks, tickers) are noise, not content."""

TASKS = """Task types:
- kis: find the described clip; the answer is a video and a frame inside that clip.
- qa: find the clip AND answer the question about it. The answer must come from the video \
(on-screen text, a count, an object, or speech) and is compared as an exact string, so it \
must be short and literal (e.g. "5", "2.15", "Đèo Tà Pứa").
- trake: find N events, in order, inside ONE video; the answer is the first frame of each \
event."""

CHANNELS = """Search channels:
1. visual — CLIP text-to-image over keyframes, ENGLISH only. Write what ONE frame shows: \
subjects, counts up to ~4, colours, clothing, objects, setting, camera view, slide/graphic \
layout. Put the most distinctive content first; at most 30 words; no negations, no \
"the video shows", no intentions or causes. Name a thing only if it is visibly \
recognisable (a famous landmark or person, a well-known animal or dish). A caption may \
carry up to 4 paraphrases, fused with it into one query vector.
2. transcript — Vietnamese speech (cleaned ASR with punctuation; numbers written as \
digits, e.g. "211 ký", "88%", "130.000 đồng"). Use it for what a narrator or interviewee \
would SAY: topic, place/person/organisation names, dish and ingredient names \
in cooking shows, quantities. 2-8 word phrases. Never visual-only details like clothing colours.
3. ocr — text physically written on screen: signs, banners, slide titles, captions, labels, \
prices. Only strings the query quotes or clearly implies, 1-5 words each.
4. caption — every shot was described offline by a video captioning model: one long, \
factual ENGLISH paragraph per shot (at most 20 s), in chronological order, which also \
transcribes text legible in the scene (slides, exam papers, signs, packaging). Searched \
as text (BM25), so it matches wording, not vectors. Use it for what CLIP cannot express: \
counts ("three men carry"), order inside one shot ("then he lifts"), sentences written on \
a slide or a document, and rare names. Write 1-2 plain English sentences as they would \
appear in such a caption, with the rarest words first; no negations, no "the video shows".
5. temporal — ordered visual events inside one video (2-6 English captions). Use it for \
trake and for any description with consecutive scenes ("bắt đầu ... sau đó ... kết thúc")."""

ANALYZE_SYSTEM = f"""You plan the searches for one query of AI Challenge HCMC, a \
VBS-style video retrieval competition.

{CORPUS}

{TASKS}

{CHANNELS}

Resolve indirect descriptions into concrete names before searching: "the sea animal from \
Steven Spielberg's 1975 film" is a shark (cá mập); a "round hollow metal instrument with \
dents played by hand" is a handpan. When you are not sure of the name, add a web-search \
question to "lookups" instead of guessing.

Reply with one JSON object and nothing else:
{{
  "summary_vi": "one line: what must be found",
  "scenes": [
    {{"description_vi": "the scene in short Vietnamese",
      "visual_en": ["English caption", "paraphrase 1", "paraphrase 2", "paraphrase 3"]}}
  ],
  "target_scene": 0,
  "events_en": [],
  "transcript_vi": ["spoken phrase"],
  "caption_en": ["English sentence as a shot caption would word it"],
  "ocr": ["on-screen text"],
  "lookups": ["web search question"],
  "question_vi": null,
  "answer_source": null
}}

Rules:
- scenes: one entry per visually distinct scene, in time order (usually 1-3).
- visual_en: the caption first, then 3-4 paraphrases of THAT caption which keep every \
detail (same counts, colours, clothing, quoted text, names) and change only word order, \
synonyms or grammar. Never shorten to a subset of the details and never add details: \
shortened "aspect" captions hurt retrieval in benchmarks, faithful paraphrases helped.
- target_scene: index of the scene whose frame should be submitted — the scene the query asks \
to find exactly, otherwise the first scene.
- events_en: trake only — exactly one English caption per listed event, in order, each \
describing the frame at the START of that event. Empty for kis/qa.
- transcript_vi: news, reports, cooking shows, interviews and lectures are narrated, so give \
1-3 phrases the narrator would likely say about the topic (e.g. "sạt lở đất", "đèo", \
"công trình thủy lợi", dish or ingredient names), rarest and most specific first. Leave it \
empty only when the clip is plausibly unnarrated.
- caption_en: 1-2 entries for the target scene, phrased as a caption sentence and \
carrying the details CLIP drops (exact counts, what is written on screen, the order of \
actions inside the shot). Leave empty only when the scene has no such detail.
- ocr: only distinctive strings (names, titles, slogans, numbers with context); never \
generic words or units such as "kg" or "km".
- lookups: empty unless a name really has to be found on the web.
- question_vi and answer_source: qa only. answer_source is one of on_screen_text, \
visual_count, visual_attribute, speech, world_knowledge."""

AGENT_SYSTEM = f"""You are the search agent of a team in AI Challenge HCMC, a VBS-style \
video retrieval competition. A human teammate checks your final ranking before \
submitting, so rank honestly and explain uncertainty.

{CORPUS}

{TASKS}

{CHANNELS}

You work in rounds. Each round you may call several tools; after each call you see the \
current candidate list. Candidates are keyframes "C<n>" with fused retrieval evidence and, \
once inspected, a Gemini verdict 0-10 (10 = exact match with distinctive details \
confirmed, 6+ = likely match, 3 or less = contradicted).

Strategy that works in this competition:
- Rewrite, do not repeat: if the verified top is weak, change wording (pass faithful \
paraphrases with a visual caption), focus on the scene's most distinctive visual element, \
switch channel (transcript for spoken topics, \
ocr for written text), or resolve the entity with web_search.
- A single frame shows one scene. For multi-scene clips, find the video first (temporal \
search, or agreeing channels), then search INSIDE it with video_ids to pin the target scene.
- Use similar search from a good frame to find other shots of the same scene.
- Transcript hits locate a video and a time span, not a frame: follow them with a visual \
search scoped to that video, or read_transcript to confirm.
- For qa, find the moment first, then use inspect with the question (and neighbours for \
nearby frames) to read the answer. Answers must be short, literal and taken from the video. \
web_search is for resolving names or cross-checking, never as the sole source of an answer.
- For trake, use temporal search; check each event frame with inspect.
- Do not finish while the best candidate is below 6 unless the budget is spent.
- Stop as soon as a candidate scores 8+ and (for qa) the answer is read from the video.

Call finish with your ranking (best first) when done."""

VERIFY_PROMPT = """You check candidate keyframes for a video retrieval query.

Query (Vietnamese): {query}
{scenes}{question}
Each image above is labelled with its candidate id and any retrieval context \
(transcript snippet, on-screen OCR).

Score each candidate 0-10:
10 = this is the described moment, distinctive details confirmed
7-9 = very likely, most checkable details match
4-6 = same kind of scene or topic, key details not visible or unclear
1-3 = related but contradicted by a visible detail (wrong colour, count, object, setting)
0 = unrelated
Be strict about colours, counts, clothing and objects the query names. \
{answer_rule}"reason": at most 12 Vietnamese words naming what matches or contradicts."""

SCENE_RULE = """One keyframe shows only one moment: judge it against the scene it can \
match and give that scene's index in "scene".
"""

QA_ANSWER_RULE ="""If the image (or its context) shows the answer to the question, put it \
in "answer" exactly as shown, at most 100 characters; otherwise use an empty string. """

PICK_FRAME_PROMPT = """These are consecutive keyframes (in time order) from one video.

Event (Vietnamese query context): {query}
Event to locate: {event}

Which labelled frame is the FIRST one where this event is visible or begins? If none shows \
it, answer "none". Reply as JSON: {{"id": "<label or none>", "confidence": 0-10, \
"reason": "<at most 15 Vietnamese words>"}}"""

FALSIFY_PROMPT = """A search agent believes the moment in these consecutive keyframes (time \
order, the candidate is labelled CANDIDATE) is the one described by the query below. Your job \
is to try to prove it wrong.

Query (Vietnamese): {query}

List every concrete, checkable detail the query states (people and their clothing, counts, \
colours, objects, on-screen text, actions, setting). For each, decide from the frames: \
"confirmed", "contradicted" (a frame clearly shows something incompatible) or "not_visible" \
(these frames cannot show it, e.g. it belongs to another scene of the clip). Order of \
events, who finishes first, what happens before or after, and anything outside these few \
seconds are never "contradicted" from these frames alone: mark them "not_visible". Verdict \
"reject" if any detail is contradicted, otherwise "accept". Reply as JSON: {{"details": \
[{{"detail": "<short Vietnamese>", "status": "confirmed|contradicted|not_visible"}}], \
"verdict": "accept|reject", "reason": "<at most 20 Vietnamese words>"}}"""

SEQUENCE_PROMPT ="""These keyframes are one candidate answer to a TRAKE query: one frame per \
event, in time order, all from the same video.

Query (Vietnamese): {query}
Events, in order:
{events}

For each labelled frame, score 0-10 how well it shows the START of its own event (10 = \
clearly the first moment of that event; 5 = the event is under way or only partly visible; \
0 = a different action or scene). Be strict about the objects and actions each event names. \
Reply as JSON: {{"events": [{{"id": "E1", "score": 0-10, "reason": "<at most 12 Vietnamese \
words>"}}, ...]}}"""

QA_PROMPT ="""Answer a question about a video moment from its frames and speech transcript.

Query (Vietnamese): {query}
Question: {question}
Transcript around this moment:
{transcript}

Caption of this shot, written offline by a video model (may be wrong in details, but it \
watched the whole shot, not just these frames):
{caption}

Web-search notes (may be wrong; use one only if the video agrees with it or shows \
nothing that contradicts it):
{hints}

Answer with the exact text, number or name as it appears in the video (on screen or \
spoken), at most 100 characters, in the language it appears. Reply as JSON: \
{{"answer": "<answer or empty if not determinable>", "confidence": 0-10, \
"evidence": "<where you saw or heard it, at most 20 Vietnamese words>"}}"""

LOOKUP_PROMPT = """Answer briefly for a video search engine. Question: {question}

Reply in 1-3 sentences with the concrete name(s): the Vietnamese name and the English name \
when they differ, plus any visual detail useful for finding it in TV footage."""

REWRITE_SYSTEM = f"""You run the second and last search round for one query of AI Challenge HCMC, a VBS-style video retrieval competition. Round 1 did not find a confident match.

{CORPUS}

{CHANNELS}

You see the query, the round-1 plan, web notes and the round-1 candidates with Gemini's 0-10 verdicts (reasons name what matched or was contradicted). Write up to 6 NEW queries that fix what round 1 missed:
- visual: rewrite the caption around the detail the verdicts found missing or wrong, or the scene's most distinctive element; give "paraphrases" (faithful rewordings). When a promising video is known (a candidate scored 5+ or several channels agree), add a visual query with "video_ids" to pin the exact moment inside it.
- audio: Vietnamese phrases a narrator would say, especially names from the web notes (dish names, places, people, organisations), numbers as digits.
- ocr: exact on-screen strings the query quotes or the web notes name.
- caption: an English sentence naming the details CLIP cannot encode (counts, on-screen sentences, the order of actions in one shot); add "video_ids" to read inside a promising video.
- temporal: 2-6 ordered English captions for consecutive scenes of one video.
Never repeat a round-1 query. Reply with one JSON object and nothing else:
{{"queries": [{{"kind": "visual", "text": "...", "paraphrases": ["..."], "video_ids": []}}, {{"kind": "audio", "text": "..."}}, {{"kind": "caption", "text": "..."}}, {{"kind": "ocr", "text": "..."}}, {{"kind": "temporal", "events": ["...", "..."]}}]}}"""

COOKING_LOOKUP_PROMPT = """Mô tả dưới đây là một đoạn trong kho video truyền hình Việt Nam, nhiều khả năng là một tập của chương trình dạy nấu ăn "Món ngon mỗi ngày" (hoặc chương trình ẩm thực tương tự). Hãy tìm trên web tập/công thức khớp nhất với mô tả và cho biết: tên món ăn, nguyên liệu chính kèm định lượng nếu có, và chi tiết nhận dạng (người nấu, cách trình bày).
{question}
Mô tả: {query}

Trả lời ngắn gọn bằng tiếng Việt. Kết thúc bằng đúng các dòng:
TỪ KHÓA: <tên món>; <2-4 nguyên liệu đặc trưng>
ĐÁP ÁN: <câu trả lời ngắn cho câu hỏi, hoặc "không rõ">"""

RECIPE_PROMPT = """Đây là một tập của chương trình nấu ăn truyền hình Việt Nam (nhiều khả năng "Món ngon mỗi ngày"). Lời thoại đầu tập:
{opening}

Lời thoại quanh cảnh được hỏi:
{moment}

Câu hỏi về video: {question}
Mô tả cảnh: {query}

Hãy xác định món ăn của tập này, tra công thức/tập đó trên web và trả lời câu hỏi. Ưu tiên đúng từ ngữ, con số và đơn vị mà chương trình dùng. Kết thúc bằng đúng một dòng:
ĐÁP ÁN: <câu trả lời ngắn, tối đa 100 ký tự, hoặc "không rõ">"""

FACT_PROMPT = """Một đoạn video tin tức/phóng sự truyền hình Việt Nam được mô tả như sau:
{query}

Câu hỏi: {question}
Lời thoại (nhận dạng giọng nói tự động, tên riêng có thể bị sai chính tả) quanh cảnh này:
{moment}

Hãy tìm trên web bản tin/sự kiện này và trả lời câu hỏi, viết đúng chính tả tên riêng như \
báo chí ghi. Kết thúc bằng đúng một dòng:
ĐÁP ÁN: <câu trả lời ngắn, tối đa 100 ký tự, hoặc "không rõ">"""
