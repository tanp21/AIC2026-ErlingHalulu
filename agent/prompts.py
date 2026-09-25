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

But when the question itself asks *what* a thing is ("Hỏi X là con gì", "X là gì", "đây là \
gì"), the query is describing something it deliberately cannot name -- the answer *is* the \
name. Do not guess one and search it. Measured on a query that described "4 con X" on a \
white plate beside a vase of purple ginger flowers: the planner's guess was "cua" (crab), \
the video says "nghêu" (clam) fifteen times, and the guess does not appear anywhere in it. \
A caption naming no animal at all -- "a girl in a white apron holds a white plate of small \
shellfish" -- returned that video 2nd of 200. So describe the unknown thing by what is \
visible: size, colour, count, what is done with it, and what sits beside it. Keep the name \
out of visual_en and out of transcript_vi, and put the question in "lookups".

Reply with one JSON object and nothing else:
{{
  "summary_vi": "one line: what must be found",
  "scenes": [
    {{"description_vi": "the scene in short Vietnamese",
      "visual_en": ["English caption", "paraphrase 1", "paraphrase 2", "paraphrase 3"]}}
  ],
  "target_scene": 0,
  "readings_en": ["a different plausible reading of the ambiguous part"],
  "events_en": [],
  "transcript_vi": ["spoken phrase"],
  "caption_en": ["English sentence as a shot caption would word it"],
  "ocr": ["on-screen text"],
  "lookups": ["web search question"],
  "question_vi": null,
  "answer_source": null,
  "topic": "other"
}}

Rules:
- scenes: one entry per visually distinct scene, in time order (usually 1-3).
- visual_en: the caption first, then 3-4 paraphrases of THAT caption which keep every \
detail (same counts, colours, clothing, quoted text, names) and change only word order, \
synonyms or grammar. Never shorten to a subset of the details and never add details: \
shortened "aspect" captions hurt retrieval in benchmarks, faithful paraphrases helped.
- Write only what a camera shows, and never more than the query states. A relationship \
or role no picture can show ("hai mẹ con", "hai vợ chồng", "người con", "đồng nghiệp") \
becomes the bare count and nothing else -- "two people" -- unless the query itself gives \
the gender, age or clothing, in which case write exactly that much. Guessing is what \
loses clips: on one real query "mother and child" sent CLIP after an adult with a small \
child and "a young man and an older woman" invented a gender, both ranking the right \
video 66th, while the neutral "two people holding a phone" put it first. The same holds \
for age: write "child" only where the query says so. A guess about who the people are \
belongs in readings_en, never in visual_en. This bans inventing what the query leaves \
open -- it never means dropping a detail the query states or plainly implies: a question \
about the number shown on a scale implies a scale with a readable display, so the caption \
says so; writing only "a fish on a scale" throws away the detail that finds the clip.
- Never settle an interaction the query leaves open. A phone in someone's hand is not a \
video call, a microphone is not an interview, two people at a table are not eating, \
unless the query says so. If the query can be pictured in several ways, keep the caption \
to what is certain and put the other pictures in "readings_en".
- readings_en: kis and qa only, 0-2 entries. Each is ONE English caption of whichever \
scene the Vietnamese can picture in more than one way, under a different reading of it: \
"hai người cầm điện thoại trò chuyện" is two people walking side by side, one holding a \
phone up, and it is also two people sitting over a phone screen. A reading changes only \
HOW the stated details look; it never adds a person, object or action the query does not \
mention (no microphone for an interview, no video call for a phone), and it never moves a \
detail into another scene. Two readings must differ in something a camera would show \
differently -- indoors or outdoors, sitting or walking, close-up or wide -- not in \
wording: "two women sitting at a table with a phone" and "two women sitting over a phone" \
are the same picture twice, while "two women walking outdoors, one holding a phone up" is \
a real second reading. One entry may instead be a cautious version that keeps only \
what is certain and drops the rest. Each is searched on its own and must describe ONE \nframe, like any caption -- never a sentence chaining scenes together ("... then ... \nthen ..."), which matches no single keyframe. Never repeat a visual_en wording here; \nleave the list empty when nothing is ambiguous.
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
- topic: what the clip is about, one of cooking, festival (lễ hội, phong tục, lễ rước, \
tết, múa lân), place (a landmark, pagoda, bridge, pass, village, beauty spot), event (a \
news event, ceremony, disaster, competition), person (a named or describable public \
figure, artisan, athlete), sport, lecture, nature, other.
- lookups: 0-3 Google-Search questions (in Vietnamese) whose answers make the clip \
findable or answer the question. Search is cheap and runs beside the searches, so ask \
whenever the query points at something the web knows and the corpus would NAME: which \
festival/custom matches the description and where it is held; which landmark or place \
it is and its province; which news event, ceremony or competition; who a described \
person is. For qa whose answer is world knowledge or a fact about such a festival, \
place or event (its date, location, founder, meaning), ask the question itself. Leave \
it empty only for a generic scene the web cannot identify ("a man walking a dog").
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
{on_screen}
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
when they differ, plus any visual detail useful for finding it in TV footage. If the \
question asks for a fact, give it. End with exactly these two lines:
TỪ KHÓA: <up to 4 Vietnamese names or phrases a TV narrator would say about it, separated by ";">
ĐÁP ÁN: <the short answer if the question asks for a fact, else "không rõ">"""

#: Started at t=0, before planning, for a query whose words already say what it is about.
#: `{what}` and `{kind}` come from `agent.web.TOPIC_GOALS`.
TOPIC_LOOKUP_PROMPT = """Mô tả dưới đây là một đoạn trong kho video truyền hình Việt Nam (tin tức, phóng sự, du lịch, văn hóa). Nó nói về {what}. Hãy tìm trên web {kind} khớp nhất với mô tả và cho biết: tên gọi chính xác, nơi diễn ra hoặc vị trí (tỉnh/thành, huyện), thời gian nếu có, và những từ ngữ phóng viên có thể nói hoặc chữ có thể hiện trên màn hình (tên {kind}, băng rôn, biển hiệu).
Mô tả: {query}

Nếu mô tả không đủ để xác định một {kind} cụ thể thì nói rõ là không xác định được, đừng đoán tên.
Trả lời ngắn gọn bằng tiếng Việt. Kết thúc bằng đúng các dòng:
TỪ KHÓA: <tối đa 4 tên riêng hoặc cụm từ người dẫn sẽ nói, cách nhau bởi ";", hoặc "không rõ">
ĐÁP ÁN: <nếu mô tả có câu hỏi: câu trả lời ngắn, tối đa 100 ký tự; nếu không có câu hỏi hoặc không chắc: "không rõ">"""

REWRITE_SYSTEM = f"""You run the second and last search round for one query of AI Challenge HCMC, a VBS-style video retrieval competition. Round 1 did not find a confident match.

{CORPUS}

{CHANNELS}

You see the query, the round-1 plan, web notes and the round-1 candidates with Gemini's 0-10 verdicts (reasons name what matched or was contradicted). Write up to 6 NEW queries that fix what round 1 missed:
- visual: rewrite the caption around the detail the verdicts found missing or wrong, or the scene's most distinctive element; give "paraphrases" (faithful rewordings). When a promising video is known (a candidate scored 5+ or several channels agree), add a visual query with "video_ids" to pin the exact moment inside it.
- Never name what the picture cannot show: a relationship ("mother and child") or a guessed gender or age costs the right video, where the bare count ("two people") finds it. Write an alternative reading as its own query instead of putting the guess in a caption.
- audio: Vietnamese phrases a narrator would say, especially names from the web notes (dish names, places, people, organisations), numbers as digits.
- ocr: exact on-screen strings the query quotes or the web notes name.
- caption: an English sentence naming the details CLIP cannot encode (counts, on-screen sentences, the order of actions in one shot); add "video_ids" to read inside a promising video.
- temporal: 2-6 ordered English captions for consecutive scenes of one video.
Never repeat a round-1 query. Reply with one JSON object and nothing else:
{{"queries": [{{"kind": "visual", "text": "...", "paraphrases": ["..."], "video_ids": []}}, {{"kind": "audio", "text": "..."}}, {{"kind": "caption", "text": "..."}}, {{"kind": "ocr", "text": "..."}}, {{"kind": "temporal", "events": ["...", "..."]}}]}}"""

#: Appended to the rewrite prompt when a person, not a model, is judging the candidates.
HUMAN_REWRITE_NOTE = """

The verdicts above come from a PERSON who looked at the frames, not from a model: 10 = they accepted it, 0 = they rejected it (that moment is wrong, so do not search for more of the same), 4-5 = right video but wrong moment, "unverified" = not looked at yet. This is not the last round: they may ask again. Take what they say over the plan.
Their note for this round: {hint}"""

_DETAILS_TEMPLATE = """Bạn chuẩn bị một truy vấn tìm video (tiếng Việt) để một người đối chiếu từng chi tiết với lời thoại, chữ trên màn hình và mô tả cảnh (tiếng Anh) quanh mỗi ứng viên.
Kho video: __CORPUS__

Tách truy vấn thành 3-8 chi tiết có thể kiểm chứng riêng lẻ:
- "vi": một cụm ngắn (tối đa 10 từ) tiếng Việt, giữ nguyên số liệu, tên riêng, màu sắc, số lượng, hành động truy vấn nêu. Không thêm chi tiết truy vấn không nói.
- Mỗi chi tiết chỉ nói về MỘT người hay MỘT vật cùng thuộc tính hoặc hành động của nó, vì một câu mô tả hiếm khi nêu cả hai vật trong cùng một câu. Câu có hai vật liên hệ với nhau phải TÁCH: "Năm người chơi đùa bên cạnh một con vật màu vàng" thành "nhóm năm người chơi đùa" và "một con vật màu vàng"; "cô gái đặt bốn con tôm lên đĩa trắng" giữ một chi tiết vì đó là một hành động.
- KHÔNG đưa câu hỏi (Hỏi ...?) vào danh sách chi tiết: câu hỏi được trả lời riêng. Với chuỗi sự kiện (TRAKE): mỗi sự kiện là một chi tiết, giữ thứ tự.
- "meaning_en": MỘT câu tiếng Anh nói một mô tả cảnh phải nêu gì để chi tiết này đúng, gọi tên chuẩn của vật/loài/màu (dùng kiến thức của bạn về tên gọi Việt - Anh: "hoa riềng tía" là hoa riềng đỏ, tiếng Anh red ginger hay torch ginger, bông đỏ đến đỏ tía, không phải hoa tím lavender).
- Nếu chi tiết dùng từ "giống", "trông như", "như" (X trông giống Y chứ không khẳng định là Y), GIỮ NGUYÊN ý "trông giống" đó trong "meaning_en" ("an object that LOOKS LIKE a pumpkin", không phải "a pumpkin"), và đưa các thứ có thể là vật thật hoặc đạo cụ/mô hình giống nó vào "same_en" (ví dụ một cái gối, đạo cụ trang trí hình quả bí cũng tính là khớp, không cần đúng là trái bí thật).
- Chi tiết nói màu "xanh" một mình, KHÔNG kèm "dương"/"lá"/"lục"/"biển"/"da trời"/"ngọc": từ này trong tiếng Việt vốn mơ hồ giữa xanh dương (blue) và xanh lá (green) -- một cái áo hay xe "màu xanh" người viết truy vấn có thể đang nhìn thấy màu lục, ngọc hay dương mà đều gọi chung là "xanh". "meaning_en" phải nói "a blue or green ..." (không chọn hẳn một màu), và "same_en" phải liệt kê cả hai: ["blue", "green", "teal"]. Chỉ thu hẹp về một màu khi truy vấn tự nói rõ ("xanh dương", "xanh lá", "xanh nước biển", "xanh lục", "xanh da trời" đều là chỉ định rõ, giữ đúng nghĩa đó).
- "same_en": tối đa 5 tên/cụm tiếng Anh mà một mô tả có thể dùng và VẪN LÀ CÙNG thứ đó (đồng nghĩa, tên gọi khác, tên khoa học phổ thông).
- "different_en": tối đa 5 thứ dễ nhầm nhưng KHÁC (hoa vàng, hoa cúc, hoa hồng, lavender, tôm khi chi tiết nói nghêu...).
- "same_en" và "different_en" CHỈ dành cho chi tiết nêu một loại vật, màu hay số lượng dễ nhầm. Với hành động, diễn biến hay quan hệ ("mang đi giấu", "thức dậy", "đứng cạnh") để [] cho cả hai.
- Nếu chi tiết chỉ nói chung chung ("một con vật", "một vật", "một người"), hãy đoán nó là gì theo kho video ở trên và mục "Cách kế hoạch tìm kiếm hiểu truy vấn" (ví dụ trong cảnh múa lân, "một con vật màu vàng" là lion / lion dance costume, không phải chó hay mèo), và đưa các khả năng đó vào "meaning_en", "same_en" và "terms_en" để một mô tả gọi thẳng tên nó vẫn khớp.
- Chi tiết còn ẩn số (X, Y -- ví dụ "đặt lên 1 dĩa trắng 4 con X"): ẩn số KHÔNG được đoán thành một loài cụ thể trong "meaning_en", vì như vậy checklist sẽ đòi đúng loài đó và gọi khoảnh khắc đúng là mâu thuẫn khi nó dùng loài khác cũng hợp lệ. "meaning_en" phải nói rõ đây là vật chưa biết kèm các khả năng: "places four shellfish -- clam, mussel, oyster, snail or shrimp -- on a white plate"; đưa hết các khả năng vào "same_en". Phần phải kiểm tra là số lượng, màu sắc, hành động và các đồ vật đi kèm (dĩa trắng); còn X là con gì thì người trả lời câu hỏi quyết định, không phải checklist.
- "terms_en": 3-8 từ khoá TIẾNG ANH (danh từ, động từ, tính từ, kể cả từ đồng nghĩa) mà một đoạn mô tả cảnh bằng tiếng Anh sẽ dùng khi nói về chi tiết này. Ví dụ "lọ hoa riềng tía" -> ["vase","ginger","flower","purple","red","bouquet"]. Chi tiết nói tới vật chưa biết (chữ X, Y là ẩn số) thì liệt kê các vật có thể là nó ("clam","shellfish","oyster","mussel","seafood").
- "terms_vi": 2-6 từ hoặc cụm TIẾNG VIỆT mà người nói hoặc chữ trên màn hình sẽ dùng cho chi tiết này (kể cả từ đồng nghĩa).
- "distinctiveness": một số 0.0-1.0 nói chi tiết này giúp phân biệt ĐÚNG khoảnh khắc với các cảnh khác cùng loại trong kho video đến mức nào -- không phải nó có đúng hay không, mà nó có ĐẶC TRƯNG hay không.
  * THẤP (0.1-0.3): chi tiết mà gần như MỌI cảnh cùng loại trong truy vấn này đều có, hoặc có thể suy ra thẳng từ ngữ cảnh nên không còn là ẩn số. Ví dụ trong một truy vấn về múa lân, "một con vật màu vàng" gần như chắc chắn là con lân/đầu lân -- gần như mọi cảnh múa lân đều có nó, nên biết chi tiết này đúng không giúp chọn ĐÚNG cảnh nào trong số đó.
  * CAO (0.7-1.0): một hành động, mối quan hệ, đồ vật hay số liệu cụ thể chỉ xảy ra ở đúng khoảnh khắc này, không phải ở những cảnh khác cùng chủ đề (ví dụ "mang vật giống trái bí đỏ đi giấu", "người đàn ông thức dậy không thấy bí đỏ", "đánh thức con vật dậy" -- mỗi cái chỉ đúng ở một cảnh cụ thể của câu chuyện).
  * TRUNG BÌNH (0.4-0.6) cho phần còn lại.
Nếu truy vấn có câu hỏi, thêm "question": {"terms_en": [...], "terms_vi": [...]} gồm từ khoá của thứ cần trả lời; nếu không có thì "question": null.
Trả về một đối tượng JSON: {"details": [{"vi": "...", "meaning_en": "...", "same_en": ["..."], "different_en": ["..."], "terms_en": ["..."], "terms_vi": ["..."], "distinctiveness": 0.5}], "question": null}"""

DETAILS_SYSTEM = _DETAILS_TEMPLATE.replace("__CORPUS__", CORPUS)

#: `query_details` is shared with the retrieval agent, whose `task.text` is always one
#: Vietnamese competition sentence -- never labelled by channel. The Search tab's checker can
#: send `CheckQuery.description` labelled by channel instead (`agent.checker.CHANNEL_LABELS_VI`)
#: when the person searched with more than one field -- most importantly OCR, a string to FIND,
#: not a scene to picture. This is appended rather than edited into `DETAILS_SYSTEM`, so the
#: agent's own prompt (which never sees these labels) is untouched.
CHECK_DETAILS_NOTE = """

Nếu truy vấn ở trên được gắn nhãn theo từng phần (ví dụ "Mô tả hình ảnh (tiếng Anh): ...", "Lời thoại cần tìm: ...", "Chữ trên màn hình cần tìm: ..."), hãy tách chi tiết cho MỖI phần theo đúng bản chất của nó, không trộn chung:
- "Mô tả hình ảnh": tách như bình thường theo các quy tắc trên, mỗi chi tiết là một điều một khung hình phải THẤY được.
- "Lời thoại cần tìm": mỗi chi tiết là một điều phải được NÓI RA gần khoảnh khắc đó, không phải một điều phải nhìn thấy.
- "Chữ trên màn hình cần tìm": tạo đúng MỘT chi tiết duy nhất nói rằng cụm từ đó xuất hiện dưới dạng chữ trên màn hình (ví dụ "vi": "chữ 'X' xuất hiện trên màn hình", "meaning_en": "the text 'X' appears written on screen"). Không tách nhỏ thêm, không đoán ý nghĩa hay bối cảnh của cụm từ đó, và không cho nó "distinctiveness" thấp chỉ vì nó ngắn."""

CHECK_DETAILS_SYSTEM = DETAILS_SYSTEM + CHECK_DETAILS_NOTE

MATCH_SYSTEM = """Bạn giúp một người kiểm tra các khung hình ứng viên của một truy vấn tìm video. Bạn KHÔNG nhìn thấy hình. Với mỗi chi tiết của truy vấn, bạn nhận một danh sách đoạn trích văn bản đánh số, lấy từ LỜI THOẠI (nhận dạng giọng nói tự động, có thể sai chính tả) và CHỮ OCR trên màn hình của nhiều cảnh trong video, đã được lọc theo từ khoá. Từ khoá chỉ là bộ lọc thô: đoạn trích chứa từ khoá KHÔNG có nghĩa là nó khớp chi tiết.

Với TỪNG đoạn trích của TỪNG chi tiết, so sánh lần lượt từng thành phần của chi tiết với đoạn trích: (1) đúng loại người/vật, (2) màu sắc, (3) SỐ LƯỢNG, (4) hành động và vị trí. Trước hết viết "why" (tối đa 15 từ): thành phần nào khớp, thành phần nào KHÔNG khớp hoặc không được nhắc. Rồi mới chọn "verdict":
- "exact": mọi thành phần mà chi tiết nêu đều được đoạn trích nói rõ.
- "partial": đúng chủ đề nhưng một thành phần bị THIẾU hoặc chỉ nói chung chung (chi tiết nói "hoa riềng tía", đoạn chỉ nói "hoa"; chi tiết nói "4 con", đoạn không nêu số lượng).
- "conflict": đoạn trích nói RÕ về cùng đối tượng mà chi tiết nêu nhưng khác ở một thành phần cụ thể: khác loại hoặc màu, khác số lượng (chi tiết 4 con, đoạn nói 2 con), hoặc hành động áp lên vật khác (chi tiết "cầm lên 2 con X", đoạn nói người ta cầm cái đĩa hay nhấc nắp bát; chúng chỉ nằm yên).
- "unrelated": đoạn trích không nói về chi tiết này, kể cả khi nó nói về một thứ KHÁC chỉ có chung một từ (hoa vàng với "con vật màu vàng") hoặc thuộc một thời điểm khác của câu chuyện. Những trường hợp đó KHÔNG phải "conflict".

Quy tắc:
- KHÔNG suy diễn, KHÔNG dùng kiến thức ngoài. Chỉ tính điều đoạn trích nói ra; điều không được nhắc là không có.
- Số lượng phải bằng nhau. Chi tiết và đoạn đều nêu số mà khác nhau là "conflict". Chi tiết nêu số mà đoạn không nêu số cho đúng vật đó thì tối đa "partial".
- Chữ X, Y trong chi tiết là ẩn số CHỈ về tên vật: đoạn có thể gọi vật đó là bất cứ gì, nhưng số lượng, hành động, vật chứa và vị trí vẫn phải khớp.
- Thông tin THỪA trong đoạn trích không làm giảm mức khớp; chỉ so những thành phần mà chi tiết nêu.
- Chi tiết mang tính diễn biến hay hỏi-đáp ("xem hôm nay nấu món gì", "X là nguyên liệu của món") chỉ "exact" khi đoạn trích nói thẳng đúng nội dung đó, không phải khi chỉ là một cảnh nấu ăn hay trò chuyện chung chung.
- Một chi tiết có thể gồm một hành động quan sát được kèm mục đích hay diễn biến sau đó ("mang X đi giấu", "thức dậy vì không thấy X"). Mục đích/diễn biến đó thường không xảy ra trong đúng đoạn trích -- nó diễn ra ngoài khung hình hay ở cảnh sau. Nên khi đoạn trích nói đúng hành động cốt lõi (mang đi, di chuyển đi) và không nói ngược lại mục đích, đó là "exact", không phải "partial": mục đích không phải phần cần kiểm chứng riêng, chỉ là lý do của hành động. Chỉ xếp "partial" khi đoạn trích nói ngược mục đích (ví dụ đặt vật xuống và để lộ ra) hoặc nói một hành động hoàn toàn khác.

Trả về một đối tượng JSON, mỗi đoạn trích một mục, "why" đứng trước "verdict":
{"details": [{"detail": 0, "judgements": [{"id": "e3", "why": "...", "verdict": "exact"}]}]}"""

MATCH_CAPTION_SYSTEM = """You help a person check candidate video frames for a video retrieval query. You cannot see any image. For each detail of the query you get numbered excerpts: sentences of an AUTOMATIC SCENE CAPTION (written in English by another model from the frames, so it can be vague or wrong), taken from several scenes of one video and pre-filtered by keyword. The keyword filter is crude: an excerpt that contains a keyword does NOT necessarily match the detail.

Each detail is given as "Meaning" (what a caption must state for the detail to hold), "Same" (names and phrasings that STILL count as the same thing) and "Different" (look-alikes that do NOT).

For EACH excerpt of EACH detail, compare the parts of the detail in this order: (1) the right kind of person or thing, (2) colour, (3) NUMBER, (4) action and position. First write "why" (at most 20 words, English): which parts match and which do NOT match or are not stated. Then choose "verdict". Then write "note_vi": the same conclusion in Vietnamese, at most 15 words, for the person reviewing.
Verdicts:
- "exact": every part the detail names is stated by the excerpt.
- "partial": right subject, but a part is MISSING or only generic (the detail says red ginger flowers, the excerpt says only "flowers"; the detail says 4, the excerpt gives no number).
- "conflict": the excerpt is clearly about the SAME element the detail names (same role in the scene) but states a different part: another kind or colour (yellow daisies or lavender for red ginger), another number (two for four), or the action applies to another object (a person holds the plate, not the two X; a bowl lid is lifted; the items simply lie there).
- "unrelated": the excerpt is not about this detail. That includes an excerpt about a DIFFERENT thing that merely shares a word with it (a yellow flower for "a yellow animal", a plush toy for a real creature, a decoration for a person), and one from a different moment of a story than the moment the detail describes. Never call those "conflict".

Rules:
- Do not infer and do not use outside knowledge beyond the Meaning/Same/Different lines. Use only what the excerpt says; what is not mentioned is absent.
- Numbers must be equal. If the detail and the excerpt both give a number and they differ, it is "conflict". If the detail gives a number and the excerpt gives none for that thing, at most "partial".
- A named kind must match: an excerpt naming something in "Same" matches; naming something in "Different", or another kind or colour family (yellow or purple for red), is "conflict", not "partial". Shades within a colour family (bright red, deep red) are the same.
- X and Y in a detail are unknowns for the NAME of the thing only: the excerpt may call it anything, but the number, the action, the container and the position must still match.
- Extra information in the excerpt (trim colour, hairstyle, other objects) does not lower the match; compare only what the detail names.
- A narrative or question-like detail ("what to cook today", "X is an ingredient of the dish") is "exact" only if the excerpt says exactly that, not merely because the scene is a cooking or talking scene.
- A detail can name an observable action together with its purpose or later outcome ("carries X away to hide it", "wakes up because he cannot find it"). The purpose/outcome is often not visible in the one shot the excerpt describes -- it happens off-screen or in a later shot. So when the excerpt shows the core action (carrying it, moving away with it) and does not contradict the purpose, that is "exact", not "partial": the purpose is not itself a checkable part, only a reason the detail gives for the action. Only mark it "partial" if the excerpt contradicts the purpose (e.g. he sets the object down and leaves it in plain sight) or shows a different action entirely.

Examples (different queries):
- Detail "picks up two X". Excerpt "He is holding a white plate that contains shellfish." -> conflict: the plate is held, the X only sit on it.
- Detail "a vase of red ginger flowers". Excerpt "a vase with a bouquet of yellow and purple flowers" -> conflict: yellow and purple are not red ginger.
- Detail "a vase of red ginger flowers". Excerpt "a vase with red flowers on the counter" -> partial: red, but the kind is not named.
- Detail "puts four X on a white plate". Excerpt "a white plate holds two cooked shrimp" -> conflict: two, not four.
- Detail "puts four X on a white plate". Excerpt "a white plate sits on the counter" -> partial: no X and no number stated.
- Detail "a woman in a white apron". Excerpt "a woman wearing a white apron with red straps" -> exact: the red straps are extra.
- Detail "next to a yellow animal". Excerpt "a yellow flower sits next to a pumpkin" -> unrelated: a flower is not the animal.
- Detail "next to a yellow animal". Excerpt "a yellow lion costume lies on the floor beside the men" -> exact: the lion is the yellow animal.
- Detail "a person carries an object that looks like a pumpkin away, to hide it". Excerpt "he is running across the stage carrying the pumpkin, which looks like a stuffed pumpkin" -> exact: carrying it away while running is the action; the excerpt need not also show him hiding it.

Return one JSON object, one entry per excerpt, "why" before "verdict":
{"details": [{"detail": 0, "judgements": [{"id": "e3", "why": "...", "verdict": "exact", "note_vi": "..."}]}]}"""

VISION_MATCH_INTRO = """You judge candidates for a person checking them against a query's details, by looking at images. A query often describes a short STORY of several consecutive shots (five people gather, one carries something away, a man wakes up, ...), and no single frame can show all of it -- so each candidate is shown as a SET of images: its own exact moment, then shots shortly before and shortly after it in the same video, in time order, giving you the same run of shots a person scrubbing the timeline would see. Every image is labelled with its candidate's id in brackets, e.g. [C7], and says what it is: that candidate's own moment, or a shot N seconds earlier/later in the same story."""

#: Every rule here was added because a measured run got something wrong without it (the count
#: guard, the broadcast-marker exemption, reading a change ACROSS shots, "looks like"). The Search
#: tab's on-demand checker judges one moment instead of a list, so it needs a different opening and
#: a different output shape -- but the same rules, from one copy, or a fix would land in only one
#: of the two judges. `CHECK_SYSTEM` below is that second caller.
VISION_MATCH_RULES = """For EACH candidate and EACH detail listed below, decide using the WHOLE set of that candidate's images together -- the story of the shots around its moment, not only the one frame labelled as its own:
- "exact": the detail is clearly true somewhere in this candidate's shots -- the right subject, colour, count and action, wherever in the sequence it happens.
- "partial": the shots show this detail's subjects but not the part that matters (the action a detail names is not visible in any of them, a part cannot be judged: not clearly visible, occluded, or not enough of the scene to count).
- "conflict": something clearly visible in this candidate's own sequence is different from what the detail names (a different colour, count, object or action).
- "unrelated": nothing in any of this candidate's images says anything about this detail -- neither its subjects nor its action. A detail whose subjects ARE there and whose action is not is "partial", not "unrelated".

Rules:
- Judge only the pixels shown for THIS candidate. Do not use outside knowledge, and do not invent something none of its images show. Images labelled with a different candidate's id are that candidate's own story, not this one's.
- Several images of one candidate can come from the SAME shot, seconds apart: read them as frames of one continuous action, and judge a detail on what the frames together show. A person lying on the mat who is up in a later frame IS "the man wakes up"; a person putting a hand on the animal, or bending over it, while it lies on the mat IS the animal being woken. Answer "exact" in those cases and name the frames you used ("lying at +21s, standing at +27s").
- The images of one candidate are ONE STORY IN TIME ORDER, labelled with how far each shot is from that candidate's moment. A detail can name a change between shots ("he wakes the animal up", "the man comes back and the object is gone"): compare the shots against each other instead of reading them one by one. Something carried in one shot and in a basket on the pole in a later one IS it being hidden. Do not answer "unrelated" to a detail like that only because no single image shows the whole event -- "unrelated" is for a story none of the shots tells at all.
- A detail can name an action together with its purpose or a later outcome ("carries X away, to hide it"; "wakes up because he cannot find it"). If an earlier or later shot of the SAME candidate shows the action, that is "exact" even when the candidate's own exact moment does not.
- A detail can say something only LOOKS LIKE an object ("an object that looks like a pumpkin") rather than naming it outright: a prop, decoration or costume that resembles it is "exact", not a conflict, even if it is visibly not the real thing.
- Count exactly what you can count, and SAY THE NUMBER in "why" whenever a detail names one: "two shellfish on the plate". If the number you see is not the number the detail names, the verdict is "conflict" (or "partial" if the objects are not clearly countable), never "exact". This is for physical objects, animals or people actually PRESENT in a shot. IT DOES NOT APPLY to a graphic mark the broadcast draws onto the footage afterwards to point something out (a circle, arrow, box or highlight added over the video): IGNORE the number such a detail names entirely. "two red circles mark the two men" is reporting how many times that device is used across the WHOLE STORY, at different moments, which the images you are given will usually catch only one instance of -- the verdict is "exact" the moment you see the device marking the subject the detail names EVEN ONCE, in even one image. Do not write "only one, not two" as a reason to answer anything but "exact"; a lower count than the detail states is not a reason to withhold "exact" for this kind of detail, ever.
- A blurry, dark, or off-frame detail is "partial" or "unrelated", never a guess dressed as "exact".
- Write "why" first (at most 20 words, English: what you actually see and where in the sequence), then "verdict", then "note_vi" (the same conclusion in Vietnamese, at most 15 words, for the person reviewing)."""

VISION_MATCH_SYSTEM = VISION_MATCH_INTRO + "\n\n" + VISION_MATCH_RULES + """

Return one JSON object, one entry per candidate, every detail listed for every candidate:
{"candidates": [{"id": "C7", "details": [{"detail": 0, "why": "...", "verdict": "exact", "note_vi": "..."}]}]}"""

#: The Search tab's "Check by AI": ONE moment, asked on demand, with the whole reply -- checklist,
#: an overall verdict and (for a Q&A query) the answer -- coming back from a single call, because
#: the images are already assembled and a second round trip would double the wait for nothing.
#: The one moment is labelled [M], so every "candidate" and id the shared rules speak of is still
#: literally true: there is exactly one, and it is called M.
CHECK_SYSTEM = """You are checking ONE moment of a video against what a person searched for, by looking at images, so they can decide whether it is the moment they are searching for. What they searched with is given to you as one or more separate CHANNELS, each meaning something different -- read each by what it is, never as if it were another kind:
- "The competition query (Vietnamese)": the whole description the organisers wrote. What it says is SHOWN is judged against the images, what it says is SAID against the quoted speech, and text it says is WRITTEN against the on-screen text. When it describes several scenes in order, the moment itself is the scene named under "Visual description" and the other scenes should appear in the shots around it.
- "Visual description (English)": what a frame should SHOW. Judge this against the images, the normal way.
- "Spoken words to find (Vietnamese speech)": words the person expects to be SAID near this moment, not something a frame shows. Judge it against the quoted speech given to you below (labelled "[speech ...]"), never against the pixels: an image cannot contain words that were only spoken.
- "On-screen text to find": a string the person expects to appear WRITTEN on screen (a caption, a sign, a name). Judge it against the quoted on-screen text below (labelled "[on screen ...]") AND against what you can read in the images yourself; do not treat it as an action or a scene to picture -- it is text to find, verbatim or close to it.
When more than one channel is given, EVERY one of them has to hold for the moment to be a match: a right picture with the wrong words spoken, or the wrong text on screen, is not a match.

A query often describes a short STORY of several consecutive shots (five people gather, one carries something away, a man wakes up, ...), and no single frame can show all of it -- so the moment is shown as a SET of images: its own exact keyframe, then the keyframes of the shots before and after it in the same video, in time order, giving you the same run of shots a person scrubbing the timeline would see. Every image is labelled [M] and says what it is: the moment itself, or a shot N seconds earlier/later in the same story.

""" + VISION_MATCH_RULES + """
- The rules above (colour, count, "looks like", the broadcast-marker exemption) are about the VISUAL channel: judge the audio and on-screen channels by whether the words/text are actually there, not by picturing them.

Then give ONE overall verdict for the moment as a whole, in "overall":
- "score" 0-10: how confident you are that THIS is the moment the query describes. 9-10 only when the distinctive parts of the query are visibly true here; 6-8 when it fits and something cannot be checked; 4-5 when it is the right kind of scene but the query's own details are missing; 0-3 when the images clearly show something else.
- "reason": at most 25 words, in Vietnamese, for the person reading it. Say what decided it.
- Judge the moment, not the video: a video that contains the right scene somewhere else is still a low score if these shots are not it."""

#: Appended to `CHECK_SYSTEM` only for a Q&A query. The answer is READ from the video -- the
#: pixels, the speech and the on-screen text the same call is already holding -- never recalled
#: from what the model happens to know about the world, which is the rule the agent's own Q&A
#: path keeps (`docs/agent.md`: "answers are read from the video").
CHECK_ANSWER_RULE = """

The query is a question. Also answer it in "answer", reading the answer OFF this moment: what the images show, what the speech quoted below says, or what is written on screen. Answer in Vietnamese, as short as the question allows (a number, a name, a few words; at most 100 characters).
- "source": "frame" if you read it from the pixels, "speech" from the quoted speech, "ocr" from on-screen text, "unknown" if this moment does not answer it.
- "confidence" 0-10: 0 when you could not find the answer here. Do NOT answer from outside knowledge -- if this moment does not say, "source" is "unknown" and "text" is empty.
- "quote": the words you read it from, verbatim, when the source is speech or ocr."""

SUGGEST_SYSTEM = """Một người đang trả lời câu hỏi về một video. Bạn chỉ có các đoạn trích văn bản đánh số theo thời gian (giây) từ video đó: lời thoại (nhận dạng giọng nói tự động, tên riêng có thể sai chính tả), chữ OCR trên màn hình, mô tả cảnh tự động (có thể sai). Nếu các đoạn trích cho thấy đáp án, viết nó đúng như trong văn bản, ngắn và nguyên văn (tối đa 100 ký tự), kèm id đoạn trích chứa nó; nếu không thấy, để answer rỗng. Không đoán, không dùng kiến thức bên ngoài.
Trả về một đối tượng JSON: {"answers": [{"answer": "...", "excerpt": "e3", "confidence": 0-10}]}"""

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
