# Cleaning the 2026 audio transcripts

The 2026 transcripts are raw ASR output. This report records what is actually
wrong with them, what cleaning each defect justified, and — for most of the
defects — why fixing them did not improve retrieval.

The headline: **the defects are real and large, and almost none of them are what
limits audio recall.** Four of the five transforms built here ship disabled.

## The corpus

`data/2026/audio/` holds two parquets, not audio:

| File | Shape |
| --- | --- |
| `transcription.parquet` | 873 rows, `(audio_key, text)` |
| `step_data.parquet` | 1,656,860 rows, `(audio_key, offset, token, log_prob)` |

`log_prob` is misnamed. Its values run 0 to 15.55 with a median of 0.0097, and
the ASR placeholder `<` averages 2.89 against 0.11 for real speech: it is a
*negative* log likelihood, so **lower is more confident**. The code calls it
`nll`.

Timing lives only on the token stream, so every transform works on a word stream
carrying offsets and writes back the same schema. `make_passages` and
`build_transcript_cues` are untouched.

## Measured defects

| Defect | Extent |
| --- | --- |
| No punctuation, no capitalisation | **0 of 873** transcripts contain `.` `,` `!` `?` |
| Numbers spelled out | **0 of 873** contain a digit; 77,361 number words = **5.5% of the corpus**; 1,088 year phrases, 1,609 `tháng`, 633 `phần trăm` |
| Sponsor boilerplate | **606 of 873** videos carry a recurring 8-gram; `"chương trình được tài trợ bởi công ty"` appears in **403** |
| ASR placeholder `<` | 55 tokens; `L24_V022` is nothing else |
| Near-empty transcripts | 30 videos under 30 words |
| Hallucination loops | **none** — 0 videos repeat a 6-gram six times running |

## The benchmark

`benchmarks/audio_ground_truth.json` holds **48 labelled queries** recovered from
the competition answer keys: 2025 round 1 (`queries/round1-2025.csv` against
`queries/query-p*-groupA/`) and 2026 round 1 (`queries/Round12026_answer.zip`
against `queries/SOTUYEN1-bo-de-thi.zip`). Round 1 is judged on L21–L30, exactly
the range the transcripts cover, so all 47 distinct gold videos are scoreable.

The metric is **gold-video recall**. The Q&A answer strings are deliberately not
stored: they are read off the screen rather than spoken — `query-p1-17-qa`'s
answer "Đèo Tà Pứa" appears **zero** times in `L22_V008`'s transcript — so they
measure OCR, not transcript retrieval.

Baseline, BM25 over the same 60s/15s passages the indexer builds:

| | @1 | @5 | @10 | @50 | @100 | median rank |
| --- | --- | --- | --- | --- | --- | --- |
| raw transcripts | 39.6% | 56.2% | 66.7% | 79.2% | 87.5% | 2 |

**48 queries means one query is 2.08 points.** Nothing below moves more than one
or two queries, which is why so little of it ships.

One caveat that took a wrong conclusion to learn: recall@k only sees a change
when a gold video crosses 1, 5, 10, 50 or 100. On a set this small that hides
real movement, so a flat recall table is not by itself evidence of no effect —
check the per-query ranks with `--detail` before concluding, as the restoration
section does.

## What did not work

Each of these was first judged on recall@k, then re-checked on per-query ranks
after that metric turned out to hide movement (see the restoration section). The
verdicts survived the better metric — only restoration has a lean worth acting
on:

| transform | queries better | worse | unchanged | sign test |
| --- | --- | --- | --- | --- |
| boilerplate removal, ≥5 videos | 11 | 6 | 31 | p = 0.33 |
| boilerplate removal, ≥10 videos | 8 | 9 | 31 | p = 1.00 |
| confidence filter, `nll<=3` | 3 | 3 | 42 | p = 1.00 |
| number normalisation | 6 | 8 | 34 | p = 0.79 |
| *capitalisation + punctuation* | *19* | *9* | *20* | *p = 0.087* |


### Boilerplate removal — the premise was wrong

Stripping recurring cross-video phrases removes 6.4–12% of the corpus and
changes recall by at most one query, in both directions:

| config | @1 | @10 | @100 |
| --- | --- | --- | --- |
| raw | 39.6% | 66.7% | 87.5% |
| `boiler>=40v` | 39.6% | 64.6% | 87.5% |
| `boiler>=20v` | 39.6% | 62.5% | 87.5% |
| `boiler>=10v` | 41.7% | 64.6% | 87.5% |
| `boiler>=5v` | 41.7% | 66.7% | 87.5% |

The reasoning behind the transform — "a phrase in 403 documents dominates BM25"
— had it backwards. A phrase in 403 of 873 documents has an IDF near zero and
was already contributing nothing. **IDF is the mechanism that exists for this.**

### Confidence filtering — nothing to filter

At `nll<=3` only 1,573 words go (0.1%) and recall is unchanged. Tighter bounds
start cutting real speech: `nll<=0.3` drops 15% of the corpus and takes @1 from
39.6% down to 33.3%.

### Number normalisation — correct, but nothing to hit

25 of 48 queries contain a digit, but almost all are TRAKE event numbering
(`E1:`, `E2:`) or object counts. Only three carry a content-bearing number, and
**none of the three videos speaks it**:

| Query | Number | Where it really comes from |
| --- | --- | --- |
| `2025:query-p1-23-kis` | 1975 | the film's metadata |
| `2025:query-p1-17-kis` | 2024 | an on-screen banner |
| `2025:query-p1-22-qa` | 200g | a recipe card |

Sweep result: @5 +1 query, @10 −1 query. The transform is right and the corpus
has 1,088 year phrases for it to work on; this benchmark simply cannot see it.

### Place-name correction — dropped before it was built

The motivating example, `"xã i agri huyện chường tỉnh gia lai"`, does not say
what it looks like it says: **`"tỉnh gia lai"` is spelled correctly.** The errors
are in the district and commune names (Ia Grai, Chư Prông).

Across the corpus, 52 of 63 provinces appear and match exactly; the `"tỉnh X"`
spans that fail to match are legitimate compounds — `tỉnh ủy`, `tỉnh thành`,
`tỉnh lộ` — not misspellings. And **2 of 48 queries** mention a province at all,
both already matching. A province gazetteer would have corrected nothing, and
no query names a district.

## What did work

### Capitalisation and punctuation — but not for BM25

This one cannot be measured lexically, and that is not a limitation of the
bench. `tokenize()` here and Elasticsearch's `standard` analyser both lowercase
and discard punctuation, so restoration is a **provable no-op** for BM25:

```
unpunctuated: ['xin', 'chào', 'các', 'bạn', 'hôm', 'nay', ...]
capu output : ['xin', 'chào', 'các', 'bạn', 'hôm', 'nay', ...]  # identical
```

Measured against the embeddings (`--dense`, the same 48 queries):

| config | @1 | @5 | @10 | @50 | @100 | median rank |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 37.5% | 56.2% | 66.7% | 83.3% | 83.3%¹ | 4.0 |
| + restoration | 37.5% | 60.4% | 62.5% | 83.3% | 83.3% | **2.0** |

Read on its own that table says "noise", the same verdict as every other
transform. **It is the wrong table.** With 48 queries, recall@k only registers a
movement that crosses 1, 5, 10, 50 or 100; everything else is invisible. The
per-query ranks (`--detail`) tell a different story:

- **19 queries improved, 9 got worse**, 20 unchanged.
- Two-sided sign test: **p = 0.087**. Not significant at 0.05, so this is not
  proof — but it is a two-to-one lean, not a coin flip.
- Among queries landing near the top, where a competitor actually looks
  (rank ≤ 10 on either side): **9 improved, 4 got worse**.
- Median rank halves, 4.0 to 2.0.

So: restoration **probably** helps retrieval a little, and the benchmark is too
small to prove it. It ships on the strength of the subtitle result below, with
the retrieval evidence as a supporting lean rather than a claim.

¹ The dense branch alone is slightly worse than BM25 alone at rank 1 (37.5%
against 39.6%). Production fuses the two with RRF, which is why it beats either.

**Model bake-off** (`scripts/benchmarks/bench_audio_capu.py`), scored by
stripping formatting from the 8,436 words of correctly punctuated Vietnamese in
`queries/` and `RULES_GUIDELINES.md` and asking each model to restore it:

| model | word count kept | cap. F1 | cap. acc | `.` | `,` | `:` | `?` | macro | words/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dragonSwing/xlm-roberta-capu` | 98.0% | 0.681 | 93.3% | 0.548 | 0.426 | 0.266 | 0.778 | 0.504 | 874 |
| `dragonSwing/vibert-capu` | 97.3% | 0.676 | 93.1% | 0.563 | 0.433 | 0.286 | 0.778 | 0.515 | 760 |

The two are **statistically indistinguishable** — two samples on preservation,
0.011 on macro F1. `xlm-roberta` is selected for its preservation rate and 15%
higher throughput, not because it is better. Capitalisation is genuinely good
(93% accuracy); punctuation is mediocre (macro F1 ~0.5), dragged down by `:` and
by `?` having only 9 instances in the reference.

**Subtitles are the clear win.** `build_transcript_cues` splits on sentence
punctuation and only then falls back to an 8-second gap or 160 characters. With
no punctuation anywhere, the fallback is the only rule that ever fires, and the
character rule cuts mid-word — one `L21_V001` cue ends `"ghép cho bệnh n"`.

| video | cues ending at a sentence, raw | after restoration |
| --- | --- | --- |
| `L21_V001` | 0 / 152 (0%) | 128 / 208 (**61.5%**) |
| `L26_V194` | 0 / 35 (0%) | 37 / 50 (**74.0%**) |
| `L30_V096` | 0 / 11 (0%) | 7 / 14 (**50.0%**) |

Median cue length falls from 143 to 106 characters, and proper nouns come back:
*"Chào mừng quý vị đến với chương trình Sáu mươi giây của Đài truyền hình thành
phố Hồ Chí Minh."*

Both models were published in 2022 and need three shims to load on current
transformers, all documented in `scripts/audio/capu.py`: a real-file snapshot
(their code resolves `verb-form-vocab.txt` next to `__file__`, which the HF blob
cache breaks), `mean_resizing=False` on the embedding resize, and re-applying
`@dataclass` to `Seq2LabelsOutput`.

**The word-count guard earns its place.** Twice over the full corpus,
`GecBERTModel.merge_chunks` raised `UnboundLocalError: tail_idx` — its overlap
matcher found no anchor between two sub-chunks. Their handler catches the
exception, prints it, and leaves `result` untouched, which **silently drops that
sub-chunk's words**. `apply_capu_output` sees the shortfall and returns the raw
chunk instead, so the loss is contained to formatting rather than content. Total
words after restoration are unchanged at 1,406,044, which is exactly what a
count-preserving overlay plus whole-chunk rejection should produce.

### A real bug the hand-check caught

Sampling 729 conversions from the corpus found `"một trăm hai"` being read as
102. In speech it is **120** — a bare digit after the hundreds is the tens place,
and 102 requires the filler (`"một trăm linh hai"`). The context in the corpus
confirms it: `"từ tám mươi lăm đến một trăm hai"`.

A sweep for the rest of that family settled three more cases:

| Pattern | Count | Decision |
| --- | --- | --- |
| `X trăm <digit>` | 60 | Read the digit as tens (`một trăm bốn` → 140) |
| `ngàn/nghìn <digit>` | 109 | **Refuse.** `ngàn hai` may be 1200, but `ngàn năm` (26 of them) is "a thousand years" — and there is no filler form to tell them apart, unlike the hundreds case |
| `X chục` | 115 | Convert (`hai chục` → 20). `mấy chục` and `vài chục` carry no number and are left alone |
| `X rưỡi` | 10 | Leave. Too few to matter and `năm rưỡi` is 5.5 or "half a year" |

`"hai trăm năm"` needs no special handling: the existing `năm` guard already
declines to read it as five, yielding `"200 năm"`, which is right if it meant two
hundred years and merely incomplete if it meant 250.

## What ships

| Transform | Ships | Why |
| --- | --- | --- |
| Capitalisation + punctuation | **on** | Subtitles go from 0% to 50–74% of cues ending at a sentence; retrieval leans positive (19 queries better, 9 worse, p = 0.087) |
| Number normalisation | off | Correct, but no benchmark query can see it |
| Boilerplate removal | off | IDF already handles it |
| Confidence filtering | off | Nothing worth filtering |
| Place-name correction | not built | Provinces are already correct |

`benchmarks/audio_config.json` records this; `scripts/audio/build_clean_transcripts.py`
applies it and writes `step_data_clean.parquet` + `transcription_clean.parquet`.
The restored corpus has **exactly** 1,406,044 words, unchanged from the raw one.

Turning any of the disabled transforms back on is a one-line config change, and
re-running `bench_audio_search --sweep` is how you would justify it.

### A practical note on memory

Running restoration and passage embedding in one process was killed by the OOM
killer at 37% of the second embedding pass, on a machine with about 7 GB free.
Do them separately — `build_clean_transcripts` first, then
`bench_audio_search --dense --transcripts <clean> --baseline-transcripts <raw>`,
which never has both models resident. Passage embeddings are cached by content
hash under `data/2026/audio/.bench_embeddings/`, so the raw side is computed once.

`audio_search.transcript_path()` prefers `step_data_clean.parquet` and falls
back to the raw file, so a clone with no cleaned data keeps working. The index
gains a `text_raw` field at 0.5 weight carrying the pre-cleaning wording, so a
query phrased the way the recogniser heard it still matches anything the cleaner
rewrote.

## Where the headroom actually is

Audio recall is not limited by transcript formatting. It is limited by queries
describing things nobody says out loud — which is what the Q&A answer keys show
directly, and what the three number queries show again. The transcripts are
already good enough that 87.5% of gold videos land in the top 100.
