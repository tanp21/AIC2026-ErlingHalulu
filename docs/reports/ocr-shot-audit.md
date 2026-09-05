# OCR shot audit: 7 pilot shots, engine bake-off, full-corpus go/no-go

Snapshot: 2026-09-03 05:50 UTC. Local hardware: RTX 3060 (12 GB).

## What this is

The user asked to try several OCR approaches on 7 hand-picked shots before deciding
whether to run OCR over the full corpus:

| Video | Shot | Status |
| --- | --- | --- |
| `L27_V010` | 83 | Regression — exactly the existing "pillar-couplet" ground-truth frame |
| `L30_V078` | 20 | Regression — exactly the existing "recipe-card" ground-truth frame |
| `L30_V078` | 21 | New — same recipe card, now under motion blur (rack focus) |
| `L30_V072` | 19 | New — charity event banner |
| `K17_V003` | 102 | Regression — exactly the existing "signboard-script" (known-hard, rejected) frame |
| `K19_V016` | 158 | Regression — exactly the existing "gauge-digits" ground-truth frame |
| `L23_V013` | 20 | New — cycling-race finish arch with an LED sign |

Each shot has exactly 3 pre-extracted keyframes (the same ones used for visual-search
embeddings). All 3 per shot were pulled in, hand-labelled against the existing
`benchmarks/ocr_ground_truth.json` schema, and appended there — 17 new frames on top
of the original 5, for 22 total (12 now `acceptance: true`, gating the sweep). See
that file's `shot-audit-2026-09` tag for the added entries, and `tests/test_ocr_benchmark.py`'s
`test_has_the_acceptance_frames` for the locked-in acceptance set.

The user also asked to compare 4 engines outside the existing tuned pipeline: **Gemini**
(as a VLM reader), **PaddleOCR**, **EasyOCR**, and **Tesseract**. These are implemented in
the new `scripts/ocr/alt_engines.py` and wired into `scripts/benchmarks/bench_ocr.py`'s
grid alongside the existing `OcrConfig` variants (see that module's docstring for the
bare-callable convention).

## Blocker (resolved): the tuned pipeline needed VietOCR's config/weights unblocked

Every `OcrConfig`-based grid entry — `baseline` through `postcorrect+reader`, the
pipeline `ocr_config.json` already names as the production winner — initially failed
outright. VietOCR's recognizer (`vietocr.tool.config.Cfg.load_config_from_name`)
fetches its base config YAML, and the model weight checkpoints (`vgg_seq2seq.pth`,
`vgg_transformer.pth`), from `https://vocr.vn` at **every process run** — no local
cache or vendored fallback existed anywhere in `.venv-ocr` or on this machine. That
host's TLS certificate has expired (verified independently with a raw `ssl`
handshake, not an artifact of this sandbox), so every fetch failed closed.

**Fixed properly, without any insecure workaround.** VietOCR's own GitHub release
(`v0.3.2`) hosts both `.pth` checkpoints as release assets on `github.com`'s own
(valid) certificate — downloaded, size-matched against the release metadata, and
loaded with `torch.load(..., weights_only=True)` to confirm they're clean before
trusting them. The base/model config YAMLs came from the same project's GitHub
source. Two changes landed:

- `scripts/ocr/vietocr_configs/{base,vgg-seq2seq,vgg-transformer}.yml` — vendored
  copies of the tiny (1-2 KB) hyperparameter configs VietOCR would otherwise fetch.
- `scripts/ocr/engines.py`'s new `_load_vietocr_config()` reproduces
  `Cfg.load_config_from_name`'s base+model merge from these local files, falling back
  to the network path if they're ever missing (so a future upstream cert renewal isn't
  silently masked). Covered by `tests.test_ocr_benchmark.VietocrConfigTests`
  (`skipUnless` on `vietocr` being importable, so it only actually runs under
  `.venv-ocr`).
- The two `.pth` files were placed at `/tmp/vgg_seq2seq.pth` /
  `/tmp/vgg_transformer.pth` — VietOCR's own `download_weights` already checks that
  exact path by filename before fetching, so no code change was needed for the weight
  half, only file placement. **This is session-local** (`/tmp` is ephemeral); see
  "Follow-up" below for making it permanent.

Verified end-to-end offline (`.venv-ocr`, no network) against the known
`pillar-couplet` frame: output matches the hand-labelled reference exactly except
`QUỶ` → `QUỲ`, which is the same unrecoverable recognizer quirk this frame's own
ground-truth comment already documented before this audit — a strong sign the fix
reproduces the pipeline faithfully rather than some subtly different configuration.

**Follow-up still worth doing** (not done this session): re-host the two `.pth` files
somewhere this repo controls permanently (e.g. the HF Hub, alongside the other models
this project already caches offline under `HF_HUB_OFFLINE=1`) instead of relying on
`/tmp`, which is cleared on reboot and not shared across the team's machines.

## Regression check: fast tiers now measured on the expanded 22-frame ground truth

With the blocker fixed, the 8 fast `OcrConfig` variants (no VLM) were run against all
22 frames. All fail the **expanded** 12-frame acceptance gate — expected, since that
gate now covers 9 harder holdout frames the pipeline was never tuned against, not
just the original 3:

| Config | Gate | Acc. required | Acc. folded | Acc. phrase | CER (all) | fps |
| --- | --- | --- | --- | --- | --- | --- |
| **v5-mobile-det** | FAIL | 0.796 | **0.905** | 0.486 | 0.407 | 0.47 |
| no-clahe | FAIL | 0.751 | 0.833 | 0.528 | 0.416 | 0.39 |
| padded-crop | FAIL | 0.674 | 0.833 | 0.500 | 0.414 | 0.36 |
| transformer-rec | FAIL | 0.729 | 0.823 | 0.528 | 0.410 | 0.25 |
| v5-server-det | FAIL | 0.719 | 0.821 | 0.500 | 0.457 | 0.16 |
| no-conf-floor | FAIL | 0.717 | 0.781 | 0.542 | 0.479 | 0.37 |
| baseline | FAIL | 0.717 | 0.771 | 0.542 | 0.430 | 0.35 |
| no-rescan | FAIL | 0.705 | 0.759 | 0.542 | 0.450 | 0.60 |

Sorted by acceptance folded recall here (not `rank_configs`'s own gate-first ordering,
since every config fails the gate — the interesting signal at this stage is which one
generalizes best to the holdout frames). **`v5-mobile-det` stands out**: it was
measured *worse* than the `baseline` v4 detector in the original 3-4 frame tuning
(the reason `det_model: "v4"` is the current default — see `engines.py`'s docstring),
but on this larger, harder 12-frame set it leads by a wide margin (0.905 vs 0.771
folded recall) and is also faster. That original tuning decision was made on very few
frames; this is exactly the kind of signal a larger holdout set exists to catch, and
it's worth a dedicated re-sweep rather than a one-line read here — the original
tradeoff (v5 resolves the flood-gauge digit unaided but emits extra spurious boxes
that break two-column reading order) may no longer be the right call once averaged
over a broader, more realistic frame mix.

`baseline`'s failures split into two groups:

- **Already-known hard cases, reproduced exactly**: `pillar-couplet` misses only
  `QUỶ` (documented unreadable-by-this-recognizer since before this audit),
  `signboard-script`/`-near1`/`-near2` all miss only `Mộc Châu` (the script-font place
  name the original ground truth already says only a VLM correction stage recovers),
  `signboard-title` misses `NHỮNG DẤU ẤN LỊCH SỬ`. This is the pipeline behaving
  exactly as previously characterized — not a regression.
- **New failure surface from the holdout frames**: the charity banner and finish-arch
  frames fail broadly (missing "FANA", "TON DONG A", "Đà Nẵng", the addresses, etc.).
  This recognizer is tuned on dense/scene Vietnamese text; short all-caps signage,
  Latin sponsor logos, and addresses were never part of its tuning set and it shows.

### `postcorrect` / `postcorrect+reader` — the previous winners, now surprisingly not the best

| Config | Acc. required | Acc. folded | Acc. phrase | CER (all) | fps |
| --- | --- | --- | --- | --- | --- |
| postcorrect+reader | 0.792 | 0.833 | **0.625** | 0.633 | 0.04 |
| postcorrect | 0.750 | 0.781 | 0.625 | 0.377 | 0.06 |

Both still fail the gate, on almost exactly the same frames `baseline` did (`Mộc
Châu`, the pillar's edge-clipped words, the charity banner's address, the finish-arch
sponsor text) — the VLM correction stage recovers some required-string recall over
`baseline` (0.75-0.79 vs 0.717) and clearly helps phrase-level ordering (0.625 vs
0.542), but **`v5-mobile-det`'s plain detector swap still beats both VLM tiers on
folded recall** (0.905 vs 0.833), at roughly 10x the throughput and with no GPU-heavy
VLM in the loop at all. `postcorrect+reader`'s CER (0.633) is also markedly worse
than `postcorrect` alone (0.377) or even `baseline` (0.430) — the Vintern-1B reader's
additive merge, which helped on the original narrow tuning set, appears to be
introducing net-new noise on this broader, more varied holdout set rather than
recovering coverage.

**This is the main actionable finding of the audit**: the existing tuning invested
its cost budget in VLM correction on top of a detector (`v4`) that a 3-4 frame sample
happened to favor. On a broader sample, swapping the detector alone (no VLM at all)
generalizes better than adding the expensive correction stage — and nobody has yet
tried combining them (`postcorrect`/`postcorrect+reader` on top of `det_model:
"v5-mobile"` isn't a config that exists in the current grid). That combination is the
natural next experiment before concluding the current tuning is optimal.

## Engine bake-off: 4 new engines × 22 frames

All 4 ran to completion. Ranked by the existing `rank_configs` rule (gate, then
accuracy, then throughput), reading the gate primarily through **folded recall**
(diacritic-insensitive) per the user's stated priority — OCR output only needs to be
close enough for Elasticsearch's `asciifolding` keyword search, not diacritically
exact:

| Rank | Engine | Gate | Acc. required | Acc. folded | Acc. phrase | CER (all) | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **gemini** | FAIL (close) | 0.958 | **0.979** | 0.917 | 0.228 | 0.47 |
| 2 | paddleocr | FAIL | 0.466 | 0.737 | 0.417 | 0.630 | 2.17 |
| 3 | easyocr | FAIL | 0.444 | 0.551 | 0.417 | 0.603 | 1.26 |
| 4 | tesseract | FAIL | 0.280 | 0.286 | 0.417 | 0.955 | 3.84 |

("Acc." columns are means over the 12 `acceptance: true` frames only; CER/fps are over
all 22.) Full per-frame data in `benchmarks/ocr_benchmark.json`.

### Gemini (`gemini-3.1-flash-lite`) — by far the strongest reader

Nearly clears the gate. Its 4 failures on the 12 acceptance frames:

- **`pillar-couplet-wide`** — missed "HỎA" and "BẠT", the top row of each column,
  which sits right at this frame's top pixel edge. This frame's own ground-truth
  comment already flagged this as unverified-for-clipping; the result is consistent
  with genuine edge-clipping rather than a reading failure.
- **`charity-banner-2`/`charity-banner-3`** — missed "Khánh Hoà" (and the small header
  line on one frame). Checked by re-cropping the source frame at 4x: the banner
  genuinely prints **"Hoà"** (grave accent on the *a*), and Gemini read it as
  **"Hòa"** (grave accent on the *o*) — a real but minor orthographic slip between two
  valid Vietnamese tone-placement conventions for the same word, not a missed reading.
  Exact-string scoring treats these as different strings.
- **`gauge-digits-headline`** — missed digit "10" out of the 7-digit gauge; the frame
  also has a breaking-news headline bar layered over the usual crawl, which may be
  eating into the model's attention.
- A repetition-loop failure mode identical to VietOCR's documented one (see
  `scripts.ocr.engines.is_degenerate`) showed up on `finish-arch-partial` and
  `finish-arch-clear-1` — Gemini read the frame correctly, then got stuck re-emitting
  "TON DONG A" (a sponsor logo stencilled repeatedly along the road barrier) until
  hitting the token budget. Fixed generally in `alt_engines.read_gemini` via
  `_collapse_repeated_lines`, which dedupes consecutive repeated lines rather than
  discarding the whole (mostly-correct) response.

Gemini is not a full-corpus candidate regardless of accuracy: it is a paid, online,
per-call API, which conflicts with this project's fully-offline production container
(`HF_HUB_OFFLINE=1`, read-only filesystem). It is, however, a strong candidate for
exactly the role `postcorrect`/`postcorrect+reader` already play in the tuned
pipeline — a low-confidence shortlist re-reader, not a bulk engine.

### PaddleOCR, EasyOCR, Tesseract — all fail the accuracy gate by a wide margin

All three are meaningfully faster than the tuned pipeline's slow tiers, but none is
close to gate-worthy:

- **PaddleOCR** (`lang="vi"`, PP-OCRv3 multilingual/"latin" recognizer): fastest of the
  three real OCR engines (2.17 fps). Reproduces the exact failure mode this repo's own
  bake-off already documented for the *other* PP-OCR recognizer in `engines.py`'s
  docstring — it strips nearly every Vietnamese diacritic (`"HOA HONG NHAT TAO"` for
  `"HỎA HỒNG NHẬT TẢO"`), which is why folded recall (0.737) is much higher than exact
  recall (0.466). Under the user's relaxed accuracy bar this is more usable than the
  exact numbers suggest, but still well short of the 1.0 folded-recall gate.
- **EasyOCR**: better diacritic recall than PaddleOCR on some words (e.g. correctly
  read "HỎA", "KIẾM", "GIANG", "QUỶ") but inconsistent — folded recall (0.551) is
  actually *lower* than PaddleOCR's, driven by missed/garbled small text and reading
  order.
- **Tesseract**: clearly the floor, as expected going in — required recall 0.28,
  folded 0.286, CER 0.955 (worse than a fresh transcription in terms of edit distance).
  Confirms Tesseract is not viable for this dataset even as a fallback.

Installing these took real environment work worth recording for next time: PaddleOCR
needed a downgrade to the 2.x API line (3.x's `PaddleOCR.ocr()` → `.predict()`
rewrite crashes on CPU inference on this machine with a oneDNN/PIR backend error), and
PaddleOCR/EasyOCR need **separate** venvs from each other (`.venv-ocr-paddle`,
`.venv-ocr-easy`) — their opencv/numpy/scipy pins collide even when isolated together
from the main `.venv-ocr`. Both also need `libgl1` at the OS level (installed via
`sudo apt install libgl1 tesseract-ocr tesseract-ocr-vie`) and
`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` at run time — this Nix-based dev
environment's default dynamic linker doesn't search the host's standard library path.

## All 14 candidates, ranked

Every config/engine tested this session, by acceptance folded recall (all fail the
formal 1.0 gate; this orders by how close each comes):

| Candidate | Acc. required | Acc. folded | Acc. phrase | CER (all) | fps | Offline-eligible? |
| --- | --- | --- | --- | --- | --- | --- |
| gemini | 0.958 | **0.979** | 0.917 | 0.228 | 0.47 | No (paid, online) |
| v5-mobile-det | 0.796 | 0.905 | 0.486 | 0.407 | 0.47 | Yes |
| no-clahe | 0.751 | 0.833 | 0.528 | 0.416 | 0.39 | Yes |
| padded-crop | 0.674 | 0.833 | 0.500 | 0.414 | 0.36 | Yes |
| postcorrect+reader | 0.792 | 0.833 | 0.625 | 0.633 | 0.04 | Yes (slow) |
| transformer-rec | 0.729 | 0.823 | 0.528 | 0.410 | 0.25 | Yes |
| v5-server-det | 0.719 | 0.821 | 0.500 | 0.457 | 0.16 | Yes |
| no-conf-floor | 0.717 | 0.781 | 0.542 | 0.479 | 0.37 | Yes |
| postcorrect | 0.750 | 0.781 | 0.625 | 0.377 | 0.06 | Yes (slow) |
| baseline | 0.717 | 0.771 | 0.542 | 0.430 | 0.35 | Yes |
| no-rescan | 0.705 | 0.759 | 0.542 | 0.450 | 0.60 | Yes |
| paddleocr | 0.466 | 0.737 | 0.417 | 0.630 | 2.17 | Yes |
| easyocr | 0.444 | 0.551 | 0.417 | 0.603 | 1.26 | Yes |
| tesseract | 0.280 | 0.286 | 0.417 | 0.955 | 3.84 | Yes |

## Full-corpus cost estimate

Corpus scale: 2025 (605 videos, 186,082 shots, 558,249 keyframes) + 2026 (873 videos,
96,624 shots, 289,883 keyframes) = **1,478 videos, 282,706 shots, 848,132 keyframes**
combined. A production run would OCR one representative keyframe per shot, not all 3
used for this audit — so **~282,706 frames**, not 848k.

This is an estimate from a small, non-representative sample (22 frames, several of
them the *hardest* known cases by design), not a confidence interval — a real
full-corpus estimate needs a stratified sample the way
`docs/reports/caprl-video-4b-*-estimate.md` used for CapRL captioning.

These fps figures come from 22-frame runs (each config loads its models once, so
load time is amortized far better here than in the original 5-frame micro-benchmark
that produced the ~0.04-0.25 fps numbers in the committed `ocr_config.json` — those
older numbers should be considered stale now that this session has real 22-frame
measurements).

| Candidate | Measured fps | 282,706 frames, local RTX 3060 |
| --- | --- | --- |
| tesseract | 3.84 | ~20 h — fails accuracy gate badly |
| paddleocr | 2.17 | ~36 h (~1.5 days) — fails accuracy gate |
| no-rescan | 0.60 | ~131 h (~5.5 days) — fails accuracy gate |
| v5-mobile-det | 0.47 | ~167 h (~7 days) — **closest offline candidate**, still fails accuracy gate |
| gemini | 0.47 | ~167 h (~7 days) wall clock, plus per-call API cost — disqualified for the offline container regardless |
| baseline | 0.35 | ~224 h (~9.3 days) — fails accuracy gate |
| easyocr | 1.26 | ~62 h (~2.6 days) — fails accuracy gate |
| postcorrect | 0.06 | ~1,309 h (~55 days) — fails accuracy gate |
| postcorrect+reader | 0.04 | ~1,963 h (~82 days) — fails accuracy gate |

This is an estimate from a small, non-representative 22-frame sample (several of them
the *hardest* known cases by design), not a confidence interval — a real full-corpus
estimate needs a stratified sample the way `docs/reports/caprl-video-4b-*-estimate.md`
used for CapRL captioning. No Kaggle/Vast.ai pricing was pulled this session: every
candidate still fails the accuracy gate, so a remote-GPU cost comparison doesn't
change the answer yet — that's the next thing worth pricing out once an
accuracy-passing candidate exists.

## Go/no-go: **not yet**

**Accuracy** is the blocker, not cost. No offline-eligible candidate clears the
1.0-folded-recall gate on the 12 acceptance frames — `v5-mobile-det` comes closest
(0.905) at a very workable ~7-day full-corpus estimate on a single RTX 3060, but that
is still a real gap, and it fails on genuinely different material than the original
tuning covered (short all-caps signage, sponsor logos, addresses) rather than on edge
cases the tuning already knew were hard.

The formal 3-4-frame acceptance set this pipeline was tuned against was too narrow to
catch that `v5-mobile-det` generalizes better than the current default `v4` detector,
or that `postcorrect+reader`'s reader stage adds noise rather than coverage on newer
material. This audit's 22-frame set (12 acceptance) is a better foundation going
forward — it should replace the 3-4-frame set as what future tuning is measured
against, not sit alongside it as a one-off check.

**Next steps, in order:**
1. Re-host `vgg_seq2seq.pth`/`vgg_transformer.pth` somewhere this repo controls
   permanently (they currently work from this session's `/tmp` placement only — see
   "Follow-up" above), so the fix survives a reboot and reaches teammates' machines.
2. Try `postcorrect`/`postcorrect+reader` **on top of `det_model: "v5-mobile"`**
   instead of only on top of the current `v4` baseline — that combination doesn't
   exist in the grid yet, and combines the two best per-axis findings from this audit
   (best detector generalization + best VLM correction) into one config worth scoring
   against the same 22-frame set.
3. If a config clears the gate, get an honest full-corpus cost estimate (stratified
   sample, sensitivity range, Kaggle/Vast.ai pricing) before committing to the run,
   the same way the CapRL estimate reports did.
