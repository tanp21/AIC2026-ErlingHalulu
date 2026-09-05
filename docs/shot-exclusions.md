# Shot exclusions

The visual index can omit reviewed intro, preview and presenter shots without
deleting or recomputing their source embeddings. Transcript data and the
Elasticsearch index are never changed.

## Generate suggestions

For the 2026 `60 Giây` videos (`L21` and `L22`):

```bash
uv run python -m scripts.indexing.suggest_shot_exclusions \
  --year 2026 --groups L21,L22 \
  --label-dir artifacts/shot-labels
```

The command averages the three existing PE-Core keyframes in each shot and
compares them with separate presenter-style and program-intro prototypes. A
high threshold handles general presenter suggestions; weaker presenter matches
are accepted only when their tiny colour-layout descriptor matches a
high-confidence presenter from the same video. This reuses the existing
keyframes and prevents red transition graphics or similarly coloured story
footage from being labelled as presenters. High-confidence intro matches are
also suggested throughout the video, covering red section bumpers such as
`L21_V002` shot 113. A high-confidence bumper also anchors a following preview
montage when at least four short shots lead to a presenter within 25 seconds;
this covers `L21_V002` shots 139–145 without swallowing the longer news block
after shot 113. It also
uses the second `60 Giây` bumper to propose the opening preview block. For
mid-program previews, it reads `data/<year>/audio/step_data.parquet`, finds
phrases such as `phần sau` and `tiếp theo trong chương trình`, then accepts only
a short-shot montage that ends before a presenter within 25 seconds. Override
the transcript with `--transcript-path`; if no transcript exists, only the
visual rules run. For `L21/L22`, the generator also detects the fixed red
`TIẾP THEO` corner and lower-third graphics on the representative keyframe. It
keeps only fast runs with at least four visual matches, bridges at most one
short missed shot, and includes an adjacent intro-like bumper. This catches
mid-program previews even when the transcript has no announcement or the
bumper is at the end of the montage. The final L21/L22 shot is also suggested
as `outro` when it is at most eight seconds and its embedding has at least
`0.95` cosine similarity to the reviewed red station end card. Every output is
`suggested`; suggestions do not affect an index until the owner reviews them.

For the program openings in `L25–L30`, run the same command with
`--groups L25,L26,L27,L28,L29,L30`. Fixed opening prefixes cover L25, L26,
L27 and L29. L28 locates the recurring first content shot and stops before it;
L30 includes its recurring title card. A video without a confident anchor is
left unlabelled rather than guessed. These groups receive only `intro`
suggestions.

Open the **Cắt clip** tab as the configured administrator. The shot timeline
shows suggestion counts and can filter by status. Inspect the three review
frames, select a reason, then choose **Loại khỏi index** or **Giữ trong index**.
The bulk buttons apply only to the current video's still-unreviewed suggestions.

Labels live under `artifacts/shot-labels/<group>/<video>.json`. Each file binds
itself to a hash of the shot boundaries; rebuilding TransNetV2 shots makes stale
labels fail closed instead of applying them to the wrong ranges.

Create the writable host directory before the first container start:

```bash
mkdir -p artifacts/shot-labels
```

## Build and activate a filtered profile

Build beside the current profile first:

```bash
uv run python -m scripts.indexing.build_flat_profile \
  --source data/2026/embeddings \
  --shot-label-dir artifacts/shot-labels \
  --name l2026-filtered --label 'L · embeddings 2026 · lọc shot'
```

The summary and `info.json` report `excluded_count` and counts by reason. Verify
the new profile in the UI. Once accepted, set `default` to `l2026-filtered` in
`indexes/profiles/manifest.json` and restart the app.

Audio and Hybrid searches retain the full transcript. Their displayed frame is
resolved through the selected filtered profile, so it cannot be a keyframe from
an excluded shot. Apply the same workflow to `K01–K20` only after the 2026
labels and thresholds have been reviewed; the 2026 seed references are not
silently applied when they do not exist in the target dataset.
