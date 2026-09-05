"""Turn raw per-frame OCR detections into searchable segments.

Three independent pure transforms, applied in order by
``scripts.indexing.build_ocr_index``:

1. :func:`suppress_static_overlays` — drop station logos and on-screen clocks,
   which appear in nearly every frame of a broadcast and would otherwise
   dominate BM25 scoring.
2. :func:`reading_order` — recursive XY-cut, so multi-column layouts read
   column-by-column instead of interleaving. This is what makes the two-column
   pillar in ``L27_V010`` come out as two intact couplets.
3. :func:`group_segments` — collapse runs of consecutive keyframes carrying the
   same text into one document, the OCR analogue of
   ``backend.audio_search.make_passages``.

No I/O and no engine imports, so all of it is unit-testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

# Reused so the bench, the indexer, and the tests all agree on what "the same
# text" means.
from scripts.ocr.scoring import fold, normalize

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

#: Text that changes every frame but always occupies the same spot: on-screen
#: clocks and date bugs. Matched against the whole line, not a substring, so a
#: news ticker that merely mentions a time is not affected.
VOLATILE_PATTERNS = (
    re.compile(r"^\d{1,2}[:.]\d{2}([:.]\d{2})?$"),          # 06:39:01, 6.39
    re.compile(r"^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$"),      # 27/08, 27-08-2026
)


@dataclass(frozen=True)
class Line:
    """One detected text line: the polygon the detector returned plus its transcription."""

    text: str
    box: tuple[tuple[float, float], ...]
    conf: float = 1.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(x0, y0, x1, y1)`` of the detection polygon."""
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def height(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return y1 - y0

    @property
    def width(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return x1 - x0


@dataclass(frozen=True)
class Segment:
    """A run of consecutive keyframes sharing the same on-screen text."""

    video_id: str
    start_frame: int
    end_frame: int
    anchor_frame: int
    text: str
    lines: tuple[str, ...]
    frame_count: int
    mean_conf: float

    @property
    def segment_id(self) -> str:
        return f"{self.video_id}:{self.start_frame:06d}"


def to_lines(raw: list[dict]) -> list[Line]:
    """Build :class:`Line` objects from the JSON shape written by ``run_ocr``."""
    return [
        Line(
            text=item["text"],
            box=tuple((float(x), float(y)) for x, y in item["box"]),
            conf=float(item.get("conf", 1.0)),
        )
        for item in raw
        if str(item.get("text", "")).strip()
    ]


# --------------------------------------------------------------------------
# 1. Static overlay suppression
# --------------------------------------------------------------------------


def _cell(line: Line, width: int, height: int, grid: int) -> tuple[int, int]:
    """Quantise a line's centre to a ``grid``x``grid`` cell over the frame."""
    x0, y0, x1, y1 = line.bbox
    cx = min(max((x0 + x1) / 2, 0), width - 1)
    cy = min(max((y0 + y1) / 2, 0), height - 1)
    return int(cx * grid / width), int(cy * grid / height)


def _row(line: Line, height: int, grid: int) -> int:
    """Quantise a line's vertical centre to a row, ignoring x entirely.

    A real news crawl's *text* occupies a different horizontal span on every
    frame -- each headline is a different length, and it is mid-scroll besides
    -- so its bounding box's centre-x wanders across most of the frame width.
    Measured on a real broadcast (``L22_V029`` around frame 24798), two ticker
    items on the same crawl line came back with centres 378px apart, in
    different 80px-wide grid columns. Keying on the full ``(x, y)`` cell (as
    the logo and clock rules do, correctly -- their content sits still) then
    almost never sees the same column twice, so occupancy never clears the
    threshold and the crawl is missed entirely. The crawl's *row* is what
    stays fixed, so that's what the scrolling rule keys on instead.
    """
    x0, y0, x1, y1 = line.bbox
    cy = min(max((y0 + y1) / 2, 0), height - 1)
    return int(cy * grid / height)


def _is_volatile(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in VOLATILE_PATTERNS)


def suppress_static_overlays(
    records: list[dict],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    grid: int = 16,
    text_threshold: float = 0.5,
    volatile_cell_threshold: float = 0.8,
    scrolling_cell_threshold: float = 0.8,
    scrolling_novelty: float = 0.8,
    min_frames: int = 8,
) -> list[dict]:
    """Drop per-video burned-in furniture from a video's frame records.

    Two rules, because the two kinds of furniture behave differently:

    * A station logo has *constant text in a constant place*, so it is caught by
      counting ``(normalised text, cell)`` and dropping anything appearing in
      more than ``text_threshold`` of frames.
    * An on-screen clock has *different text every frame in a constant place*, so
      text-keying misses it entirely. It is caught by requiring the text to look
      like a time or date and its cell to be occupied in more than
      ``volatile_cell_threshold`` of frames.

    * A scrolling news ticker is the inverse of the logo: its *row* is occupied
      in nearly every frame while its text almost never repeats, so neither of
      the rules above can see it. Unlike the logo and clock, a crawl's
      bounding box does not sit at a fixed ``(x, y)`` cell -- each headline is
      a different length and the crawl is mid-scroll besides, so its centre-x
      wanders across most of the frame width and a full-cell key almost never
      repeats. This rule keys on the row alone (``y`` only) for that reason,
      and is caught by requiring high row occupancy *and* high novelty — the
      count of distinct texts seen in that row divided by the number of
      frames the row was occupied.

    Suppressing the ticker is a decision about this competition rather than a
    property of the video. Queries ask about text belonging to the scene —
    signage, documents, banners held up to camera — not about the unrelated
    headlines crawling along the bottom of a news broadcast. Indexing the crawl
    adds a large volume of text that can only ever produce false matches, and it
    dominates BM25 because it is long and dense. Anything wanting the crawl back
    can raise ``scrolling_novelty`` above 1.0 to disable the rule.

    Videos shorter than ``min_frames`` keyframes are returned untouched; the
    frequency statistics are meaningless on a handful of frames and would strip
    genuine text.
    """
    if len(records) < min_frames:
        return [dict(record) for record in records]

    total = len(records)
    text_counts: dict[tuple[str, tuple[int, int]], int] = {}
    cell_counts: dict[tuple[int, int], int] = {}
    volatile_cells: dict[tuple[int, int], int] = {}
    row_counts: dict[int, int] = {}
    row_texts: dict[int, set[str]] = {}

    parsed: list[list[Line]] = []
    for record in records:
        lines = to_lines(record.get("lines", []))
        parsed.append(lines)
        seen_cells: set[tuple[int, int]] = set()
        seen_keys: set[tuple[str, tuple[int, int]]] = set()
        seen_rows: set[int] = set()
        for line in lines:
            cell = _cell(line, width, height, grid)
            row = _row(line, height, grid)
            key = (normalize(line.text), cell)
            # Count each key at most once per frame, so a word repeated within a
            # single frame cannot by itself look like a persistent overlay.
            if key not in seen_keys:
                seen_keys.add(key)
                text_counts[key] = text_counts.get(key, 0) + 1
            if cell not in seen_cells:
                seen_cells.add(cell)
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if row not in seen_rows:
                seen_rows.add(row)
                row_counts[row] = row_counts.get(row, 0) + 1
            row_texts.setdefault(row, set()).add(normalize(line.text))
            if _is_volatile(line.text):
                volatile_cells[cell] = volatile_cells.get(cell, 0) + 1

    def is_overlay(line: Line) -> bool:
        cell = _cell(line, width, height, grid)
        key = (normalize(line.text), cell)
        if text_counts.get(key, 0) > text_threshold * total:
            return True
        if _is_volatile(line.text):
            occupancy = cell_counts.get(cell, 0)
            volatile_share = volatile_cells.get(cell, 0)
            # Both conditions matter: the slot must be near-permanently occupied
            # *and* near-always occupied by clock-shaped text.
            if occupancy > volatile_cell_threshold * total and volatile_share > 0.5 * occupancy:
                return True
        row = _row(line, height, grid)
        occupancy = row_counts.get(row, 0)
        if occupancy > scrolling_cell_threshold * total:
            # Novelty near 1.0 means the row showed something different almost
            # every frame — a crawl. A caption that persists for a while, or a
            # ticker paused on one headline, repeats and stays well below it.
            novelty = len(row_texts.get(row, ())) / occupancy
            if novelty >= scrolling_novelty:
                return True
        return False

    filtered = []
    for record, lines in zip(records, parsed):
        kept = [line for line in lines if not is_overlay(line)]
        filtered.append({
            **record,
            "lines": [
                {"text": line.text, "box": [list(p) for p in line.box], "conf": line.conf}
                for line in kept
            ],
        })
    return filtered


def in_region(
    line: Line,
    region: tuple[float, float, float, float],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> bool:
    """Whether a line's centre falls inside a normalised ``(x0, y0, x1, y1)`` region.

    Coordinates are fractions of the frame so a mask written against 1280x720
    keeps working if the keyframe size ever changes. The *centre* is tested
    rather than any overlap, so a mask drawn tightly around a logo cannot also
    swallow a headline that merely brushes its edge.
    """
    x0, y0, x1, y1 = line.bbox
    cx = (x0 + x1) / 2 / width
    cy = (y0 + y1) / 2 / height
    rx0, ry0, rx1, ry1 = region
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def drop_regions(
    lines: list[Line],
    regions,
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> list[Line]:
    """Remove lines whose centre lies in any masked region.

    The declarative counterpart to :func:`suppress_static_overlays`: same intent
    — get burned-in station furniture out of the index — but stated up front
    instead of inferred from frequency. Two situations need it.

    A single frame carries no frequency statistics, so the benchmark cannot use
    the statistical rule at all; without a mask it scores the engine on a channel
    logo and a clock that never reach the index, which measures the wrong thing.

    And the statistical rule has a real blind spot: it keys a logo on its
    *transcription*, so a bug the recogniser reads inconsistently (``HTV9`` on
    one frame, ``HIVS`` on the next) splits across several keys and may clear
    neither threshold. A region says "nothing here is content" regardless of what
    came back.
    """
    regions = [tuple(float(v) for v in r) for r in (regions or ())]
    if not regions:
        return list(lines)
    return [
        line for line in lines
        if not any(in_region(line, r, width=width, height=height) for r in regions)
    ]


# --------------------------------------------------------------------------
# 2. Reading order (recursive XY-cut)
# --------------------------------------------------------------------------


def _widest_gap(
    intervals: list[tuple[float, float]], lower: float, upper: float
) -> tuple[float, float] | None:
    """Widest uncovered span strictly inside ``[lower, upper]``, as ``(gap, position)``.

    Intervals are merged before scanning, so a box that spans a would-be gap
    closes it. That is what stops a full-width headline from being split down
    the middle of the page.
    """
    if not intervals:
        return None
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    best: tuple[float, float] | None = None
    for left, right in zip(merged, merged[1:]):
        gap = right[0] - left[1]
        if gap > 0 and (best is None or gap > best[0]):
            best = (gap, (left[1] + right[0]) / 2)
    return best


def _median_height(lines: list[Line]) -> float:
    heights = sorted(line.height for line in lines if line.height > 0)
    if not heights:
        return 1.0
    return heights[len(heights) // 2]


def _centre_y(line: Line) -> float:
    x0, y0, x1, y1 = line.bbox
    return (y0 + y1) / 2


def _group_rows(lines: list[Line], tolerance_ratio: float) -> list[list[Line]]:
    """Cluster lines into visual rows by vertical centre, then sort each left to right.

    Centres rather than bounding-box overlap, because a page photographed at an
    angle has *perspective*, not just rotation: on the reference recipe card the
    per-line skew runs from -17.4 to -21.4 degrees, so deskewing by the median
    still leaves a couple of degrees on the outermost lines. Across a 420px-wide
    line that residual tilts the bounding box by ~15px, enough to overlap its
    neighbours and scramble an overlap-based grouping. A line's centre barely
    moves under the same residual, so it orders reliably.
    """
    if not lines:
        return []
    heights = sorted(l.height for l in lines if l.height > 0)
    unit = heights[len(heights) // 2] if heights else 1.0
    tolerance = tolerance_ratio * unit
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: (_centre_y(l), l.bbox[0])):
        centre = _centre_y(line)
        for row in rows:
            row_centre = sum(_centre_y(l) for l in row) / len(row)
            if abs(centre - row_centre) <= tolerance:
                row.append(line)
                break
        else:
            rows.append([line])
    rows.sort(key=lambda row: sum(_centre_y(l) for l in row) / len(row))
    # Centre-y breaks ties: consecutive lines of a justified block share an
    # identical left edge, so sorting on x alone leaves their order undefined.
    return [sorted(row, key=lambda l: (l.bbox[0], _centre_y(l))) for row in rows]


def line_angle(line: Line) -> float:
    """Skew of a detection polygon in degrees, from its top edge.

    Returns 0 for boxes with fewer than four points. Angles are wrapped into
    (-90, 90]; a text line and the same line upside down are not distinguished,
    which is fine because only the skew of the page matters here.
    """
    if len(line.box) < 4:
        return 0.0
    # Identify corners by coordinate sum and difference rather than by "which is
    # highest": on a rotated quad the two topmost points can be the top-left and
    # bottom-left, whose edge is vertical, which reads as an 78-degree skew on a
    # page tilted by 12.
    points = list(line.box)
    top_left = min(points, key=lambda p: p[0] + p[1])
    top_right = min(points, key=lambda p: p[1] - p[0])
    (x0, y0), (x1, y1) = top_left, top_right
    if (x1 - x0) == 0 and (y1 - y0) == 0:
        return 0.0
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def _rotate(line: Line, degrees: float, cx: float, cy: float) -> Line:
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    rotated = tuple(
        (cx + (x - cx) * cos - (y - cy) * sin, cy + (x - cx) * sin + (y - cy) * cos)
        for x, y in line.box
    )
    return Line(text=line.text, box=rotated, conf=line.conf)


def deskew_angle(lines: list[Line], min_lines: int = 3) -> float:
    """Median skew across detections, or 0 when the page is effectively upright.

    Documents photographed at an angle — a sheet of paper held up to camera —
    produce lines whose axis-aligned bounding boxes overlap heavily in both
    directions, so the projection profiles an XY-cut relies on have no gaps and
    reading order collapses into whatever the sort order happened to be. Rotating
    the coordinates by the page's own skew restores those gaps.

    Only the median is used, so a few differently-oriented detections (a station
    logo burned into the corner of an otherwise tilted shot) do not steer it.
    """
    if len(lines) < min_lines:
        return 0.0
    angles = sorted(line_angle(l) for l in lines)
    return angles[len(angles) // 2]


def _rows_spanning(lines: list[Line], position: float, row_overlap: float) -> int:
    """How many visual rows have content on both sides of ``position``.

    This is what separates a column gutter from a hole. Real columns are a
    repeated structure: every row of a two-column layout straddles the gutter. A
    single wide line interrupted by something — a news ticker broken by a station
    logo — straddles it exactly once, and splitting there tears one logical line
    in half and scatters whatever lies between the pieces.
    """
    spanning = 0
    for row in _group_rows(lines, row_overlap):
        if any(l.bbox[2] <= position for l in row) and any(l.bbox[0] > position for l in row):
            spanning += 1
    return spanning


def reading_order(
    lines: list[Line],
    *,
    column_gap_ratio: float = 1.2,
    row_gap_ratio: float = 0.08,
    row_overlap: float = 0.35,
    min_column_rows: int = 2,
    deskew_threshold: float = 3.0,
    max_depth: int = 8,
) -> list[Line]:
    """Sort detections into human reading order via recursive XY-cut.

    At each step the widest *valid* horizontal and vertical gap are measured
    against thresholds scaled by the median line height, so the same settings
    work for a full-screen headline and for 10px body text. A column cut wins
    whenever it qualifies and its gutter is at least as wide as the best row
    gap; otherwise a qualifying row gap is used. Each side is then ordered
    recursively — left before right for a column cut, top before bottom for a
    row cut. When neither gap qualifies the lines are one block and are grouped
    into rows.

    The two thresholds are deliberately very different, and the comparison is on
    *absolute* gap width rather than on how far each gap overshoots its own
    threshold. Normalising first makes the two comparable in the wrong way: with
    a permissive row threshold, an ordinary 20px line gap "overshoots" by more
    than a 280px column gutter does, and a genuine three-column layout gets read
    row-major. Comparing raw widths keeps the wide gutter winning.

    Columns must clear a wide gutter because mistaking word spacing for a column
    would scramble a line. Rows need almost nothing, because splitting two
    stacked blocks top-then-bottom is what reading order does anyway, so an
    over-eager row cut is harmless. That asymmetry is load-bearing: on the
    reference pillar frame the gap between the heading and the body beneath it
    is only 7px against 55px-tall text, and if that row cut does not fire, the
    533px-wide heading spans the gutter and blocks the column split below it
    forever, interleaving the two couplets.
    """

    def cut(group: list[Line], depth: int) -> list[Line]:
        if len(group) <= 1 or depth >= max_depth:
            return sorted(group, key=lambda l: (l.bbox[1], l.bbox[0]))
        unit = _median_height(group)
        x_gap = _widest_gap(
            [(l.bbox[0], l.bbox[2]) for l in group],
            min(l.bbox[0] for l in group),
            max(l.bbox[2] for l in group),
        )
        y_gap = _widest_gap(
            [(l.bbox[1], l.bbox[3]) for l in group],
            min(l.bbox[1] for l in group),
            max(l.bbox[3] for l in group),
        )
        column_ok = bool(x_gap) and x_gap[0] >= column_gap_ratio * unit
        if column_ok and _rows_spanning(group, x_gap[1], row_overlap) < min_column_rows:
            # A gap only one row wide is a hole in a line, not a gutter.
            column_ok = False
        row_ok = bool(y_gap) and y_gap[0] >= row_gap_ratio * unit

        # Compare each candidate gap against the extent of the block it would
        # divide, not against the other gap directly. A gutter's significance is
        # relative: 157px of white space is a real column break across a 500px
        # block and an incidental one across the full 1280px width of a frame.
        # Raw-width comparison mis-ranks exactly that case — on the gauge frame a
        # 157px hole in the news ticker (punched by a station logo) outranked the
        # 247px band separating the ticker from the gauge above it, so the ticker
        # was read as a column and interleaved with the digits.
        span_x = max(l.bbox[2] for l in group) - min(l.bbox[0] for l in group) or 1.0
        span_y = max(l.bbox[3] for l in group) - min(l.bbox[1] for l in group) or 1.0
        column_weight = x_gap[0] / span_x if x_gap else 0.0
        row_weight = y_gap[0] / span_y if y_gap else 0.0

        if column_ok and (not row_ok or column_weight >= row_weight):
            position = x_gap[1]
            left = [l for l in group if l.bbox[2] <= position]
            right = [l for l in group if l.bbox[2] > position]
            if left and right:
                return cut(left, depth + 1) + cut(right, depth + 1)
        if row_ok:
            position = y_gap[1]
            top = [l for l in group if l.bbox[3] <= position]
            bottom = [l for l in group if l.bbox[3] > position]
            if top and bottom:
                return cut(top, depth + 1) + cut(bottom, depth + 1)
        if column_ok:
            position = x_gap[1]
            left = [l for l in group if l.bbox[2] <= position]
            right = [l for l in group if l.bbox[2] > position]
            if left and right:
                return cut(left, depth + 1) + cut(right, depth + 1)
        return [line for row in _group_rows(group, row_overlap) for line in row]

    lines = list(lines)
    if not lines:
        return []
    # Order in a deskewed frame when the page is tilted, then report the original
    # lines in that order — callers get their own Line objects back untouched.
    skew = deskew_angle(lines)
    if abs(skew) >= deskew_threshold:
        xs = [x for l in lines for x, _ in l.box]
        ys = [y for l in lines for _, y in l.box]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        straightened = [_rotate(l, -skew, cx, cy) for l in lines]
        back = {id(s): original for s, original in zip(straightened, lines)}
        return [back[id(s)] for s in cut(straightened, 0)]
    return cut(lines, 0)


# --------------------------------------------------------------------------
# 2b. Within-frame diacritic harmonisation
# --------------------------------------------------------------------------

#: Splits a line into alternating word / separator pieces, keeping the
#: separators so the line can be rebuilt byte-for-byte apart from the words that
#: actually change. Interior punctuation has to split too: the recogniser drops
#: spaces around it often enough that ``bếp.nhấc`` is a single whitespace token,
#: and edge-stripping alone would leave both of those words unmatchable.
_WORD_SPLIT = re.compile(r"([^\w]+)", re.UNICODE)


def _match_case(source: str, target: str) -> str:
    """Re-apply ``source``'s capitalisation to ``target``.

    A word at the start of a line is capitalised and the same word mid-sentence
    is not, so the majority form has to be re-cased before it can stand in for
    the minority one — otherwise fixing ``Nhãn`` to ``nhân`` would trade a wrong
    tone mark for a wrong capital.
    """
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def harmonize_diacritics(
    lines: list[Line],
    *,
    min_length: int = 2,
    min_count: int = 2,
    majority_ratio: float = 2.0,
) -> list[Line]:
    """Make tone marks agree with the majority spelling of the same word in the frame.

    The recogniser's dominant residual failure is a single wrong tone mark on a
    word it reads correctly elsewhere on the same page: on the reference recipe
    card it returns ``nhân`` three times and ``Nhãn`` once, ``bếp`` three times
    and ``bép`` once, ``hành`` three times and ``hánh`` once. Each minority
    reading is a real Vietnamese word, so no dictionary can reject it — but the
    frame is its own evidence, and three-to-one is not a close call.

    Tokens are keyed on their diacritic-folded, punctuation-stripped form, so
    only spellings that differ *purely* in tone marks can ever be merged;
    ``vào``/``vàn`` fold differently and are left alone. A minority spelling is
    rewritten only when the majority appears at least ``min_count`` times and at
    least ``majority_ratio`` times as often, which keeps a one-against-one tie
    untouched — with no majority there is nothing to learn from.

    Confidence is deliberately *not* consulted. On the card the minority ``Nhãn``
    scores 0.85 against the majority's 0.85 mean, so a confidence guard would
    veto the single most valuable correction available. Repetition is the
    stronger signal here, and it is the one being measured.

    This can be wrong: two genuinely different words that differ only in tone
    (``má`` and ``mà``) in a three-to-one split would be merged. The cost is
    bounded — one tone mark on the rarer token, in a field Elasticsearch also
    indexes ascii-folded — and it is measurably outweighed. Set
    ``min_count`` high, or skip the pass, to trade the other way.
    """
    counts: dict[str, dict[str, int]] = {}
    for line in lines:
        for word in _WORD_SPLIT.split(line.text)[::2]:
            key = fold(word)
            if len(key) < min_length:
                continue
            counts.setdefault(key, {})
            counts[key][word] = counts[key].get(word, 0) + 1

    winners: dict[str, str] = {}
    for key, spellings in counts.items():
        if len(spellings) < 2:
            continue
        ranked = sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))
        (best, best_count), (_, runner_up_count) = ranked[0], ranked[1]
        if best_count >= min_count and best_count >= majority_ratio * runner_up_count:
            winners[key] = best
    if not winners:
        return list(lines)

    rewritten = []
    for line in lines:
        pieces = _WORD_SPLIT.split(line.text)
        for index in range(0, len(pieces), 2):
            word = pieces[index]
            winner = winners.get(fold(word))
            if winner and winner != word:
                pieces[index] = _match_case(word, winner)
        rewritten.append(Line(text="".join(pieces), box=line.box, conf=line.conf))
    return rewritten


def render_text(lines: list[Line]) -> str:
    """Join ordered lines into the single string that gets indexed and phrase-matched."""
    return " ".join(line.text.strip() for line in lines if line.text.strip()).strip()


# --------------------------------------------------------------------------
# 3. Run-length segment grouping
# --------------------------------------------------------------------------


def _tokens(lines: list[Line]) -> set[str]:
    return {normalize(line.text) for line in lines if line.text.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_segments(
    video_id: str,
    records: list[dict],
    *,
    similarity: float = 0.6,
    max_frame_gap: int | None = None,
    **order_kwargs,
) -> list[Segment]:
    """Collapse consecutive keyframes carrying the same text into segments.

    Walks the video's keyframes in ``frame_idx`` order, extending the current
    run while the Jaccard similarity of normalised line sets stays at or above
    ``similarity``. Frames with no text end the current run and produce nothing.

    ``anchor_frame`` is the member frame with the highest total confidence-weighted
    text length — the frame whose thumbnail best shows why the segment matched,
    which is what the UI displays for a hit.

    ``max_frame_gap`` optionally breaks a run when two consecutive *keyframes* are
    far apart in the source video, so a caption that happens to recur after a long
    unrelated stretch does not merge into one implausibly long segment.
    """
    ordered = sorted(records, key=lambda r: int(r["frame_idx"]))
    segments: list[Segment] = []
    run: list[tuple[int, list[Line], set[str]]] = []

    def flush():
        if not run:
            return
        best = max(
            run,
            key=lambda item: sum(len(l.text) * l.conf for l in item[1]),
        )
        ordered_lines = reading_order(best[1], **order_kwargs)
        text = render_text(ordered_lines)
        if not text:
            return
        confidences = [l.conf for _, lines, _ in run for l in lines]
        segments.append(Segment(
            video_id=video_id,
            start_frame=run[0][0],
            end_frame=run[-1][0],
            anchor_frame=best[0],
            text=text,
            lines=tuple(l.text.strip() for l in ordered_lines if l.text.strip()),
            frame_count=len(run),
            mean_conf=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        ))

    for record in ordered:
        frame_idx = int(record["frame_idx"])
        lines = to_lines(record.get("lines", []))
        tokens = _tokens(lines)
        if not tokens:
            flush()
            run = []
            continue
        if run:
            too_far = max_frame_gap is not None and frame_idx - run[-1][0] > max_frame_gap
            if too_far or _jaccard(run[-1][2], tokens) < similarity:
                flush()
                run = []
        run.append((frame_idx, lines, tokens))
    flush()
    return segments
