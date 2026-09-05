import unittest

from scripts.ocr.postprocess import (
    Line,
    drop_regions,
    group_segments,
    harmonize_diacritics,
    in_region,
    reading_order,
    render_text,
    suppress_static_overlays,
    to_lines,
)


def box(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def line(text, x0, y0, x1, y1, conf=0.95):
    return Line(text=text, box=box(x0, y0, x1, y1), conf=conf)


def raw(text, x0, y0, x1, y1, conf=0.95):
    return {"text": text, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "conf": conf}


# Coordinates traced from data/2026/keyframes/L27/L27_V010/005576.png (1280x720):
# an "HTV / Online" bug at top left, a full-width heading, and the couplet in two
# vertical columns of one word each.
PILLAR_LINES = [
    line("HTV", 15, 8, 75, 38),
    line("Online", 10, 55, 108, 85),
    line("ANH HÙNG DÂN TỘC NGUYỄN TRUNG TRỰC", 398, 14, 918, 36),
    line("HỎA", 432, 62, 512, 100),
    line("HỒNG", 428, 118, 530, 158),
    line("NHẬT", 428, 175, 522, 215),
    line("TẢO", 432, 230, 505, 270),
    line("OANH", 425, 285, 528, 325),
    line("THIÊN", 422, 340, 528, 378),
    line("ĐỊA", 432, 392, 505, 432),
    line("KIẾM", 770, 62, 862, 100),
    line("BẠT", 770, 118, 845, 158),
    line("KIÊN", 765, 175, 855, 215),
    line("GIANG", 762, 230, 888, 270),
    line("KHẤP", 768, 285, 880, 325),
    line("QUỶ", 775, 340, 862, 380),
    line("THẦN", 768, 392, 888, 432),
]


# The boxes the detector actually returns for that frame, as opposed to the
# idealised ones above. Two differences matter and both broke a first attempt:
# the heading sits only 7px above the body (not 20px), and the couplet lines
# overlap each other vertically by a pixel or two.
PILLAR_LINES_MEASURED = [
    line("HTV", 18, 3, 127, 45),
    line("ANH HÙNG DÂN TỘC NGUYỄN TRUNG TRỰC", 388, 4, 921, 42),
    line("Online", 2, 49, 118, 87),
    line("HỎA", 425, 49, 528, 107),
    line("KIẾM", 765, 49, 874, 107),
    line("BẠT", 776, 106, 869, 165),
    line("HỒNG", 410, 107, 537, 158),
    line("KIÊN", 766, 160, 870, 215),
    line("NHẬT", 414, 163, 529, 213),
    line("TẢO", 423, 215, 514, 267),
    line("GIANG", 755, 223, 888, 268),
    line("KHẤP", 759, 269, 885, 327),
    line("OANH", 407, 273, 536, 323),
    line("THIÊN", 403, 324, 532, 380),
    line("QUỶ", 775, 324, 880, 382),
    line("THẦN", 758, 379, 886, 439),
    line("ĐỊA", 425, 384, 509, 436),
]


class ReadingOrderTests(unittest.TestCase):
    def test_two_column_pillar_keeps_couplets_intact(self):
        """The frame-2 acceptance regression.

        Sorting these detections naively by y would produce
        'HỎA KIẾM HỒNG BẠT …'. The XY-cut must emit each column whole.
        """
        text = render_text(reading_order(PILLAR_LINES))
        self.assertIn("HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA", text)
        self.assertIn("KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN", text)

    def test_full_width_heading_is_not_split_down_the_middle(self):
        # The heading spans the gutter, so the top-level column cut must not fire
        # inside it; it has to be separated by a row cut first.
        text = render_text(reading_order(PILLAR_LINES))
        self.assertIn("ANH HÙNG DÂN TỘC NGUYỄN TRUNG TRỰC", text)

    def test_naive_vertical_sort_would_have_failed(self):
        # Documents why the XY-cut is needed at all: assert the wrong algorithm
        # really does interleave, so this test fails loudly if someone
        # "simplifies" reading_order back into a y-sort.
        naive = sorted(PILLAR_LINES, key=lambda l: (l.bbox[1], l.bbox[0]))
        self.assertNotIn("HỎA HỒNG NHẬT TẢO", render_text(naive))

    def test_measured_pillar_boxes_keep_couplets_intact(self):
        """Same assertion as above but on the detector's real output.

        The idealised fixture passed while this one failed: the real
        heading-to-body gap is 7px against 55px-tall text, so the row cut that
        frees the columns below only fires with a permissive row threshold.
        """
        text = render_text(reading_order(PILLAR_LINES_MEASURED))
        self.assertIn("HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA", text)
        self.assertIn("KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN", text)

    def test_narrow_heading_gap_still_separates_heading_from_body(self):
        # Isolates the 7px-gap case: a wide heading over two columns.
        lines = [
            line("WIDE HEADING ACROSS THE PAGE", 400, 4, 920, 42),
            line("left-a", 420, 49, 530, 104),
            line("left-b", 420, 110, 530, 165),
            line("right-a", 760, 49, 880, 104),
            line("right-b", 760, 110, 880, 165),
        ]
        text = render_text(reading_order(lines))
        self.assertEqual(
            text, "WIDE HEADING ACROSS THE PAGE left-a left-b right-a right-b"
        )

    def test_ticker_split_by_a_logo_is_not_read_as_two_columns(self):
        """Geometry from the gauge frame (K19_V016).

        A bottom news ticker is interrupted by a station logo, leaving a 157px
        hole. Above it sits a column of gauge digits. Ranking gaps by raw width
        treats that hole as a column gutter and interleaves the ticker with the
        digits; ranking each gap against the block it divides does not.
        """
        lines = [
            line("HTV9", 1060, 61, 1152, 92),
            line("06:39:01", 1051, 92, 1131, 112),
            line("20", 969, 358, 998, 386),
            line("18", 944, 397, 969, 428),
            line("16", 967, 443, 992, 473),
            line("14", 943, 486, 965, 524),
            line("12", 964, 536, 991, 564),
            line("10", 938, 580, 965, 607),
            line("8", 970, 631, 981, 646),
            line("cận trẻ em gây hoang mang dư luận", 0, 655, 441, 689),
            line("Quảng Trị phát hiện lán trại vàng tặc", 598, 654, 1268, 691),
        ]
        text = render_text(reading_order(lines))
        # The gauge column must come out contiguous and top-to-bottom. Whether it
        # precedes or follows the ticker is genuinely ambiguous here — the "8"
        # ends at y=646 and the ticker starts at y=654, so no row gap separates
        # them — and both readings are defensible. Contiguity is what retrieval
        # depends on, so that is what is pinned.
        self.assertIn("20 18 16 14 12 10 8", text)
        # The two ticker halves must not be split across the digits.
        self.assertNotIn("cận trẻ em gây hoang mang dư luận 20", text)

    def test_ticker_with_a_clean_logo_gap_keeps_both_items_in_order(self):
        """Geometry from K17_V003 frame 11191.

        The same hazard as the gauge frame but in its clearest form: two separate
        news items in one ticker band, parted by a circular '60' logo. Here the
        logo is detected as its own line, so masking it leaves a genuine 155px
        hole between the halves — wider than the 37px band is tall, and with no
        other content anywhere near that x. Nothing but the min_column_rows rule
        stops that from reading as a two-column layout, which would emit the
        right-hand item first.
        """
        lines = [
            line("KHU DU LỊCH QUỐC GIA", 446, 477, 704, 511),
            line("trữ nước vách núi - giải cơn khát\" cho người dân vùng núi cao",
                 4, 653, 775, 690),
            line("Thái Bình, Kon Tum: Chuẩn bị", 930, 655, 1278, 687),
        ]
        text = render_text(reading_order(lines))
        self.assertTrue(text.startswith("KHU DU LỊCH QUỐC GIA"), text)
        self.assertLess(text.index("trữ nước"), text.index("Thái Bình"), text)

    def test_single_line_is_returned_unchanged(self):
        only = [line("solo", 10, 10, 100, 40)]
        self.assertEqual(render_text(reading_order(only)), "solo")

    def test_empty_input(self):
        self.assertEqual(reading_order([]), [])

    def test_ordinary_single_column_reads_top_to_bottom(self):
        lines = [
            line("second", 100, 200, 600, 240),
            line("first", 100, 100, 600, 140),
            line("third", 100, 300, 600, 340),
        ]
        self.assertEqual(render_text(reading_order(lines)), "first second third")

    def test_words_on_one_line_read_left_to_right(self):
        # Word spacing must never be mistaken for a column gutter.
        lines = [
            line("cuốn", 300, 100, 380, 140),
            line("Nhân", 100, 100, 190, 140),
            line("bánh", 200, 100, 290, 140),
        ]
        self.assertEqual(render_text(reading_order(lines)), "Nhân bánh cuốn")

    def test_three_columns(self):
        lines = []
        for col, x in enumerate((100, 500, 900)):
            for row, y in enumerate((100, 160, 220)):
                lines.append(line(f"c{col}r{row}", x, y, x + 120, y + 40))
        text = render_text(reading_order(lines))
        self.assertEqual(text, "c0r0 c0r1 c0r2 c1r0 c1r1 c1r2 c2r0 c2r1 c2r2")

    def test_scales_to_small_text(self):
        # Same layout as the pillar but at a tenth of the size: thresholds are
        # relative to median line height, so the outcome must not change.
        small = [
            Line(text=l.text, box=tuple((x / 10, y / 10) for x, y in l.box), conf=l.conf)
            for l in PILLAR_LINES
        ]
        self.assertIn("HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA", render_text(reading_order(small)))


class ToLinesTests(unittest.TestCase):
    def test_drops_blank_detections(self):
        lines = to_lines([raw("real", 0, 0, 10, 10), raw("   ", 0, 0, 10, 10)])
        self.assertEqual([l.text for l in lines], ["real"])

    def test_defaults_confidence(self):
        lines = to_lines([{"text": "x", "box": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
        self.assertEqual(lines[0].conf, 1.0)


class DropRegionsTests(unittest.TestCase):
    """Declarative overlay masking, used by the benchmark and by per-video config.

    Coordinates below are the real ones measured on the acceptance frames.
    """

    # The gauge frame's top-right corner: an "HTV9" bug over a clock.
    GAUGE_MASK = [(0.78, 0.02, 0.96, 0.20)]
    GAUGE = [
        line("HIVS", 1057, 59, 1154, 95),
        line("06:39", 1051, 91, 1134, 113),
        line("20", 969, 358, 999, 384),
        line("8", 970, 632, 982, 648),
        line("cận trẻ em gây hoang mang dư luận", 0, 654, 442, 690),
    ]

    def test_masks_the_logo_and_the_clock(self):
        kept = drop_regions(self.GAUGE, self.GAUGE_MASK)
        self.assertEqual([l.text for l in kept],
                         ["20", "8", "cận trẻ em gây hoang mang dư luận"])

    def test_keeps_everything_when_no_regions_given(self):
        for empty in (None, [], ()):
            self.assertEqual(len(drop_regions(self.GAUGE, empty)), len(self.GAUGE))

    def test_returns_a_new_list(self):
        kept = drop_regions(self.GAUGE, None)
        self.assertIsNot(kept, self.GAUGE)

    def test_pillar_bug_mask_spares_the_heading(self):
        # The heading starts at x=388 (0.303) — well clear of a mask that must
        # still cover "Online", whose box begins at x=2.
        lines = [
            line("HTV", 18, 4, 127, 45),
            line("Online", 2, 49, 118, 87),
            line("ANH HÙNG DÂN TỘC NGUYỄN TRUNG TRỰC", 388, 4, 921, 42),
        ]
        kept = drop_regions(lines, [(0.0, 0.0, 0.15, 0.17)])
        self.assertEqual([l.text for l in kept],
                         ["ANH HÙNG DÂN TỘC NGUYỄN TRUNG TRỰC"])

    def test_masks_by_centre_not_by_overlap(self):
        # A headline that merely brushes the mask's edge must survive; keying on
        # any intersection would let a corner mask delete a full-width line.
        grazing = line("headline", 100, 0, 900, 40)
        self.assertEqual(drop_regions([grazing], [(0.0, 0.0, 0.15, 0.17)]), [grazing])

    def test_accepts_lists_as_regions(self):
        # Regions arrive from JSON, where a box is a list, not a tuple.
        self.assertEqual(drop_regions(self.GAUGE, [[0.78, 0.02, 0.96, 0.20]]),
                         drop_regions(self.GAUGE, self.GAUGE_MASK))

    def test_multiple_regions_are_unioned(self):
        kept = drop_regions(self.GAUGE, [(0.78, 0.02, 0.96, 0.20), (0.0, 0.85, 0.4, 1.0)])
        self.assertEqual([l.text for l in kept], ["20", "8"])

    def test_in_region_respects_frame_size(self):
        # Normalised coordinates must follow the frame, not assume 1280x720.
        small = line("x", 50, 50, 70, 70)
        self.assertTrue(in_region(small, (0.0, 0.0, 0.5, 0.5), width=200, height=200))
        self.assertFalse(in_region(small, (0.0, 0.0, 0.5, 0.5), width=100, height=100))


class HarmonizeDiacriticsTests(unittest.TestCase):
    """Lines below are the real recogniser output for the recipe-card frame."""

    RECIPE = [
        line("Nhãn bánh cuốn", 327, 72, 417, 121, conf=0.85),
        line("Nạc dâm xay 200 gr (ướp bột nêm)", 207, 101, 403, 190, conf=0.85),
        line("hánh tây", 217, 168, 282, 206, conf=0.83),
        line("Cách xào nhân", 192, 220, 272, 266, conf=0.85),
        line("Đặt bép, cho dầu ân, đợi chảo nóng già, cho hành tây xào tái",
             194, 120, 613, 295, conf=0.80),
        line("nhân ráo khô thì tất bếp.nhấc khỏi bếp đổ ngay chên hành tây",
             206, 154, 621, 331, conf=0.81),
        line("thì nhân sẽ rất thơm", 218, 249, 418, 345, conf=0.88),
        line("thu cho sau cùng khi vừa tắt bếp mùi tiêu sẽ thơm hơn",
             275, 288, 500, 394, conf=0.85),
    ]

    def _text(self, lines):
        return " ".join(l.text for l in harmonize_diacritics(lines))

    def test_fixes_the_title_from_the_body(self):
        # 'nhân' appears three times below; 'Nhãn' once in the title.
        self.assertIn("Nhân bánh cuốn", self._text(self.RECIPE))

    def test_preserves_capitalisation_of_the_token_it_replaces(self):
        # The majority spelling is lowercase 'nhân'; the title must not become
        # lowercase in the course of fixing its tone mark.
        self.assertNotIn("nhãn", self._text(self.RECIPE).lower().split(" bánh")[0])

    def test_fixes_a_token_carrying_punctuation(self):
        # 'bép,' has a trailing comma and 'bếp.' a period; both must key to the
        # same word as bare 'bếp', and the punctuation must survive.
        self.assertIn("Đặt bếp,", self._text(self.RECIPE))

    def test_fixes_a_midline_token(self):
        self.assertIn("hành tây", self._text(self.RECIPE))
        self.assertNotIn("hánh", self._text(self.RECIPE))

    def test_leaves_a_word_with_no_majority_alone(self):
        # 'dâm' appears once and nothing folds to 'dam' beside it, so there is no
        # evidence either way and the pass must not invent any.
        self.assertIn("Nạc dâm xay", self._text(self.RECIPE))

    def test_does_not_break_a_one_against_one_tie(self):
        tied = [line("chén", 0, 0, 50, 20), line("chèn", 0, 30, 50, 50)]
        self.assertEqual(self._text(tied), "chén chèn")

    def test_only_merges_spellings_differing_in_diacritics(self):
        # 'vào' and 'vàn' fold to different keys and must never be merged, however
        # lopsided the counts.
        lines = [line("vào vào vào", 0, 0, 90, 20), line("vàn", 0, 30, 30, 50)]
        self.assertEqual(self._text(lines), "vào vào vào vàn")

    def test_uppercase_minority_stays_uppercase(self):
        lines = [line("HỒNG", 0, 0, 60, 20),
                 line("hông hông hông", 0, 30, 90, 50)]
        self.assertIn("HÔNG", self._text(lines))

    def test_leaves_short_tokens_alone(self):
        # One- and two-letter Vietnamese particles differ meaningfully by tone
        # ('ở' vs 'ó'); the evidence is too thin to act on.
        lines = [line("à", 0, 0, 10, 20), line("a a a", 0, 30, 40, 50)]
        self.assertEqual(self._text(lines), "à a a a")

    def test_respects_a_raised_min_count(self):
        lines = [line("bép", 0, 0, 30, 20), line("bếp bếp", 0, 30, 60, 50)]
        self.assertIn("bếp bếp bếp",
                      " ".join(l.text for l in harmonize_diacritics(lines)))
        self.assertIn("bép",
                      " ".join(l.text for l in harmonize_diacritics(lines, min_count=5)))

    def test_boxes_and_confidence_survive(self):
        harmonized = harmonize_diacritics(self.RECIPE)
        self.assertEqual([l.box for l in harmonized], [l.box for l in self.RECIPE])
        self.assertEqual([l.conf for l in harmonized], [l.conf for l in self.RECIPE])

    def test_pillar_frame_is_untouched(self):
        # Every word on the couplet frame is unique, so nothing votes on anything
        # — including the QUỲ/QUỶ error this pass cannot help with.
        lines = [line(w, 0, i * 50, 100, i * 50 + 40)
                 for i, w in enumerate("KIẾM BẠT KIÊN GIANG KHẤP QUỲ THẦN".split())]
        self.assertEqual(self._text(lines), "KIẾM BẠT KIÊN GIANG KHẤP QUỲ THẦN")

    def test_empty_input(self):
        self.assertEqual(harmonize_diacritics([]), [])


class StaticOverlayTests(unittest.TestCase):
    def _broadcast(self, frames=20):
        """A synthetic newscast: fixed logo, ticking clock, changing ticker."""
        records = []
        for i in range(frames):
            records.append({
                "frame_idx": i * 25,
                "lines": [
                    raw("HTV9", 1180, 50, 1260, 90),                     # station logo
                    raw(f"06:39:{i:02d}", 1150, 95, 1260, 125),          # clock
                    raw(f"tin thời sự số {i}", 100, 650, 900, 690),      # ticker
                ],
            })
        return records

    def test_drops_constant_logo(self):
        filtered = suppress_static_overlays(self._broadcast())
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertNotIn("HTV9", texts)

    def test_drops_ticking_clock_despite_changing_text(self):
        # The clock's text differs in every frame, so frequency-of-text alone
        # cannot catch it; the volatile-pattern plus occupied-cell rule must.
        filtered = suppress_static_overlays(self._broadcast())
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertFalse([t for t in texts if t.startswith("06:39")])

    def test_drops_scrolling_ticker(self):
        # The crawl is the inverse of the logo: same slot every frame, different
        # text every frame, so neither the constant-text nor the clock rule sees
        # it. Excluded on instruction — queries ask about scene text, not about
        # unrelated headlines running under a broadcast.
        filtered = suppress_static_overlays(self._broadcast())
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertNotIn("tin thời sự số 7", texts)

    def test_scrolling_rule_can_be_disabled(self):
        filtered = suppress_static_overlays(self._broadcast(), scrolling_novelty=1.1)
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertIn("tin thời sự số 7", texts)

    def test_keeps_a_caption_that_lingers(self):
        # A lower third that holds for several frames at a time repeats, so its
        # novelty stays well below the threshold. This is the distinction the
        # rule turns on, and the reason it is novelty rather than position.
        records = []
        for i in range(20):
            records.append({"frame_idx": i * 25, "lines": [
                raw("HTV9", 1180, 50, 1260, 90),
                raw(f"phóng sự {i // 5}", 100, 650, 900, 690),
            ]})
        filtered = suppress_static_overlays(records)
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertIn("phóng sự 2", texts)

    def test_keeps_scene_text_that_changes_with_the_shot(self):
        # Scene text moves around the frame between shots, so no single cell is
        # occupied often enough to trip the occupancy half of the rule.
        records = []
        for i in range(20):
            records.append({"frame_idx": i * 25, "lines": [
                raw(f"biển hiệu {i}", 100 + (i % 4) * 200, 200 + (i % 3) * 150,
                    300 + (i % 4) * 200, 240 + (i % 3) * 150),
            ]})
        filtered = suppress_static_overlays(records)
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertIn("biển hiệu 7", texts)

    def test_keeps_text_repeated_in_only_a_minority_of_frames(self):
        records = self._broadcast()
        for record in records[:5]:
            record["lines"].append(raw("BẢN TIN ĐẶC BIỆT", 300, 300, 800, 350))
        filtered = suppress_static_overlays(records)
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertIn("BẢN TIN ĐẶC BIỆT", texts)

    def test_short_videos_are_left_alone(self):
        # With three keyframes, "appears in >50% of frames" is meaningless and
        # would delete genuine text.
        records = self._broadcast(frames=3)
        filtered = suppress_static_overlays(records)
        texts = {l["text"] for record in filtered for l in record["lines"]}
        self.assertIn("HTV9", texts)

    def test_same_text_in_a_different_place_is_kept(self):
        # Keying on (text, cell) rather than text alone: a word that happens to
        # match the logo but appears elsewhere in the scene is content.
        records = self._broadcast()
        records[3]["lines"].append(raw("HTV9", 200, 300, 300, 340))
        filtered = suppress_static_overlays(records)
        kept = [l for l in filtered[3]["lines"] if l["text"] == "HTV9"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["box"][0], [200, 300])

    def test_preserves_other_record_fields(self):
        filtered = suppress_static_overlays(self._broadcast())
        self.assertEqual(filtered[2]["frame_idx"], 50)


class GroupSegmentsTests(unittest.TestCase):
    def test_merges_consecutive_identical_frames(self):
        records = [
            {"frame_idx": f, "lines": [raw("BIỂN ĐÔNG", 100, 100, 400, 140)]}
            for f in (10, 20, 30, 40)
        ]
        segments = group_segments("L21_V001", records)
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (10, 40))
        self.assertEqual(segments[0].frame_count, 4)

    def test_splits_when_text_changes(self):
        records = [
            {"frame_idx": 10, "lines": [raw("MỘT", 100, 100, 400, 140)]},
            {"frame_idx": 20, "lines": [raw("MỘT", 100, 100, 400, 140)]},
            {"frame_idx": 30, "lines": [raw("HAI", 100, 100, 400, 140)]},
        ]
        segments = group_segments("L21_V001", records)
        self.assertEqual([s.text for s in segments], ["MỘT", "HAI"])
        self.assertEqual(segments[0].end_frame, 20)
        self.assertEqual(segments[1].start_frame, 30)

    def test_partial_overlap_above_threshold_extends_the_run(self):
        # Three of four lines shared -> Jaccard 3/5 = 0.6, at the threshold.
        common = [raw(t, 100, 100 + 50 * i, 400, 140 + 50 * i) for i, t in enumerate("abc")]
        records = [
            {"frame_idx": 10, "lines": common + [raw("d", 100, 300, 400, 340)]},
            {"frame_idx": 20, "lines": common + [raw("e", 100, 300, 400, 340)]},
        ]
        self.assertEqual(len(group_segments("L21_V001", records)), 1)

    def test_blank_frames_break_runs_and_produce_nothing(self):
        records = [
            {"frame_idx": 10, "lines": [raw("MỘT", 100, 100, 400, 140)]},
            {"frame_idx": 20, "lines": []},
            {"frame_idx": 30, "lines": [raw("MỘT", 100, 100, 400, 140)]},
        ]
        segments = group_segments("L21_V001", records)
        self.assertEqual(len(segments), 2)

    def test_anchor_is_the_frame_with_the_most_confident_text(self):
        # Same text throughout so the run stays intact; only confidence varies,
        # isolating the anchor choice.
        def frame(frame_idx, conf):
            return {"frame_idx": frame_idx, "lines": [
                raw("BIỂN", 100, 100, 400, 140, conf=conf),
                raw("ĐÔNG", 100, 160, 400, 200, conf=conf),
            ]}

        segments = group_segments(
            "L21_V001", [frame(10, 0.40), frame(20, 0.99), frame(30, 0.50)]
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].anchor_frame, 20)

    def test_max_frame_gap_breaks_a_run(self):
        records = [
            {"frame_idx": 10, "lines": [raw("MỘT", 100, 100, 400, 140)]},
            {"frame_idx": 9000, "lines": [raw("MỘT", 100, 100, 400, 140)]},
        ]
        self.assertEqual(len(group_segments("L21_V001", records, max_frame_gap=500)), 2)

    def test_records_are_sorted_before_grouping(self):
        records = [
            {"frame_idx": 30, "lines": [raw("HAI", 100, 100, 400, 140)]},
            {"frame_idx": 10, "lines": [raw("MỘT", 100, 100, 400, 140)]},
        ]
        segments = group_segments("L21_V001", records)
        self.assertEqual([s.text for s in segments], ["MỘT", "HAI"])

    def test_segment_id_is_stable_and_sortable(self):
        records = [{"frame_idx": 1817, "lines": [raw("x", 0, 0, 10, 10)]}]
        self.assertEqual(group_segments("L30_V078", records)[0].segment_id, "L30_V078:001817")

    def test_segment_text_uses_reading_order(self):
        records = [{"frame_idx": 5576, "lines": [
            {"text": l.text, "box": [list(p) for p in l.box], "conf": l.conf}
            for l in PILLAR_LINES
        ]}]
        segment = group_segments("L27_V010", records)[0]
        self.assertIn("HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA", segment.text)

    def test_empty_input(self):
        self.assertEqual(group_segments("L21_V001", []), [])


if __name__ == "__main__":
    unittest.main()


class DeskewTests(unittest.TestCase):
    """Tilted pages must be ordered in their own frame, not the camera's."""

    @staticmethod
    def _tilted(text, x, y, w, h, degrees, cx=400.0, cy=300.0):
        import math
        r = math.radians(degrees)
        cos, sin = math.cos(r), math.sin(r)
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return Line(text=text, box=tuple(
            (cx + (px - cx) * cos - (py - cy) * sin,
             cy + (px - cx) * sin + (py - cy) * cos)
            for px, py in corners), conf=0.9)

    def test_upright_page_reports_no_skew(self):
        from scripts.ocr.postprocess import deskew_angle
        self.assertAlmostEqual(deskew_angle(PILLAR_LINES_MEASURED), 0.0, places=6)

    def test_detects_page_skew(self):
        from scripts.ocr.postprocess import deskew_angle
        lines = [self._tilted(f"line{i}", 200, 200 + 40 * i, 300, 30, -12)
                 for i in range(5)]
        self.assertAlmostEqual(deskew_angle(lines), -12.0, places=4)

    def test_a_few_odd_angles_do_not_steer_the_median(self):
        from scripts.ocr.postprocess import deskew_angle
        lines = [self._tilted(f"l{i}", 200, 200 + 40 * i, 300, 30, -12) for i in range(6)]
        lines.append(self._tilted("logo", 20, 20, 100, 30, 0))   # upright overlay
        self.assertAlmostEqual(deskew_angle(lines), -12.0, places=4)

    def test_tilted_document_reads_top_to_bottom(self):
        # Without deskewing, the rotated bounding boxes overlap in y and the
        # order collapses; this is the recipe-card failure in miniature.
        lines = [self._tilted(f"line{i}", 150, 150 + 45 * i, 420, 32, -14)
                 for i in range(7)]
        text = render_text(reading_order(lines))
        self.assertEqual(text, " ".join(f"line{i}" for i in range(7)))

    @staticmethod
    def _two_block_page():
        """A tilted page: short indented ingredient block above a wide method block."""
        lines = [DeskewTests._tilted("title", 260, 120, 180, 26, -13)]
        lines += [DeskewTests._tilted(f"ing{i}", 180, 175 + 34 * i, 250, 24, -13)
                  for i in range(3)]
        lines += [DeskewTests._tilted(f"step{i}", 180, 300 + 34 * i, 520, 24, -13)
                  for i in range(3)]
        return lines

    def test_tilted_two_block_page_keeps_blocks_in_order(self):
        """The recipe-card failure proper.

        Rotation makes the two blocks' axis-aligned bounding boxes overlap in y,
        so without deskewing they interleave — which is how the ingredient list
        ended up printed after the method section.
        """
        text = render_text(reading_order(self._two_block_page()))
        self.assertEqual(text, "title ing0 ing1 ing2 step0 step1 step2")

    def test_naive_order_would_have_failed_on_that_page(self):
        naive = sorted(self._two_block_page(), key=lambda l: (l.bbox[1], l.bbox[0]))
        self.assertNotEqual(
            render_text(naive), "title ing0 ing1 ing2 step0 step1 step2"
        )

    def test_small_skew_is_ignored(self):
        # A 1-degree wobble must not trigger a rotation pass.
        lines = [self._tilted(f"line{i}", 150, 150 + 45 * i, 420, 32, -1)
                 for i in range(5)]
        text = render_text(reading_order(lines))
        self.assertEqual(text, " ".join(f"line{i}" for i in range(5)))

    def test_returns_the_caller_s_own_line_objects(self):
        lines = [self._tilted(f"line{i}", 150, 150 + 45 * i, 420, 32, -14)
                 for i in range(5)]
        ordered = reading_order(lines)
        self.assertEqual({id(l) for l in ordered}, {id(l) for l in lines})


class TextBlockTests(unittest.TestCase):
    """Cropping detections into blocks for the second reader."""

    def _image(self):
        import numpy as np
        return np.zeros((720, 1280, 3), dtype="uint8")

    def _boxes(self, *rects):
        return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in rects]

    def test_neighbouring_lines_form_one_block(self):
        from scripts.ocr.reader import text_blocks
        # The signboard: red caps and the tagline directly beneath them.
        blocks = text_blocks(self._image(),
                             self._boxes((446, 477, 704, 511), (459, 508, 696, 526)))
        self.assertEqual(len(blocks), 1)

    def test_distant_regions_stay_separate(self):
        from scripts.ocr.reader import text_blocks
        blocks = text_blocks(self._image(),
                             self._boxes((446, 477, 704, 511), (100, 60, 300, 90)))
        self.assertEqual(len(blocks), 2)

    def test_block_covers_every_member_with_margin(self):
        from scripts.ocr.reader import text_blocks
        (x0, y0, x1, y1), _ = text_blocks(
            self._image(), self._boxes((446, 477, 704, 511), (459, 508, 696, 526)))[0]
        self.assertLessEqual(x0, 446)
        self.assertLessEqual(y0, 477)
        self.assertGreaterEqual(x1, 704)
        self.assertGreaterEqual(y1, 526)

    def test_small_text_is_upscaled(self):
        from scripts.ocr.reader import text_blocks
        (_, crop), = text_blocks(self._image(), self._boxes((400, 300, 500, 312)))
        self.assertGreater(crop.shape[0], 12)

    def test_crop_stays_inside_the_frame(self):
        from scripts.ocr.reader import text_blocks
        (x0, y0, x1, y1), _ = text_blocks(self._image(), self._boxes((0, 0, 60, 20)))[0]
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1280)
        self.assertLessEqual(y1, 720)

    def test_no_boxes(self):
        from scripts.ocr.reader import text_blocks
        self.assertEqual(text_blocks(self._image(), []), [])


class TileImageTests(unittest.TestCase):
    def test_wide_crop_gets_a_wide_grid(self):
        from PIL import Image
        from scripts.ocr.reader import tile_image
        tiles = tile_image(Image.new("RGB", (1200, 200)), max_tiles=6)
        self.assertGreater(len(tiles), 1)

    def test_square_crop_needs_one_tile_and_no_thumbnail(self):
        from PIL import Image
        from scripts.ocr.reader import tile_image
        self.assertEqual(len(tile_image(Image.new("RGB", (448, 448)), max_tiles=6)), 1)

    def test_tiles_are_448_square(self):
        from PIL import Image
        from scripts.ocr.reader import tile_image
        for tile in tile_image(Image.new("RGB", (1200, 200)), max_tiles=6):
            self.assertEqual(tile.size, (448, 448))

    def test_respects_the_tile_budget(self):
        from PIL import Image
        from scripts.ocr.reader import tile_image
        # Grid tiles plus at most one thumbnail.
        tiles = tile_image(Image.new("RGB", (1600, 120)), max_tiles=4)
        self.assertLessEqual(len(tiles), 5)
