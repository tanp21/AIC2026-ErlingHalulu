import unittest

from scripts.ocr.postcorrect import (
    CorrectionGuard,
    accept_correction,
    char_drift,
    word_retention,
)

# Abridged from the pillar frame: the draft the pipeline produces, and the
# correction that only fixes the one wrong tone mark.
DRAFT = "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA KIẾM BẠT KIÊN GIANG KHẤP QUỲ THẦN"
FIXED = "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN"


class WordRetentionTests(unittest.TestCase):
    def test_identical_text_fully_retained(self):
        self.assertEqual(word_retention(DRAFT, DRAFT), 1.0)

    def test_tone_mark_change_counts_as_retained(self):
        # Diacritic-blind by design: a tone-mark fix is the point of the exercise
        # and must not read as a lost word.
        self.assertEqual(word_retention(DRAFT, FIXED), 1.0)

    def test_dropped_words_reduce_retention(self):
        self.assertAlmostEqual(word_retention("a b c d", "a b"), 0.5)

    def test_empty_draft_is_fully_retained(self):
        self.assertEqual(word_retention("", "anything"), 1.0)


class CharDriftTests(unittest.TestCase):
    def test_tone_only_change_is_zero_drift(self):
        self.assertEqual(char_drift(DRAFT, FIXED), 0.0)

    def test_rewrite_drifts(self):
        self.assertGreater(char_drift(DRAFT, "hoàn toàn khác biệt không liên quan"), 0.5)

    def test_empty_draft(self):
        self.assertEqual(char_drift("", "x"), 0.0)


class AcceptCorrectionTests(unittest.TestCase):
    def test_accepts_a_single_tone_mark_repair(self):
        self.assertTrue(accept_correction(DRAFT, FIXED))

    def test_accepts_a_real_word_repair(self):
        # 'HIVS' -> 'HTV9' on the gauge frame: a genuine fix the VLM made.
        draft = "HIVS 06:39 20 18 16 14 12 10 8 cận trẻ em gây hoang mang dư luận"
        fixed = "HTV9 06:39 20 18 16 14 12 10 8 cận trẻ em gây hoang mang dư luận"
        self.assertTrue(accept_correction(draft, fixed))

    def test_rejects_an_unchanged_result(self):
        self.assertFalse(accept_correction(DRAFT, DRAFT))

    def test_rejects_empty_output(self):
        self.assertFalse(accept_correction(DRAFT, ""))
        self.assertFalse(accept_correction(DRAFT, "   "))

    def test_rejects_content_loss(self):
        # The observed failure mode on the densest frame: the model summarised
        # instead of correcting, and word recall fell from 0.756 to 0.513.
        truncated = " ".join(DRAFT.split()[:7])
        self.assertFalse(accept_correction(DRAFT, truncated))

    def test_rejects_added_commentary(self):
        chatty = (
            "Here is the corrected transcription of the Vietnamese text in the "
            "image, with all tone marks fixed as requested: " + FIXED
        )
        self.assertFalse(accept_correction(DRAFT, chatty))

    def test_rejects_wholesale_rewrite_of_similar_length(self):
        rewrite = " ".join(["KHÁC"] * len(DRAFT.split()))
        self.assertFalse(accept_correction(DRAFT, rewrite))

    def test_guard_thresholds_are_configurable(self):
        truncated = " ".join(DRAFT.split()[:7])
        permissive = CorrectionGuard(
            max_word_delta=1.0, min_word_retention=0.0, max_char_drift=1.0
        )
        self.assertTrue(accept_correction(DRAFT, truncated, permissive))

    def test_empty_draft_accepts_nothing(self):
        self.assertFalse(accept_correction("", "anything"))



class RestoreDeletionsTests(unittest.TestCase):
    """The corrector may fix characters; it may not quietly drop content."""

    def test_restores_a_dropped_line(self):
        from scripts.ocr.postcorrect import restore_deletions
        # Abridged from the recipe card: the corrector fixed 'dâm' to 'dăm' and
        # in the same pass deleted the card's closing line.
        draft = "Nạc dâm xay 200 gr bỏ cho sau cùng khi vừa tắt bếp"
        candidate = "Nạc dăm xay 200 gr"
        restored = restore_deletions(draft, candidate)
        self.assertEqual(restored, "Nạc dăm xay 200 gr bỏ cho sau cùng khi vừa tắt bếp")

    def test_keeps_the_corrections_it_made(self):
        from scripts.ocr.postcorrect import restore_deletions
        draft = "Nạc dâm xay 200 gr bỏ cho sau cùng khi vừa tắt bếp"
        candidate = "Nạc dăm xay 200 gr"
        self.assertIn("dăm", restore_deletions(draft, candidate))

    def test_leaves_a_short_deletion_alone(self):
        from scripts.ocr.postcorrect import restore_deletions
        # One or two words is more likely a genuine de-duplication.
        draft = "bột nêm nêm tiêu nước mắm"
        candidate = "bột nêm tiêu nước mắm"
        self.assertEqual(restore_deletions(draft, candidate), candidate)

    def test_a_tone_repair_is_not_a_deletion(self):
        from scripts.ocr.postcorrect import restore_deletions
        self.assertEqual(restore_deletions(DRAFT, FIXED), FIXED)

    def test_unchanged_when_nothing_was_dropped(self):
        from scripts.ocr.postcorrect import restore_deletions
        draft = "HIVS 06:39 cận trẻ em"
        candidate = "HTV9 06:39 cận trẻ em"
        self.assertEqual(restore_deletions(draft, candidate), candidate)

    def test_lost_spans_reports_an_outright_deletion(self):
        from scripts.ocr.postcorrect import lost_spans
        self.assertEqual(lost_spans("a b c d e f g h", "a b h"), [(2, 7)])

    def test_lost_spans_catches_loss_disguised_as_a_replacement(self):
        from scripts.ocr.postcorrect import lost_spans
        # The real recipe-card failure: 13 draft words became 2, and because
        # 'tusitretv' partially matches 'tusitre', the diff reports one merged
        # 'replace' rather than a replace plus a delete.
        draft = "thôi tusitretv thu cho sau cùng khi vừa tắt bếp mùi tiêu sẽ thơm hơn"
        candidate = "thôi tusitre TV"
        self.assertTrue(lost_spans(draft, candidate))

    def test_restores_loss_disguised_as_a_replacement(self):
        from scripts.ocr.postcorrect import restore_deletions
        draft = "thôi tusitretv thu cho sau cùng khi vừa tắt bếp mùi tiêu sẽ thơm hơn"
        candidate = "thôi tusitre TV"
        restored = restore_deletions(draft, candidate)
        self.assertIn("mùi tiêu sẽ thơm hơn", restored)

    def test_a_same_length_replacement_is_not_loss(self):
        from scripts.ocr.postcorrect import lost_spans, restore_deletions
        draft = "cho dầu ân đợi chảo nóng"
        candidate = "cho dầu ăn đợi chảo nóng"
        self.assertEqual(lost_spans(draft, candidate), [])
        self.assertEqual(restore_deletions(draft, candidate), candidate)

    def test_empty_inputs(self):
        from scripts.ocr.postcorrect import restore_deletions
        self.assertEqual(restore_deletions("", "x"), "x")
        self.assertEqual(restore_deletions("x", ""), "")

    def test_a_truncation_survives_restore_and_is_then_a_no_op_correction(self):
        # Restoring everything the corrector dropped can make the candidate
        # identical to the draft, which accept_correction must then reject
        # rather than treat as an improvement.
        from scripts.ocr.postcorrect import restore_deletions
        truncated = " ".join(DRAFT.split()[:7])
        restored = restore_deletions(DRAFT, truncated)
        self.assertEqual(restored, DRAFT)
        self.assertFalse(accept_correction(DRAFT, restored))


class ReconcileViewsTests(unittest.TestCase):
    """Two views of the recipe card, arbitrated by the recogniser's own draft."""

    # The recogniser read 'già' correctly and 'chến' as a fold-match for 'chén'.
    DRAFT = "đợi chảo nóng già cho hành tây xào tái đổ ra chến"

    def test_draft_decides_where_the_views_disagree(self):
        from scripts.ocr.postcorrect import reconcile_views
        original = "đợi chảo nóng già cho hành tây xào tái đổ ra chảo"
        deskewed = "đợi chảo nóng giã cho hành tây xào tái đổ ra chén"
        merged = reconcile_views(self.DRAFT, original, deskewed)
        # 'già' from the original, 'chén' from the deskewed: each is the reading
        # the draft corroborates, and neither view has both.
        self.assertIn("chảo nóng già", merged)
        self.assertIn("đổ ra chén", merged)

    def test_agreement_is_kept_as_is(self):
        from scripts.ocr.postcorrect import reconcile_views
        same = "đợi chảo nóng già"
        self.assertEqual(reconcile_views(self.DRAFT, same, same), same)

    def test_secondary_wins_when_the_draft_supports_neither(self):
        from scripts.ocr.postcorrect import reconcile_views
        merged = reconcile_views("xxx", "hòa", "hỏa")
        self.assertEqual(merged, "hỏa")

    def test_content_present_in_only_one_view_is_kept(self):
        from scripts.ocr.postcorrect import reconcile_views
        merged = reconcile_views(self.DRAFT, "đợi chảo nóng già cho hành", "đợi chảo nóng già")
        self.assertIn("cho hành", merged)

    def test_empty_views(self):
        from scripts.ocr.postcorrect import reconcile_views
        self.assertEqual(reconcile_views("d", "", "b"), "b")
        self.assertEqual(reconcile_views("d", "a", ""), "a")


class MergeByConfidenceTests(unittest.TestCase):
    """The pillar-frame trade: keep QUỶ's fix without losing KHẤP."""

    # Confidences as the recogniser actually reported them for that frame.
    WORDS = [("KIẾM", 0.89), ("BẠT", 0.84), ("KIÊN", 0.79), ("GIANG", 0.89),
             ("KHẤP", 0.86), ("QUỲ", 0.69), ("THẦN", 0.89)]

    def test_takes_the_fix_only_where_the_recogniser_was_unsure(self):
        from scripts.ocr.postcorrect import merge_by_confidence
        # The model repaired QUỲ->QUỶ but spoiled KHẤP->KHẨP in the same pass.
        corrected = "KIẾM BẠT KIÊN GIANG KHẨP QUỶ THẦN"
        merged = merge_by_confidence(self.WORDS, corrected, max_conf=0.80)
        self.assertEqual(merged, "KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN")

    def test_returns_none_when_word_counts_disagree(self):
        from scripts.ocr.postcorrect import merge_by_confidence
        # Without alignment a substitution is indistinguishable from an
        # insertion, so the caller must fall back to the draft.
        self.assertIsNone(merge_by_confidence(self.WORDS, "KIẾM BẠT KIÊN"))

    def test_high_confidence_draft_always_wins(self):
        from scripts.ocr.postcorrect import merge_by_confidence
        words = [("ĐỊA", 0.98)]
        self.assertEqual(merge_by_confidence(words, "KHÁC", max_conf=0.80), "ĐỊA")

    def test_low_confidence_defers_to_the_model(self):
        from scripts.ocr.postcorrect import merge_by_confidence
        words = [("QUỲ", 0.30)]
        self.assertEqual(merge_by_confidence(words, "QUỶ", max_conf=0.80), "QUỶ")

    def test_empty_input(self):
        from scripts.ocr.postcorrect import merge_by_confidence
        self.assertEqual(merge_by_confidence([], ""), "")


if __name__ == "__main__":
    unittest.main()


class AbsorbReaderTests(unittest.TestCase):
    """An independent reader may add coverage; it may never overwrite the spine."""

    # Verbatim from the signboard frame.
    SPINE = "KHU DU LỊCH QUỐC GIA MỘC CHÂU"
    READER = ("KHU DU LỊCH QUỐC GIA Mai Chiu Điểm đến thiên nhiên "
              "khu vực hành hương địa phương")

    def test_adds_text_the_spine_never_saw(self):
        from scripts.ocr.postcorrect import absorb_reader
        merged = absorb_reader(self.SPINE, self.READER)
        self.assertIn("Điểm đến thiên nhiên", merged)

    def test_never_overwrites_the_spine(self):
        from scripts.ocr.postcorrect import absorb_reader
        merged = absorb_reader(self.SPINE, self.READER)
        # The reader's 'Mai Chiu' competes for the place name's slot and loses;
        # the spine's reading came from the grounded corrector.
        self.assertIn("MỘC CHÂU", merged)
        self.assertNotIn("Mai Chiu", merged)

    def test_agreement_changes_nothing(self):
        from scripts.ocr.postcorrect import absorb_reader
        same = "20 18 16 14 12 10 8"
        self.assertEqual(absorb_reader(same, same), same)

    def test_a_reader_that_saw_less_adds_nothing(self):
        from scripts.ocr.postcorrect import absorb_reader
        self.assertEqual(absorb_reader(self.SPINE, "KHU DU LỊCH"), self.SPINE)

    def test_a_small_disagreement_is_not_treated_as_coverage(self):
        from scripts.ocr.postcorrect import absorb_reader
        # One word against one word is an opinion, not new text.
        self.assertEqual(absorb_reader("KIẾM BẠT THẦN", "KIẾM BÁT THẦN"),
                         "KIẾM BẠT THẦN")

    def test_an_unambiguous_insertion_is_always_coverage(self):
        from scripts.ocr.postcorrect import absorb_reader
        # Nothing competes for these words' slot, so min_gain does not apply.
        spine, reader = "A B", "A X Y Z B"
        self.assertEqual(absorb_reader(spine, reader, min_gain=9), "A X Y Z B")

    def test_min_gain_controls_how_much_growth_counts_as_coverage(self):
        from scripts.ocr.postcorrect import absorb_reader
        # Here the growth is entangled with a disagreement: the reader replaces
        # 'Q' with four words. Only the excess beyond the spine's slot is
        # coverage, and min_gain decides whether it is taken at all.
        spine, reader = "A Q B", "A X Y Z W B"
        # The spine's slot is one word against the reader's four, so three words
        # of excess are read off the tail of the reader's span.
        self.assertEqual(absorb_reader(spine, reader, min_gain=3), "A Q Y Z W B")
        self.assertEqual(absorb_reader(spine, reader, min_gain=9), spine)

    def test_empty_operands(self):
        from scripts.ocr.postcorrect import absorb_reader
        self.assertEqual(absorb_reader(self.SPINE, ""), self.SPINE)
        self.assertEqual(absorb_reader("", "abc"), "abc")
