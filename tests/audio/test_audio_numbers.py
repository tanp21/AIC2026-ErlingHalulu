import unittest

from scripts.audio.numbers import format_number, parse_number


def parse(text: str):
    return parse_number(text.split())


class ParseNumberTests(unittest.TestCase):
    def test_single_digits(self):
        self.assertEqual(parse("không"), 0)
        self.assertEqual(parse("một"), 1)
        self.assertEqual(parse("chín"), 9)

    def test_teens_use_muoi(self):
        self.assertEqual(parse("mười"), 10)
        self.assertEqual(parse("mười lăm"), 15)
        self.assertEqual(parse("mười tám"), 18)

    def test_tens_use_muoi_with_a_leading_digit(self):
        self.assertEqual(parse("hai mươi"), 20)
        self.assertEqual(parse("hai mươi mốt"), 21)
        self.assertEqual(parse("hai mươi tư"), 24)
        self.assertEqual(parse("năm mươi lăm"), 55)

    def test_chuc_is_a_spoken_alternative_to_muoi(self):
        self.assertEqual(parse("hai chục"), 20)
        self.assertEqual(parse("ba chục"), 30)
        self.assertEqual(parse("năm chục"), 50)

    def test_a_quantified_dozen_carries_no_number(self):
        # "mấy chục" and "vài chục" are "a few dozen"; there is nothing to write.
        self.assertIsNone(parse("mấy chục"))
        self.assertIsNone(parse("vài chục"))

    def test_colloquial_tens_drop_the_muoi(self):
        # "ba mốt" is how a presenter says "ba mươi mốt".
        self.assertEqual(parse("ba mốt"), 31)
        self.assertEqual(parse("bốn lăm"), 45)

    def test_hundreds(self):
        self.assertEqual(parse("một trăm"), 100)
        self.assertEqual(parse("chín trăm bốn mươi lăm"), 945)

    def test_filler_marks_an_empty_tens_place(self):
        self.assertEqual(parse("một trăm linh năm"), 105)
        self.assertEqual(parse("một trăm lẻ tám"), 108)

    def test_a_bare_digit_after_hundreds_is_the_tens_place(self):
        # Found in the corpus as "từ tám mươi lăm đến một trăm hai", which is
        # 85 to 120. Reading it as 102 would be a silent corruption.
        self.assertEqual(parse("một trăm hai"), 120)
        self.assertEqual(parse("ba trăm tư"), 340)

    def test_the_filler_is_what_distinguishes_120_from_102(self):
        self.assertEqual(parse("một trăm hai"), 120)
        self.assertEqual(parse("một trăm linh hai"), 102)

    def test_the_same_shorthand_above_a_thousand_is_refused_not_guessed(self):
        # "một nghìn hai" is 1200 in speech, but 1002 has no filler form to tell
        # them apart, so the span is left as words.
        self.assertIsNone(parse("một nghìn hai"))

    def test_years(self):
        self.assertEqual(parse("một ngàn chín trăm bốn mươi lăm"), 1945)
        self.assertEqual(parse("hai ngàn không trăm hai mươi tư"), 2024)
        self.assertEqual(parse("hai nghìn không trăm hai mươi tư"), 2024)

    def test_a_bare_scale_word_means_one_of_it(self):
        self.assertEqual(parse("ngàn chín trăm"), 1900)

    def test_millions_and_billions(self):
        self.assertEqual(parse("một trăm triệu"), 100_000_000)
        self.assertEqual(parse("hai tỷ"), 2_000_000_000)
        self.assertEqual(parse("mười lăm triệu"), 15_000_000)

    def test_scales_compose(self):
        self.assertEqual(parse("hai triệu ba trăm ngàn"), 2_300_000)

    def test_unparseable_runs_return_none(self):
        self.assertIsNone(parse("con cá heo"))
        self.assertIsNone(parse("trăm"))
        self.assertIsNone(parse("mươi"))
        self.assertIsNone(parse(""))

    def test_a_stray_word_inside_a_run_rejects_the_whole_run(self):
        # Better to leave the wording alone than to invent a number from it.
        self.assertIsNone(parse("hai mươi con tư"))


class FormatNumberTests(unittest.TestCase):
    def test_ordinary_values_become_plain_digits(self):
        self.assertEqual(format_number(2024), "2024")
        self.assertEqual(format_number(1945), "1945")
        self.assertEqual(format_number(31), "31")
        self.assertEqual(format_number(0), "0")

    def test_round_millions_and_billions_keep_their_scale_word(self):
        # "100 triệu đồng" is what a person writes; "100000000" is not.
        self.assertEqual(format_number(100_000_000), "100 triệu")
        self.assertEqual(format_number(2_000_000_000), "2 tỷ")

    def test_values_that_are_not_round_stay_as_digits(self):
        self.assertEqual(format_number(2_300_000), "2300000")
        self.assertEqual(format_number(1_500_000_000), "1500 triệu")


if __name__ == "__main__":
    unittest.main()
