import os
import unittest

import audio_search


GOLDEN_QUERY = (
    "Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. "
    "Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. "
    "Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu "
    "ánh sáng cực quang ở vùng cực"
)


@unittest.skipUnless(os.environ.get("RUN_AUDIO_INTEGRATION") == "1", "requires Qwen + Elasticsearch")
class AudioIntegrationTests(unittest.TestCase):
    def test_golden_private_spaceflight_query(self):
        hits = audio_search.search_audio(GOLDEN_QUERY, 10)

        rank = next(i for i, hit in enumerate(hits, 1) if hit.video_id == "L21_V015")
        hit = next(hit for hit in hits if hit.video_id == "L21_V015")
        self.assertLessEqual(rank, 3)
        self.assertLessEqual(abs(hit.anchor_sec - 857.47), 15)


if __name__ == "__main__":
    unittest.main()
