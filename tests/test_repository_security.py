import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_EXTRACTOR = ROOT / "scripts" / "extraction" / "extract_keyframes_remote.py"
COLAB_NOTEBOOK = ROOT / "notebooks" / "colab_extract_keyframes.ipynb"


class RepositorySecurityTests(unittest.TestCase):
    def test_remote_tools_do_not_embed_google_oauth_credentials(self):
        for path in (REMOTE_EXTRACTOR, COLAB_NOTEBOOK):
            source = path.read_text()
            with self.subTest(path=path.name):
                self.assertNotIn("client_id =", source)
                self.assertNotIn("client_secret =", source)
                self.assertNotIn('\\"refresh_token\\"', source)

    def test_tools_do_not_embed_owner_home_paths(self):
        paths = (
            ROOT / "compose.yaml",
            ROOT / "scripts" / "extraction" / "zip_keyframes.py",
            ROOT / "scripts" / "remote" / "upload_shots.sh",
        )
        for path in paths:
            with self.subTest(path=path.name):
                self.assertNotIn("/home/tanp21", path.read_text())


if __name__ == "__main__":
    unittest.main()
