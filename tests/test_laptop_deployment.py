"""The laptop media host: a switch instead of a tailnet (docs/laptop-media-host.md).

These are repository invariants, in the style of `test_frontend_serving.py`: the pieces
that make that deployment safe are spread across a compose override, a second HAProxy
config and the env template, and the failure mode of getting one wrong is silent. A
`haproxy.cfg` mounted by mistake looks like a working app that 403s the whole team; a
dropped `http-request auth` looks like a working app that trusts anyone on the LAN.
"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.media import check_media_host, make_lan_users

ROOT = Path(__file__).resolve().parents[1]

MEDIA_COMPOSE = ROOT / "compose.media.yaml"
LAN_HAPROXY = ROOT / "haproxy.lan.cfg"


def directives(path: Path) -> str:
    """A config file's directives, without its prose.

    The comments in `haproxy.lan.cfg` exist to explain what the Tailscale config does
    *differently*, so they name the very things these tests assert are absent.
    """
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("#")
    )


class MediaComposeTests(unittest.TestCase):
    def test_the_override_takes_out_what_a_media_host_has_no_use_for(self):
        text = MEDIA_COMPOSE.read_text()

        # `!reset`/`!override` are what make this a diff. A plain value on a list field
        # would *merge*, so the model mounts and the Elasticsearch dependency would come
        # straight back and the laptop would fail to start on a missing `${VAR:?}`.
        self.assertIn("depends_on: !reset null", text)
        self.assertIn("gpus: !reset null", text)
        self.assertIn('profiles: ["elasticsearch"]', text)
        self.assertIn("volumes: !override", text)

    def test_the_override_does_not_mount_a_model_directory(self):
        # It must not name the mounts at all: the `${VAR:?}` guards in compose.yaml would
        # otherwise have to be satisfied with a real Hugging Face cache on a laptop that
        # has never loaded a model.
        self.assertNotIn("/models/hub", MEDIA_COMPOSE.read_text())

    def test_the_ingress_is_pinned_to_one_address_and_authenticates(self):
        text = MEDIA_COMPOSE.read_text()

        self.assertIn("AIC_INGRESS_BIND:?", text)
        self.assertIn("haproxy.lan.cfg", text)
        self.assertIn("haproxy-users.cfg", text)
        # The password list is a second config file, not an `include`d one: without this
        # `-f`, HAProxy exits at start-up on the undefined `aic_users` userlist.
        self.assertIn("/usr/local/etc/haproxy/users.cfg", text)

    def test_the_env_template_is_tracked(self):
        # `.gitignore` excludes `.env.*` wholesale; the laptop is bootstrapped from this
        # file, so an un-excluded template would be missing from a fresh clone.
        ignored = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("!.env.laptop.example", ignored)
        self.assertTrue((ROOT / ".env.laptop.example").exists())


class LanIngressTests(unittest.TestCase):
    def test_a_request_must_authenticate_and_cannot_name_itself(self):
        text = directives(LAN_HAPROXY)

        self.assertIn("acl authenticated http_auth(aic_users)", text)
        self.assertIn("acl health path -i /healthz", text)
        self.assertIn("http-request auth realm AIC if !health !authenticated", text)
        # Order matters: the client's header has to be gone before the authenticated one
        # is set, or a teammate could present `X-AIC-User: <the admin>` and be believed.
        delete = text.index("http-request del-header X-AIC-User")
        stamp = text.index("http-request set-header X-AIC-User %[http_auth_user]")
        self.assertLess(delete, stamp)

    def test_it_does_not_fall_back_to_a_trusted_identity(self):
        # The whole difference from `haproxy.cfg`. That file believes
        # `Tailscale-User-Login` and stamps the *admin* for a local request without one --
        # a trust decision that only holds when tailscaled is what reached loopback.
        text = directives(LAN_HAPROXY)

        self.assertNotIn("Tailscale-User-Login", text)
        self.assertNotIn("AIC_ADMIN_USER}", text.split("acl admin")[0])
        self.assertNotIn("acl local_proxy", text)

    def test_the_two_ingress_configs_still_share_the_video_limit(self):
        # Scoped to the frontend's two backends: teammates share one uplink to the laptop,
        # and `/video/` is the only route that moves hundreds of megabytes per request.
        for path in (LAN_HAPROXY, ROOT / "haproxy.cfg"):
            text = directives(path)
            self.assertIn("use_backend teammate_video if video", text)
            self.assertIn("bwlim-out teammate_video_limit limit 20m", text)


class LanUserListTests(unittest.TestCase):
    def test_usernames_come_from_the_allowlist_and_the_admin_is_marked(self):
        users = ["owner@example.com", "a@example.com", "b@example.com", "c@example.com", "d@example.com"]
        text = make_lan_users.build_userlist(
            users, {user: "password123" for user in users}, admin="owner@example.com"
        )

        self.assertIn("userlist aic_users", text)
        lines = [line for line in text.splitlines() if line.strip().startswith("user ")]
        self.assertEqual(len(lines), 5)
        for user, line in zip(users, lines):
            self.assertTrue(line.startswith(f"  user {user} "), line)
        self.assertIn("admin", [line for line in lines if "owner@" in line][0])

    def test_every_line_is_a_password_directive_haproxy_accepts(self):
        text = make_lan_users.build_userlist(
            ["owner@example.com"], {"owner@example.com": "password123"}, "owner@example.com"
        )
        crypted = text.splitlines()[-1].split()[2]

        self.assertIn(crypted, ("password", "insecure-password"))

    def test_a_username_that_would_break_the_file_is_refused(self):
        # A space splits the line, a `#` comments out the rest of it, and `$` opens the
        # crypt field. All three are silent corruption rather than an error from HAProxy.
        for user in ("two words", "hash#name", "dollar$name"):
            with self.subTest(user=user):
                with self.assertRaises(SystemExit):
                    make_lan_users._check_username(user)

    def test_env_parsing_ignores_comments_and_unquotes_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.secure"
            path.write_text(
                "# a comment\n"
                "\n"
                'AIC_ALLOWED_USERS="a@example.com,b@example.com"\n'
                "AIC_ADMIN_USER=plain@example.com  \n"
            )

            env = make_lan_users.read_env(path)

        self.assertEqual(env["AIC_ALLOWED_USERS"], "a@example.com,b@example.com")
        self.assertEqual(env["AIC_ADMIN_USER"], "plain@example.com")


class FrameSourceSignatureTests(unittest.TestCase):
    """The DRES precondition, checked the way `backend/dres.py::_source` checks it."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = self.root / "data"
        self.index_dir = self.root / "indexes"
        (self.index_dir / "frame_maps").mkdir(parents=True)

    def _manifest(self, video_id="L21_V001", **overrides):
        path = self.data_dir / "2026" / "videos" / "L21" / f"{video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        signature = {
            # Recorded where the corpus was built; read where it now lives. Only the last
            # four components have to agree.
            "path": f"/home/someone/aic/data/2026/videos/L21/{video_id}.mp4",
            "size": path.stat().st_size,
            "mtimeNs": path.stat().st_mtime_ns,
            **overrides,
        }
        (self.index_dir / "frame_maps" / "manifest.json").write_text(
            json.dumps({"version": 2, "videos": {video_id: signature}})
        )
        return path

    def test_a_copied_corpus_at_a_new_root_passes(self):
        self._manifest()

        healthy, lines = check_media_host.check_signatures(self.index_dir, self.data_dir)

        self.assertTrue(healthy, lines)

    def test_a_redownloaded_corpus_with_the_extraction_time_fails(self):
        self._manifest(mtimeNs=1)

        healthy, lines = check_media_host.check_signatures(self.index_dir, self.data_dir)
        report = "\n".join(lines)

        self.assertFalse(healthy)
        self.assertIn("L21_V001", report)
        # The point of failing is telling the reader what to run about it.
        self.assertIn("build_frame_maps", report)

    def test_a_video_whose_length_changed_fails_even_with_a_matching_mtime(self):
        # What an audio remux leaves behind: the same video timeline, a different file.
        # Nothing about the frame map is wrong, but DRES will not accept it, so neither
        # does this -- and `--resign-only` refuses it too, which is why the report says to
        # prefer `build_frame_maps` when ffprobe is available.
        path = self._manifest()
        mtime = path.stat().st_mtime_ns
        path.write_bytes(b"the same video with a corrected audio track")
        os.utime(path, ns=(mtime, mtime))

        healthy, _ = check_media_host.check_signatures(self.index_dir, self.data_dir)

        self.assertFalse(healthy)

    def test_a_video_that_is_not_there_is_not_a_pass(self):
        path = self._manifest()
        path.unlink()

        healthy, _ = check_media_host.check_signatures(self.index_dir, self.data_dir)

        self.assertFalse(healthy)

    def test_a_missing_manifest_is_reported_rather_than_assumed_fine(self):
        self._manifest()
        (self.index_dir / "frame_maps" / "manifest.json").unlink()

        healthy, lines = check_media_host.check_signatures(self.index_dir, self.data_dir)

        self.assertFalse(healthy)
        self.assertIn("missing", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
