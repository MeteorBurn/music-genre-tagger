from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.playlists import parse_playlist


class PlaylistParsingTests(TestCase):
    def test_parses_utf8_bom_comments_relative_paths_and_file_uris(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            playlist_path = root / "seed.m3u8"
            relative_track = root / "relative" / "one.flac"
            uri_track = root / "uri track.mp3"
            relative_track.parent.mkdir()
            relative_track.touch()
            uri_track.touch()
            playlist_path.write_text(
                "#EXTM3U\n"
                "#EXTINF:123,Artist - Track\n"
                "relative/one.flac\n"
                f"{uri_track.as_uri()}\n",
                encoding="utf-8-sig",
            )

            entries = parse_playlist(playlist_path)

            self.assertEqual(entries, [relative_track.resolve(), uri_track.resolve()])

    def test_rejects_remote_playlist_urls(self) -> None:
        with TemporaryDirectory() as temp_dir:
            playlist_path = Path(temp_dir) / "remote.m3u"
            playlist_path.write_text(
                "https://example.invalid/audio.mp3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "remote URL"):
                parse_playlist(playlist_path)
