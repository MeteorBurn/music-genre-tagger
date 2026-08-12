import asyncio
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import httpx

from tools.maest522.annotation_api import create_app
from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS


def write_test_wav(path: Path, sample_value: int = 0) -> None:
    frame_count = math.ceil(16_000 * 0.1)
    sample = int(sample_value).to_bytes(2, "little", signed=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(sample * frame_count)


async def api_request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://annotation.local",
    ) as client:
        return await client.request(method, path, **kwargs)


class AnnotationApiTests(TestCase):
    def test_project_trusted_import_goals_and_single_label_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "annotations.db"
            audio_path = root / "audio" / "track.wav"
            write_test_wav(audio_path)
            playlist_path = root / "positive.m3u"
            playlist_path.write_text(str(audio_path) + "\n", encoding="utf-8")
            app = create_app(database_path)

            project_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    "/api/projects",
                    json={"name": "community-extension"},
                )
            )
            self.assertEqual(project_response.status_code, 201)
            project_id = project_response.json()["project_id"]
            preflight_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/trusted-playlists/preflight",
                    json={
                        "playlist_path": str(playlist_path),
                        "label": NEW_LABELS[0],
                        "state": "positive",
                    },
                )
            )
            self.assertEqual(preflight_response.status_code, 200)
            self.assertEqual(preflight_response.json()["new_count"], 1)
            commit_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/trusted-playlists/commit",
                    json={
                        "playlist_path": str(playlist_path),
                        "label": NEW_LABELS[0],
                        "state": "positive",
                        "expected_playlist_sha256": preflight_response.json()[
                            "playlist_sha256"
                        ],
                    },
                )
            )
            self.assertEqual(commit_response.status_code, 200)
            self.assertEqual(commit_response.json()["new_count"], 1)

            progress_response = asyncio.run(
                api_request(
                    app,
                    "GET",
                    f"/api/projects/{project_id}/confirmed-progress",
                )
            )
            self.assertEqual(progress_response.status_code, 200)
            self.assertEqual(progress_response.json()[0]["positive_count"], 1)
            self.assertEqual(progress_response.json()[0]["positive_target"], 1000)
            goal_response = asyncio.run(
                api_request(
                    app,
                    "PATCH",
                    f"/api/projects/{project_id}/goals/{NEW_LABELS[0]}",
                    json={"positive_target": 1200, "negative_target": 900},
                )
            )
            self.assertEqual(goal_response.status_code, 200)
            self.assertEqual(goal_response.json()["positive_target"], 1200)

            source_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/sources",
                    json={
                        "source_path": str(audio_path.parent),
                        "suggested_label": NEW_LABELS[1],
                        "candidate_role": "positive_candidate",
                    },
                )
            )
            self.assertEqual(source_response.status_code, 200)

            store = AnnotationStore(database_path)
            now = datetime.now(timezone.utc).isoformat()
            with store.connection() as connection:
                connection.execute(
                    "UPDATE tracks SET group_id = ?, split = 'train'",
                    ("b" * 64,),
                )
                connection.execute(
                    "UPDATE projects SET split_frozen_at = ? WHERE id = ?",
                    (now, project_id),
                )

            round_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/rounds",
                    json={
                        "label": NEW_LABELS[1],
                        "round_number": 1,
                        "split": "train",
                    },
                )
            )
            self.assertEqual(round_response.status_code, 201)
            next_response = asyncio.run(
                api_request(
                    app,
                    "GET",
                    f"/api/projects/{project_id}/queue/next",
                )
            )
            self.assertEqual(next_response.status_code, 200)
            queue_item = next_response.json()
            self.assertNotIn("path", queue_item)
            self.assertNotIn("acquisition_score", queue_item)
            self.assertEqual(queue_item["filename"], "track.wav")
            self.assertEqual(queue_item["active_label"], NEW_LABELS[1])
            self.assertEqual(queue_item["state"], "unreviewed")
            queue_item_id = queue_item["queue_item_id"]

            annotation_response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/queue/{queue_item_id}/annotations",
                    json={
                        "label": NEW_LABELS[1],
                        "state": "uncertain",
                        "note": "reviewed by ear",
                    },
                )
            )
            self.assertEqual(annotation_response.status_code, 200)
            queue_progress_response = asyncio.run(
                api_request(
                    app,
                    "GET",
                    f"/api/projects/{project_id}/progress",
                )
            )
            self.assertEqual(
                queue_progress_response.json(),
                {"total": 1, "completed": 1, "remaining": 0},
            )

            route_paths = {route.path for route in app.routes}
            self.assertIn(f"/api/projects/{{project_id}}/fingerprints", route_paths)
            self.assertIn(f"/api/projects/{{project_id}}/splits/freeze", route_paths)
            self.assertIn(f"/api/projects/{{project_id}}/audio/{{track_id}}", route_paths)
            self.assertIn(
                f"/api/projects/{{project_id}}/trusted-playlists/preflight",
                route_paths,
            )

    def test_imports_uploaded_m3u_text_relative_to_selected_base_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_test_wav(root / "one.wav", sample_value=1)
            write_test_wav(root / "two.wav", sample_value=2)
            app = create_app(root / "annotations.db")
            project_id = asyncio.run(
                api_request(
                    app,
                    "POST",
                    "/api/projects",
                    json={"name": "playlist-upload"},
                )
            ).json()["project_id"]

            response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/sources",
                    json={
                        "playlist_name": "seed.m3u",
                        "playlist_text": "#EXTM3U\none.wav\ntwo.wav\n",
                        "base_directory": str(root),
                        "suggested_label": NEW_LABELS[1],
                        "candidate_role": "positive_candidate",
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["discovered"], 2)
            self.assertEqual(response.json()["imported_new"], 2)

    def test_rejects_blank_server_source_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(root / "annotations.db")
            project_id = asyncio.run(
                api_request(
                    app,
                    "POST",
                    "/api/projects",
                    json={"name": "blank-source"},
                )
            ).json()["project_id"]

            response = asyncio.run(
                api_request(
                    app,
                    "POST",
                    f"/api/projects/{project_id}/sources",
                    json={
                        "source_path": "   ",
                        "suggested_label": NEW_LABELS[0],
                        "candidate_role": "positive_candidate",
                    },
                )
            )

            self.assertEqual(response.status_code, 422)
