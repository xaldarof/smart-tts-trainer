import pytest

from app.services.youtube import downloader as yd
from app.services.youtube.urls import InvalidURL, parse_youtube_url


@pytest.mark.parametrize("url,kind,ident", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "video", "jNQXAC9IVRw"),
    ("youtube.com/watch?v=jNQXAC9IVRw&t=10s", "video", "jNQXAC9IVRw"),
    ("https://youtu.be/jNQXAC9IVRw?si=abc", "video", "jNQXAC9IVRw"),
    ("https://www.youtube.com/shorts/jNQXAC9IVRw", "video", "jNQXAC9IVRw"),
    ("https://m.youtube.com/watch?v=jNQXAC9IVRw&list=PL123", "video", "jNQXAC9IVRw"),
    ("https://www.youtube.com/playlist?list=PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI", "playlist",
     "PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI"),
    ("https://www.youtube.com/@SomeChannel", "channel", "@SomeChannel"),
    ("https://www.youtube.com/@SomeChannel/videos", "channel", "@SomeChannel"),
    ("https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw", "channel", "channel/UC_x5XG1OV2P6uZZ5FSM9Ttw"),
    ("https://www.youtube.com/c/Legacy/featured", "channel", "c/Legacy"),
])
def test_parse_valid(url, kind, ident):
    p = parse_youtube_url(url)
    assert p.kind == kind and p.ident == ident
    assert p.url.startswith("https://www.youtube.com/")


def test_channel_tab_normalized():
    assert parse_youtube_url("https://www.youtube.com/@x/featured").url.endswith("/@x/videos")
    assert parse_youtube_url("https://www.youtube.com/@x/streams").url.endswith("/@x/streams")


@pytest.mark.parametrize("url", ["", "https://vimeo.com/123", "https://www.youtube.com/watch?v=short",
                                 "https://youtu.be/", "https://www.youtube.com/results?search_query=x",
                                 "https://www.youtube.com/playlist?list=a b"])
def test_parse_invalid(url):
    with pytest.raises(InvalidURL):
        parse_youtube_url(url)


def _vi(i, dur=60.0):
    return yd.VideoInfo(f"vid{i:08d}xx", f"https://www.youtube.com/watch?v=vid{i:08d}xx", f"Video {i}", "Uploader",
                        dur, "20240101", 1000, "public")


def test_entry_to_info_and_size_estimate():
    info = yd._entry_to_info({"id": "abc", "title": "T", "duration": 100})
    assert info.filesize_estimate == 100 * 128_000 // 8
    assert yd._entry_to_info({"_type": "playlist", "id": "x"}) is None
    assert yd._entry_to_info({"id": "a", "filesize_approx": 5}).filesize_estimate == 5


def test_save_discovery_dedupes(db, project):
    p = parse_youtube_url("https://www.youtube.com/@x")
    _, new1 = yd.save_discovery(db, project, p, "x", [_vi(1), _vi(2)])
    _, new2 = yd.save_discovery(db, project, p, "x", [_vi(2), _vi(3)])
    assert (new1, new2) == (2, 1)
    assert db.scalar("SELECT COUNT(*) FROM videos WHERE project_id=?", (project,)) == 3
    assert db.scalar("SELECT COUNT(*) FROM sources") == 2


def test_download_requires_authorization(db, project, tmp_path):
    p = parse_youtube_url("https://youtu.be/jNQXAC9IVRw")
    yd.save_discovery(db, project, p, "t", [_vi(1)])
    vid = db.scalar("SELECT id FROM videos")
    with pytest.raises(PermissionError):
        yd.download_audio(db, vid, tmp_path, yd.JobContext())


def test_download_reuses_existing_file(db, project, tmp_path):
    from app.database.db import get_or_create_project

    f = tmp_path / "a.m4a"
    f.write_bytes(b"x")
    p = parse_youtube_url("https://youtu.be/jNQXAC9IVRw")
    yd.save_discovery(db, project, p, "t", [_vi(1)])
    db.execute("UPDATE videos SET download_status='downloaded', download_path=?, authorized=1", (str(f),))
    other = get_or_create_project(db, "other")
    yd.save_discovery(db, other, p, "t", [_vi(1)])
    vid2 = db.scalar("SELECT id FROM videos WHERE project_id=?", (other,))
    db.update("videos", vid2, authorized=1)
    assert yd.download_audio(db, vid2, tmp_path, yd.JobContext()) == f   # duplicate across projects, no network


def test_error_classification():
    assert not yd._is_retryable("ERROR: Private video. Sign in")
    assert yd._is_retryable("HTTP Error 503: Service Unavailable")
    assert yd._is_retryable("Read timed out")
    assert "restricted" in yd._explain("ERROR: This video is members-only")
    assert "bot" in yd._explain("Sign in to confirm you're not a bot")


def test_import_local_file(db, project, tmp_path):
    f = tmp_path / "talk.wav"
    f.write_bytes(b"RIFF")
    a = yd.import_local_file(db, project, f)
    b = yd.import_local_file(db, project, f)
    assert a == b
    row = db.q1("SELECT * FROM videos WHERE id=?", (a,))
    assert row["download_status"] == "downloaded" and row["authorized"] == 1
