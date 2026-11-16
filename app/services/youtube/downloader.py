"""yt-dlp based discovery and audio-only download.

Policy: this module only downloads publicly accessible content through yt-dlp's
normal extraction. It does not pass cookies, bypass logins/age gates, rotate
IPs, or attempt DRM circumvention. Restricted videos are reported as such.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from app.database.db import Database
from app.services.youtube.urls import ParsedURL, parse_youtube_url
from app.utils.jobs import Cancelled, JobContext

log = logging.getLogger(__name__)

RESTRICTED_MARKERS = (
    "private video", "sign in to confirm your age", "members-only", "join this channel", "video unavailable",
    "this video is not available", "copyright", "has been removed", "premieres in", "drm",
    "requires payment", "confirm you", "not a bot",
)


class DownloadError(RuntimeError):
    def __init__(self, msg: str, retryable: bool):
        super().__init__(msg)
        self.retryable = retryable


@dataclass
class VideoInfo:
    youtube_id: str
    url: str
    title: str
    uploader: str | None
    duration: float | None
    upload_date: str | None
    filesize_estimate: int | None
    availability: str | None


def _base_opts() -> dict:
    return {"quiet": True, "no_warnings": True, "noprogress": True, "socket_timeout": 30, "logger": _YtdlpLogger()}


class _YtdlpLogger:
    def debug(self, msg):
        if not msg.startswith("[debug] "):
            log.debug(msg)

    def info(self, msg):
        log.debug(msg)

    def warning(self, msg):
        log.info("yt-dlp: %s", msg)

    def error(self, msg):
        log.warning("yt-dlp: %s", msg)


def _estimate_audio_size(entry: dict) -> int | None:
    for k in ("filesize", "filesize_approx"):
        if entry.get(k):
            return int(entry[k])
    dur = entry.get("duration")
    # ~128 kbps opus/m4a audio when nothing better is known
    return int(dur * 128_000 / 8) if dur else None


def _entry_to_info(e: dict) -> VideoInfo | None:
    vid = e.get("id")
    if not vid or e.get("_type") == "playlist":
        return None
    return VideoInfo(
        youtube_id=vid,
        url=e.get("webpage_url") or e.get("url") or f"https://www.youtube.com/watch?v={vid}",
        title=e.get("title") or vid,
        uploader=e.get("uploader") or e.get("channel"),
        duration=float(e["duration"]) if e.get("duration") else None,
        upload_date=e.get("upload_date"),
        filesize_estimate=_estimate_audio_size(e),
        availability=e.get("availability"),
    )


def discover(url: str, ctx: JobContext, limit: int | None = None) -> tuple[ParsedURL, str, list[VideoInfo]]:
    """List videos behind a URL without downloading media. Returns (parsed, title, videos)."""
    import yt_dlp

    parsed = parse_youtube_url(url)
    opts = _base_opts() | {"extract_flat": "in_playlist", "skip_download": True, "ignoreerrors": True}
    if limit:
        opts["playlistend"] = int(limit)
    ctx.progress(0.05, f"Querying {parsed.kind}…")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(parsed.url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(_explain(str(e)), retryable=_is_retryable(str(e))) from e
    if not info:
        raise DownloadError("YouTube returned no information for this URL (deleted, private, or region-blocked).",
                            retryable=False)
    ctx.check()
    entries = info.get("entries") if info.get("_type") == "playlist" else [info]
    videos: list[VideoInfo] = []
    seen = set()
    for e in entries or []:
        if not e:
            continue
        # Channel root pages can nest tabs as playlists; flatten one level.
        if e.get("_type") == "playlist" and e.get("entries"):
            for sub in e["entries"]:
                vi = _entry_to_info(sub) if sub else None
                if vi and vi.youtube_id not in seen:
                    seen.add(vi.youtube_id)
                    videos.append(vi)
            continue
        vi = _entry_to_info(e)
        if vi and vi.youtube_id not in seen:
            seen.add(vi.youtube_id)
            videos.append(vi)
    ctx.progress(1.0, f"Found {len(videos)} videos")
    return parsed, info.get("title") or parsed.ident, videos


def save_discovery(db: Database, project_id: int, parsed: ParsedURL, title: str, videos: list[VideoInfo]) -> tuple[int, int]:
    """Persist discovered videos. Existing videos (same YouTube id in project) are left untouched.
    Returns (source_id, number_of_new_videos)."""
    now = time.time()
    with db.tx() as c:
        src = c.execute("INSERT INTO sources(project_id, url, kind, title, added_at) VALUES(?,?,?,?,?)",
                        (project_id, parsed.url, parsed.kind, title, now)).lastrowid
        new = 0
        for v in videos:
            cur = c.execute(
                "INSERT OR IGNORE INTO videos(project_id, source_id, youtube_id, url, title, uploader, duration, "
                "upload_date, filesize_estimate, availability, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (project_id, src, v.youtube_id, v.url, v.title, v.uploader, v.duration, v.upload_date,
                 v.filesize_estimate, v.availability, now),
            )
            new += cur.rowcount
    return int(src), new


def _is_retryable(msg: str) -> bool:
    m = msg.lower()
    if any(k in m for k in RESTRICTED_MARKERS):
        return False
    return any(k in m for k in ("timed out", "timeout", "connection", "temporarily", "http error 5", "network",
                                "reset by peer", "remote end closed", "incomplete", "http error 429", "getaddrinfo"))


def _explain(msg: str) -> str:
    m = msg.lower()
    if "not a bot" in m or "confirm you" in m:
        return ("YouTube is asking for bot verification for this network. This app does not bypass it; "
                "try again later or download the media yourself and use 'Import local file'.")
    if any(k in m for k in ("private video", "sign in", "members-only", "join this channel", "age")):
        return "This video is restricted (private, members-only, or age-gated) and cannot be downloaded by this app."
    if "unavailable" in m or "removed" in m:
        return "The video is unavailable (removed, region-blocked, or deleted)."
    if "unable to download webpage" in m or "getaddrinfo" in m or "connection" in m:
        return "Network error contacting YouTube. Check your internet connection and retry."
    return msg.replace("ERROR: ", "").strip()


def download_audio(db: Database, video_id: int, out_dir: Path, ctx: JobContext, retries: int = 4) -> Path:
    """Download best audio-only stream for one video, with resume and exponential backoff."""
    import yt_dlp

    row = db.q1("SELECT * FROM videos WHERE id=?", (video_id,))
    if row is None:
        raise ValueError(f"Video #{video_id} not found")
    if not row["authorized"]:
        raise PermissionError("Confirm that you are authorized to use this video's audio before downloading it.")
    if row["download_status"] == "downloaded" and row["download_path"] and Path(row["download_path"]).exists():
        ctx.progress(1.0, "Already downloaded")
        return Path(row["download_path"])
    # duplicate detection across projects: reuse an existing file for the same YouTube id
    dup = db.q1("SELECT download_path FROM videos WHERE youtube_id=? AND download_status='downloaded' AND id<>?",
                (row["youtube_id"], video_id))
    if dup and dup["download_path"] and Path(dup["download_path"]).exists():
        db.update("videos", video_id, download_status="downloaded", download_path=dup["download_path"],
                  error=None, updated_at=time.time())
        ctx.progress(1.0, "Reused existing download")
        return Path(dup["download_path"])

    db.transition("videos", video_id, ["discovered", "queued", "failed", "cancelled"], "downloading",
                  column="download_status", updated_at=time.time(), error=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    final: dict = {}

    def hook(d):
        if ctx.cancelled:
            raise Cancelled("Download cancelled")
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            speed = d.get("speed") or 0
            ctx.progress(done / total if total else 0.0,
                         f"{done / 1e6:.1f}/{total / 1e6:.1f} MB at {speed / 1e6:.2f} MB/s" if total else "Downloading…")
        elif d["status"] == "finished":
            final["path"] = d.get("filename")

    opts = _base_opts() | {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "continuedl": True,            # resume .part files
        "nopart": False,
        "overwrites": False,
        "retries": 3,
        "fragment_retries": 3,
        "progress_hooks": [hook],
        "noplaylist": True,
    }
    attempt = 0
    while True:
        attempt += 1
        db.execute("UPDATE videos SET attempts=attempts+1 WHERE id=?", (video_id,))
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(row["url"], download=True)
                path = Path(final.get("path") or ydl.prepare_filename(info))
            if not path.exists():
                raise DownloadError(f"yt-dlp reported success but {path.name} is missing.", retryable=True)
            db.transition("videos", video_id, ["downloading"], "downloaded", column="download_status",
                          download_path=str(path), updated_at=time.time(), error=None,
                          duration=info.get("duration") or row["duration"])
            ctx.progress(1.0, "Downloaded")
            return path
        except Cancelled:
            db.update("videos", video_id, download_status="cancelled", updated_at=time.time())
            raise
        except Exception as e:  # yt-dlp wraps our Cancelled inside DownloadError sometimes
            if ctx.cancelled:
                db.update("videos", video_id, download_status="cancelled", updated_at=time.time())
                raise Cancelled("Download cancelled") from e
            msg = str(e)
            retryable = getattr(e, "retryable", None)
            if retryable is None:
                retryable = _is_retryable(msg)
            if retryable and attempt <= retries:
                delay = min(60.0, 2 ** attempt + random.uniform(0, 1))
                log.warning("Download of %s failed (%s); retry %d/%d in %.0fs", row["youtube_id"], msg, attempt,
                            retries, delay)
                ctx.progress(0.0, f"Retrying in {delay:.0f}s ({attempt}/{retries})")
                end = time.time() + delay
                while time.time() < end:
                    if ctx.cancelled:
                        db.update("videos", video_id, download_status="cancelled", updated_at=time.time())
                        raise Cancelled("Download cancelled")
                    time.sleep(0.2)
                continue
            explained = _explain(msg)
            db.update("videos", video_id, download_status="failed", error=explained, updated_at=time.time())
            db.log_error(f"download:{row['youtube_id']}", explained, msg)
            raise DownloadError(explained, retryable=False) from e


def import_local_file(db: Database, project_id: int, path: Path) -> int:
    """Register a local media file the user owns as a 'video' row (already downloaded)."""
    import hashlib

    if not path.exists():
        raise FileNotFoundError(path)
    h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:11]
    now = time.time()
    existing = db.q1("SELECT id FROM videos WHERE project_id=? AND youtube_id=?", (project_id, f"local-{h}"))
    if existing:
        return int(existing["id"])
    return db.insert("videos", project_id=project_id, youtube_id=f"local-{h}", url=path.resolve().as_uri(),
                     title=path.name, selected=1, authorized=1, download_status="downloaded",
                     download_path=str(path.resolve()), updated_at=now)
