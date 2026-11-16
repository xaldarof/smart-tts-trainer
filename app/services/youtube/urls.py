"""YouTube URL classification (no network)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class InvalidURL(ValueError):
    pass


@dataclass(frozen=True)
class ParsedURL:
    kind: str          # video | playlist | channel
    url: str           # canonical URL to hand to yt-dlp
    ident: str         # video id, playlist id, or channel handle/id


def parse_youtube_url(raw: str) -> ParsedURL:
    s = (raw or "").strip()
    if not s:
        raise InvalidURL("Enter a YouTube video, playlist, or channel URL.")
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    u = urlparse(s)
    host = (u.hostname or "").lower()
    if host not in _YT_HOSTS:
        raise InvalidURL(f"'{raw}' is not a YouTube URL (host: {host or 'none'}).")
    qs = parse_qs(u.query)
    path = u.path.rstrip("/")

    if host.endswith("youtu.be"):
        vid = path.lstrip("/").split("/")[0]
        if _VIDEO_ID.match(vid):
            return ParsedURL("video", f"https://www.youtube.com/watch?v={vid}", vid)
        raise InvalidURL(f"Could not find a video id in '{raw}'.")

    if path == "/watch" and "v" in qs and _VIDEO_ID.match(qs["v"][0]):
        vid = qs["v"][0]
        # A watch URL with list= is treated as a single video; users paste playlist URLs for playlists.
        return ParsedURL("video", f"https://www.youtube.com/watch?v={vid}", vid)
    m = re.match(r"^/(shorts|live|embed|v)/([A-Za-z0-9_-]{11})$", path)
    if m:
        return ParsedURL("video", f"https://www.youtube.com/watch?v={m.group(2)}", m.group(2))

    if path == "/playlist" and "list" in qs:
        pid = qs["list"][0]
        if not re.match(r"^[A-Za-z0-9_-]{10,}$", pid):
            raise InvalidURL(f"Invalid playlist id '{pid}'.")
        return ParsedURL("playlist", f"https://www.youtube.com/playlist?list={pid}", pid)

    m = re.match(r"^/(@[A-Za-z0-9._\-·]+|channel/UC[A-Za-z0-9_-]{22}|c/[^/]+|user/[^/]+)(/(videos|streams|shorts|featured|playlists))?$",
                 path)
    if m:
        base = m.group(1)
        tab = m.group(3) or "videos"
        if tab in ("featured", "playlists"):
            tab = "videos"
        return ParsedURL("channel", f"https://www.youtube.com/{base}/{tab}", base)

    raise InvalidURL(f"Unsupported YouTube URL form: '{raw}'. Use a video, playlist, or channel URL.")
