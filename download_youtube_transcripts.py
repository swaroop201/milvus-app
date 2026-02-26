import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from yt_dlp import YoutubeDL


load_dotenv()


# Map creator names to their YouTube channel URLs.
# If any URL is wrong, update it here.
CREATOR_CHANNELS: Dict[str, str] = {
    "Ezachly": "https://www.youtube.com/@eczachly_",
    "JoeReisData": "https://www.youtube.com/c/TernaryData",
    # "Data with Danny": "https://www.youtube.com/@DataWithDanny",  # TODO: replace with correct channel URL if available
    "Matthew Berman": "https://www.youtube.com/@matthew_berman",
}

MAX_VIDEOS_PER_CREATOR = 5
DOWNLOAD_DIR = Path("downloads/youtube_audio").resolve()
TRANSCRIPTS_DIR = Path("downloads/transcripts").resolve()


def get_audio_client_and_model() -> tuple[OpenAI, str]:
    """
    Return an OpenAI-compatible client and model name for transcription.
    Prefer Groq (free) if GROQ_API_KEY is set; otherwise fall back to OpenAI.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        # Groq supports Whisper via this model name
        return client, "whisper-large-v3"

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY (recommended, free) or OPENAI_API_KEY for transcription.")
    return OpenAI(api_key=api_key), "whisper-1"


def fetch_recent_video_urls(channel_url: str, max_videos: int = MAX_VIDEOS_PER_CREATOR) -> List[str]:
    """
    Use yt-dlp to fetch the most recent video URLs from a channel.
    This does NOT download the videos, only lists metadata.
    """
    # Use the channel's "Videos" tab, which yt-dlp handles more reliably.
    channel_videos_url = channel_url.rstrip("/") + "/videos"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": max_videos,
        # Ensure we hit the uploads/videos playlist, not shorts etc.
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_videos_url, download=False)

    entries = info.get("entries") or []
    urls: List[str] = []
    for entry in entries:
        # Prefer full URLs if present.
        url = entry.get("webpage_url") or entry.get("url") or ""
        if url.startswith("http"):
            urls.append(url)
        else:
            # Fall back to ID if present.
            video_id = entry.get("id")
            if video_id:
                urls.append(f"https://www.youtube.com/watch?v={video_id}")
        if len(urls) >= max_videos:
            break
    return urls


def download_audio(video_url: str, creator_name: str) -> Path | None:
    """
    Download the audio track of a YouTube video.
    We keep whatever audio format YouTube provides (mp3/m4a/webm, etc.).
    """
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize creator name for filenames
    creator_safe = creator_name.replace(" ", "_")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_DIR / f"{creator_safe}_%(title)s_%(id)s.%(ext)s"),
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    # Use the file yt-dlp actually wrote (whatever extension it chose)
    filename = ydl.prepare_filename(info)
    audio_path = Path(filename)
    if not audio_path.exists():
        return None
    return audio_path


def transcribe_with_timestamps(client: OpenAI, model: str, audio_path: Path) -> dict:
    """
    Use OpenAI Whisper API to create a timed transcript.
    Returns the full verbose JSON response (segments with start/end).
    """
    # Very large files can exceed the API's size limits (HTTP 413).
    # Skip anything over ~25MB to avoid repeated failures.
    try:
        if audio_path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Audio file too large for transcription API; skipping.")
    except OSError:
        pass

    with audio_path.open("rb") as f:
        # verbose_json gives segments with timing information
        transcript = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )
    # The OpenAI client returns a pydantic-style object; convert to plain dict
    return transcript.model_dump()


def save_transcript_files(creator: str, video_url: str, audio_path: Path, transcript: dict) -> None:
    """
    Save transcript as:
    - JSON with full metadata (segments, language, etc.)
    - Simple SRT-style file built from segments
    """
    import json

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    creator_safe = creator.replace(" ", "_")
    base_name = audio_path.stem

    json_path = TRANSCRIPTS_DIR / f"{creator_safe}_{base_name}.json"
    srt_path = TRANSCRIPTS_DIR / f"{creator_safe}_{base_name}.srt"

    payload = {
        "creator": creator,
        "video_url": video_url,
        "audio_file": str(audio_path),
        "transcript": transcript,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Build a simple SRT from segments if available
    segments = transcript.get("segments") or []
    if segments:
        def format_ts(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

        with srt_path.open("w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                text = seg.get("text", "").strip()
                if not text:
                    continue
                f.write(f"{idx}\n")
                f.write(f"{format_ts(start)} --> {format_ts(end)}\n")
                f.write(f"{text}\n\n")


def main() -> None:
    client, model = get_audio_client_and_model()

    for creator, channel_url in CREATOR_CHANNELS.items():
        print(f"Processing creator: {creator} ({channel_url})")
        try:
            video_urls = fetch_recent_video_urls(channel_url, MAX_VIDEOS_PER_CREATOR)
        except Exception as e:
            print(f"  Failed to fetch videos for {creator}: {e}")
            continue

        if not video_urls:
            print(f"  No videos found for {creator}.")
            continue

        for video_url in video_urls:
            print(f"  Downloading audio for: {video_url}")
            try:
                audio_path = download_audio(video_url, creator)
            except Exception as e:
                print(f"    Failed to download audio: {e}")
                continue

            if not audio_path:
                print("    Audio file not found after download.")
                continue

            print(f"    Transcribing: {audio_path.name}")
            try:
                transcript = transcribe_with_timestamps(client, model, audio_path)
            except Exception as e:
                print(f"    Transcription failed: {e}")
                continue

            save_transcript_files(creator, video_url, audio_path, transcript)
            print(f"    Saved transcripts for {audio_path.name}")


if __name__ == "__main__":
    main()

