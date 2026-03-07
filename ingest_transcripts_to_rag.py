import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Reuse the same RAG configuration and Milvus helpers
from main import (  # type: ignore
    get_rag_collection_name,
    ingest_documents_with_metadata,
)


load_dotenv()


TRANSCRIPTS_DIR = Path("downloads/transcripts").resolve()


def _extract_video_id(video_url: str) -> str:
    """
    Best-effort extraction of a YouTube video ID from a URL.
    Handles standard watch URLs and youtu.be short links.
    """
    if not video_url:
        return ""
    # Common case: https://www.youtube.com/watch?v=VIDEO_ID&...
    if "v=" in video_url:
        part = video_url.split("v=", 1)[1]
        return part.split("&", 1)[0]
    # Short links: https://youtu.be/VIDEO_ID
    if "youtu.be/" in video_url:
        part = video_url.split("youtu.be/", 1)[1]
        return part.split("?", 1)[0].split("&", 1)[0]
    return ""


def build_document_text(payload: dict) -> str | None:
    """
    Legacy helper (no longer used for ingestion now that we ingest per segment).
    Kept for reference; current pipeline works per Whisper segment instead of
    chunking the whole transcript text.
    """
    transcript = payload.get("transcript") or {}
    full_text: str = transcript.get("text") or ""
    full_text = (full_text or "").strip()
    return full_text or None


def main() -> None:
    if not TRANSCRIPTS_DIR.exists():
        print(f"No transcripts directory found at {TRANSCRIPTS_DIR}")
        return

    json_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    if not json_files:
        print(f"No transcript JSON files found in {TRANSCRIPTS_DIR}")
        return

    all_texts: List[str] = []
    all_channel_names: List[str] = []
    all_video_ids: List[str] = []
    all_start_secs: List[float] = []
    all_end_secs: List[float] = []

    for path in json_files:
        print(f"Processing transcript file: {path.name}")
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  Failed to read JSON: {e}")
            continue

        creator = payload.get("creator", "Unknown creator")
        video_url = payload.get("video_url", "")
        video_id = _extract_video_id(video_url)
        transcript = payload.get("transcript") or {}
        segments = transcript.get("segments") or []
        if not segments:
            print("  No segments found; skipping.")
            continue

        # Group multiple segments into larger chunks (e.g. more than 3 lines) for better context.
        MIN_SEGMENTS_PER_CHUNK = 5
        MAX_CHARS_PER_CHUNK = 1400

        buffer_texts: list[str] = []
        buffer_start: float | None = None
        buffer_end: float | None = None
        buffer_chars = 0
        chunk_count = 0

        def flush_chunk() -> None:
            nonlocal buffer_texts, buffer_start, buffer_end, buffer_chars, chunk_count
            if not buffer_texts or buffer_start is None or buffer_end is None:
                return
            text = " ".join(buffer_texts).strip()
            if not text:
                buffer_texts = []
                buffer_start = None
                buffer_end = None
                buffer_chars = 0
                return
            all_texts.append(text)
            all_channel_names.append(creator)
            all_video_ids.append(video_id)
            all_start_secs.append(buffer_start)
            all_end_secs.append(buffer_end)
            chunk_count += 1
            buffer_texts = []
            buffer_start = None
            buffer_end = None
            buffer_chars = 0

        for idx, seg in enumerate(segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)

            if buffer_start is None:
                buffer_start = start
            buffer_end = end
            buffer_texts.append(text)
            buffer_chars += len(text)

            is_last_segment = idx == len(segments) - 1
            if (
                len(buffer_texts) >= MIN_SEGMENTS_PER_CHUNK
                and buffer_chars >= MAX_CHARS_PER_CHUNK
            ) or is_last_segment:
                flush_chunk()

        print(f"  Generated {chunk_count} chunks from transcript (channel: {creator}).")

    if not all_texts:
        print("No chunks to ingest.")
        return

    print(
        f"Ingesting {len(all_texts)} chunks into RAG collection "
        f"'{get_rag_collection_name()}' (with channel_name, video_id, and timestamps)..."
    )
    inserted = ingest_documents_with_metadata(
        all_texts, all_channel_names, all_video_ids, all_start_secs, all_end_secs
    )
    print(f"Done. Inserted {inserted} chunks.")


if __name__ == "__main__":
    main()

