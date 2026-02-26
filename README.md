# YouTube RAG — FastAPI + Zilliz

A FastAPI app with a chat UI and **RAG** over YouTube transcript segments. It uses **Zilliz Cloud** (or Milvus) for vector search and supports **free APIs** (Groq + Hugging Face) or OpenAI.

## What’s in this repo

- **Web app** — Chat UI at `/`, RAG at `POST /task`, docs at `/docs`.
- **YouTube pipeline** — Download recent videos from chosen creators, transcribe with Groq/OpenAI Whisper, store timed segments.
- **RAG ingest** — Chunk/segment transcripts, embed with sentence-transformers (or OpenAI), insert into Zilliz with metadata (channel, video_id, timestamps). Answers can cite YouTube URLs with `&t=` so links start at the right time.

## How to run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and set:

- **Chat:** `GROQ_API_KEY` (free at [Groq Console](https://console.groq.com)) or `OPENAI_API_KEY`
- **Embeddings:** `HUGGINGFACE_TOKEN` (free at [Hugging Face](https://huggingface.co/settings/tokens)) or `OPENAI_API_KEY`
- **Zilliz:** `ZILLIZ_URI`, `ZILLIZ_API_KEY` (from [Zilliz Cloud](https://cloud.zilliz.com))
- **RAG collection (if using existing collection):** `RAG_COLLECTION_NAME`, `RAG_ID_FIELD`, `RAG_VECTOR_FIELD`, `RAG_TEXT_FIELD`, `RAG_CHANNEL_NAME_MAX_LENGTH`

Never commit `.env`; it is in `.gitignore`.

### 3. (Optional) Download and transcribe YouTube videos

```bash
python download_youtube_transcripts.py
```

This downloads audio for the latest videos from the configured creators, transcribes with Groq/OpenAI Whisper, and writes JSON + SRT under `downloads/transcripts/` and `downloads/youtube_audio/`. Requires ffmpeg on PATH.

### 4. (Optional) Ingest transcripts into RAG

```bash
python ingest_transcripts_to_rag.py
```

Reads `downloads/transcripts/*.json`, builds one vector row per segment with channel, video_id, and timestamps, and inserts into the Zilliz collection. Run after creating/clearing the collection as needed.

### 5. Start the server

```bash
python main.py
```

- **App:** http://localhost:8000  
- **API docs:** http://localhost:8000/docs  
- **RAG:** `POST /task` with `{"message": "your question"}` — answers can include YouTube URLs with `&t=` and time ranges.

---

## Pushing to GitHub

1. **Create a repo** on GitHub (e.g. `your-username/youtube-rag`). Do not add a README or .gitignore if you already have them locally.

2. **From the project folder** (PowerShell):

   ```powershell
   cd path\to\vibe-coding

   git init
   git add .
   git status
   ```

   Confirm `.env` and `downloads/` do **not** appear (they are in `.gitignore`).

3. **Commit and push:**

   ```powershell
   git commit -m "Add YouTube RAG app and transcript pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub repo.

4. **Later changes:**

   ```powershell
   git add .
   git commit -m "Short description of changes"
   git push
   ```

---

## Deploying to Render

1. In [Render](https://render.com), create a **Web Service** and connect your GitHub repo.

2. **Build command:**  
   `pip install -r requirements.txt`

3. **Start command:**  
   `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Environment:** Add the same variables as in `.env` (e.g. `GROQ_API_KEY`, `HUGGINGFACE_TOKEN`, `ZILLIZ_URI`, `ZILLIZ_API_KEY`, `RAG_COLLECTION_NAME`, `RAG_ID_FIELD`, `RAG_VECTOR_FIELD`, `RAG_TEXT_FIELD`, `RAG_CHANNEL_NAME_MAX_LENGTH`). Do not commit `.env`; set them in the Render dashboard.

5. Deploy. The app uses your Zilliz collection; no need to push `downloads/` — data lives in Zilliz.

---

## API overview

| Endpoint        | Description |
|----------------|-------------|
| `GET /`        | Chat UI.    |
| `GET /docs`    | OpenAPI docs. |
| `POST /ask`    | Plain chat (no RAG): `{"message": "..."}`. |
| `POST /task`   | RAG: `{"message": "..."}` → answer with citations (YouTube URL + time). |
| `POST /ingest` | Ingest text or documents into RAG (optional `channel_name`). |
| `GET /health/milvus` | Check Zilliz/Milvus connectivity. |

Server runs on port **8000** by default (Render uses `$PORT`).
