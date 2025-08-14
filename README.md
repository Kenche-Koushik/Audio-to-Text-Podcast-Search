# 🎙️ Audio-to-Text RAG for Podcast Search

Upload podcast episodes → transcribe + diarize → time-aware chunking → vector search across episodes → query with timestamped results in a Streamlit UI.

## Features
- Whisper via **faster-whisper** (word timestamps)
- **pyannote.audio** diarization (speaker turns)
- Time-aware chunking with overlap
- Vector search in **Chroma** (local, persistent)
- Optional re-ranking via CrossEncoder
- Streamlit UI with episode, speakers, and `MM:SS` timestamps
- Ready for **Hugging Face Spaces** deployment

## Project Structure
```
podcast-rag/
├─ app/
│  ├─ streamlit_app.py
│  ├─ components.py
│  └─ prompts.py
├─ pipeline/
│  ├─ ingest.py
│  ├─ align.py
│  ├─ chunk.py
│  ├─ embed_index.py
│  └─ retrieve.py
├─ storage/
│  ├─ chroma/         # local vector DB (gitignored)
│  └─ data/           # uploaded audio + episode JSON
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## Local Setup
```bash
# 0) System dependency
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg

# 1) Create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your Hugging Face token for pyannote (required)
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Windows (Powershell):  $Env:HF_TOKEN="hf_xxx"

# 4) Run
streamlit run app/streamlit_app.py
```

## Usage
1. Open the app.
2. In the sidebar, upload one or more audio files (`.mp3`, `.wav`, `.m4a`) and give each a title.
3. Click **Process & Index** to:
   - Convert audio → 16k mono WAV
   - Transcribe (Whisper) with word timestamps
   - Diarize (pyannote) to get speaker turns
   - Align words ↔ speakers, chunk sentences, build windows
   - Embed & upsert into Chroma
4. Search in the main panel; results show episode title, `MM:SS` range, speakers, and snippet.

> Tip: You can re-index after adding more episodes—Chroma persists in `storage/chroma/`.

## Deploy on Hugging Face Spaces (Streamlit)
1. Create a **New Space** → **Streamlit**.
2. Push this repo’s files to the Space (or use the zip in this package).
3. In the Space **Settings → Variables and secrets**, add:
   - `HF_TOKEN` = your Hugging Face access token (needed for pyannote).
4. The Space will install from `requirements.txt` and launch Streamlit automatically.
5. If diarization/ASR is slow on CPU, switch the Space **Hardware** to a small GPU.

### Alternative: Docker
```bash
docker build -t podcast-rag .
docker run --rm -it -p 7860:7860 -e HF_TOKEN=$HF_TOKEN podcast-rag
```

## Notes & Limits
- Accuracy depends on audio quality; domain-specific terms may reduce ASR accuracy.
- Spaces free tier may time out on very long files; consider shorter uploads or GPU hardware.
- If you need managed vector DB, swap Chroma for Pinecone/Weaviate and update `embed_index.py` / `retrieve.py`.

## License
MIT
