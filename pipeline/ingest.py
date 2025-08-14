import os, json, uuid, tempfile
from pathlib import Path
from faster_whisper import WhisperModel
import ffmpeg
from pyannote.audio import Pipeline as DiarizationPipeline

AUDIO_DIR = Path("storage/data")
JSON_DIR = Path("storage/data")

def load_audio(input_path):
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    (
        ffmpeg.input(str(input_path))
              .output(tmp_wav, ar=16000, ac=1, format="wav", loglevel="error")
              .overwrite_output()
              .run()
    )
    return tmp_wav

def transcribe_with_whisper(wav_path, device="auto", model_size="medium"):
    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments, info = model.transcribe(
        wav_path, beam_size=5, word_timestamps=True, vad_filter=True
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "text": w.word, "start": float(w.start), "end": float(w.end)
            })
    return words, info.language

def diarize_with_pyannote(wav_path, hf_token_env="HF_TOKEN"):
    import os
    token = os.getenv(hf_token_env)
    if not token:
        # No token -> skip diarization
        return []
    try:
        from pyannote.audio import Pipeline as DiarizationPipeline
        pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        diarization = pipeline({"audio": wav_path})
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
        return sorted(turns, key=lambda x: x["start"])
    except Exception as e:
        print(f"[diarization] disabled: {e}")
        return []

def save_episode_json(episode_title, wav_path, words, turns, language):
    episode_id = str(uuid.uuid4())[:8]
    out = {
        "episode_id": episode_id,
        "episode_title": episode_title,
        "audio_path": wav_path,
        "language": language,
        "words": words,
        "turns": turns
    }
    out_path = JSON_DIR / f"{episode_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path, episode_id

def process_episode(file_path, title):
    wav = load_audio(file_path)
    words, lang = transcribe_with_whisper(wav)
    turns = diarize_with_pyannote(wav)
    if not turns:
        # Single-speaker fallback covering the clip
        if words:
            turns = [{"speaker": "SPK0", "start": words[0]["start"], "end": words[-1]["end"]}]
        else:
            turns = [{"speaker": "SPK0", "start": 0.0, "end": 0.0}]
    return save_episode_json(title, wav, words, turns, lang)
