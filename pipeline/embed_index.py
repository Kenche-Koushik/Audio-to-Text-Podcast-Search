import json
import chromadb
from sentence_transformers import SentenceTransformer

def get_chroma():
    client = chromadb.PersistentClient(path="storage/chroma")
    return client.get_or_create_collection(name="podcast_chunks", metadata={"hnsw:space":"cosine"})

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

def upsert_episode(chunks, episode_meta):
    coll = get_chroma()
    emb = Embedder()
    ids, docs, metas = [], [], []
    for idx, ch in enumerate(chunks):
        ids.append(f"{episode_meta['episode_id']}_{idx}")
        docs.append(ch["text"])

        speakers = ch.get("speakers", {}) or {}
        top_speaker = max(speakers, key=speakers.get) if speakers else ""
        top_speaker_secs = float(speakers[top_speaker]) if top_speaker else 0.0

        metas.append({
            **episode_meta,
            "start_time": float(ch["start"]),
            "end_time": float(ch["end"]),
            "tokens": int(ch.get("tokens", 0)),
            # primitives only:
            "n_speakers": int(len(speakers)),
            "top_speaker": str(top_speaker),
            "top_speaker_secs": float(top_speaker_secs),
            # JSON string for the full map (not a dict)
            "speakers_json": json.dumps(speakers, ensure_ascii=False),
        })
    embs = emb.encode(docs)
    coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
