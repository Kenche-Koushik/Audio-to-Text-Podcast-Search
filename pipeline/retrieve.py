from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

def get_chroma():
    client = chromadb.PersistentClient(path="storage/chroma")
    return client.get_or_create_collection(name="podcast_chunks")

class Retriever:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2", rerank=False):
        self.embedder = SentenceTransformer(embed_model)
        self.coll = get_chroma()
        self.rerank = rerank
        self.cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if rerank else None

    def search(self, query, k=15, out_k=6, filters=None):
        qv = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.coll.query(query_embeddings=qv, n_results=k, where=filters or {})
        hits = list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))
        if self.rerank:
            pairs = [[query, d] for _, d, _ in hits]
            scores = self.cross.predict(pairs).tolist()
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        return hits[:out_k]
