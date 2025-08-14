# --- robust project-root import shim ---
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------

import streamlit as st
import json as _json
from pathlib import Path

from pipeline.ingest import process_episode
from pipeline.align import assign_speakers, sentences_from_words
from pipeline.chunk import time_aware_windows
from pipeline.embed_index import upsert_episode
from pipeline.retrieve import Retriever
from app.components import ts_to_mmss


st.set_page_config(page_title="Podcast RAG", page_icon="🎙️", layout="wide")
st.title("🎙️ Audio-to-Text RAG for Podcast Search")

with st.sidebar:
    hf_token_ui = st.text_input("Hugging Face token (for diarization)", type="password", value=os.getenv("HF_TOKEN", ""))
    if hf_token_ui:
        os.environ["HF_TOKEN"] = hf_token_ui

    st.caption("Diarization: " + ("ON (token detected)" if os.getenv("HF_TOKEN") else "OFF (no token)"))
    st.header("Upload Episodes")
    upl = st.file_uploader("Add audio files", type=["mp3","wav","m4a"], accept_multiple_files=True)
    titles = {}
    if upl:
        for f in upl:
            titles[f.name] = st.text_input(f"Title for {f.name}", value=f.name)

    if st.button("Process & Index"):
        if not upl: 
            st.error("Upload at least one file."); 
            st.stop()
        os.makedirs("storage/data", exist_ok=True)
        for f in upl:
            raw_path = Path("storage/data") / f.name
            with open(raw_path, "wb") as w: w.write(f.read())
            ep_json_path, ep_id = process_episode(raw_path, titles[f.name])
            data = _json.load(open(ep_json_path, "r", encoding="utf-8"))
            words = assign_speakers(data["words"], data["turns"])
            sents = sentences_from_words(words)
            chunks = time_aware_windows(sents)
            meta = {"episode_id": data["episode_id"], "episode_title": data["episode_title"]}
            upsert_episode(chunks, meta)
        st.success("Indexed! Go search ➡️")

st.divider()
st.header("🔎 Search across episodes")

query = st.text_input("Ask about topics, quotes, names, steps…", placeholder="e.g., What did they say about transfer learning?")
colA, colB = st.columns([2,1])
with colB:
    k = st.slider("Max results", 3, 15, 8)
    rerank = st.toggle("Re-rank (better precision)")
with colA:
    if st.button("Search") and query:
        r = Retriever(rerank=rerank)
        hits = r.search(query, k=15, out_k=k)
        if not hits: 
            st.info("No matches yet.")
        else:
            for hid, text, meta in hits:
                with st.container(border=True):
                    st.markdown(f"**{meta['episode_title']}**  ·  ⏱️ {ts_to_mmss(meta['start_time'])} – {ts_to_mmss(meta['end_time'])}")
                    st.write(text)
                    spk_meta = meta.get("speakers_json") or meta.get("speakers")
                    spk_dict = {}
                    if isinstance(spk_meta, str):
                        try:
                            spk_dict = _json.loads(spk_meta)
                        except Exception:
                            spk_dict = {}
                    elif isinstance(spk_meta, dict):
                        spk_dict = spk_meta

                    if spk_dict:
                        st.caption("Speakers: " + ", ".join(str(k) for k in spk_dict.keys()))
                    else:
                        top = meta.get("top_speaker")
                        if top:
                            st.caption(f"Top speaker: {top}")
