from typing import List, Dict

def assign_speakers(words: List[Dict], turns: List[Dict]) -> List[Dict]:
    j = 0
    for w in words:
        while j + 1 < len(turns) and turns[j]["end"] < w["start"]:
            j += 1
        speaker = turns[j]["speaker"] if turns and turns[j]["start"] <= w["start"] <= turns[j]["end"] else "UNK"
        w["speaker"] = speaker
    return words

def sentences_from_words(words: List[Dict], max_chars=280):
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        cur.append(w); cur_len += len(w["text"]) + 1
        if w["text"].endswith(('.', '?', '!', '…')) or cur_len >= max_chars:
            s_text = " ".join(x["text"] for x in cur)
            s_start = cur[0]["start"]; s_end = cur[-1]["end"]
            speakers = {}
            for x in cur: speakers[x["speaker"]] = speakers.get(x["speaker"], 0) + (x["end"]-x["start"])
            chunks.append({"text": s_text, "start": s_start, "end": s_end, "speakers": speakers})
            cur, cur_len = [], 0
    if cur:
        s_text = " ".join(x["text"] for x in cur)
        chunks.append({"text": s_text, "start": cur[0]["start"], "end": cur[-1]["end"], "speakers": {cur[0]['speaker']: cur[-1]['end']-cur[0]['start']}})
    return chunks
