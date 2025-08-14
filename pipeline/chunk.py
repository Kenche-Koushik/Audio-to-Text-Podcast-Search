def time_aware_windows(sentences, target_tokens=420, overlap=0.2):
    def est_tokens(t): return max(1, int(len(t.split()) * 1.3))
    windows = []
    i = 0
    while i < len(sentences):
        cur, cur_tok = [], 0
        start_t = sentences[i]["start"]
        j = i
        while j < len(sentences) and cur_tok < target_tokens:
            cur.append(sentences[j])
            cur_tok += est_tokens(sentences[j]["text"])
            j += 1
        end_t = cur[-1]["end"]
        text = " ".join(s["text"] for s in cur)
        speakers = {}
        for s in cur:
            for k, v in s["speakers"].items():
                speakers[k] = speakers.get(k, 0) + v
        windows.append({"text": text, "start": start_t, "end": end_t, "speakers": speakers, "tokens": cur_tok})
        step = max(1, int(len(cur) * (1 - overlap)))
        i += step
    return windows
