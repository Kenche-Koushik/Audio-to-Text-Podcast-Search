def ts_to_mmss(t):
    m = int(t // 60); s = int(t % 60)
    return f"{m:02d}:{s:02d}"

def audio_fragment_tag(audio_path, start):
    return f"Start at {ts_to_mmss(start)}"
