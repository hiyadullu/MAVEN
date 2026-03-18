EMOTION_MAP = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3
}

def parse_emotion(line):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None
    return EMOTION_MAP.get(parts[2], None)
