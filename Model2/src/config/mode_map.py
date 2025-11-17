MODES = [
    "major", "dorian", "phrygian", "lydian",
    "mixolydian", "minor", "locrian"
]

NUM_TO_SCALE_MAP = {
    0: "C major", 1: "C dorian", 2: "C phrygian", 3: "C lydian", 4: "C mixolydian", 5: "C minor", 6: "C locrian",
    7: "C# major", 8: "C# dorian", 9: "C# phrygian", 10: "C# lydian", 11: "C# mixolydian", 12: "C# minor",
    13: "C# locrian",
    14: "D major", 15: "D dorian", 16: "D phrygian", 17: "D lydian", 18: "D mixolydian", 19: "D minor", 20: "D locrian",
    21: "D# major", 22: "D# dorian", 23: "D# phrygian", 24: "D# lydian", 25: "D# mixolydian", 26: "D# minor",
    27: "D# locrian",
    28: "E major", 29: "E dorian", 30: "E phrygian", 31: "E lydian", 32: "E mixolydian", 33: "E minor", 34: "E locrian",
    35: "F major", 36: "F dorian", 37: "F phrygian", 38: "F lydian", 39: "F mixolydian", 40: "F minor", 41: "F locrian",
    42: "F# major", 43: "F# dorian", 44: "F# phrygian", 45: "F# lydian", 46: "F# mixolydian", 47: "F# minor",
    48: "F# locrian",
    49: "G major", 50: "G dorian", 51: "G phrygian", 52: "G lydian", 53: "G mixolydian", 54: "G minor", 55: "G locrian",
    56: "G# major", 57: "G# dorian", 58: "G# phrygian", 59: "G# lydian", 60: "G# mixolydian", 61: "G# minor",
    62: "G# locrian",
    63: "A major", 64: "A dorian", 65: "A phrygian", 66: "A lydian", 67: "A mixolydian", 68: "A minor", 69: "A locrian",
    70: "A# major", 71: "A# dorian", 72: "A# phrygian", 73: "A# lydian", 74: "A# mixolydian", 75: "A# minor",
    76: "A# locrian",
    77: "B major", 78: "B dorian", 79: "B phrygian", 80: "B lydian", 81: "B mixolydian", 82: "B minor", 83: "B locrian"
}

SCALE_TO_NUM_MAP = {v: k for k, v in NUM_TO_SCALE_MAP.items()}

NUM_TO_TONAL_CENTER_MAP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

TONAL_CENTER_TO_NUM_MAP = {v: k for k, v in NUM_TO_TONAL_CENTER_MAP.items()}

NUM_TO_MUSICAL_MODE_MAP = {
    0: "major",
    1: "dorian",
    2: "phrygian",
    3: "lydian",
    4: "mixolydian",
    5: "minor",
    6: "locrian",
}

MUSICAL_MODE_TO_NUM_MAP = {v: k for k, v in NUM_TO_MUSICAL_MODE_MAP.items()}


def get_scale_from_nums(center, mode):
    return f"{NUM_TO_TONAL_CENTER_MAP[center]} {NUM_TO_MUSICAL_MODE_MAP[mode]}"


def get_scales_from_nums(center, mode):
    res = []
    for i in range(center.shape[0]):
        c, m = center[i], mode[i]
        res.append(get_scale_from_nums(c.item(), m.item()))
    return res
