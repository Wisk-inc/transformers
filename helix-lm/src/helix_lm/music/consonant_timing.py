# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Consonants ahead of the beat. A singer starts a syllable's consonant before the note so that its vowel lands on the
note's onset; a synthesiser that starts the consonant on the onset sounds late, and its sung onsets measure late.
:func:`advance` moves each syllable-initial consonant run into the end of the phoneme before it, by up to
``max_lead_s`` and never more than half of that phoneme.
"""

from __future__ import annotations

VOWELS = set("aeiouAEIOU") | {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh",
                                "uw", "ax", "ix", "ux"}


def is_vowel(phoneme: str) -> bool:
    p = phoneme.lower().rstrip("012")
    return p in VOWELS or (len(p) == 1 and p in "aeiouy")


def advance(phonemes: list[str], durations: list[float], note_starts: list[int], max_lead_s: float = 0.08):
    """``phonemes`` with ``durations`` (seconds); ``note_starts`` are the indices of the phonemes that begin a note.
    Returns new durations: before each note, the consonants that start it are taken out of the previous phoneme's
    tail, so the note's vowel keeps its place."""
    d = list(durations)
    for start in note_starts:
        if start <= 0:
            continue
        run = 0
        while start + run < len(phonemes) and not is_vowel(phonemes[start + run]):
            run += 1
        if run == 0 or start + run >= len(phonemes):
            continue
        lead = min(max_lead_s, sum(d[start : start + run]), d[start - 1] * 0.5)
        # the phoneme before gives up ``lead``: the consonants and the vowel after them all start that much
        # earlier, and the vowel takes the time back at its end, so the note's vowel reaches the onset sooner and
        # everything after it keeps its time
        d[start - 1] -= lead
        d[start + run] += lead
    return d


__all__ = ["advance", "is_vowel"]
