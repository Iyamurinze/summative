"""
Lightweight Kinyarwanda language guard.

The fine-tuned sentiment model has no built-in way to tell whether its input is actually
Kinyarwanda — it will happily produce a confident-looking prediction for English, French, or
gibberish input, since AfroXLMR is inherently multilingual. This module provides a cheap,
dependency-free heuristic check (NOT a real language-ID model) so the UI can warn the user
when their input probably isn't Kinyarwanda, without blocking the prediction.

The reference vocabulary below is the 250 most frequent words extracted directly from the
AfriSenti Kinyarwanda train/validation/test text (see notebooks/kinyarwanda_sentiment.ipynb
for how it could be regenerated) — mostly grammatical/function words ("mu", "ni", "ko", "na",
"ariko"...) that appear across virtually all Kinyarwanda text regardless of topic, which makes
them a reasonable signal independent of what the tweet is actually about. A handful of
incidental English/slang tokens that leaked into the raw frequency list (e.g. "the", "amp",
"bro") were manually filtered out, since they would otherwise cause false negatives on
English input.
"""

import re

# Top 250 most frequent words in the Kinyarwanda AfriSenti train/val/test text, used as a
# lightweight heuristic to flag likely non-Kinyarwanda input before classification.
KINYARWANDA_COMMON_WORDS = {
    'mu', 'ni', 'ko', 'muri', 'ngo', 'na', 'cyane', 'kuri',
    'ku', 'ya', 'ari', 'rwanda', 'kandi', 'ariko', 'wa', 'yo',
    'iyo', 'uyu', 'se', 'aho', 'neza', 'kuko', 'ibyo', 'iki',
    'munsi', 'nawe', 'uko', 'nta', 'gusa', 'no', 'cg', 'hari',
    'umuntu', 'uri', 'iyi', 'niba', 'ubu', 'ikibazo', 'ubwo', 'abantu',
    'rwot', 'ese', 'igihe', 'icyo', 'umunsi', 'aba', 'gahunda', 'rwose',
    'kuba', 'we', 'nyuma', 'koko', 'imana', 'none', 'umwana', 'wowe',
    'kubera', 'nuko', 'mwiza', 'ahubwo', 'covid', 'dore', 'byose', 'ntabwo',
    'ba', 'amakuru', 'hamwe', 'byo', 'za', 'umugore', 'ubundi', 'cya',
    'amafaranga', 'uwo', 'kigali', 'wo', 'murakoze', 'njye', 'ubuzima', 'buri',
    'mbere', 'ufite', 'si', 'reka', 'akazi', 'amahoro', 'ka', 'bose',
    'bari', 'nka', 'ibi', 'nziza', 'inama', 'rero', 'ukuri', 'nukuri',
    'ndetse', 'kimwe', 'cyo', 'hano', 'iri', 'niyo', 'nabi', 'nonese',
    'maze', 'yose', 'kurya', 'abana', 'gute', 'ibintu', 'kdi', 'wawe',
    'aha', 'kumva', 'ndi', 'nyamara', 'afite', 'nubwo', 'yari', 'kwirinda',
    'abo', 'zo', 'imbere', 'ubwenge', 'kuva', 'inyuma', 'numva', 'kugira',
    'bwo', 'baba', 'umva', 'kuki', 'nanjye', 'ye', 'mbona', 'umwanya',
    'amazi', 'rya', 'kubona', 'kera', 'amahirwe', 'ntacyo', 'bwa', 'abandi',
    'nyine', 'ejo', 'sha', 'bafite', 'mwese', 'wumva', 'bo', 'abaturage',
    'gukora', 'turi', 'ukuntu', 'buriya', 'kureba', 'ibyiza', 'bya', 'benshi',
    'mwaramutse', 'ikipe', 'gihe', 'oya', 'twese', 'sinzi', 'umutima', 'umubyeyi',
    'igihugu', 'yawe', 'umukobwa', 'ntago', 'yaba', 'kwa', 'naho', 'wanjye',
    'atari', 'umuyobozi', 'hejuru', 'umugabo', 'ubwoba', 'mama', 'wese', 'kuvuga',
    'ibindi', 'ibibazo', 'gukoresha', 'imyaka', 'kugirango', 'ubanza', 'ayo', 'hanze',
    'mwe', 'nk', 'by', 'ndumva', 'uzi', 'mugire', 'inshuti', 'kurwanya',
    'imbaraga', 'hasi', 'umwe', 'mfite', 'impamvu', 'abagore', 'niko', 'saa',
    'byiza', 'namwe', 'izina', 'amabwiriza', 'leta', 'umuriro', 'reba', 'ikiganiro',
    'noneho', 'harya', 'ushaka', 'undi', 'nibyo', 'ra', 'hagati', 'isi',
    'gufata', 'uba', 'urubyiruko', 'ubusa', 'erega', 'izo', 'urakoze', 'muntu',
    'mwana', 'inda', 'icyaha', 'ahantu', 'buryo', 'byinshi', 'rusange', 'perezida',
    'wasanga', 'kurusha', 'aka', 'nabo', 'cyangwa', 'uwiteka', 'kugeza', 'nubundi',
    'harimo', 'mba',
}

_WORD_RE = re.compile(r"[a-zàâéèêëîïôûùç']+")

# Below this many analyzable words, the overlap signal is too noisy to trust either way
# (e.g. a 2-word input has a real chance of zero overlap even if it is genuine Kinyarwanda).
MIN_WORDS_TO_JUDGE = 4


def kinyarwanda_overlap(text: str) -> tuple[int, int]:
    """Return (matched_word_count, total_analyzable_word_count) for the given text."""
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]
    matches = sum(1 for w in words if w in KINYARWANDA_COMMON_WORDS)
    return matches, len(words)


def looks_like_kinyarwanda(text: str) -> bool:
    """
    Heuristic only — not a real language-ID model. Returns True (benefit of the doubt)
    whenever the input is too short to judge reliably, or whenever at least one word
    overlaps with the reference vocabulary. Returns False only when there are enough
    words to judge and literally none of them match.
    """
    matches, total = kinyarwanda_overlap(text)
    if total < MIN_WORDS_TO_JUDGE:
        return True
    return matches > 0
