"""
nepali_phrases.py  --  Curated Nepali phrase table + normalization for SoundSight TTS.

SoundSight's spoken Nepali is a SMALL FIXED set (denominations, navigation building
blocks, warnings). Pre-recording those once with a good TTS and PLAYING THE FILE beats
live synthesis and works offline -- so /tts tries, in order:
    pre-recorded file (audio/ne/<slug>.mp3)  ->  gTTS (online)  ->  espeak-ng (offline)

`gen_voice.py` synthesizes the table into audio/ne/. `normalize_ne()` cleans OCR'd
Devanagari (zero-width joiners, stray spaces) before lookup / synthesis.
"""

import re
import unicodedata
from pathlib import Path

AUDIO_DIR = Path("audio/ne")

# Canonical Devanagari phrase -> slug (filename stem under audio/ne/).
# Keep these in sync with vision_core.BANKNOTE_SPEECH / NE_ZONES and the warnings.
PHRASES = {
    # denominations
    "पाँच रुपैयाँ": "rs5",
    "दस रुपैयाँ": "rs10",
    "बीस रुपैयाँ": "rs20",
    "पचास रुपैयाँ": "rs50",
    "एक सय रुपैयाँ": "rs100",
    "पाँच सय रुपैयाँ": "rs500",
    "एक हजार रुपैयाँ": "rs1000",
    # money state
    "कुनै नोट छैन": "no_note",
    "कुनै नोट देखिएन। राम्रो उज्यालोमा नोट स्थिर राख्नुहोस्।": "no_note_long",
    # navigation building blocks
    "अगाडि": "ahead",
    "बायाँतिर": "left",
    "दायाँतिर": "right",
    "मानिस": "person",
    "नजिक": "close",
    "ठीक अगाडि": "right_in_front",
    # warnings
    "सावधान": "careful",
    "होसियार": "hosiyar",
    # crossing
    "रातो बत्ती, कृपया पर्खनुहोस्।": "red_wait",
    "हरियो बत्ती, काट्न ठीक देखिन्छ।": "green_cross",
}


def normalize_ne(text):
    """NFC-normalize, drop zero-width joiners, collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("‍", "").replace("‌", "")  # ZWJ / ZWNJ
    return re.sub(r"\s+", " ", text).strip()


def clean_ocr_ne(text):
    """Light cleanup of OCR'd Nepali before TTS: normalize, strip isolated stray
    Latin letters/symbols that EasyOCR sometimes injects between Devanagari, so the
    voice doesn't read 'x' / 'l' mid-sentence. Keeps Latin words that stand alone."""
    text = normalize_ne(text)
    # remove a lone Latin letter wedged between Devanagari/space (OCR noise)
    text = re.sub(r"(?<=[ऀ-ॿ\s])[A-Za-z](?=[ऀ-ॿ\s])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prerecorded_path(text):
    """Return the audio file for a known phrase, or None. Matches on normalized text."""
    slug = PHRASES.get(normalize_ne(text))
    if not slug:
        return None
    for ext in (".mp3", ".wav"):
        p = AUDIO_DIR / (slug + ext)
        if p.exists():
            return p
    return None
