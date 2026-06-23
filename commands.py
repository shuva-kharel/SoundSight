"""
commands.py  --  ONE shared voice-command parser for SoundSight.

Single source of truth used by BOTH front-ends:
  * laptop: index.html does Web Speech recognition and POSTs the transcript to
    server.py's /command, which calls parse_command().
  * Pi:     pi_app.py runs Vosk offline recognition and calls parse_command() directly.

    parse_command(text) -> {
        "action":  one of ACTIONS, or "none",
        "target":  object word (find), person name (remember), or amount (pay), else None,
        "wake":    True if the wake word was present in the utterance,
        "matched": True if an action was recognized,
        "raw":     the original text,
    }

Matching is fuzzy/keyword-based (NOT exact strings) and BILINGUAL (English + Nepali,
Devanagari). Precedence is set by _ACTION_ORDER. Add keywords freely.
"""

import difflib
import re

ACTIONS = (
    # core modes
    "describe", "read", "money", "navigate", "find",
    # conversation / TTS control
    "repeat", "stop", "louder", "softer", "slower", "faster",
    # street crossing
    "cross",
    # faces
    "who", "remember",
    # label reading
    "label",
    # money tally / change + multi-note count
    "money_add", "money_total", "money_clear", "money_undo", "money_pay", "money_count",
    # meta
    "help", "sos",
)

# Wake word (+ common mishearings; recognizers often hear "site/side" for "sight").
WAKE_WORDS = [
    "hey sight", "hey site", "hey sights", "hey side", "hi sight", "hey psych",
    "okay sight", "ok sight", "a sight", "हे साइट", "साइट",
]

# action -> trigger phrases / keywords (English + Nepali). Phrases (with spaces) are
# substring-matched; single words are word- and fuzzy-matched.
KEYWORDS = {
    "stop":        ["stop", "quiet", "shut up", "silence", "be quiet", "enough",
                    "रोक", "रोक्नु", "चुप", "बन्द"],
    # --- money tally (check these multi-word ones before single-note "money") ----
    "money_add":   ["add this note", "add note", "add this", "count this note", "add this one",
                    "नोट जोड", "यो जोड", "जोड"],
    "money_count": ["count the notes", "count the money", "count notes", "count money",
                    "count all", "how many notes", "count", "गन्नुहोस्", "नोट गन", "कति वटा नोट"],
    "money_total": ["total", "what's the total", "how much total", "grand total", "sum",
                    "कुल", "जम्मा", "कति भयो"],
    "money_clear": ["clear", "reset", "start over", "clear total", "खाली", "रिसेट", "सफा"],
    "money_undo":  ["undo", "undo last", "remove last", "take back", "हटाउ", "फिर्ता"],
    "money_pay":   ["i need to pay", "calculate change", "change for", "i have to pay",
                    "pay", "change", "तिर्नु", "फिर्ता कति", "चेन्ज"],
    # --- crossing ---------------------------------------------------------------
    "cross":       ["can i cross", "is it safe to cross", "traffic light", "should i cross",
                    "cross the road", "cross", "crossing", "light",
                    "बाटो काट", "काट्न", "ट्राफिक", "बत्ती", "हरियो बत्ती"],
    # --- faces ------------------------------------------------------------------
    "remember":    ["remember this person", "remember this person as", "remember this face",
                    "save this person", "this person is", "यो मान्छे सम्झ", "नाम राख"],
    "who":         ["who is this", "who's this", "who is here", "who's here", "who is that",
                    "who's there", "recognize", "यो को हो", "को छ", "को हो"],
    # --- label reading ----------------------------------------------------------
    "label":       ["read the label", "read label", "check expiry", "expiry", "expiration",
                    "expiry date", "is this expired", "check the date", "best before",
                    "लेबल", "म्याद", "कहिले सम्म", "एक्सपायरी"],
    # --- single-note money ("how much is this") --------------------------------
    "money":       ["how much", "how much money", "how much is this", "what note", "which note",
                    "money", "rupee", "rupees", "cash", "what's this worth",
                    "कति", "कति हो", "पैसा", "नोट", "कति रुपैयाँ", "कति पैसा"],
    # --- read text --------------------------------------------------------------
    "read":        ["read this", "read it", "read", "reading", "read text",
                    "पढ", "पढ्", "पढ्नु", "पढ्नुहोस्", "लेख पढ"],
    # --- navigate ---------------------------------------------------------------
    "navigate":    ["navigate", "start walking", "start navigation", "guide me", "guide",
                    "walk", "walking", "lead me", "हिँड", "हिंड", "बाटो देखाउ", "निर्देशन"],
    # --- describe ---------------------------------------------------------------
    "describe":    ["what's in front", "what is in front", "whats in front", "in front of me",
                    "what's ahead", "what is ahead", "whats ahead", "describe", "look",
                    "look around", "what's around", "what do you see",
                    "अगाडि के छ", "वर्णन", "के देख", "के छ अगाडि", "हेर"],
    # --- TTS control ------------------------------------------------------------
    "louder":      ["louder", "speak up", "volume up", "more volume", "ठूलो", "चर्को"],
    "softer":      ["softer", "quieter", "volume down", "less volume", "lower volume",
                    "सानो स्वर", "बिस्तारै बोल"],
    "slower":      ["slower", "slow down", "too fast", "ढिलो", "बिस्तारै"],
    "faster":      ["faster", "speed up", "too slow", "छिटो", "चाँडो"],
    # --- repeat -----------------------------------------------------------------
    "repeat":      ["repeat", "say again", "say that again", "again", "come again",
                    "दोहोर्याउ", "फेरि भन", "फेरि"],
    # --- emergency (check BEFORE help so "help me" -> SOS, not the command list) -
    "sos":         ["sos", "help me", "emergency", "i need help", "call for help",
                    "save me", "बचाउ", "आपत", "सहयोग चाहियो", "गुहार"],
    # --- help -------------------------------------------------------------------
    "help":        ["help", "what can you do", "what can i say", "commands", "options",
                    "मद्दत", "के गर्न सक्छौ", "के भन्न सक्छु"],
}

# find triggers handled separately (need target extraction)
FIND_TRIGGERS = ["find my", "find the", "find a", "find", "where is my", "where is the",
                 "where is", "where's my", "where's", "locate my", "locate", "look for",
                 "खोज", "कहाँ छ"]
# "remember this person as Ram" / "this person is Ram" -> capture the name after these
NAME_TRIGGERS = ["remember this person as", "remember this as", "remember this person",
                 "save this person as", "this person is", "name is", "call them",
                 "नाम", "सम्झ"]
_STOPWORDS = {"my", "the", "a", "an", "please", "is", "at", "this", "that", "to", "for"}

# Precedence: most specific / safety-relevant first.
_ACTION_ORDER = [
    "stop", "sos", "help", "remember", "money_add", "money_count", "money_total",
    "money_clear", "money_undo", "money_pay", "cross", "who", "label", "find", "money",
    "read", "navigate", "describe", "louder", "softer", "slower", "faster", "repeat",
]


def _normalize(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^\wऀ-ॿ]+", " ", text)  # keep Devanagari, drop punctuation
    return re.sub(r"\s+", " ", text).strip()


def _detect_and_strip_wake(t):
    for w in WAKE_WORDS:
        if w in t:
            return True, _normalize(t.replace(w, " "))
    # fuzzy: catch "hey sighed", "a site" etc. on the first two words
    head = " ".join(t.split()[:2])
    if head and difflib.get_close_matches(head, WAKE_WORDS, n=1, cutoff=0.8):
        return True, _normalize(" ".join(t.split()[2:]))
    return False, t


def _has(t, words, keyword):
    """True if keyword (phrase = substring; single word = word/fuzzy) is in t."""
    if " " in keyword:
        return keyword in t
    if keyword in words:
        return True
    if keyword.isascii():   # fuzzy only for ASCII (Devanagari needs exact-ish)
        return bool(difflib.get_close_matches(keyword, words, n=1, cutoff=0.85))
    return False


def _match(t, words, action):
    return any(_has(t, words, kw) for kw in KEYWORDS.get(action, ()))


def _extract_after(t, triggers):
    """Return the cleaned words after the first matching trigger, or None."""
    for trig in triggers:
        idx = t.find(trig)
        if idx != -1:
            rest = t[idx + len(trig):].strip().split()
            rest = [w for w in rest if w not in _STOPWORDS]
            if rest:
                return " ".join(rest[:3])
            return None
    return None


def _extract_amount(t):
    m = re.search(r"\d+", t.replace(",", ""))
    return m.group(0) if m else None


def parse_command(text):
    raw = text or ""
    t = _normalize(raw)
    wake, t = _detect_and_strip_wake(t)
    words = set(t.split())

    action, target = "none", None
    if not t:
        pass
    elif _match(t, words, "stop"):
        action = "stop"
    elif _match(t, words, "remember"):
        action, target = "remember", _extract_after(t, NAME_TRIGGERS)
    elif _match(t, words, "money_pay"):
        action, target = "money_pay", _extract_amount(t)
    else:
        # find needs target extraction; try it before the generic single-word checks
        tgt = _extract_after(t, FIND_TRIGGERS)
        if tgt is not None and not _match(t, words, "money") and not _match(t, words, "label"):
            action, target = "find", tgt
        else:
            for a in _ACTION_ORDER:
                if a in ("stop", "remember", "money_pay", "find"):
                    continue
                if _match(t, words, a):
                    action = a
                    break

    return {"action": action, "target": target, "wake": wake,
            "matched": action != "none", "raw": raw}
