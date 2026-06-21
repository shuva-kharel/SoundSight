"""
labels.py  --  Medicine / packaged-goods label understanding for SoundSight Read mode.

Runs ON TOP of the existing /ocr + EasyOCR result (text in, spoken summary out) -- it
does NOT change normal Read. parse_label() pulls out, and speaks IN THIS ORDER:
  1. product / medicine name  (the most prominent line)
  2. expiry status            ("This expired 2 months ago" / "Expires next month")
  3. dosage / quantity        ("500mg", "twice daily")

Handles many date formats and BOTH calendars: Gregorian (AD) and Bikram Sambat (BS,
the Nepali calendar ~56.7 years ahead). EXP vs MFG is decided by nearby keywords.
Pure Python (datetime + regex), so it runs on the Pi too.
"""

import datetime
import logging
import re

log = logging.getLogger("soundsight.labels")

EXP_KEYWORDS = ["exp", "expiry", "expires", "expiration", "use by", "best before", "bb",
                "use before", "म्याद", "समाप्ति"]
MFG_KEYWORDS = ["mfg", "mfd", "manufactured", "manufacture", "mfg date", "packed", "उत्पादन"]
DOSAGE_RE = re.compile(
    r"\b\d+\s?(?:mg|mcg|µg|g|ml|iu|%)\b"
    r"|\b(?:once|twice|thrice|\d+\s*times?)\s+(?:a\s+)?(?:day|daily|week|weekly)\b"
    r"|\b\d+\s*(?:tablet|tablets|capsule|capsules|tab|caps|drops?|spoons?)\b",
    re.IGNORECASE,
)
MONTHS = {m.lower(): i for i, m in enumerate(
    ["", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

# Heuristic: a 4-digit year >= 2070 is almost certainly Bikram Sambat (BS) on a
# Nepali label; BS - 56.7y ~ AD. (2024 AD ~ 2081 BS.) We flag it as BS and convert
# approximately, telling the user it's the Nepali calendar.
BS_YEAR_MIN = 2070
BS_TO_AD_OFFSET = 56   # approximate; good enough for "expired / not expired" judgement


def _month_from_token(tok):
    tok = tok.lower()[:3]
    return MONTHS.get(tok)


def _candidate_dates(text):
    """Yield (date, is_bs, span_start) for every date-like substring found."""
    t = text
    out = []
    # numeric DD/MM/YYYY or D-M-YY or YYYY/MM/DD or MM/YYYY  (sep: / - .)
    # middle group allows up to 4 digits so "03/2026" (MM/YYYY) is caught too.
    for m in re.finditer(r"\b(\d{1,4})[/\-.](\d{1,4})(?:[/\-.](\d{1,4}))?\b", t):
        a, b, c = m.group(1), m.group(2), m.group(3)
        d = _build_numeric(a, b, c)
        if d:
            out.append((d[0], d[1], m.start()))
    # "EXP 2026-08", "Aug 2026", "08 Aug 2026", "2026 Aug"
    for m in re.finditer(r"\b(\d{1,2})?\s*([A-Za-z]{3,9})\s*(\d{2,4})\b", t):
        mon = _month_from_token(m.group(2))
        if mon:
            day = int(m.group(1)) if m.group(1) else 1
            yr = _norm_year(m.group(3))
            d = _safe_date(yr, mon, day)
            if d:
                out.append((d[0], d[1], m.start()))
    return out


def _norm_year(y):
    y = int(y)
    if y < 100:           # two-digit year -> 2000s
        y += 2000
    return y


def _safe_date(year, month, day):
    is_bs = year >= BS_YEAR_MIN
    ad_year = year - BS_TO_AD_OFFSET if is_bs else year
    try:
        return datetime.date(ad_year, max(1, min(12, month)), max(1, min(28, day))), is_bs
    except (ValueError, TypeError):
        return None


def _build_numeric(a, b, c):
    a, b = int(a), int(b)
    if c is None:                       # MM/YYYY or YYYY/MM
        if a > 31:                      # YYYY/MM
            return _safe_date(_norm_year(a), b, 1)
        return _safe_date(_norm_year(b), a, 1)  # MM/YYYY
    c = int(c)
    if a > 31:                          # YYYY-MM-DD
        return _safe_date(_norm_year(a), b, c)
    return _safe_date(_norm_year(c), b, a)      # DD-MM-YYYY


def _classify_date_kind(text, span_start):
    """Look ~25 chars before the date for EXP/MFG keywords."""
    window = text[max(0, span_start - 25):span_start].lower()
    if any(k in window for k in EXP_KEYWORDS):
        return "exp"
    if any(k in window for k in MFG_KEYWORDS):
        return "mfg"
    return "unknown"


def _months_between(d, today):
    return (d.year - today.year) * 12 + (d.month - today.month)


def _expiry_phrase(d, is_bs, today):
    cal_en = " (Nepali calendar)" if is_bs else ""
    cal_ne = " (विक्रम सम्वत्)" if is_bs else ""
    months = _months_between(d, today)
    if d < today:
        n = abs(months) or 1
        return (f"Warning: this expired about {n} month{'s' if n != 1 else ''} ago{cal_en}.",
                f"सावधान: यो करिब {n} महिना अघि म्याद सकियो{cal_ne}।")
    if months <= 1:
        return (f"This expires very soon, within a month{cal_en}.",
                f"यसको म्याद चाँडै, एक महिनाभित्र सकिन्छ{cal_ne}।")
    return (f"This is good for about {months} more months{cal_en}.",
            f"यो करिब {months} महिना सम्म ठीक छ{cal_ne}।")


def _product_name(text):
    """Most prominent product line: the first reasonably long, mostly-alphabetic
    line (OCR returns top-to-bottom order from the server's box sorting)."""
    for line in re.split(r"[\n.;|]| {3,}", text):
        s = line.strip()
        letters = sum(c.isalpha() for c in s)
        if len(s) >= 4 and letters >= 3 and letters >= len(s) * 0.5:
            return s
    return ""


def parse_label(text, today=None):
    """Parse an OCR'd label. Returns dict with product/expiry/dosage + a combined
    bilingual spoken summary (product -> expiry -> dosage)."""
    today = today or datetime.date.today()
    text = text or ""

    # --- expiry vs mfg dates ------------------------------------------------- #
    exp_date = exp_bs = None
    seen = _candidate_dates(text)
    for d, is_bs, start in seen:
        if _classify_date_kind(text, start) == "exp":
            exp_date, exp_bs = d, is_bs
            break
    if exp_date is None and seen:        # no EXP keyword: assume the LATEST date is expiry
        d, is_bs, _ = max(seen, key=lambda x: x[0])
        exp_date, exp_bs = d, is_bs

    if exp_date is not None:
        exp_en, exp_ne = _expiry_phrase(exp_date, exp_bs, today)
        exp_iso = exp_date.isoformat()
    else:
        exp_en, exp_ne, exp_iso = "Expiry date not found.", "म्याद मिति फेला परेन।", None

    # --- dosage / quantity --------------------------------------------------- #
    dosages = [m.group(0).strip() for m in DOSAGE_RE.finditer(text)]
    dosages = list(dict.fromkeys(dosages))[:3]   # dedupe, keep first few

    # --- product name -------------------------------------------------------- #
    product = _product_name(text)

    parts_en, parts_ne = [], []
    if product:
        parts_en.append(product)
        parts_ne.append(product)
    parts_en.append(exp_en)
    parts_ne.append(exp_ne)
    if dosages:
        parts_en.append("Dosage: " + ", ".join(dosages))
        parts_ne.append("मात्रा: " + ", ".join(dosages))

    return {
        "product": product,
        "expiry": exp_iso,
        "expiry_is_bs": bool(exp_bs),
        "expiry_text": exp_en,
        "dosage": dosages,
        "text": " ".join(parts_en),
        "text_ne": " ".join(parts_ne),
    }
