#!/usr/bin/env python
"""
gen_voice.py  --  Pre-generate SoundSight's fixed Nepali phrases to audio/ne/*.mp3.

Run ONCE (needs internet -- uses gTTS, a good Nepali voice). After this, /tts plays
these files instead of synthesizing live, so the fixed vocabulary (denominations,
navigation, warnings) sounds clean AND works offline.

    python gen_voice.py            # generate any missing files
    python gen_voice.py --force    # regenerate all

Add phrases by editing nepali_phrases.PHRASES, then re-run.
"""

import argparse

from nepali_phrases import AUDIO_DIR, PHRASES


def main():
    ap = argparse.ArgumentParser(description="Pre-record fixed Nepali phrases (gTTS).")
    ap.add_argument("--force", action="store_true", help="regenerate even if the file exists")
    args = ap.parse_args()

    try:
        from gtts import gTTS
    except ImportError:
        print("gTTS not installed. Run: pip install gTTS  (needs internet to generate).")
        raise SystemExit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    made, skipped, failed = 0, 0, 0
    for phrase, slug in PHRASES.items():
        out = AUDIO_DIR / f"{slug}.mp3"
        if out.exists() and not args.force:
            skipped += 1
            continue
        try:
            gTTS(text=phrase, lang="ne").save(str(out))
            print(f"  [ok]  {slug:16} {phrase}")
            made += 1
        except Exception as exc:   # offline / quota
            print(f"  [FAIL]{slug:16} {phrase}  -> {exc}")
            failed += 1
    print(f"\nDone. {made} generated, {skipped} already present, {failed} failed -> {AUDIO_DIR}/")
    if failed:
        print("Failures are usually no internet. Re-run when online.")


if __name__ == "__main__":
    main()
