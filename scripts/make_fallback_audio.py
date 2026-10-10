"""Record the "sorry, please try again" clips once, so they can be played even when every API is down.

Run after TTS works:  python -m scripts.make_fallback_audio
Writes assets/sorry_mr.wav, assets/sorry_hi.wav and assets/sorry_en.wav using the configured TTS backend.
"""
import shutil

from agent import tts
from agent.prompts import LANGUAGE
from config import ASSETS_DIR


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    for lang, words in LANGUAGE.items():
        target = ASSETS_DIR / f"sorry_{lang}.wav"
        shutil.move(tts.synthesize(words["sorry"], lang), target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
