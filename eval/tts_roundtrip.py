"""Is the voice understandable? Speak a sentence, listen to it with the speech model, and compare the two texts.

    python -m eval.tts_roundtrip            # says what it would cost and stops
    python -m eval.tts_roundtrip --spend    # speaks and listens for real (paid: see the cost it prints)

A sentence the voice says clearly comes back from the listener almost letter for letter, so the
character error rate (CER) of the round trip is a cheap, automatic check of the voice. It measures
the voice and the listener together, on clean synthetic speech: it says nothing about how well the
listener hears a real student, which needs recorded clips (run_eval.py).
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from eval.run_eval import normalise

EVAL_DIR = Path(__file__).parent
# Sentences of the kind the app speaks. No numerals: the listener may write "35" for "thirty five",
# which is a difference in spelling, not in what was heard.
SENTENCES = {
    "mr": [
        "पृष्ठताण म्हणजे द्रवाच्या पृष्ठभागावर कार्य करणारा ताण होय.",
        "प्रकाशसंश्लेषणात वनस्पती सूर्यप्रकाशाच्या मदतीने अन्न तयार करतात.",
        "तापमान वाढले की वायूचा दाब वाढतो.",
        "हे तुमच्या नोट्समध्ये नाही.",
    ],
    "hi": [
        "पृष्ठ तनाव द्रव की सतह पर लगने वाला बल है।",
        "प्रकाश संश्लेषण में पौधे सूर्य के प्रकाश से भोजन बनाते हैं।",
        "तापमान बढ़ने पर गैस का दाब बढ़ता है।",
        "यह आपके नोट्स में नहीं है।",
    ],
    "en": [
        "Surface tension is the force acting along the surface of a liquid.",
        "Lenz's law says the induced current opposes the change that causes it.",
        "When the temperature rises, the pressure of the gas rises.",
        "This is not in your notes.",
    ],
}
SECONDS_PER_CHAR = 0.075    # roughly how long the voice takes per character, for the cost estimate


def estimate() -> tuple[int, float]:
    """(characters to be spoken, rupees for speaking and listening) for the whole set."""
    from agent import budget

    characters = sum(len(sentence) for group in SENTENCES.values() for sentence in group)
    rupees = characters * budget.VOICE_RUPEES_PER_CHAR + characters * SECONDS_PER_CHAR * budget.LISTENING_RUPEES_PER_SECOND
    return characters, rupees


def run(voice: str) -> list[dict]:
    import jiwer

    from agent import budget, stt, tts
    from agent.audio import load_16k_mono, to_wav_bytes
    from config import SAMPLE_RATE

    results = []
    for language, sentences in SENTENCES.items():
        for sentence in sentences:
            said = tts.clean_for_speech(sentence, language)
            if not budget.daily.spend(len(said) * budget.VOICE_RUPEES_PER_CHAR, "voice"):
                print("Stopped: today's budget for the paid services is used up.")
                return results
            path = Path(tts.synthesize(sentence, language, voice))
            heard_audio = load_16k_mono(str(path))
            path.unlink(missing_ok=True)
            seconds = len(heard_audio) / SAMPLE_RATE
            if not budget.daily.spend(seconds * budget.LISTENING_RUPEES_PER_SECOND, "listening"):
                print("Stopped: today's budget for the paid services is used up.")
                return results
            heard, _ = stt._sarvam(to_wav_bytes(heard_audio), language)    # the paid listener itself, never the free backup
            reference, hypothesis = normalise(said), normalise(heard)
            results.append({
                "language": language, "said": said, "heard": heard, "seconds": round(seconds, 1),
                "cer": round(jiwer.cer(reference, hypothesis), 3), "wer": round(jiwer.wer(reference, hypothesis), 3),
            })  # fmt: skip
            print(f"[{language}] CER {100 * results[-1]['cer']:.0f}%  said: {said}\n{'':>14}heard: {heard}")
    return results


def report(results: list[dict], voice: str) -> None:
    import jiwer

    from agent import budget

    print(f"\nVoice: {voice}. Each sentence was spoken by the voice and written down again by the listener.\n")
    print(f"{'Language':<12}{'CER':>8}{'WER':>8}   sentences")
    print("-" * 44)
    for language in SENTENCES:
        rows = [r for r in results if r["language"] == language]
        if rows:
            said, heard = [normalise(r["said"]) for r in rows], [normalise(r["heard"]) for r in rows]
            print(f"{language:<12}{100 * jiwer.cer(said, heard):>7.1f}%{100 * jiwer.wer(said, heard):>7.1f}%   {len(rows)}")
    characters = sum(len(r["said"]) for r in results)
    seconds = sum(r["seconds"] for r in results)
    rupees = characters * budget.VOICE_RUPEES_PER_CHAR + seconds * budget.LISTENING_RUPEES_PER_SECOND
    print(f"\nSent to Sarvam: {characters} characters spoken, {seconds:.0f} s listened to: about Rs {rupees:.2f}.")


def save(results: list[dict]) -> Path:
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-voice.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spend", action="store_true", help="really call the paid voice and listener")
    parser.add_argument("--voice", default="", help="the speaker to test; the app's default when not given")
    args = parser.parse_args()
    from agent import tts

    voice = args.voice if args.voice in tts.VOICES else tts.DEFAULT_VOICE
    characters, rupees = estimate()
    print(f"{sum(len(group) for group in SENTENCES.values())} sentences, {characters} characters: about Rs {rupees:.2f} of Sarvam credit.")
    if not args.spend:
        print("Nothing was sent. Run again with --spend to do it.")
        return
    results = run(voice)
    if results:
        report(results, voice)
        print(f"Per-sentence results: {save(results)}")


if __name__ == "__main__":
    main()
