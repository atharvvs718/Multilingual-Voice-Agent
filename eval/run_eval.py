"""Measure the whole loop on a labelled test set and print one table.

    python -m eval.run_eval --notes notes/                     # audio clips through the full pipeline
    python -m eval.run_eval --notes notes/ --mode text         # reference transcripts, no STT (experiment 2)
    python -m eval.run_eval --notes notes/ --no-hint           # speech to text without the language hint (experiment 3)
    python -m eval.run_eval --notes notes/ --model llama-3.3-70b-versatile   # another LLM (experiment 1)
    python -m eval.run_eval --notes book.pdf --mode text --rows 1-30 --delay 60   # half the set, slowly

labels.csv columns: clip_id, language, reference_transcript, expected_tool, expected_query_or_chapter,
expected_page, answerable, conversation. expected_page may list several pages that answer the question,
separated by |. Rows that share a `conversation` value run in order in one chat session, which is how
follow-ups and quiz answers are tested. TTS is skipped: it is not what is being measured, and it
costs credit.

The free tier of the model service allows about 8,000 tokens a minute and 200,000 a day, and one turn
uses 4,000 to 8,000. Past either limit the weaker backup model answers, which is not what should be
measured: --delay 60 keeps to the minute limit, --rows splits the set over two days, and the report
says how many turns the main model answered.
"""
import argparse
import csv
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
EVAL_SESSION = "eval"
_PUNCTUATION = re.compile(r"[^\w\sऀ-ॿ]|[।॥_]")
_JOINERS = str.maketrans("", "", "\u200d\u200c")      # the invisible joiners some fonts put inside words


def normalise(text: str) -> str:
    """Lower-case and strip punctuation so the error rates count wrong words and letters, not commas and dandas.

    The same letter can be stored in more than one way in Devanagari, and some fonts put invisible
    joiners inside words: both are made the same on the two sides first.
    """
    text = unicodedata.normalize("NFC", text).translate(_JOINERS)
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def tool_correct(expected: str, actual: str | None) -> bool:
    return (expected.strip() or "none") == (actual or "none")


def argument_correct(row: dict, tool_args: dict | None) -> bool | None:
    """Exact match for a quiz chapter. Search queries need a human, so they return None."""
    if row["expected_tool"] != "quiz_me" or not row["expected_query_or_chapter"].strip():
        return None
    return str((tool_args or {}).get("chapter")) == row["expected_query_or_chapter"].strip()


def citation_correct(expected_page: str, sources: list[dict], reply: str) -> bool | None:
    """One of the pages that answer the question was retrieved and is named in the reply."""
    pages = [page.strip() for page in expected_page.split("|") if page.strip()]
    if not pages:
        return None
    return any(
        any(str(s["page"]) == page for s in sources) and bool(re.search(rf"(?<!\d){re.escape(page)}(?!\d)", reply))
        for page in pages
    )


def rows_wanted(rows: list[dict], wanted: str) -> list[dict]:
    """The rows of a slice written as "first-last", counted from 1; all of them when no slice is given."""
    if not wanted:
        return rows
    first, _, last = wanted.partition("-")
    return rows[int(first) - 1: int(last or first)]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def rate(flags: list[bool | None]) -> str:
    """'87% (26/30)' over the rows where the check applies."""
    known = [f for f in flags if f is not None]
    return f"{100 * sum(known) / len(known):.0f}% ({sum(known)}/{len(known)})" if known else "n/a"


def run(args: argparse.Namespace) -> list[dict]:
    # Imported here so --model can change LLM_MODEL before config is read.
    from agent.errors import BadClip
    from agent.lang import language_matches
    from agent.pipeline import answer_turn
    from agent.prompts import LANGUAGE
    from agent.session import Session
    from rag.ingest import IngestError, delete_session, ingest

    delete_session(EVAL_SESSION)
    notes_languages: set[str] = set()
    notes = Path(args.notes)
    for path in sorted(notes.iterdir()) if notes.is_dir() else [notes]:
        if path.suffix.lower() in (".pdf", ".txt", ".md"):
            try:
                added = ingest(str(path), EVAL_SESSION)
                notes_languages.add(added.language)
                print(f"indexed {path.name}: {added.pages} pages, {added.chunks} passages, in {added.language}")
            except IngestError as error:
                print(f"skipped {path.name}: {error}")

    with open(args.labels, encoding="utf-8-sig", newline="") as f:
        rows = rows_wanted(list(csv.DictReader(f)), args.rows)
    sessions: dict[str, Session] = {}
    results = []
    for number, row in enumerate(rows, start=1):
        # Notes are shared (same session id); chat history is per conversation.
        key = row.get("conversation", "").strip() or row["clip_id"]
        session = sessions.setdefault(key, Session(id=EVAL_SESSION, notes_languages=sorted(notes_languages)))
        hint = None if args.no_hint else {"mr": "mr", "hi": "hi"}.get(row["language"])
        try:
            if args.mode == "audio":
                turn = answer_turn(session, audio_path=str(EVAL_DIR / "clips" / row["clip_id"]), hint=hint)
            else:
                turn = answer_turn(session, text=row["reference_transcript"], hint=hint)
        except BadClip as error:
            print(f"[{number}/{len(rows)}] {row['clip_id']}: rejected ({error})")
            continue

        # A code-mixed clip has no single right answer; the reply should match what was detected.
        expected_language = turn.language if row["language"] == "mixed" else row["language"]
        answerable = row["answerable"].strip().lower() not in ("0", "no", "false")
        declined = any(LANGUAGE[lang]["missing"].rstrip(".") in turn.reply_text for lang in LANGUAGE)
        results.append({
            "clip_id": row["clip_id"],
            "language": row["language"],
            "reference": row["reference_transcript"],
            "transcript": turn.transcript,
            "expected_tool": row["expected_tool"].strip() or "none",
            "tool": turn.tool_name or "none",
            "tool_args": turn.tool_args,
            "tool_ok": tool_correct(row["expected_tool"], turn.tool_name),
            "argument_ok": argument_correct(row, turn.tool_args),
            "citation_ok": citation_correct(row["expected_page"], turn.sources, turn.reply_text) if answerable else None,
            "declined_ok": None if answerable else declined,
            "language_ok": language_matches(turn.reply_text, expected_language),
            "reply": turn.reply_text,
            "model": turn.model,
            "tokens": turn.tokens_in + turn.tokens_out,
            "stt_ms": turn.stt_ms, "llm_ms": turn.llm_ms, "tool_ms": turn.tool_ms, "total_ms": turn.total_ms,
            "error": turn.error or "",
        })  # fmt: skip
        print(f"[{number}/{len(rows)}] {row['clip_id']}: tool={results[-1]['tool']} ok={results[-1]['tool_ok']}")
        time.sleep(args.delay)      # stay under the free tier's tokens-per-minute limit
    delete_session(EVAL_SESSION)
    return results


def report(results: list[dict], mode: str) -> None:
    print(f"\n{'Metric':<34}Value")
    print("-" * 60)
    if mode == "audio":
        import jiwer

        for language in sorted({r["language"] for r in results}):
            subset = [r for r in results if r["language"] == language and r["reference"].strip()]
            if subset:
                said = [normalise(r["reference"]) for r in subset]
                heard = [normalise(r["transcript"]) for r in subset]
                # By letters as well as by words: one wrong vowel sign makes a whole Marathi word "wrong".
                print(f"{'CER ' + language:<34}{100 * jiwer.cer(said, heard):.1f}% ({len(subset)} clips)")
                print(f"{'WER ' + language:<34}{100 * jiwer.wer(said, heard):.1f}%")
    print(f"{'Tool selection accuracy':<34}{rate([r['tool_ok'] for r in results])}")
    print(f"{'Quiz chapter accuracy':<34}{rate([r['argument_ok'] for r in results])}")
    print(f"{'Citation accuracy':<34}{rate([r['citation_ok'] for r in results])}")
    print(f"{'Declined when not in notes':<34}{rate([r['declined_ok'] for r in results])}")
    print(f"{'Language match':<34}{rate([r['language_ok'] for r in results])}")
    print(f"{'Turns with an error':<34}{rate([bool(r['error']) for r in results])}")
    from config import LLM_MODEL  # read here, after --model has had its say

    print(f"{'Answered by the main model':<34}{rate([r['model'] == LLM_MODEL if r['model'] else None for r in results])}")
    for stage in ("stt_ms", "llm_ms", "tool_ms", "total_ms"):
        values = [r[stage] for r in results if r[stage]]
        if values:
            print(f"{stage + ' median / p95':<34}{statistics.median(values):.0f} / {percentile(values, 95):.0f}")
    tokens = [r["tokens"] for r in results]
    print(f"{'Tokens per turn mean / max':<34}{statistics.mean(tokens):.0f} / {max(tokens)}")
    print("\nSearch queries and grounding need a manual read: see the tool_args and reply columns in the results file.")


def save(results: list[dict], mode: str) -> Path:
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(EVAL_DIR / "labels.csv"))
    parser.add_argument("--notes", required=True, help="file or folder of notes the questions are about")
    parser.add_argument("--mode", choices=("audio", "text"), default="audio")
    parser.add_argument("--no-hint", action="store_true", help="do not pass the language hint to the speech model")
    parser.add_argument("--model", help="LLM to test instead of the configured default")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds to wait between clips")
    parser.add_argument("--rows", default="", help='only these rows of the labels, as "first-last", counted from 1')
    args = parser.parse_args()
    if args.model:
        os.environ["LLM_MODEL"] = args.model
        os.environ["LLM_FALLBACK_MODEL"] = args.model   # no silent fallback while comparing models

    results = run(args)
    if not results:
        sys.exit("No clips were evaluated.")
    report(results, args.mode)
    print(f"\nPer-clip results: {save(results, args.mode)}")


if __name__ == "__main__":
    main()
