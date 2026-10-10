"""How long a turn takes, stage by stage, from the app's own log of real turns.

    python -m eval.latency                           # every logged turn
    python -m eval.latency --since "2026-10-04 13:10+05:30"

Every turn the app answers is logged with the time each stage took (db/log.py). This reads those
rows: nothing is asked and nothing is spent. The log keeps turns for a week, and they are the
questions asked while building and testing the app, so the figures describe this laptop and this
network, not a load test.
"""
import argparse
import statistics
import sys

_TURNS = """
SELECT stt_ms, llm_ms, tool_ms, tts_ms, model FROM turns
WHERE session_id NOT LIKE 'eval%%' AND error IS NULL AND created_at >= %s
ORDER BY id
"""


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def line(name: str, milliseconds: list[float]) -> str:
    if not milliseconds:
        return f"{name:<38}no turns"
    median, slow = statistics.median(milliseconds) / 1000, percentile(milliseconds, 95) / 1000
    return f"{name:<38}{median:>6.1f} s{slow:>8.1f} s   ({len(milliseconds)} turns)"


def stages(rows: list[tuple]) -> dict[str, list[float]]:
    """The times to report, from rows of (stt_ms, llm_ms, tool_ms, tts_ms, model). A turn with stt_ms was spoken."""
    spoken = [row for row in rows if row[0]]
    voiced = [row for row in spoken if row[3]]
    to_text = [(row[0] or 0) + (row[1] or 0) + (row[2] or 0) for row in spoken]
    return {
        "listening": [row[0] for row in spoken],
        "model, both calls": [row[1] for row in spoken if row[1]],
        "search or other tool": [row[2] for row in spoken if row[2]],
        "voice": [row[3] for row in voiced],
        "spoken question to written answer": to_text,
        "spoken question to voice ready": [(row[0] or 0) + (row[1] or 0) + (row[2] or 0) + row[3] for row in voiced],
        "typed question to written answer": [(row[1] or 0) + (row[2] or 0) for row in rows if not row[0] and row[1]],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2000-01-01", help="only turns logged from this date and time on")
    args = parser.parse_args()
    from db.conn import run

    rows = run(_TURNS, (args.since,))
    print(f"{len(rows)} logged turns without an error since {args.since}\n")
    print(f"{'Stage':<38}{'median':>8}{'p95':>10}")
    print("-" * 70)
    for name, times in stages(rows).items():
        print(line(name, times))
    models = [row[4] for row in rows if row[4]]
    print("\nWritten by: " + ", ".join(f"{model} {models.count(model)}" for model in sorted(set(models))))


if __name__ == "__main__":
    main()
