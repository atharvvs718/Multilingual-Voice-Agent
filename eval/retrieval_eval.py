"""Measure the search on its own: when the page that answers a question is known, how often is it among the passages found?

    python -m eval.retrieval_eval --book path/to/textbook.pdf

questions.csv columns: id, kind (term or paraphrase), question, pages. `pages` lists the printed
pages that answer the question, separated by |. A "term" question names the thing asked about
("What is surface tension?"); a "paraphrase" describes it without the book's word for it ("Why
can a greased needle float on water?").

Three ways of ranking are compared on the same questions: by meaning only (embeddings), by words
only, and the app's own search, which adds the two rankings together. No language model is called
and nothing goes to a paid service, so this can be run as often as needed.

The agent sees the first 3 passages (5 for a broad question), so "top 3" is the figure that matters.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
SESSION = "eval-retrieval"
DEPTH = 10              # how far down each ranking is looked at
CUTS = (1, 3, 5)

_BY_MEANING = """
SELECT coalesce(page_label, page::text) FROM note_chunks
WHERE session_id = %(session)s
ORDER BY embedding <=> %(vector)s
LIMIT %(k)s
"""
_BY_WORDS = """
SELECT coalesce(page_label, page::text) FROM note_chunks
WHERE session_id = %(session)s AND to_tsvector('english', text) @@ plainto_tsquery('english', %(query)s)
ORDER BY ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', %(query)s)) DESC
LIMIT %(k)s
"""


def first_hit(found: list[str], right: set[str]) -> int | None:
    """Position (from 1) of the first passage that is on one of the right pages; None if there is none."""
    return next((place for place, page in enumerate(found, start=1) if str(page) in right), None)


def summarise(places: list[int | None]) -> dict[str, float]:
    """Share of questions answered within the first 1, 3 and 5 passages, and the mean of 1/position (MRR)."""
    total = len(places) or 1
    scores = {f"top {cut}": sum(place is not None and place <= cut for place in places) / total for cut in CUTS}
    scores["MRR"] = sum(1 / place for place in places if place) / total
    return scores


def line(name: str, places: list[int | None]) -> str:
    scores = summarise(places)
    hits = "".join(f"{100 * scores[f'top {cut}']:>7.0f}%" for cut in CUTS)
    return f"{name:<28}{hits}{scores['MRR']:>9.2f}   ({len(places)} questions)"


def run(args: argparse.Namespace) -> tuple[list[dict], str]:
    # Imported here: the database and the embedding model are only needed for a real run, not by the tests.
    from agent.tools.notes import count_chunks, has_source, search_notes
    from db.conn import run as query_database
    from rag.embed import embed_query
    from rag.ingest import delete_session, ingest

    book = Path(args.book)
    if not has_source(SESSION, book.name):
        delete_session(SESSION)
        added = ingest(str(book), SESSION)
        print(f"indexed {book.name}: {added.pages} pages, {added.chunks} passages")
    described = f"{book.name} ({count_chunks(SESSION)} passages)"

    with open(args.questions, encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    results = []
    for row in questions:
        right = {page.strip() for page in row["pages"].split("|") if page.strip()}
        asked = {"session": SESSION, "query": row["question"], "k": DEPTH}
        by_meaning = [found[0] for found in query_database(_BY_MEANING, {**asked, "vector": embed_query(row["question"])})]
        by_words = [found[0] for found in query_database(_BY_WORDS, asked)]
        by_both = [str(passage["page"]) for passage in search_notes(SESSION, row["question"], k=DEPTH)]
        results.append({
            "id": row["id"], "kind": row["kind"], "question": row["question"], "pages": row["pages"],
            "meaning": first_hit(by_meaning, right), "words": first_hit(by_words, right), "both": first_hit(by_both, right),
            "found_first_3": "|".join(by_both[:3]),
        })  # fmt: skip
    if not args.keep:
        delete_session(SESSION)
    return results, described


def report(results: list[dict], described: str) -> None:
    print(f"\n{len(results)} questions on {described}. A question counts when a passage from its answer page is found.\n")
    print(f"{'Ranked by':<28}{'top 1':>8}{'top 3':>8}{'top 5':>8}{'MRR':>9}")
    print("-" * 78)
    print(line("meaning only (embeddings)", [r["meaning"] for r in results]))
    print(line("words only", [r["words"] for r in results]))
    print(line("both: the app's search", [r["both"] for r in results]))
    print()
    for kind in sorted({r["kind"] for r in results}):
        for name, way in (("meaning", "meaning"), ("words", "words"), ("both", "both")):
            print(line(f"{kind}: {name}", [r[way] for r in results if r["kind"] == kind]))
    missed = [r for r in results if not r["both"] or r["both"] > 3]
    print(f"\nNot in the app's first 3 passages: {len(missed)}")
    for r in missed:
        print(f"  {r['id']} {r['question']}  (answer on {r['pages']}; found {r['found_first_3']}; right page at position {r['both'] or 'none in 10'})")


def save(results: list[dict]) -> Path:
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-retrieval.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True, help="the file the questions are about")
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.csv"))
    parser.add_argument("--keep", action="store_true", help="leave the book indexed, so the next run starts at once")
    args = parser.parse_args()
    results, described = run(args)
    report(results, described)
    print(f"\nPer-question results: {save(results)}")


if __name__ == "__main__":
    main()
