"""Queries over one session's notes. Every query is scoped by session_id: that is the privacy boundary.

In every passage returned here, "page" is the page to cite: the number printed on the page
when the book has one, otherwise its position in the file.

Try it by hand:  python -m agent.tools.notes "प्रकाशसंश्लेषण म्हणजे काय" --session dev
"""
import argparse
import re

from agent.lang import HINDI_WORDS, MARATHI_WORDS, detect_language
from db.conn import run
from rag.embed import embed_query

TOP_K = 3
WIDE_K = 5      # for a broad question that gets a longer reply: more passages to draw on
RRF_K = 60      # the usual constant for reciprocal rank fusion: how quickly a low rank stops counting
_PASSAGE = "source, coalesce(page_label, page::text) AS page, chapter, text"
_FIELDS = ("source", "page", "chapter", "text")


def _hybrid(words: str, wanted: str) -> str:
    """The search: every passage of the session ranked by meaning and by words, the two ranks added together."""
    return f"""
WITH ranked AS (
    SELECT id,
           1 - (embedding <=> %(vector)s) AS score,
           row_number() OVER (ORDER BY embedding <=> %(vector)s) AS by_meaning,
           CASE WHEN {words} @@ {wanted} THEN
               row_number() OVER (ORDER BY ts_rank_cd({words}, {wanted}) DESC)
           END AS by_words
    FROM note_chunks
    WHERE session_id = %(session)s
)
SELECT {_PASSAGE}, score
FROM note_chunks JOIN ranked USING (id)
ORDER BY 1.0 / (%(rrf)s + by_meaning) + coalesce(1.0 / (%(rrf)s + by_words), 0) DESC
LIMIT %(k)s
"""


# English: a passage counts when it holds every content word of the query, compared by word stem.
_ENGLISH = _hybrid("to_tsvector('english', text)", "plainto_tsquery('english', %(query)s)")
# Marathi and Hindi: a passage counts when it holds any content word of the query, and more of them rank
# higher. Words are compared by how they begin, because the endings change (तत्त्वज्ञान, तत्त्वज्ञानाची), and with
# the spelling folded on both sides (see _fold).
_FOLDED_TEXT = "regexp_replace(translate(text, E'\\u200d\\u200c', ''), '(.)्\\1', '\\1', 'g')"
_INDIC = _hybrid(f"to_tsvector('simple', {_FOLDED_TEXT})", "to_tsquery('simple', %(words)s)")

_JOINERS = str.maketrans("", "", "\u200d\u200c")
_DOUBLED = re.compile(r"(.)्\1")
_WORD = re.compile(r"[0-9A-Za-z\u0900-\u097F]+")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Words that say nothing about the topic: the function words lang.py tells Marathi from Hindi by, and asking words.
_FILLER = MARATHI_WORDS | HINDI_WORDS | {"सांग", "समजावून", "बताइए", "समझाओ", "व्याख्या", "म्हणजेच", "बद्दल", "विषयी", "बारे"}
MIN_WORD_CHARS = 3


def _fold(text: str) -> str:
    """Remove spelling differences that should not stop a match.

    Books set in some fonts carry invisible joiner characters inside words, and a doubled
    consonant is often written singly (तत्त्वज्ञान and तत्वज्ञान, महत्त्व and महत्व).
    """
    return _DOUBLED.sub(r"\1", text.translate(_JOINERS))


def indic_words(query: str) -> str:
    """The content words of a Marathi or Hindi query as a word search: any of them, matched by how they begin."""
    words = [_fold(word).lower() for word in _WORD.findall(query) if word not in _FILLER]
    return " | ".join(f"{word}:*" for word in dict.fromkeys(words) if len(word) >= MIN_WORD_CHARS)


def search_notes(session_id: str, query: str, k: int = TOP_K) -> list[dict]:
    """The search_notes tool: up to k passages as {source, page, chapter, text, score}.

    Two rankings are combined. Meaning (embeddings) finds passages that say the same thing in
    other words. Words lift the passages that actually contain the terms asked about, which is
    what finds a named term such as "threshold frequency" among all the other passages about
    frequency. score is the cosine similarity.
    """
    query = query.translate(_JOINERS)
    words = indic_words(query) if _DEVANAGARI.search(query) else ""
    sql, wanted = (_INDIC, {"words": words}) if words else (_ENGLISH, {"query": query})
    rows = run(sql, {"vector": embed_query(query), "session": session_id, "rrf": RRF_K, "k": k, **wanted})
    return [dict(zip((*_FIELDS, "score"), row, strict=True)) for row in rows]


def chapter_chunks(session_id: str, chapter: int, limit: int = 200) -> list[dict]:
    """Passages of one chapter in page order."""
    rows = run(
        f"SELECT {_PASSAGE} FROM note_chunks WHERE session_id = %s AND chapter = %s"
        " ORDER BY note_chunks.page, id LIMIT %s",
        (session_id, chapter, limit),
    )
    return [dict(zip(_FIELDS, row, strict=True)) for row in rows]


def random_chunks(session_id: str, limit: int) -> list[dict]:
    """A few passages picked at random, for quizzing on notes that have no chapter headings."""
    rows = run(
        f"SELECT {_PASSAGE} FROM note_chunks WHERE session_id = %s ORDER BY random() LIMIT %s",
        (session_id, limit),
    )
    return [dict(zip(_FIELDS, row, strict=True)) for row in rows]


def spread_chunks(session_id: str, limit: int) -> list[dict]:
    """Up to `limit` passages spaced evenly through all the notes, in reading order."""
    rows = run(
        f"WITH numbered AS (SELECT id, row_number() OVER (ORDER BY id) AS position, count(*) OVER () AS total"
        f" FROM note_chunks WHERE session_id = %s)"
        f" SELECT {_PASSAGE} FROM note_chunks JOIN numbered USING (id)"
        f" WHERE (position - 1) %% greatest(1, ceil(total / %s::numeric)::int) = 0 ORDER BY id LIMIT %s",
        (session_id, limit, limit),
    )
    return [dict(zip(_FIELDS, row, strict=True)) for row in rows]


def available_chapters(session_id: str) -> list[int]:
    rows = run(
        "SELECT DISTINCT chapter FROM note_chunks WHERE session_id = %s AND chapter IS NOT NULL ORDER BY chapter",
        (session_id,),
    )
    return [row[0] for row in rows]


def notes_outline(session_id: str) -> list[dict]:
    """The notes_outline tool: each file's chapters in reading order.

    One row per chapter as {source, chapter, title, first_page, last_page, pages}. chapter is None
    for the pages of a file that belong to no chapter, and title is None when the book does not state one.
    """
    rows = run(
        "SELECT source, chapter, max(chapter_title),"
        " (array_agg(coalesce(page_label, page::text) ORDER BY page))[1],"
        " (array_agg(coalesce(page_label, page::text) ORDER BY page DESC))[1],"
        " count(DISTINCT page)"
        " FROM note_chunks WHERE session_id = %s GROUP BY source, chapter ORDER BY min(id)",
        (session_id,),
    )
    return [dict(zip(("source", "chapter", "title", "first_page", "last_page", "pages"), row, strict=True)) for row in rows]


def list_sources(session_id: str) -> list[dict]:
    """The files a session has added, oldest first, as {source, pages, passages}."""
    rows = run(
        "SELECT source, count(DISTINCT page), count(*) FROM note_chunks WHERE session_id = %s"
        " GROUP BY source ORDER BY min(id)",
        (session_id,),
    )
    return [dict(zip(("source", "pages", "passages"), row, strict=True)) for row in rows]


def notes_languages(session_id: str) -> list[str]:
    """The languages a session's notes are written in ('mr', 'hi', 'en'), judged from a sample of each file.

    Indexing reports this once; this recovers it for notes that are already stored, after a restart.
    """
    rows = run(
        "SELECT string_agg(left(text, 300), ' ') FROM ("
        " SELECT source, text, row_number() OVER (PARTITION BY source ORDER BY id) AS position"
        " FROM note_chunks WHERE session_id = %s) AS numbered"
        " WHERE position %% 20 = 1 GROUP BY source",
        (session_id,),
    )
    return list(dict.fromkeys(detect_language(sample) for (sample,) in rows))


def has_source(session_id: str, source: str) -> bool:
    """Has this session already indexed a file with this name?"""
    return bool(run("SELECT 1 FROM note_chunks WHERE session_id = %s AND source = %s LIMIT 1", (session_id, source)))


def count_chunks(session_id: str) -> int:
    return run("SELECT count(*) FROM note_chunks WHERE session_id = %s", (session_id,))[0][0]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--session", default="dev")
    args = parser.parse_args()
    for hit in search_notes(args.session, args.query):
        print(f"[{hit['score']:.3f}] {hit['source']} p.{hit['page']} ch.{hit['chapter']}\n  {hit['text'][:200]}\n")
