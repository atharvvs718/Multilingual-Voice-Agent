"""The list of each visitor's notebooks.

Only the list lives here: a notebook's id, whose it is, and its title. What a notebook holds
is stored under its id elsewhere (note_chunks.session_id, turns.session_id, the uploads folder).
"""
from db.conn import run


def stored_notebooks(owner: str) -> list[tuple[str, str]]:
    """(id, title) of one visitor's notebooks, oldest first."""
    rows = run("SELECT id, title FROM notebooks WHERE owner = %s ORDER BY created_at, id", (owner,))
    return [(row[0], row[1]) for row in rows]


def save_notebook(notebook_id: str, owner: str, title: str) -> None:
    """Add a notebook, or change its title. An id that belongs to someone else is left as it is."""
    run(
        """
        INSERT INTO notebooks (id, owner, title) VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET title = excluded.title WHERE notebooks.owner = excluded.owner
        """,
        (notebook_id, owner, title),
    )


def remove_notebook(notebook_id: str, owner: str) -> None:
    run("DELETE FROM notebooks WHERE id = %s AND owner = %s", (notebook_id, owner))


def forget_notebooks(owner: str) -> None:
    """Remove every notebook of one visitor from the list."""
    run("DELETE FROM notebooks WHERE owner = %s", (owner,))


def purge_empty_notebooks(hours: int) -> None:
    """Remove notebooks older than this whose notes are gone: notes are deleted after a day, and an empty name is not worth keeping."""
    run(
        """
        DELETE FROM notebooks AS kept
        WHERE kept.created_at < now() - make_interval(hours => %s)
          AND NOT EXISTS (SELECT 1 FROM note_chunks WHERE note_chunks.session_id = kept.id)
        """,
        (hours,),
    )
