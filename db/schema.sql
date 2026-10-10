CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk of a student's uploaded notes. Scoped to a session and deleted with it.
CREATE TABLE IF NOT EXISTS note_chunks (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    source      TEXT NOT NULL,          -- file name shown in the citation
    page        INT  NOT NULL,
    chapter     INT,                    -- NULL when no chapter heading was found
    text        TEXT NOT NULL,
    embedding   vector(384) NOT NULL,   -- intfloat/multilingual-e5-small
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS note_chunks_session ON note_chunks (session_id, chapter);

-- One row per voice turn (filled in by db/log.py in unit 5).
CREATE TABLE IF NOT EXISTS turns (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    language    TEXT,
    transcript  TEXT,
    tool_name   TEXT,
    tool_args   JSONB,
    reply_text  TEXT,
    model       TEXT,
    tokens_in   INT,
    tokens_out  INT,
    stt_ms      INT,
    llm_ms      INT,
    tool_ms     INT,
    tts_ms      INT,
    total_ms    INT,
    error       TEXT,
    feedback    SMALLINT               -- 1 thumbs up, -1 thumbs down
);

-- Added 3 Oct 2026: the page to cite, when it differs from the position in the file
-- (a textbook's printed "32" is often the 41st page of the PDF).
ALTER TABLE note_chunks ADD COLUMN IF NOT EXISTS page_label TEXT;

-- Added 4 Oct 2026: the chapter's title, when its first page states one. Lets the app answer
-- questions about the book itself (how many chapters, what they are called, which is longest).
ALTER TABLE note_chunks ADD COLUMN IF NOT EXISTS chapter_title TEXT;

-- Added 4 Oct 2026: what the paid services cost each day, so the daily budget survives a restart (agent/budget.py).
CREATE TABLE IF NOT EXISTS spend (
    day     DATE NOT NULL,
    service TEXT NOT NULL,              -- 'voice' or 'listening', and 'total' for the day's sum of both
    rupees  NUMERIC NOT NULL DEFAULT 0,
    PRIMARY KEY (day, service)
);

-- Added 5 Oct 2026: a visitor's notebooks (db/notebooks.py). Each notebook has its own notes, files and
-- logged turns, all stored under the notebook's id; owner is a hash of the browser's cookie.
CREATE TABLE IF NOT EXISTS notebooks (
    id         TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS notebooks_owner ON notebooks (owner, created_at);
