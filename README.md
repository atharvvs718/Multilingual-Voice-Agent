# Voice study companion

A voice-first study companion for Marathi- and Hindi-medium students. Upload your own textbook
chapter or notes, ask a question out loud, and hear a short answer in the same language with the
page it came from. It can also quiz you on a chapter and check your spoken answer.

No models are trained here. Speech recognition, the LLM and the voice are pretrained models; this
repository is the pipeline, retrieval, prompts, token budgeting and evals around them.

## How one turn works

```
mic / upload ─► ffmpeg ─► Silero-VAD ─► Sarvam Saaras ─► language detection
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                 agent call 1 (gpt-oss-120b) ──► answer directly
                              │
                              ├─► search_notes ─► meaning + exact words (this session's notes only)
                              ├─► open_page ─► the page as a picture, read by a model that can see (examples, equations)
                              ├─► notes_outline ─► files, chapters, titles, page ranges
                              ├─► summarise_notes ─► a chapter's sections, or the book's chapters, as a digest
                              └─► quiz_me      ─► passages from one chapter
                              ▼
                 agent call 2: 2–4 sentences, grounded, ends with the page
                              ▼
                 language guard ─► turn log ─► TTS (Sarvam Bulbul v3)
```

| Stage | Technology | File |
|---|---|---|
| Decode, trim silence | ffmpeg, Silero-VAD | `agent/audio.py` |
| Speech to text | Sarvam `saaras:v4` (code-mix mode); Groq `whisper-large-v3` as backup | `agent/stt.py`, `agent/sarvam.py` |
| Language detection and guard | marker words | `agent/lang.py` |
| LLM, retries, model fallback | Groq `gpt-oss-120b` → `gpt-oss-20b` | `agent/llm.py`, `agent/retry.py` |
| Agent loop | plain Python, no framework | `agent/pipeline.py` |
| Prompts and tool definitions | | `agent/prompts.py` |
| Chat memory and quiz state | last 8 messages + summary | `agent/session.py` |
| Tools | `search_notes`, `open_page`, `notes_outline`, `summarise_notes`, `quiz_me` | `agent/tools/` |
| Notes ingestion | PDFium in its own process, printed page numbers, chapters, e5-small | `rag/` |
| Notes guide | title, summary and suggested questions for new notes | `agent/overview.py` |
| Notebooks | each has its own notes, conversations and memory; the list is a table | `server.py`, `db/notebooks.py` |
| Storage and turn log | PostgreSQL + pgvector | `db/` |
| Text to speech | Sarvam Bulbul v3, five speakers to choose from | `agent/tts.py` |
| Page and API | FastAPI; plain HTML, CSS and JavaScript, no build step | `server.py`, `web/` |
| Daily budget and per-visitor limits | one row per day in PostgreSQL; hourly counts in memory | `agent/budget.py`, `agent/limits.py` |
| Evaluation | jiwer + a labelled test set | `eval/run_eval.py` |

## Run it locally

Needs Python 3.12, ffmpeg on the PATH, and Docker Desktop.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # on macOS/Linux: .venv/bin/pip
cp .env.example .env                                  # then fill in GROQ_API_KEY and SARVAM_API_KEY
docker compose up -d                                  # PostgreSQL + pgvector, tables created on first start
.venv/Scripts/python server.py                        # http://localhost:7860
```

On Windows, `start.bat` does the last two steps with a double-click and opens the app in Edge by itself (in cmd use backslashes:
`.venv\Scripts\python server.py`). Run one copy only; a second start is refused. Use Edge or Chrome:
Brave's privacy shield hands the page a silent microphone.

**Notebooks.** A notebook is one set of sources with the conversations about them, like a notebook in
NotebookLM. Questions asked in one notebook search only that notebook's files and remember only that
notebook's conversations. New, Rename and Delete are at the top left; a new notebook takes its name from
the first file added to it. A browser can have 8. The list of notebooks is in the database, so it is still
there after a restart; conversations are kept in memory and are not.

**Using it.** Add a source on the left (a whole textbook PDF up to 25 MB is fine; "no copying"
PDFs open too). The page shows what the notes cover and three questions to try. Pick the language
you speak in the Studio, then press the microphone, ask, and press it again, or type. The written
reply appears first with the page it came from. A spoken question is answered aloud; a typed one is
read out when you press play under the answer. The Studio also has Quiz me and Summarise, for one
chapter or the whole book. Notes may be in English, Marathi or Hindi; questions can mix in English.

**Listening.** Questions are transcribed by Sarvam Saaras v4 (₹30 per hour of audio, about 4 paise a
question). If Sarvam fails or there is no key, Groq Whisper takes over; it is noticeably worse at Marathi.

**Voices.** Replies are spoken by Sarvam Bulbul v3, so `SARVAM_API_KEY` must be set in `.env`; a
reply costs about ₹1. The Studio has a list of five speakers (Priya, Pooja, Neha, Shubh, Aditya) and
`TTS_VOICE` sets the one selected at start. Nothing runs on the local GPU for the voice.

**The page.** `server.py` is a small API over the pipeline; the page it serves is plain HTML, CSS and
JavaScript in `web/`, with no framework and no build step.

**Changing the look.** Colours, fonts and the width of the reading column are the variables at the top of
`web/style.css`; the structure of the page is `web/index.html`; what it does (asking, the microphone, the
narration, the animations) is `web/app.js`. Save and reload the browser: no restart is needed.

**Cost and caps.** The voice is about 90% of what a turn costs, so only a spoken question is answered
aloud; a typed one is spoken when the student presses play under the answer.

- `DAILY_BUDGET_RUPEES` (30): the most the paid services, voice and listening together, may cost in a
  day for all visitors. The total is one row in the database, checked and raised in a single step, so a
  restart or a second copy of the app cannot hand out a second budget. Past it the app carries on with
  what is free: Whisper listens, and the reply is read by the browser's own voice if it has one for that
  language, otherwise shown in writing only. If the database cannot be reached, nothing paid is used.
- No one address may use more than half of the day's budget on the voice.
- Per address, per hour: `MAX_TURNS_PER_HOUR` (30) questions, `MAX_SPOKEN_PER_HOUR` (20) spoken
  replies, `MAX_UPLOADS_PER_HOUR` (6) files, 10 wrong access codes. These are counted in memory, so a
  restart clears them; the daily budget is the limit that holds.
- `ACCESS_CODE`: if set, the app asks for the code before anything else works.
- `TURN_LOG_DAYS` (7): logged questions and replies are deleted after this many days.
- Sizes: a file up to 25 MB, a recording up to 30 seconds, a typed question up to 1,000 characters.
  A request body is cut off while it is being read once it passes the limit for its path.

Sarvam credit is prepaid, so the most that can ever be spent is the credit in the account; the daily
budget decides how fast it may go.

**Scanned or old-font PDFs.** Install Tesseract with the `hin` and `mar` language packs. Without it,
pages with no readable text are refused with a clear message.

**Fallback audio.** Once TTS works, run `python -m scripts.make_fallback_audio` once. It records the
"sorry, please try again" clips into `assets/` so they can be played even when every API is down.

## Tests and linting

```bash
.venv/Scripts/python -m pytest          # no network, GPU or database needed
.venv/Scripts/python -m ruff check .
```

## Evaluation

Three layers, cheapest first. The search alone, on questions whose answer page is known
(`eval/questions.csv`); it calls no model and costs nothing:

```bash
python -m eval.retrieval_eval --book path/to/textbook.pdf
```

Then the whole pipeline on the 60 labelled questions in `eval/labels.csv`, typed (`--mode text`) or
spoken. (Those questions are about a textbook that is not in this repository; `eval/labels.example.csv`
holds eight about the sample notes in `notes/`, for trying the script: `--notes notes/ --labels
eval/labels.example.csv`.) For the spoken run, record each line of `reference_transcript` into `eval/clips/` under its
`clip_id`:

```bash
python -m eval.run_eval --notes path/to/chapters/                 # full pipeline on audio
python -m eval.run_eval --notes path/to/chapters/ --mode text     # reference transcripts, no STT
python -m eval.run_eval --notes path/to/chapters/ --no-hint       # speech to text without the language hint
python -m eval.run_eval --notes path/to/chapters/ --model llama-3.3-70b-versatile
```

It prints WER per language, tool and quiz-chapter accuracy, citation accuracy, "not in your notes"
on unanswerable questions, language match, latency (median and p95 per stage) and tokens per turn,
and writes per-clip results to `eval/results/`. Keep the eval clips out of every prompt and example.

### Results

**Search, measured on 5 October 2026** (`python -m eval.retrieval_eval --book <textbook.pdf>`): 112 questions
on a 364-page Class 12 physics textbook (1,386 passages), each labelled with the page that answers it.
A question counts when a passage from that page is among those found. The agent is given the first 3.

| Ranked by | top 1 | top 3 | top 5 | MRR |
|---|---|---|---|---|
| Meaning only (embeddings) | 77% | 95% | 98% | 0.86 |
| Words only | 60% | 79% | 83% | 0.69 |
| Both: the app's search | 76% | 93% | 97% | 0.85 |

For the app's search by kind of question: 96% in the top 3 for 50 questions that name the term, 92% for
50 short keyword queries (the form the model actually sends), 83% for 12 paraphrases that avoid the
book's own word.

What this shows, and what it does not. The search finds the right page for about 19 questions in 20.
On an English book, adding the word search to the meaning search changes nothing that can be measured:
93% against 95% is two questions. The word search was added for bare terms and for Marathi, where it
fixed a failed question on one book; that has not been benchmarked. The questions and their page labels
were written from the book's text by the same hands that built the search.

**The whole pipeline on typed questions, 6 October 2026** (`python -m eval.run_eval --notes <textbook.pdf>
--mode text --rows 1-30 --delay 60`): the first 30 of the 60 labelled questions in `eval/labels.csv`, in
English, Marathi, Hindi and mixed. Every turn was answered by the main model.

| Measure | Result |
|---|---|
| Right tool chosen (search, quiz, outline, summary, page, or none) | 100% (30 of 30) |
| Right page retrieved and cited | 95% (18 of 19) |
| Said "not in your notes" for a topic the book does not cover | 80% (4 of 5) |
| Reply in the language of the question | 100% (30 of 30) |
| Tokens per turn | 4,112 on average, 6,873 at most |

The two real failures. A question in pure Marathi, "केंद्रकीय विखंडन म्हणजे काय?" (nuclear fission): the model
turned the term into "central decomposition", searched for that, and answered about a diffraction
pattern with a page number. And "Projectile motion क्या होता है?", which the book does not cover, was
answered from the model's own knowledge with a page attached. One label was widened after the run:
the heat-engine formula is on page 98 as well as 97, and the reply cited 98 (17 of 19 as first scored).

**Response time, from the app's own log** (`python -m eval.latency --since "2026-10-04 13:10+05:30"`): 96
turns asked while building and testing, on one laptop.

| | Median | 95th percentile | Turns |
|---|---|---|---|
| Spoken question to written answer | 3.1 s | 5.2 s | 21 |
| Spoken question to voice ready | 7.4 s | 12.3 s | 20 |
| Typed question to written answer | 1.7 s | 13.1 s | 75 |

A spoken turn is, at the median, 1.3 s of listening, 1.5 s for the model's two calls, 0.3 s of search and
4.0 s to make the voice. A third of these turns (32 of 96) were written by the backup model, because of
the free tier's limit.

**Is the voice clear?** (`python -m eval.tts_roundtrip --spend`, about Rs 2): 12 sentences spoken by Bulbul v3
and written down again by Saaras v4. Character error: Marathi 3.9%, Hindi 3.8%, English 0%. Every
difference was a spelling choice by the listener ("notes" in Latin letters, a compound written as two
words), none a misheard word. This is clean synthetic speech: it says the voice is clear, not that a
real student is heard correctly.

**Not measured yet:** speech recognition on real recordings (the 60 clips are not recorded; the script
reports character and word error per language), the other 30 typed questions, and any Marathi book.

## Privacy and copyright

- Recordings are not kept: a clip is deleted as soon as it has been transcribed. Spoken replies are
  deleted about ten minutes after they are made.
- Notes, and the uploaded file itself, belong to the browser that added them. They are deleted after
  24 hours, or at once with "Remove everything of mine", which also removes that browser's
  conversations and its logged questions.
- Questions and replies are logged in the database for 7 days (`TURN_LOG_DAYS`), to measure quality.
- To answer, the app sends data to two companies: the recording and the reply text to Sarvam; the
  question, the passages found in the notes, a picture of a page when one is opened, and (as the
  backup listener) the recording to Groq. Fonts load from Google Fonts and the formula library from
  jsDelivr. The page says this on a visitor's first visit.
- The app ships only notes written for it (`notes/`). No textbook content is stored in the
  repository; students upload their own copy.
- Keys live in `.env`, which is not in the repository, and in the hosting service's secrets when
  deployed. The helper programs that read uploaded PDFs and audio are started without them.

## Security

What is in place, in `server.py` unless another file is named:

- The page may load scripts only from itself and one pinned library, and cannot be shown inside
  another site (`Content-Security-Policy`, `frame-ancestors 'none'`).
- Requests that change anything are refused when they come from another site.
- The visitor cookie is `HttpOnly` and `SameSite=Lax`, and `Secure` when the site is on HTTPS. The
  access pass is tied to the browser it was given to and signed with a secret made at start-up.
- Only `app.js` and `style.css` are served as files; audio is served only by exact name.
- Uploaded audio is decoded with a time limit and a length limit (`agent/audio.py`); PDFs are read
  in a separate process (`rag/`).
- Messages shown to a visitor never contain a provider's error text or a file path.

What it does not have: accounts, payments, or an admin page. A visitor is a cookie. Hourly limits
are per address and in memory. It has been reviewed, not audited; treat a public link as a demo.

## Deploying

Not yet done on a public host. The container is built and tested on this laptop: `docker build -t
voice-agent .` gives a 3.5 GB image (Python, ffmpeg, Tesseract, the CPU build of PyTorch, the
embedding model) that serves the page and the API against the local database.

1. Create a hosted PostgreSQL with pgvector (Neon is the usual free choice). The app creates its
   tables when it starts.
2. Give the host this repository; it builds the `Dockerfile`. It needs about 2 GB of memory.
3. Set as secrets: `GROQ_API_KEY`, `SARVAM_API_KEY`, `DATABASE_URL`, `ACCESS_CODE`. The image already
   sets `HOST=0.0.0.0`, `PORT=7860` and `PROXY_HOPS=1` (the host's proxy is the one hop; with 0 every
   visitor looks like the same address, with a larger number a visitor can fake theirs). Set
   `DAILY_BUDGET_RUPEES` to what you are willing to spend in a day.
4. Share the app's own address, not a page that embeds it: the app refuses to be shown in a frame.
5. Open the link on a phone and run through the six use cases.

What changes without a GPU: indexing the 364-page textbook took about 105 s in the container on a
24-thread laptop processor (72 s of it embedding) against about 10 s with the GPU, and a small host
with 2 processors will take several times longer. Answering a question is unaffected: embedding one
search query takes 0.01 s. Uploaded PDFs are kept on the container's disk for page pictures, so on a
host that wipes its disk at a restart, "explain example 1.6" needs the book uploaded again.
