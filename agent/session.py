"""State for one browser session: chat memory (FR-10), the open quiz question, and the last language used."""
import logging
import uuid
from dataclasses import dataclass, field

from agent.errors import ServiceUnavailable
from agent.llm import llm
from agent.prompts import SUMMARISE
from config import LLM_FALLBACK_MODEL, MAX_HISTORY

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict] = field(default_factory=list)   # only user and assistant text, never tool output
    summary: str = ""                                    # two lines covering messages no longer in `messages`
    language: str = ""                                   # last detected language; breaks ties on short answers
    quiz: dict | None = None                             # {"question", "passages"} while an answer is awaited
    notes_languages: list[str] = field(default_factory=list)      # languages the uploaded notes are written in
    last_turn_id: int | None = None                      # turn-log row of the newest turn
    message_turns: dict[int, int] = field(default_factory=dict)   # chat bubble index -> turn-log row, for feedback

    def reset_chat(self) -> None:
        """Start a new conversation. The uploaded notes and the chosen language stay."""
        self.messages, self.summary, self.quiz = [], "", None
        self.message_turns = {}

    def add_turn(self, user_text: str, reply_text: str) -> None:
        """Store one exchange, then fold the oldest messages into the summary if history is too long."""
        self.messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply_text}]
        if len(self.messages) > MAX_HISTORY:
            old, self.messages = self.messages[:-MAX_HISTORY], self.messages[-MAX_HISTORY:]
            self._summarise(old)

    def _summarise(self, old: list[dict]) -> None:
        """Uses the fallback model so it does not eat the main model's per-minute token limit."""
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        try:
            message, _ = llm(
                [
                    {"role": "system", "content": SUMMARISE},
                    {"role": "user", "content": f"Summary so far: {self.summary or '(none)'}\n\nNew messages:\n{lines}"},
                ],
                model=LLM_FALLBACK_MODEL,
                max_tokens=300,
            )
        except ServiceUnavailable:
            # Losing old context is better than failing the student's turn.
            log.warning("could not summarise older messages; dropping them")
            return
        self.summary = (message.content or self.summary).strip()
