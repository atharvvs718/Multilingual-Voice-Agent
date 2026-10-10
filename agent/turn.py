"""The record of one voice turn: what the UI shows and what the turn log stores."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Turn:
    session_id: str
    transcript: str = ""
    language: str = "hi"
    tool_name: str | None = None
    tool_args: dict | None = None
    sources: list[dict] = field(default_factory=list)   # [{source, page, score}] returned by the tool
    reply_text: str = ""
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    vad_ms: int = 0
    stt_ms: int = 0
    llm_ms: int = 0
    tool_ms: int = 0
    tts_ms: int = 0
    total_ms: int = 0
    language_retry: bool = False                        # the language guard had to re-ask
    error: str | None = None
    id: int | None = None                               # row id in the turns table once logged
    started: float = field(default_factory=time.perf_counter, repr=False)   # for total_ms; not logged
    spoken: dict[str, str] = field(default_factory=dict, repr=False)        # voice -> audio file made; not logged

    def add_error(self, message: str) -> None:
        self.error = f"{self.error}; {message}" if self.error else message


@contextmanager
def timed(turn: Turn, stage: str):
    """Add the elapsed milliseconds of the block to turn.<stage>, e.g. timed(turn, "stt_ms")."""
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(turn, stage, getattr(turn, stage) + int((time.perf_counter() - start) * 1000))
