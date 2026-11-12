"""Prompt construction and strict answer parsing for multiple-choice QA."""

import re
from dataclasses import dataclass

from app.backends.base import ChatMessage

PROMPT_TEMPLATE_VERSION = "mcq-v1"
LETTERS = ("A", "B", "C", "D")

SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. Reply with only the letter of the correct "
    "option: A, B, C or D. Do not explain."
)

# Strict: the first alphanumeric content must be a single option letter, optionally
# preceded by "Answer:" / "The answer is", optionally wrapped in parentheses, and
# followed by nothing or by punctuation. "B", "(b)", "Answer: C.", "D. Because..." parse;
# "Both A and B", "I think the answer might be A or C" and "Bangalore" do not.
_ANSWER_RE = re.compile(
    r"^\W*(?:(?:the\s+)?(?:correct\s+)?answer\s*(?:is)?\s*[:\-]?\s*)?"
    r"\(?\s*([A-Da-d])\s*\)?(?=$|[\s\.\):,;!\-])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MCQItem:
    item_id: str
    language: str
    subject: str | None
    question: str
    options: tuple[str, str, str, str]
    answer: str  # "A".."D"


def build_messages(item: MCQItem) -> list[ChatMessage]:
    lines = [item.question.strip(), ""]
    for letter, option in zip(LETTERS, item.options, strict=True):
        lines.append(f"{letter}. {option}")
    lines += ["", "Answer with only the letter (A, B, C or D)."]
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


def render_prompt(messages: list[ChatMessage]) -> str:
    """Flat text of the prompt for storage; the backend sends the structured messages."""
    return "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)


def parse_answer(text: str | None) -> str | None:
    """Return the chosen letter in upper case, or None if the reply is not parseable."""
    if not text:
        return None
    m = _ANSWER_RE.match(text.strip())
    if not m:
        return None
    letter = m.group(1).upper()
    # Reject "A or B" style hedges: a second standalone letter right after.
    rest = text.strip()[m.end() :]
    if re.match(r"^\s*(?:or|and|/|,)\s*\(?[A-Da-d]\)?(?=$|\W)", rest, re.IGNORECASE):
        return None
    return letter
