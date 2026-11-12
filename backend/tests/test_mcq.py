import pytest

from app.evals.mcq import MCQItem, build_messages, parse_answer

ITEM = MCQItem(
    "abstract_algebra/test/0", "en", "abstract_algebra", "What is 1+1?", ("1", "2", "3", "4"), "B"
)


def test_prompt_lists_options_with_letters():
    msgs = build_messages(ITEM)
    assert msgs[0].role == "system" and msgs[1].role == "user"
    user = msgs[1].content
    assert "What is 1+1?" in user
    assert "A. 1\nB. 2\nC. 3\nD. 4" in user
    assert user.endswith("Answer with only the letter (A, B, C or D).")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        ("b", "B"),
        (" C.", "C"),
        ("(D)", "D"),
        ("Answer: A", "A"),
        ("The answer is C.", "C"),
        ("The correct answer is: B", "B"),
        ("D. Because the group has order 4.", "D"),
        ("A\n", "A"),
        ("B)", "B"),
    ],
)
def test_parses_strict_single_letter_forms(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "E",
        "Bangalore",
        "Both A and B",
        "A or C",
        "I think it is probably B",
        "4",
        "The options are A, B, C and D.",
        "ABC",
        "उत्तर: ख",
    ],
)
def test_rejects_unparseable(text):
    assert parse_answer(text) is None
