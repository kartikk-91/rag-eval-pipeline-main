"""The demo's examples must come from the active, passage-grounded evaluation set."""

import re

from evaluation.eval_harness import DEFAULT_EVAL_SET, load_eval_set


def test_every_example_question_comes_from_the_measured_eval_set():
    from api.main import _PAGE

    examples = re.findall(r'data-query="([^"]+)"', _PAGE)
    measured = {ex.query for ex in load_eval_set(DEFAULT_EVAL_SET)}

    assert len(examples) == 3
    assert not [q for q in examples if q not in measured]
    assert 'id="answerButton"' in _PAGE
    assert "may incur charges" not in _PAGE
