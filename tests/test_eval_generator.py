"""The propagation-eval generator: structural ground-truth integrity (deterministic, no LLM).
A self-built eval's credibility depends on the ground truth being well-formed."""

import pytest

from gem.eval import generate, generate_stratified, GENERATORS
from gem.scenarios import SCENARIOS


def test_generate_nonempty_and_sized():
    s = generate()
    assert len(s) >= 100                       # scaled set, per the plan's 100-150
    assert len(GENERATORS) >= 14


@pytest.mark.parametrize("scen_list", [SCENARIOS, generate()],
                         ids=["hand-built", "generated"])
def test_scenario_structure(scen_list):
    for s in scen_list:
        assert len(s.facts) == len(s.parents) == len(s.expect_invalid), s.name
        for pidx in s.parents:                 # parent indices must be valid + acyclic-ish range
            for j in pidx:
                assert 0 <= j < len(s.facts), s.name
        assert s.trigger and s.facts, s.name


def test_every_scenario_has_a_should_invalidate_or_is_negative():
    # each scenario must encode a decision (at least one True or be an all-survive negative)
    for s in generate():
        assert isinstance(s.expect_invalid, list) and len(s.expect_invalid) > 0


def test_stratified_spans_all_templates():
    strat = generate_stratified(1)
    cats = {s.category for s in strat}
    assert len(strat) == len([g for g in GENERATORS])    # 1 per template (those with >=1 item)
    assert len(cats) >= 10                                # broad category coverage
