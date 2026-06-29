"""SubEM scorer — the metric the gate reproduces. Deterministic."""

from unit0 import scorer


def test_normalize():
    assert scorer.normalize("The City of London!") == "city of london"
    assert scorer.normalize("a An THE") == ""


def test_sub_em_basic():
    assert scorer.sub_em("The answer is Berlin.", "Berlin")
    assert scorer.sub_em("berlin", "Berlin")            # case-insensitive
    assert not scorer.sub_em("Paris", "Berlin")
    assert not scorer.sub_em("anything", "")            # empty gold never matches


def test_sub_em_any():
    assert scorer.sub_em_any("it is football", ["soccer", "football"])
    assert not scorer.sub_em_any("cricket", ["soccer", "football"])


def test_score_with_list_golds():
    r = scorer.score(["Berlin", "Paris", "Rome"],
                     [["Berlin"], ["London"], ["Rome", "Roma"]])
    assert r["correct"] == 2 and r["total"] == 3
    assert abs(r["subem"] - 2 / 3) < 1e-9
    assert r["hits"] == [True, False, True]


def test_score_mixed_str_and_list():
    r = scorer.score(["Belgium"], ["Belgium"])          # gold as bare string
    assert r["correct"] == 1
