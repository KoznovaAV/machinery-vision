"""Checks of spatial corroboration and non-exact marking hints."""

from scripts.analyze_model_marking import corroborate, labels_in_text, revised_suggestion


def line(text, box=(0, 0, 100, 20), confidence=0.99):
    return dict(text=text, box=box, confidence=confidence)


def test_new_codes_are_distinguished():
    assert labels_in_text("ACROS 550") == ["acros_550"]
    assert labels_in_text("ACROS 585") == []
    assert labels_in_text("МТЗ 1523.3") == ["mtz_1523"]
    assert labels_in_text("Т-150К") == ["t_150k"]
    assert labels_in_text("ДТ-75М") == ["dt_75"]


def test_two_views_and_same_region_required():
    assert corroborate([line("820")], [])["status"] == "unconfirmed"
    assert (
        corroborate([line("820")], [line("820", box=(300, 0, 400, 20))])["status"] == "unconfirmed"
    )
    result = corroborate([line("820")], [line("820")])
    assert result["suggestion"] == "mtz_820" and result["exact_model"] is None


def test_equipment_number_or_conflict_not_a_model_claim():
    assert corroborate([line("82:")], [line("82:")])["status"] == "unconfirmed"
    assert corroborate([line("820")], [line("82")])["status"] == "unconfirmed"
    assert corroborate([line("820", confidence=0.7)], [line("820")])["status"] == "unconfirmed"


def test_unknown_marking_can_only_flag_outside_supported_list():
    e = corroborate([line("1523")], [line("1523")])
    assert revised_suggestion("mtz_1221", e, {"mtz_1221"}) == "other"


def test_marking_hint_never_promotes_family_only():
    from scripts.analyze_model_marking import guard_decision

    evidence = corroborate([line("820")], [line("820")])
    base = dict(status="family_only", suggestion="mtz_820", model=None, family="mtz_belarus")
    result = guard_decision(base, evidence, {"mtz_80", "mtz_820"})
    assert result["model"] is None and result["status"] == "family_only"
    assert result["visible_marking_model"] == "mtz_820"


def test_future_marking_vetoes_wrong_exact_answer():
    from scripts.analyze_model_marking import guard_decision

    evidence = corroborate([line("1523")], [line("1523")])
    base = dict(status="experimental_match", suggestion="mtz_1221", model="mtz_1221")
    result = guard_decision(base, evidence, {"mtz_1221"})
    assert result["status"] == "manual_review" and result["model"] is None
    assert result["suggestion"] == "mtz_1221" and result["visible_marking_model"] == "mtz_1523"
