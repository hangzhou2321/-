from __future__ import annotations

import pytest

from manisoftrl import compare_paired_outcomes, gate_paired_report


def test_strict_success_set_expansion_passes() -> None:
    decision = compare_paired_outcomes(
        {0: True, 1: False, 2: True},
        {0: True, 1: True, 2: True},
    )
    assert decision.promote is True
    assert decision.strict_success_set_expansion is True
    assert decision.gained_successes == ("1",)
    assert decision.lost_successes == ()


def test_higher_total_still_fails_when_an_old_success_is_lost() -> None:
    decision = compare_paired_outcomes(
        {0: True, 1: True, 2: False, 3: False},
        {0: False, 1: True, 2: True, 3: True},
    )
    assert decision.parent_success_count == 2
    assert decision.candidate_success_count == 3
    assert decision.promote is False
    assert decision.reason == "candidate_lost_parent_successes"
    assert decision.gained_successes == ("2", "3")
    assert decision.lost_successes == ("0",)


def test_same_success_set_does_not_promote() -> None:
    decision = compare_paired_outcomes(
        {0: True, 1: False},
        {0: True, 1: False},
    )
    assert decision.promote is False
    assert decision.reason == "candidate_added_no_successes"


def test_unpaired_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="paired evaluation keys differ"):
        compare_paired_outcomes(
            {0: True, 1: False},
            {0: True, 2: True},
        )


def _paired_report(*, formal: bool = True) -> dict[str, object]:
    return {
        "formal_evaluation": formal,
        "evaluation_scope": "test_panel",
        "common_random_numbers": True,
        "single_frozen_checkpoint_per_role": True,
        "parent_checkpoint_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "environment_contract_sha256": "c" * 64,
        "success_predicate": "raw_official_predicate",
        "parent": {
            "rows": [
                {"case": 0, "seed": 100, "official_success": True},
                {"case": 1, "seed": 101, "official_success": False},
            ]
        },
        "candidate": {
            "rows": [
                {"case": 0, "seed": 100, "official_success": True},
                {"case": 1, "seed": 101, "official_success": True},
            ]
        },
    }


def test_formal_paired_report_passes_and_recomputes_raw_rows() -> None:
    decision = gate_paired_report(_paired_report())
    assert decision.promote is True
    assert decision.formal_evaluation is True
    assert decision.evaluation_scope == "test_panel"
    assert decision.gained_successes == ("1",)


def test_nonformal_panel_cannot_trigger_formal_promotion() -> None:
    decision = gate_paired_report(_paired_report(formal=False))
    assert decision.strict_success_set_expansion is True
    assert decision.promote is False
    assert decision.reason == "invalid_evidence_contract"
    assert "formal_evaluation is not true" in decision.evidence_errors


def test_report_requires_common_random_numbers() -> None:
    report = _paired_report()
    report["common_random_numbers"] = False
    decision = gate_paired_report(report)
    assert decision.promote is False
    assert "common_random_numbers is not true" in decision.evidence_errors


def test_formal_report_requires_environment_contract_hash() -> None:
    report = _paired_report()
    del report["environment_contract_sha256"]
    decision = gate_paired_report(report)
    assert decision.promote is False
    assert (
        "environment_contract_sha256 is not a SHA-256"
        in decision.evidence_errors
    )
