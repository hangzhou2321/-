from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionDecision:
    """Fail-closed decision for one paired parent/candidate evaluation panel."""

    promote: bool
    reason: str
    strict_success_set_expansion: bool
    evidence_contract_valid: bool
    total_pairs: int
    parent_success_count: int
    candidate_success_count: int
    retained_successes: tuple[str, ...]
    gained_successes: tuple[str, ...]
    lost_successes: tuple[str, ...]
    evidence_errors: tuple[str, ...] = ()
    formal_evaluation: bool | None = None
    evaluation_scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", value):
        if part.isdigit():
            parts.append((0, int(part)))
        elif part:
            parts.append((1, part.casefold()))
    return tuple(parts)


def _normalize_outcomes(
    outcomes: Mapping[object, object], role: str
) -> dict[str, bool]:
    if not outcomes:
        raise ValueError(f"{role} outcomes cannot be empty")

    normalized: dict[str, bool] = {}
    for raw_key, raw_success in outcomes.items():
        key = str(raw_key)
        if not key:
            raise ValueError(f"{role} contains an empty evaluation key")
        if key in normalized:
            raise ValueError(f"{role} contains duplicate normalized key {key!r}")
        if type(raw_success) is not bool:
            raise ValueError(f"{role}[{key!r}] success must be a JSON/Python bool")
        normalized[key] = raw_success
    return normalized


def compare_paired_outcomes(
    parent: Mapping[object, object],
    candidate: Mapping[object, object],
) -> PromotionDecision:
    """Require the candidate success set to be a strict superset of the parent.

    Mapping keys identify paired evaluation units: a case for one rollout per
    case, or a caller-defined case/seed key for repeated stochastic rollouts.
    """

    parent_outcomes = _normalize_outcomes(parent, "parent")
    candidate_outcomes = _normalize_outcomes(candidate, "candidate")
    parent_keys = set(parent_outcomes)
    candidate_keys = set(candidate_outcomes)
    if parent_keys != candidate_keys:
        missing = sorted(parent_keys - candidate_keys, key=_natural_key)
        extra = sorted(candidate_keys - parent_keys, key=_natural_key)
        raise ValueError(
            "paired evaluation keys differ; "
            f"candidate missing={missing}, candidate extra={extra}"
        )

    parent_successes = {key for key, value in parent_outcomes.items() if value}
    candidate_successes = {
        key for key, value in candidate_outcomes.items() if value
    }
    retained = tuple(
        sorted(parent_successes & candidate_successes, key=_natural_key)
    )
    gained = tuple(
        sorted(candidate_successes - parent_successes, key=_natural_key)
    )
    lost = tuple(
        sorted(parent_successes - candidate_successes, key=_natural_key)
    )
    strict_expansion = not lost and bool(gained)
    if lost:
        reason = "candidate_lost_parent_successes"
    elif not gained:
        reason = "candidate_added_no_successes"
    else:
        reason = "strict_success_set_expansion"

    return PromotionDecision(
        promote=strict_expansion,
        reason=reason,
        strict_success_set_expansion=strict_expansion,
        evidence_contract_valid=True,
        total_pairs=len(parent_keys),
        parent_success_count=len(parent_successes),
        candidate_success_count=len(candidate_successes),
        retained_successes=retained,
        gained_successes=gained,
        lost_successes=lost,
    )


def _row_identity(row: Mapping[str, Any], use_case_only: bool) -> str:
    if use_case_only:
        return str(row["case"])

    parts: list[str] = []
    if "case" in row:
        parts.append(f"case={row['case']}")
    if "deterministic" in row:
        deterministic = row["deterministic"]
        if type(deterministic) is not bool:
            raise ValueError("row deterministic field must be a bool")
        parts.append(f"deterministic={str(deterministic).lower()}")
    if "seed" in row:
        parts.append(f"seed={row['seed']}")
    if not parts:
        raise ValueError("each row needs case and/or seed identity fields")
    return ",".join(parts)


def _rows_to_outcomes(rows: object, role: str) -> dict[str, bool]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{role}.rows must be a non-empty array")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"every {role}.rows item must be an object")

    case_labels = [str(row["case"]) for row in rows if "case" in row]
    use_case_only = len(case_labels) == len(rows) and len(set(case_labels)) == len(rows)
    outcomes: dict[str, bool] = {}
    for row in rows:
        if "official_success" not in row:
            raise ValueError(f"{role} row is missing official_success")
        key = _row_identity(row, use_case_only)
        if key in outcomes:
            raise ValueError(f"{role}.rows contains duplicate key {key!r}")
        success = row["official_success"]
        if type(success) is not bool:
            raise ValueError(f"{role} row {key!r} official_success must be a bool")
        outcomes[key] = success
    return outcomes


def gate_paired_report(
    report: Mapping[str, Any], *, require_formal: bool = True
) -> PromotionDecision:
    """Recompute a decision from raw rows in a Route-B-style paired report."""

    errors: list[str] = []
    formal = report.get("formal_evaluation")
    if type(formal) is not bool:
        errors.append("formal_evaluation must be explicit")
        formal_value: bool | None = None
    else:
        formal_value = formal
    if require_formal and formal is not True:
        errors.append("formal_evaluation is not true")
    if report.get("common_random_numbers") is not True:
        errors.append("common_random_numbers is not true")

    if require_formal:
        if report.get("single_frozen_checkpoint_per_role") is not True:
            errors.append("single_frozen_checkpoint_per_role is not true")
        for field in (
            "parent_checkpoint_sha256",
            "candidate_checkpoint_sha256",
            "environment_contract_sha256",
        ):
            value = report.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
                errors.append(f"{field} is not a SHA-256")
        predicate = report.get("success_predicate")
        if not isinstance(predicate, str) or not predicate.strip():
            errors.append("success_predicate is missing")
        evaluation_scope = report.get("evaluation_scope")
        if not isinstance(evaluation_scope, str) or not evaluation_scope.strip():
            errors.append("evaluation_scope is missing")

    parent_block = report.get("parent")
    candidate_block = report.get("candidate")
    if not isinstance(parent_block, Mapping) or not isinstance(candidate_block, Mapping):
        raise ValueError("report must contain parent and candidate objects")
    parent = _rows_to_outcomes(parent_block.get("rows"), "parent")
    candidate = _rows_to_outcomes(candidate_block.get("rows"), "candidate")
    decision = compare_paired_outcomes(parent, candidate)

    scope = report.get("evaluation_scope")
    scope_value = scope if isinstance(scope, str) else None
    if errors:
        return replace(
            decision,
            promote=False,
            reason="invalid_evidence_contract",
            evidence_contract_valid=False,
            evidence_errors=tuple(errors),
            formal_evaluation=formal_value,
            evaluation_scope=scope_value,
        )
    return replace(
        decision,
        formal_evaluation=formal_value,
        evaluation_scope=scope_value,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed parent/candidate success-set promotion gate"
    )
    parser.add_argument("paired_report", type=Path)
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="allow a local development decision; never implies official success",
    )
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.paired_report.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("paired report root must be an object")
        decision = gate_paired_report(
            data, require_formal=not args.allow_nonformal
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(json.dumps({"promote": False, "error": str(error)}, ensure_ascii=False))
        return 2

    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 0 if decision.promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
