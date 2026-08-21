from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai_parser import DEFAULT_MODEL, PromptParseResult, api_is_configured, parse_prompt  # noqa: E402
from app.prompt_test import TEST_PROMPTS  # noqa: E402


FIELDS = [
    "id", "category", "should_pass", "actual_pass", "passed_eval",
    "expected_algorithms", "actual_algorithms", "error", "model",
]


def evaluate_cases(
    cases: list[dict],
    parser: Callable[..., PromptParseResult] = parse_prompt,
    *,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    rows = []
    for case in cases:
        result = parser(case["prompt"], model=model)
        actual_algorithms = (
            [operation.algorithm.value for operation in result.pipeline.operations]
            if result.pipeline else []
        )
        expected_algorithms = case.get("expected_algorithms", [])
        decision_correct = result.ok == case["should_pass"]
        algorithms_correct = not result.ok or actual_algorithms == expected_algorithms
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "should_pass": case["should_pass"],
                "actual_pass": result.ok,
                "passed_eval": decision_correct and algorithms_correct,
                "expected_algorithms": ";".join(expected_algorithms),
                "actual_algorithms": ";".join(actual_algorithms),
                "error": result.error or "",
                "model": result.model,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá AI parser trên bộ 26 prompt của B.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "benchmarks/results/ai_prompt_evaluation.csv",
    )
    args = parser.parse_args(argv)
    if not api_is_configured():
        raise SystemExit("Chưa có OPENAI_API_KEY hoặc Colab Secret OPENAI_API_KEY.")
    cases = TEST_PROMPTS[: args.limit] if args.limit else TEST_PROMPTS
    rows = evaluate_cases(cases, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(bool(row["passed_eval"]) for row in rows)
    print(f"AI prompt eval: {passed}/{len(rows)} đạt. CSV: {args.output}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
