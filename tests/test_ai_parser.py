from __future__ import annotations

from types import SimpleNamespace

from app.ai_parser import AIParseEnvelope, DEFAULT_MODEL, PromptParseResult, parse_prompt
from app.pipeline_schema import Pipeline
from scripts.evaluate_ai_prompts import evaluate_cases


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def test_ai_parser_accepts_and_revalidates_pipeline() -> None:
    envelope = AIParseEnvelope.model_validate(
        {
            "accepted": True,
            "message": "",
            "operations": [
                {
                    "algorithm": "gaussian_blur",
                    "backend": "openmp",
                    "thread_count": 4,
                    "params": {"kernel_size": 5, "sigma": 1.5},
                }
            ],
        }
    )
    client = FakeClient(envelope)
    result = parse_prompt("Làm mờ ảnh", client=client)
    assert result.ok and result.pipeline is not None
    assert result.pipeline.operations[0].thread_count == 4
    assert client.responses.call["text_format"] is AIParseEnvelope
    assert client.responses.call["model"] == DEFAULT_MODEL


def test_ai_parser_returns_model_clarification() -> None:
    client = FakeClient(AIParseEnvelope(accepted=False, message="Thiếu threshold.", operations=[]))
    result = parse_prompt("Tìm biên giúp tôi", client=client)
    assert not result.ok
    assert result.pipeline is None
    assert result.error == "Thiếu threshold."


def test_ai_parser_rejects_empty_prompt_without_api_call() -> None:
    result = parse_prompt("   ", client=FakeClient(None))
    assert not result.ok
    assert "nhập yêu cầu" in result.error.lower()


def test_ai_parser_requires_key_when_client_is_not_injected(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = parse_prompt("Sobel threshold 100")
    assert not result.ok
    assert "OPENAI_API_KEY" in result.error


def test_prompt_evaluator_checks_decision_and_algorithm_order() -> None:
    pipeline = Pipeline.model_validate(
        {
            "operations": [
                {"algorithm": "sobel", "backend": "sequential", "params": {"threshold": 100}}
            ]
        }
    )

    def fake_parser(prompt: str, *, model: str):
        return PromptParseResult(True, pipeline, None, model)

    rows = evaluate_cases(
        [{
            "id": "T01", "category": "valid", "prompt": "Sobel",
            "should_pass": True, "expected_algorithms": ["sobel"],
        }],
        parser=fake_parser,
    )
    assert rows[0]["passed_eval"] is True
