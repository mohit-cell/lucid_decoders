from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class HallucinationSpan:
    start: int
    end: int
    label: int = 1


@dataclass(slots=True)
class TranslationExample:
    example_id: str
    source_text: str
    hypothesis_text: str
    sentence_label: int | None = None
    token_labels: list[int] | None = None
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)
    language_pair: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hallucination_spans"] = [asdict(span) for span in self.hallucination_spans]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranslationExample":
        spans = [
            HallucinationSpan(**span)
            for span in payload.get("hallucination_spans", []) or []
        ]
        return cls(
            example_id=str(payload["example_id"]),
            source_text=str(payload["source_text"]),
            hypothesis_text=str(payload["hypothesis_text"]),
            sentence_label=_maybe_int(payload.get("sentence_label")),
            token_labels=_maybe_int_list(payload.get("token_labels")),
            hallucination_spans=spans,
            language_pair=payload.get("language_pair"),
            split=payload.get("split"),
            metadata=payload.get("metadata", {}) or {},
        )


@dataclass(slots=True)
class AttentionExtraction:
    example_id: str
    source_text: str
    hypothesis_text: str
    source_tokens: list[str]
    target_tokens: list[str]
    target_offsets: list[tuple[int, int]]
    cross_attentions: Any
    self_attentions: Any
    sentence_label: int | None = None
    token_labels: list[int] | None = None
    language_pair: str | None = None
    split: str | None = None


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _maybe_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]
