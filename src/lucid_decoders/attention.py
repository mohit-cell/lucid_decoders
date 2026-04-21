"""mBART attention extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MBartAttentionConfig:
    """Configuration for teacher-forced mBART attention extraction."""

    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    source_lang: str = "en_XX"
    target_lang: str = "de_DE"
    max_source_length: int = 256
    max_target_length: int = 256
    device: str | None = None


@dataclass(frozen=True)
class AttentionResult:
    """Token strings and raw attention outputs for one translation pair."""

    source_tokens: list[str]
    target_tokens: list[str]
    cross_attentions: Any
    decoder_attentions: Any | None = None
    encoder_attentions: Any | None = None


class MBartAttentionExtractor:
    """Extract cross-attention for a provided source and generated translation."""

    def __init__(self, config: MBartAttentionConfig | None = None) -> None:
        self.config = config or MBartAttentionConfig()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the `ml` extra to use mBART attention extraction.") from exc

        self.torch = torch
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

    def extract(
        self,
        source: str,
        generated: str,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> AttentionResult:
        """Run teacher-forced decoding and return mBART attention tensors."""

        src_lang = source_lang or self.config.source_lang
        tgt_lang = target_lang or self.config.target_lang
        self._set_language_codes(src_lang, tgt_lang)

        source_inputs = self.tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_length,
        )
        target_inputs = self.tokenizer(
            text_target=generated,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_target_length,
        )

        source_inputs = {name: tensor.to(self.device) for name, tensor in source_inputs.items()}
        labels = target_inputs["input_ids"].to(self.device)
        decoder_input_ids = self._decoder_inputs_from_labels(labels)

        with self.torch.no_grad():
            outputs = self.model(
                **source_inputs,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        if not outputs.cross_attentions or outputs.cross_attentions[0] is None:
            raise RuntimeError(
                "mBART did not return cross-attention tensors. Use a Transformers/PyTorch backend "
                "that supports `output_attentions=True`, or load the model with eager attention."
            )

        source_tokens = self.tokenizer.convert_ids_to_tokens(source_inputs["input_ids"][0].detach().cpu().tolist())
        target_tokens = self.tokenizer.convert_ids_to_tokens(labels[0].detach().cpu().tolist())
        return AttentionResult(
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            cross_attentions=outputs.cross_attentions,
            decoder_attentions=outputs.decoder_attentions,
            encoder_attentions=outputs.encoder_attentions,
        )

    def extract_many(self, pairs: Iterable[tuple[str, str]]) -> list[AttentionResult]:
        """Extract attention for multiple source/generated pairs."""

        return [self.extract(source, generated) for source, generated in pairs]

    def _set_language_codes(self, source_lang: str, target_lang: str) -> None:
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = source_lang
        if hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = target_lang

    def _decoder_inputs_from_labels(self, labels: Any) -> Any:
        if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            return self.model.prepare_decoder_input_ids_from_labels(labels=labels)
        if hasattr(self.model, "_shift_right"):
            return self.model._shift_right(labels)
        raise RuntimeError("The selected seq2seq model cannot prepare decoder inputs from labels.")
