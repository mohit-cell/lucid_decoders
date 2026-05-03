from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lucid_decoders.config import FeatureConfig, MBartConfig
from lucid_decoders.features.contracts import validate_feature_frame
from lucid_decoders.features.sentence_head_features import build_sentence_head_feature_rows
from lucid_decoders.features.sentence_features import build_sentence_feature_frame
from lucid_decoders.features.token_features import build_token_feature_rows
from lucid_decoders.io import read_jsonl, write_table
from lucid_decoders.schemas import AttentionExtraction, TranslationExample


class MBartAttentionExtractor:
    def __init__(self, config: MBartConfig) -> None:
        self.config = config
        self.torch = self._import_torch()
        self.transformers = self._import_transformers()
        self.device = self._resolve_device(config.device)
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(config.model_name)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self._set_language_codes()

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for mBART extraction. Install dependencies with `pip install -e .`."
            ) from exc
        return torch

    def _import_transformers(self) -> Any:
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "transformers is required for mBART extraction. Install dependencies with `pip install -e .`."
            ) from exc
        return transformers

    def _resolve_device(self, requested: str | None) -> Any:
        if requested:
            return self.torch.device(requested)
        if self.torch.cuda.is_available():
            return self.torch.device("cuda")
        return self.torch.device("cpu")

    def _load_model(self) -> Any:
        load_kwargs = {"attn_implementation": "eager"}
        try:
            model = self.transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
        except TypeError:
            # Older transformers versions do not accept `attn_implementation` in from_pretrained.
            model = self.transformers.AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
        else:
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
        return model

    def _set_language_codes(self) -> None:
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.config.source_lang
        if hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = self.config.target_lang

    def extract(self, example: TranslationExample) -> AttentionExtraction:
        source_batch = self._tokenize_source(example.source_text)
        target_batch = self._tokenize_target(example.hypothesis_text)
        labels = target_batch["input_ids"]
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)

        with self.torch.no_grad():
            model_kwargs = {
                "input_ids": source_batch["input_ids"].to(self.device),
                "attention_mask": source_batch["attention_mask"].to(self.device),
                "decoder_input_ids": decoder_input_ids.to(self.device),
                "output_attentions": True,
                "return_dict": True,
            }
            decoder_attention_mask = target_batch.get("attention_mask")
            if decoder_attention_mask is not None:
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.to(self.device)
            outputs = self.model(**model_kwargs)

        cross_attentions = self._stack_attentions(outputs.cross_attentions)
        self_attentions = self._stack_attentions(outputs.decoder_attentions)
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_batch["input_ids"][0].tolist())
        target_tokens = self.tokenizer.convert_ids_to_tokens(labels[0].tolist())
        target_offsets = self._normalize_offsets(target_batch.get("offset_mapping"), len(target_tokens))
        token_labels = self._align_token_labels(example, target_offsets, len(target_tokens))

        return AttentionExtraction(
            example_id=example.example_id,
            source_text=example.source_text,
            hypothesis_text=example.hypothesis_text,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            target_offsets=target_offsets,
            cross_attentions=cross_attentions,
            self_attentions=self_attentions,
            sentence_label=example.sentence_label,
            token_labels=token_labels,
            language_pair=example.language_pair,
            split=example.split,
        )

    def _tokenize_source(self, text: str) -> dict[str, Any]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_length,
        )

    def _tokenize_target(self, text: str) -> dict[str, Any]:
        try:
            return self.tokenizer(
                text_target=text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_target_length,
                return_offsets_mapping=True,
            )
        except TypeError:
            if not hasattr(self.tokenizer, "as_target_tokenizer"):
                raise
            with self.tokenizer.as_target_tokenizer():
                return self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_target_length,
                    return_offsets_mapping=True,
                )

    def _stack_attentions(self, attention_layers: Any) -> np.ndarray:
        if attention_layers is None:
            raise RuntimeError(
                "Attention tensors were not returned by the model. "
                "Use an attention backend that supports `output_attentions=True`, "
                "such as `attn_implementation='eager'`."
            )
        layer_arrays = [layer[0].detach().cpu().numpy() for layer in attention_layers]
        return np.stack(layer_arrays, axis=0)

    def _normalize_offsets(
        self,
        offsets: Any,
        expected_length: int,
    ) -> list[tuple[int, int]]:
        if offsets is None:
            return [(0, 0) for _ in range(expected_length)]
        if hasattr(offsets, "tolist"):
            raw_offsets = offsets[0].tolist()
        else:
            raw_offsets = offsets[0]
        return [tuple(int(value) for value in pair) for pair in raw_offsets]

    def _align_token_labels(
        self,
        example: TranslationExample,
        target_offsets: list[tuple[int, int]],
        target_length: int,
    ) -> list[int] | None:
        if example.token_labels is not None:
            content_indices = [idx for idx, span in enumerate(target_offsets) if span[1] > span[0]]
            if len(example.token_labels) == target_length:
                return list(example.token_labels)
            if len(example.token_labels) == len(content_indices):
                aligned = [0] * target_length
                for label, idx in zip(example.token_labels, content_indices):
                    aligned[idx] = int(label)
                return aligned
            raise ValueError(
                f"Token label count mismatch for example {example.example_id}: "
                f"got {len(example.token_labels)} labels for {target_length} target tokens."
            )

        if not example.hallucination_spans:
            if example.sentence_label == 0:
                return [0] * target_length
            return None

        aligned = [0] * target_length
        for idx, token_span in enumerate(target_offsets):
            token_start, token_end = token_span
            if token_end <= token_start:
                continue
            for span in example.hallucination_spans:
                if span.label <= 0:
                    continue
                if max(token_start, span.start) < min(token_end, span.end):
                    aligned[idx] = 1
                    break
        return aligned


def load_examples(path: str | Path) -> list[TranslationExample]:
    return [TranslationExample.from_dict(record) for record in read_jsonl(path)]


def validate_example_for_extraction(
    example: TranslationExample,
    require_sentence_label: bool = False,
    require_token_labels: bool = False,
) -> str | None:
    if not example.source_text.strip():
        return "empty_source_text"
    if not example.hypothesis_text.strip():
        return "empty_hypothesis_text"
    if require_sentence_label and example.sentence_label is None:
        return "missing_sentence_label"
    if require_token_labels and example.token_labels is None and not example.hallucination_spans:
        return "missing_token_supervision"
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mBART attention features from normalized examples.")
    parser.add_argument("--input", required=True, help="Normalized JSONL from the preprocessing step.")
    parser.add_argument("--token-output", required=True, help="Path to token-level feature table.")
    parser.add_argument("--sentence-output", required=True, help="Path to sentence-level feature table.")
    parser.add_argument(
        "--sentence-head-output",
        help="Optional path to sentence-level feature rows grouped by decoder layer/head.",
    )
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--report-output", help="Optional JSON report with extraction counts and skipped rows.")
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Raise an error instead of skipping invalid examples.",
    )
    parser.add_argument(
        "--require-sentence-label",
        action="store_true",
        help="Skip/fail examples without sentence_label.",
    )
    parser.add_argument(
        "--require-token-labels",
        action="store_true",
        help="Skip/fail examples without token labels or hallucination spans.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    mbart_config = MBartConfig(
        model_name=args.model_name,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        device=args.device,
    )
    feature_config = FeatureConfig(topk=args.topk)

    extractor = MBartAttentionExtractor(mbart_config)
    examples = load_examples(args.input)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    token_rows: list[dict[str, Any]] = []
    sentence_head_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    for example in examples:
        invalid_reason = validate_example_for_extraction(
            example,
            require_sentence_label=args.require_sentence_label,
            require_token_labels=args.require_token_labels,
        )
        if invalid_reason:
            skipped_rows.append({"example_id": example.example_id, "reason": invalid_reason})
            if args.fail_on_invalid:
                raise ValueError(f"Invalid example {example.example_id}: {invalid_reason}")
            continue
        extraction = extractor.extract(example)
        token_rows.extend(build_token_feature_rows(extraction, feature_config))
        if args.sentence_head_output:
            sentence_head_rows.extend(build_sentence_head_feature_rows(extraction, feature_config))

    if not token_rows:
        raise ValueError("No token feature rows were produced. Check input examples and extraction filters.")

    token_frame = pd.DataFrame(token_rows)
    sentence_frame = build_sentence_feature_frame(token_frame)
    validate_feature_frame(token_frame, "token")
    validate_feature_frame(sentence_frame, "sentence")
    write_table(token_frame, args.token_output)
    write_table(sentence_frame, args.sentence_output)
    if args.sentence_head_output:
        sentence_head_frame = pd.DataFrame(sentence_head_rows)
        validate_feature_frame(sentence_head_frame, "sentence_head")
        write_table(sentence_head_frame, args.sentence_head_output)
    if args.report_output:
        report = {
            "input": str(args.input),
            "total_examples": len(examples),
            "processed_examples": len({row["example_id"] for row in token_rows}),
            "skipped_examples": len(skipped_rows),
            "skipped": skipped_rows,
            "token_rows": len(token_rows),
            "sentence_rows": len(sentence_frame),
            "sentence_head_rows": len(sentence_head_rows),
        }
        Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
