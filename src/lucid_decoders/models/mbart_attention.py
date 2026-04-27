from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lucid_decoders.config import FeatureConfig, MBartConfig
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
        self.model = self.transformers.AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
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
            outputs = self.model(
                input_ids=source_batch["input_ids"].to(self.device),
                attention_mask=source_batch["attention_mask"].to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                output_attentions=True,
                return_dict=True,
            )

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mBART attention features from normalized examples.")
    parser.add_argument("--input", required=True, help="Normalized JSONL from the preprocessing step.")
    parser.add_argument("--token-output", required=True, help="Path to token-level feature table.")
    parser.add_argument("--sentence-output", required=True, help="Path to sentence-level feature table.")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
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
    for example in examples:
        extraction = extractor.extract(example)
        token_rows.extend(build_token_feature_rows(extraction, feature_config))

    token_frame = pd.DataFrame(token_rows)
    sentence_frame = build_sentence_feature_frame(token_frame)
    write_table(token_frame, args.token_output)
    write_table(sentence_frame, args.sentence_output)


if __name__ == "__main__":
    main()

