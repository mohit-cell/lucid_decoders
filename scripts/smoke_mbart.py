"""Run one real mBART attention extraction smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lucid_decoders.attention import MBartAttentionConfig, MBartAttentionExtractor  # noqa: E402
from lucid_decoders.features import compute_token_features, extract_sentence_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test real mBART attention extraction.")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--source", default="The cat eats fish.")
    parser.add_argument("--generated", default="Die Katze isst Fisch.")
    parser.add_argument("--source-lang", default="en_XX")
    parser.add_argument("--target-lang", default="de_DE")
    args = parser.parse_args()

    extractor = MBartAttentionExtractor(
        MBartAttentionConfig(
            model_name=args.model_name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            max_source_length=64,
            max_target_length=64,
        )
    )
    result = extractor.extract(args.source, args.generated)
    token_features = compute_token_features(
        result.cross_attentions,
        decoder_self_attention=result.decoder_attentions,
        source_tokens=result.source_tokens,
        target_tokens=result.target_tokens,
    )
    sentence_features = extract_sentence_features(
        result.cross_attentions,
        decoder_self_attention=result.decoder_attentions,
        source_tokens=result.source_tokens,
        target_tokens=result.target_tokens,
    )

    first_cross = result.cross_attentions[0]
    shape = list(first_cross.shape) if hasattr(first_cross, "shape") else []
    payload = {
        "model": args.model_name,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "source_tokens": result.source_tokens,
        "target_tokens": result.target_tokens,
        "cross_attention_layers": len(result.cross_attentions),
        "first_cross_attention_shape": shape,
        "token_feature_count": len(token_features),
        "first_token_features": token_features[0] if token_features else {},
        "sentence_features": sentence_features,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

