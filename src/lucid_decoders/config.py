from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 13
    stratify: bool = True


@dataclass(slots=True)
class MBartConfig:
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    source_lang: str = "en_XX"
    target_lang: str = "de_DE"
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 1
    device: str | None = None


@dataclass(slots=True)
class FeatureConfig:
    topk: int = 3
    epsilon: float = 1e-12
    include_last_layer_only: bool = False


@dataclass(slots=True)
class ArtifactConfig:
    root: Path = field(default_factory=lambda: Path("artifacts"))

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

