from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class RuntimeSettings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    checkpoint_dir: Path = field(init=False)
    runtime_dir: Path = field(init=False)
    examples_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    invert_checkpoint_name: str = "model_checkpoint_01500.pt"
    mask_checkpoint_name: str = "checkpoint-175.pth"
    payload_encoder_name: str = "encoder_Q.ckpt"
    payload_decoder_name: str = "decoder_Q.ckpt"
    secret_image_name: str = "bluesky_white2.png"
    logo_name: str = "logo.png"
    default_example_name: str = "0000.png"
    max_input_side: int = 2048
    tamper_mask_input_size: int = 1024
    payload_resolution: int = 256
    payload_bit_length: int = 100
    payload_strength: float = 1.0
    tamper_mask_threshold: float = 0.03
    tamper_mask_scales: tuple[float, ...] = (1.0,)
    inpaint_model_id: str = field(
        default_factory=lambda: os.getenv(
            "OMNIGUARD_INPAINT_MODEL",
            "sd2-community/stable-diffusion-2-inpainting",
        )
    )
    use_diffusers: bool = field(default_factory=lambda: _env_flag("OMNIGUARD_USE_DIFFUSERS", True))
    allow_inpaint_model_download: bool = field(
        default_factory=lambda: _env_flag("OMNIGUARD_ALLOW_INPAINT_MODEL_DOWNLOAD", False)
    )
    public_share: bool = field(default_factory=lambda: _env_flag("OMNIGUARD_SHARE", False))
    hmac_secret: str = field(
        default_factory=lambda: os.getenv("OMNIGUARD_HMAC_SECRET", "omniguard-demo-key")
    )

    def __post_init__(self) -> None:
        self.checkpoint_dir = self.base_dir / "checkpoint"
        self.runtime_dir = self.base_dir / "runtime"
        self.examples_dir = self.base_dir / "examples"
        self.assets_dir = self.base_dir
        self.docs_dir = self.base_dir / "docs"

    @property
    def invert_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.invert_checkpoint_name

    @property
    def mask_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.mask_checkpoint_name

    @property
    def payload_encoder_path(self) -> Path:
        return self.checkpoint_dir / self.payload_encoder_name

    @property
    def payload_decoder_path(self) -> Path:
        return self.checkpoint_dir / self.payload_decoder_name

    @property
    def secret_image_path(self) -> Path:
        return self.assets_dir / self.secret_image_name

    @property
    def logo_path(self) -> Path:
        return self.assets_dir / self.logo_name

    @property
    def default_example_path(self) -> Path:
        return self.examples_dir / self.default_example_name

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.runtime_dir,
            self.runtime_dir / "reports",
            self.runtime_dir / "artifacts",
            self.runtime_dir / "masks",
            self.runtime_dir / "benchmarks",
            self.runtime_dir / "datasets",
        ):
            path.mkdir(parents=True, exist_ok=True)
