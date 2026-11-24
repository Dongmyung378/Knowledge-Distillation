"""Cross-architecture Teacher (SegFormer ViT) and Quantization-Friendly Edge CNN Students."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import NUM_CLASSES

STUDENT_ARCHS: dict[str, dict[str, Any]] = {
    "mobilenetv4_small": {
        "timm_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "out_indices": (1, 2, 3, 4),
        "default_fpn_dim": 96,
        "tier": "ultra_light",
    },
    "mobilenetv3_large": {
        "timm_name": "mobilenetv3_large_100",
        "out_indices": (1, 2, 3, 4),
        "default_fpn_dim": 128,
        "tier": "balanced",
    },
    "mobilenetv4_medium": {
        "timm_name": "mobilenetv4_conv_medium.e500_r256_in1k",
        "out_indices": (1, 2, 3, 4),
        "default_fpn_dim": 128,
        "tier": "high_accuracy",
    },
    "resnet18": {
        "timm_name": "resnet18",
        "out_indices": (1, 2, 3, 4),
        "default_fpn_dim": 128,
        "tier": "classic_cnn",
    },
    "tiny_cnn": {
        "timm_name": None,
        "out_indices": None,
        "default_fpn_dim": 48,
        "tier": "smoke_test",
    },
}

TEACHER_VARIANTS: dict[str, dict[str, Any]] = {
    "segformer_b4": {
        "hf_id": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
        "depths": (3, 8, 27, 3),
        "hidden_sizes": (64, 128, 320, 512),
        "decoder_hidden_size": 768,
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
    },
    "segformer_b2": {
        "hf_id": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "depths": (3, 4, 6, 3),
        "hidden_sizes": (64, 128, 320, 512),
        "decoder_hidden_size": 768,
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
    },
    "segformer_b5": {
        "hf_id": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "depths": (3, 6, 40, 3),
        "hidden_sizes": (64, 128, 320, 512),
        "decoder_hidden_size": 768,
        "num_attention_heads": (1, 2, 5, 8),
        "sr_ratios": (8, 4, 2, 1),
    },
    "segformer_tiny_smoke": {
        "hf_id": None,
        "depths": (1, 1, 1, 1),
        "hidden_sizes": (16, 32, 64, 128),
        "decoder_hidden_size": 64,
        "num_attention_heads": (1, 2, 4, 8),
        "sr_ratios": (8, 4, 2, 1),
    },
}

STAGE_NAMES: tuple[str, ...] = ("s4", "s8", "s16", "s32")


def _make_activation(act: str) -> nn.Module:
    if act == "relu6":
        return nn.ReLU6(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported quantization-friendly activation: {act!r}")


class ConvBNAct(nn.Sequential):
    """Strictly foldable Conv(bias=False) -> BatchNorm -> ReLU/ReLU6 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: str = "relu6",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _make_activation(act),
        )


class DepthwiseSeparableConv(nn.Sequential):
    """Quantization-friendly depthwise-separable block with BN before every activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, act: str = "relu6") -> None:
        super().__init__(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, act=act),
            ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=act),
        )


class TinyDrivingBackbone(nn.Module):
    """Pure-PyTorch compact 4-stage CNN (strides 4, 8, 16, 32) for fast offline smoke tests."""

    def __init__(self, channels: tuple[int, int, int, int] = (24, 32, 64, 96)) -> None:
        super().__init__()
        c4, c8, c16, c32 = channels
        self.channels = list(channels)
        self.stem = nn.Sequential(
            ConvBNAct(3, c4, kernel_size=3, stride=2, act="relu6"),
            DepthwiseSeparableConv(c4, c4, stride=2, act="relu6"),
        )
        self.stage2 = DepthwiseSeparableConv(c4, c8, stride=2, act="relu6")
        self.stage3 = DepthwiseSeparableConv(c8, c16, stride=2, act="relu6")
        self.stage4 = DepthwiseSeparableConv(c16, c32, stride=2, act="relu6")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s4 = self.stem(x)
        s8 = self.stage2(s4)
        s16 = self.stage3(s8)
        s32 = self.stage4(s16)
        return [s4, s8, s16, s32]


class LiteFPNDecoder(nn.Module):
    """Quantization-friendly multi-scale FPN + global context decoder.

    Every convolution uses ``Conv2d(bias=False) -> BatchNorm2d -> ReLU6`` so
    ONNX Runtime / TensorRT / PyTorch QAT can fold every BatchNorm cleanly.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        fpn_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        act: str = "relu6",
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("LiteFPNDecoder expects exactly 4 pyramid stages (s4, s8, s16, s32)")
        self.lateral_s4 = ConvBNAct(in_channels[0], fpn_dim, kernel_size=1, padding=0, act=act)
        self.lateral_s8 = ConvBNAct(in_channels[1], fpn_dim, kernel_size=1, padding=0, act=act)
        self.lateral_s16 = ConvBNAct(in_channels[2], fpn_dim, kernel_size=1, padding=0, act=act)
        self.lateral_s32 = ConvBNAct(in_channels[3], fpn_dim, kernel_size=1, padding=0, act=act)

        self.context_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[3], fpn_dim, kernel_size=1, padding=0, bias=True),
            _make_activation(act),
        )
        self.smooth_s16 = DepthwiseSeparableConv(fpn_dim, fpn_dim, stride=1, act=act)
        self.smooth_s8 = DepthwiseSeparableConv(fpn_dim, fpn_dim, stride=1, act=act)
        self.fuse_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim, stride=1, act=act),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, pyramid: Sequence[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        s4, s8, s16, s32 = pyramid
        p32 = self.lateral_s32(s32)
        ctx = F.interpolate(self.context_pool(s32), size=p32.shape[-2:], mode="bilinear", align_corners=False)
        p32 = p32 + ctx

        p16 = self.lateral_s16(s16) + F.interpolate(p32, size=s16.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth_s16(p16)

        p8 = self.lateral_s8(s8) + F.interpolate(p16, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth_s8(p8)

        p4 = self.lateral_s4(s4) + F.interpolate(p8, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        logits_s4 = self.fuse_head(p4)
        return F.interpolate(logits_s4, size=output_size, mode="bilinear", align_corners=False)


class EdgeSegStudent(nn.Module):
    """Plain deployable Edge CNN semantic segmentation student.

    ``forward(x)`` returns a single ``logits`` tensor ``[B, C, H, W]`` so ONNX
    export and runtime inference require no distillation wrapper.
    Pass ``return_features=True`` during KD training to access pyramid stages.
    """

    def __init__(
        self,
        arch: str = "mobilenetv4_small",
        *,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
        fpn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arch not in STUDENT_ARCHS:
            raise ValueError(f"Unknown student arch {arch!r}. Supported: {sorted(STUDENT_ARCHS)}")
        spec = STUDENT_ARCHS[arch]
        self.arch = arch
        self.num_classes = int(num_classes)
        self.fpn_dim = int(fpn_dim if fpn_dim is not None else spec["default_fpn_dim"])

        if arch == "tiny_cnn":
            self.encoder = TinyDrivingBackbone()
            channels = self.encoder.channels
        else:
            import timm

            self.encoder = timm.create_model(
                spec["timm_name"],
                pretrained=pretrained,
                features_only=True,
                out_indices=spec["out_indices"],
            )
            channels = list(self.encoder.feature_info.channels())

        self.feature_channels: dict[str, int] = dict(zip(STAGE_NAMES, channels))
        self.decoder = LiteFPNDecoder(
            channels,
            fpn_dim=self.fpn_dim,
            num_classes=self.num_classes,
            dropout=dropout,
            act="relu6",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        input_size = (int(x.shape[-2]), int(x.shape[-1]))
        pyramid = list(self.encoder(x))
        logits = self.decoder(pyramid, output_size=input_size)
        if not return_features:
            return logits
        return {
            "logits": logits,
            "features": dict(zip(STAGE_NAMES, pyramid)),
        }


class SegformerTeacher(nn.Module):
    """Cross-architecture Vision Transformer Teacher (SegFormer) returning logits + pyramid features."""

    def __init__(
        self,
        variant: str = "segformer_b4",
        *,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        hf_model_id: str | None = None,
        hf_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if variant not in TEACHER_VARIANTS:
            raise ValueError(f"Unknown teacher variant {variant!r}. Supported: {sorted(TEACHER_VARIANTS)}")
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        self.variant = variant
        self.num_classes = int(num_classes)
        spec = TEACHER_VARIANTS[variant]

        if hf_config is not None:
            cfg = SegformerConfig.from_dict(dict(hf_config))
            cfg.num_labels = self.num_classes
            self.model = SegformerForSemanticSegmentation(cfg)
            self.feature_channels = dict(zip(STAGE_NAMES, cfg.hidden_sizes))
        elif variant == "segformer_tiny_smoke":
            cfg = SegformerConfig(
                num_labels=self.num_classes,
                depths=list(spec["depths"]),
                hidden_sizes=list(spec["hidden_sizes"]),
                decoder_hidden_size=int(spec["decoder_hidden_size"]),
                num_attention_heads=list(spec["num_attention_heads"]),
                sr_ratios=list(spec["sr_ratios"]),
            )
            self.model = SegformerForSemanticSegmentation(cfg)
            self.feature_channels = dict(zip(STAGE_NAMES, spec["hidden_sizes"]))
        else:
            model_id = hf_model_id or spec["hf_id"]
            if pretrained:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
            else:
                cfg = SegformerConfig(
                    num_labels=self.num_classes,
                    depths=list(spec["depths"]),
                    hidden_sizes=list(spec["hidden_sizes"]),
                    decoder_hidden_size=int(spec["decoder_hidden_size"]),
                    num_attention_heads=list(spec["num_attention_heads"]),
                    sr_ratios=list(spec["sr_ratios"]),
                    mlp_ratios=[4, 4, 4, 4],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                )
                self.model = SegformerForSemanticSegmentation(cfg)
            self.feature_channels = dict(zip(STAGE_NAMES, self.model.config.hidden_sizes))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        input_size = (int(x.shape[-2]), int(x.shape[-1]))
        outputs = self.model(pixel_values=x, output_hidden_states=return_features)
        logits = F.interpolate(outputs.logits, size=input_size, mode="bilinear", align_corners=False)
        if not return_features:
            return logits
        hidden = list(outputs.hidden_states)
        if len(hidden) != 4:
            raise RuntimeError(f"Expected 4 SegFormer hidden states, got {len(hidden)}")
        return {
            "logits": logits,
            "features": dict(zip(STAGE_NAMES, hidden)),
        }


class FeatureProjectorBank(nn.Module):
    """Training-only 1x1 channel adapters aligning Student pyramid features to Teacher channels.

    Discarded after training; adds zero parameters and zero latency at deployment.
    """

    def __init__(
        self,
        student_channels: Mapping[str, int],
        teacher_channels: Mapping[str, int],
        stages: Sequence[str] = ("s8", "s16", "s32"),
    ) -> None:
        super().__init__()
        self.stages = tuple(stages)
        self.projectors = nn.ModuleDict()
        for stage in self.stages:
            if stage not in student_channels or stage not in teacher_channels:
                raise ValueError(f"Missing stage {stage!r} in student/teacher feature channels")
            self.projectors[stage] = nn.Sequential(
                nn.Conv2d(int(student_channels[stage]), int(teacher_channels[stage]), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(teacher_channels[stage])),
            )

    def forward(self, student_features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {stage: self.projectors[stage](student_features[stage]) for stage in self.stages}


def build_student(
    arch: str = "mobilenetv4_small",
    *,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    fpn_dim: int | None = None,
    dropout: float = 0.1,
) -> EdgeSegStudent:
    return EdgeSegStudent(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        fpn_dim=fpn_dim,
        dropout=dropout,
    )


def build_teacher(
    variant: str = "segformer_b4",
    *,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    hf_model_id: str | None = None,
    hf_config: Mapping[str, Any] | None = None,
) -> SegformerTeacher:
    return SegformerTeacher(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        hf_model_id=hf_model_id,
        hf_config=hf_config,
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in model.parameters() if (not trainable_only or p.requires_grad)))


def weights_sha256(model: nn.Module) -> str:
    """Deterministic SHA-256 over all model parameters and persistent buffers."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        if tensor.dtype == torch.bfloat16:
            raw = tensor.to(torch.float32).numpy().tobytes(order="C")
        else:
            raw = tensor.numpy().tobytes(order="C")
        digest.update(raw)
    return digest.hexdigest()


def audit_quantization_friendliness(model: nn.Module) -> dict[str, Any]:
    """Audit Sequential blocks for the unfused ``Conv -> Activation -> BN`` anti-pattern.

    Placing an activation between Conv2d and BatchNorm2d prevents Conv-BN folding
    during static INT8 quantization and induces severe quantization noise. This
    auditor enforces ``Conv2d(bias=False) -> BatchNorm2d -> Activation`` across
    all Sequential student blocks.
    """
    activation_types = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.LeakyReLU)
    violations: list[dict[str, str]] = []
    conv_count = 0
    bn_count = 0

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            conv_count += 1
        elif isinstance(mod, nn.BatchNorm2d):
            bn_count += 1

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Sequential):
            children = list(mod.children())
            for i in range(len(children) - 2):
                if (
                    isinstance(children[i], nn.Conv2d)
                    and isinstance(children[i + 1], activation_types)
                    and isinstance(children[i + 2], nn.BatchNorm2d)
                ):
                    violations.append(
                        {
                            "module": name or "<root>",
                            "pattern": f"{type(children[i]).__name__} -> {type(children[i + 1]).__name__} -> {type(children[i + 2]).__name__}",
                        }
                    )
    return {
        "passed": len(violations) == 0,
        "conv_count": conv_count,
        "bn_count": bn_count,
        "violations": violations,
    }


def save_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    role: str,
    arch: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "role": role,
        "arch": arch,
        "num_classes": getattr(model, "num_classes", NUM_CLASSES),
        "fpn_dim": getattr(model, "fpn_dim", None),
        "hf_config": model.model.config.to_dict() if hasattr(model, "model") and hasattr(model.model, "config") else None,
        "parameter_count": count_parameters(model),
        "weights_sha256": weights_sha256(model),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload["extra"] = dict(extra)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(dest)
    return dest


def load_model_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    role = payload["role"]
    arch = payload["arch"]
    num_classes = int(payload.get("num_classes", NUM_CLASSES))
    if role == "student":
        model = build_student(
            arch=arch,
            num_classes=num_classes,
            pretrained=False,
            fpn_dim=payload.get("fpn_dim"),
        )
    elif role == "teacher":
        model = build_teacher(
            variant=arch,
            num_classes=num_classes,
            pretrained=False,
            hf_config=payload.get("hf_config"),
        )
    else:
        raise ValueError(f"Unknown checkpoint role: {role!r}")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
