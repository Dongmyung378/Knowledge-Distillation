"""Unified Cityscapes-19 driving taxonomy, safety-critical subsets, and palettes."""

from __future__ import annotations

import numpy as np

NUM_CLASSES: int = 19
IGNORE_INDEX: int = 255

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

CLASS_NAMES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Vulnerable road users and active traffic control elements critical for driving safety.
SAFETY_CRITICAL_CLASSES: tuple[str, ...] = (
    "traffic_light",
    "traffic_sign",
    "person",
    "rider",
    "motorcycle",
    "bicycle",
)
SAFETY_CRITICAL_INDICES: tuple[int, ...] = tuple(
    CLASS_NAMES.index(name) for name in SAFETY_CRITICAL_CLASSES
)

# Official Cityscapes / KITTI Semantics raw labelId (0..33) -> trainId (0..18, 255).
_CITYSCAPES_RAW_TO_TRAIN: dict[int, int] = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

LABEL_ID_TO_TRAIN_ID: np.ndarray = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _raw_id, _train_id in _CITYSCAPES_RAW_TO_TRAIN.items():
    LABEL_ID_TO_TRAIN_ID[_raw_id] = _train_id

# Official Cityscapes visualization palette (RGB uint8).
CITYSCAPES_PALETTE: np.ndarray = np.array(
    [
        [128, 64, 128],   # 0: road
        [244, 35, 232],   # 1: sidewalk
        [70, 70, 70],     # 2: building
        [102, 102, 156],  # 3: wall
        [190, 153, 153],  # 4: fence
        [153, 153, 153],  # 5: pole
        [250, 170, 30],   # 6: traffic light
        [220, 220, 0],    # 7: traffic sign
        [107, 142, 35],   # 8: vegetation
        [152, 251, 152],  # 9: terrain
        [70, 130, 180],   # 10: sky
        [220, 20, 60],    # 11: person
        [255, 0, 0],      # 12: rider
        [0, 0, 142],      # 13: car
        [0, 0, 70],       # 14: truck
        [0, 60, 100],     # 15: bus
        [0, 80, 100],     # 16: train
        [0, 0, 230],      # 17: motorcycle
        [119, 11, 32],    # 18: bicycle
    ],
    dtype=np.uint8,
)


def map_mask_to_train_ids(mask: np.ndarray, label_format: str = "train_id") -> np.ndarray:
    """Convert raw mask pixels to canonical Cityscapes-19 trainIds (0..18, 255)."""
    arr = np.asarray(mask, dtype=np.uint8)
    if label_format == "train_id":
        out = arr.copy()
        invalid = (out >= NUM_CLASSES) & (out != IGNORE_INDEX)
        out[invalid] = IGNORE_INDEX
        return out
    if label_format == "label_id":
        return LABEL_ID_TO_TRAIN_ID[arr]
    raise ValueError(f"Unsupported label_format: {label_format!r}. Expected 'train_id' or 'label_id'.")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map a 2D trainId mask [H, W] to an RGB uint8 color visualization [H, W, 3]."""
    arr = np.asarray(mask, dtype=np.int64)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    valid = (arr >= 0) & (arr < NUM_CLASSES)
    rgb[valid] = CITYSCAPES_PALETTE[arr[valid]]
    return rgb


def safety_channel_weights(safety_boost: float = 2.0, device=None):
    """Normalized per-class channel weights emphasizing safety-critical classes."""
    import torch

    weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    if safety_boost <= 0:
        raise ValueError("safety_boost must be positive")
    for idx in SAFETY_CRITICAL_INDICES:
        weights[idx] = float(safety_boost)
    return weights / weights.mean()
