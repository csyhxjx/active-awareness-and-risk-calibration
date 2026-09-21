"""Deterministic ROI-only detector contract for Phase 6B."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from guard.json_io import canonical_dumps, write_json


DETECTOR_VERSION = "phase6b-rgb-centroid-v1"
PAID_CAMERAS = ("q_branch", "q_a", "q_b", "q_c")
FORBIDDEN_INPUT_KEYS = frozenset({
    "filename", "path", "layout_id", "layout_group_id", "hidden_state",
    "state", "route", "route_result", "collision", "completion", "outcome",
})
PREPROCESSING = {
    "name": "roi_channel_moments_v1",
    "input_dtype": "uint8",
    "input_shape": "HWC_RGB",
    "resize": None,
    "normalization": "divide_by_255",
    "features": ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"],
}


class DetectorContractError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorConfig:
    architecture: str = "camera_conditioned_nearest_centroid"
    optimizer: str = "closed_form_class_mean"
    epochs: int = 1
    seed: int = 66301
    distance_temperature: float = 0.02
    abstain_confidence: float = 0.0
    checkpoint_rule: str = "validation_max_accuracy_then_min_nll_then_earliest_epoch"


def sha256_canonical(value) -> str:
    payload = canonical_dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocessing_hash() -> str:
    return sha256_canonical(PREPROCESSING)


def validate_request(request: dict) -> tuple[str, np.ndarray]:
    if not isinstance(request, dict):
        raise DetectorContractError("detector request must be a dict")
    extra = set(request) - {"camera_id", "roi_rgb"}
    forbidden = set(request) & FORBIDDEN_INPUT_KEYS
    if forbidden:
        raise DetectorContractError(f"forbidden detector inputs: {sorted(forbidden)}")
    if extra:
        raise DetectorContractError(f"unregistered detector inputs: {sorted(extra)}")
    if set(request) != {"camera_id", "roi_rgb"}:
        raise DetectorContractError("request requires exactly camera_id and roi_rgb")
    camera = request["camera_id"]
    image = request["roi_rgb"]
    if camera not in PAID_CAMERAS:
        raise DetectorContractError(f"unregistered paid camera: {camera!r}")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise DetectorContractError("roi_rgb must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise DetectorContractError("roi_rgb must have shape HxWx3")
    return camera, np.ascontiguousarray(image)


def preprocess(request: dict) -> tuple[str, np.ndarray]:
    camera, image = validate_request(request)
    normalized = image.astype(np.float64) / 255.0
    feature = np.concatenate((normalized.mean(axis=(0, 1)), normalized.std(axis=(0, 1))))
    if not np.isfinite(feature).all():
        raise DetectorContractError("non-finite preprocessed feature")
    return camera, feature


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class RGBDetector:
    def __init__(self, classes_by_camera: dict[str, tuple[str, ...]], centroids: dict[str, list[float]], config: DetectorConfig):
        self.classes_by_camera = {key: tuple(value) for key, value in classes_by_camera.items()}
        self.centroids = {key: np.asarray(value, dtype=np.float64) for key, value in centroids.items()}
        self.config = config
        self._validate_model()

    def _validate_model(self) -> None:
        if set(self.classes_by_camera) != set(PAID_CAMERAS):
            raise DetectorContractError("model must define every paid camera")
        for camera, classes in self.classes_by_camera.items():
            if not classes or len(classes) != len(set(classes)):
                raise DetectorContractError(f"invalid classes for {camera}")
            for label in classes:
                value = self.centroids.get(f"{camera}|{label}")
                if value is None or value.shape != (6,) or not np.isfinite(value).all():
                    raise DetectorContractError(f"invalid centroid for {camera}/{label}")
        if self.config.distance_temperature <= 0.0:
            raise DetectorContractError("distance_temperature must be positive")
        if not 0.0 <= self.config.abstain_confidence <= 1.0:
            raise DetectorContractError("abstain_confidence must be in [0, 1]")

    @classmethod
    def fit(cls, samples: list[dict], config: DetectorConfig | None = None) -> "RGBDetector":
        config = config or DetectorConfig()
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        for sample in samples:
            if set(sample) != {"request", "target_symbol"}:
                raise DetectorContractError("training sample requires request and target_symbol only")
            camera, feature = preprocess(sample["request"])
            label = sample["target_symbol"]
            if not isinstance(label, str) or label == "unobserved":
                raise DetectorContractError("target_symbol must be a registered observed symbol")
            grouped.setdefault((camera, label), []).append(feature)
        classes_by_camera = {
            camera: tuple(sorted(label for candidate, label in grouped if candidate == camera))
            for camera in PAID_CAMERAS
        }
        if any(not labels for labels in classes_by_camera.values()):
            raise DetectorContractError("training data must cover every paid camera")
        centroids = {
            f"{camera}|{label}": np.mean(rows, axis=0).tolist()
            for (camera, label), rows in grouped.items()
        }
        return cls(classes_by_camera, centroids, config)

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "detector_version": DETECTOR_VERSION,
            "preprocessing": PREPROCESSING,
            "preprocessing_sha256": preprocessing_hash(),
            "config": asdict(self.config),
            "classes_by_camera": {key: list(value) for key, value in sorted(self.classes_by_camera.items())},
            "centroids": {key: value.tolist() for key, value in sorted(self.centroids.items())},
        }

    @property
    def model_hash(self) -> str:
        return sha256_canonical(self.payload())

    @property
    def config_hash(self) -> str:
        return sha256_canonical(asdict(self.config))

    def predict(self, request: dict) -> dict:
        camera, feature = preprocess(request)
        labels = self.classes_by_camera[camera]
        distances = np.asarray([
            np.square(feature - self.centroids[f"{camera}|{label}"]).mean()
            for label in labels
        ])
        logits = -distances / self.config.distance_temperature
        probabilities = _softmax(logits)
        winner = int(np.argmax(probabilities))
        confidence = float(probabilities[winner])
        predicted = labels[winner] if confidence >= self.config.abstain_confidence else "unobserved"
        output = {
            "camera_id": camera,
            "predicted_outcome": predicted,
            "confidence": confidence,
            "calibrated_probabilities": {label: float(probabilities[index]) for index, label in enumerate(labels)},
            "model_hash": self.model_hash,
            "detector_version": DETECTOR_VERSION,
        }
        if not math.isfinite(confidence) or abs(sum(output["calibrated_probabilities"].values()) - 1.0) > 1e-9:
            raise DetectorContractError("non-finite detector output")
        return output

    def save(self, path: Path) -> None:
        write_json(path, self.payload())

    @classmethod
    def load(cls, path: Path) -> "RGBDetector":
        import json

        payload = json.loads(Path(path).read_text())
        if payload.get("detector_version") != DETECTOR_VERSION:
            raise DetectorContractError("detector version mismatch")
        if payload.get("preprocessing") != PREPROCESSING or payload.get("preprocessing_sha256") != preprocessing_hash():
            raise DetectorContractError("preprocessing contract mismatch")
        return cls(
            {key: tuple(value) for key, value in payload["classes_by_camera"].items()},
            payload["centroids"],
            DetectorConfig(**payload["config"]),
        )
