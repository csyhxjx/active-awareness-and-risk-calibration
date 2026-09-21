"""Shared B=2 purchased-view broker for Phase 6B observation methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guard.active_vision.rgb_detector import PAID_CAMERAS, RGBDetector
from guard.active_vision.phase6a7 import sha256_bytes


@dataclass(frozen=True)
class PurchasedROI:
    camera_id: str
    roi_rgb: np.ndarray
    state_before: str
    state_after: str


class PurchasedViewBroker:
    def __init__(self, supplier, budget: int = 2):
        if budget != 2:
            raise ValueError("Phase 6B primary broker budget is frozen at B=2")
        self._supplier = supplier
        self.budget = budget
        self.ledger: list[dict] = []

    @property
    def remaining(self) -> int:
        return self.budget - len(self.ledger)

    def purchase(self, camera_id: str) -> PurchasedROI:
        if camera_id not in PAID_CAMERAS:
            raise ValueError(f"unregistered paid camera: {camera_id}")
        if self.remaining <= 0:
            raise RuntimeError("B=2 observation budget exhausted")
        if any(row["camera_id"] == camera_id for row in self.ledger):
            raise RuntimeError("a paid view may be purchased only once")
        result = self._supplier(camera_id)
        if not isinstance(result, PurchasedROI) or result.camera_id != camera_id:
            raise TypeError("supplier must return PurchasedROI for requested camera")
        if result.state_before != result.state_after:
            raise RuntimeError("camera query mutated simulator state")
        if not isinstance(result.roi_rgb, np.ndarray) or result.roi_rgb.dtype != np.uint8:
            raise TypeError("supplier ROI must be uint8 numpy array")
        self.ledger.append({
            "camera_id": camera_id,
            "roi_sha256": sha256_bytes(np.ascontiguousarray(result.roi_rgb).tobytes()),
            "state_fingerprint": result.state_before,
            "query_index": len(self.ledger),
        })
        return result


class RGBObservationSource:
    def __init__(self, broker: PurchasedViewBroker, detector: RGBDetector):
        self.broker = broker
        self.detector = detector

    def query(self, camera_id: str) -> dict:
        purchased = self.broker.purchase(camera_id)
        return self.detector.predict({"camera_id": camera_id, "roi_rgb": purchased.roi_rgb})


class OracleObservationSource:
    def __init__(self, broker: PurchasedViewBroker, table: dict[str, str]):
        self.broker = broker
        self._table = dict(table)

    def query(self, camera_id: str) -> dict:
        self.broker.purchase(camera_id)
        if camera_id not in self._table:
            raise KeyError(camera_id)
        return {"camera_id": camera_id, "predicted_outcome": self._table[camera_id], "source": "oracle_roi"}
