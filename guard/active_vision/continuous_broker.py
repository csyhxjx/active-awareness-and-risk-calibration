"""Input-isolating B=2 broker for Phase 6C depth features."""

from __future__ import annotations

from guard.active_vision.continuous_belief import CAMERAS, validate_observation


class ContinuousDepthBroker:
    def __init__(self, supplier, budget: int = 2):
        if budget != 2:
            raise ValueError("Phase 6C primary budget is frozen at B=2")
        self._supplier = supplier
        self.budget = budget
        self.ledger = []

    def query(self, camera: str) -> dict:
        if camera not in CAMERAS:
            raise PermissionError(camera)
        if len(self.ledger) >= self.budget:
            raise PermissionError("B=2 exhausted")
        if any(row["camera_id"] == camera for row in self.ledger):
            raise PermissionError("view already purchased")
        payload = self._supplier(camera)
        if set(payload) != {"observation", "state_before", "state_after", "feature_sha256"}:
            raise ValueError("supplier payload contains missing or forbidden fields")
        if payload["state_before"] != payload["state_after"]:
            raise RuntimeError("depth query changed simulator state")
        validate_observation(payload["observation"])
        self.ledger.append({
            "camera_id": camera,
            "feature_sha256": payload["feature_sha256"],
            "state_fingerprint": payload["state_before"],
            "query_index": len(self.ledger),
        })
        return payload["observation"]
