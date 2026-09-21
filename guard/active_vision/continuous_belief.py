"""Pure Phase 6C continuous-belief and decision contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


ROUTES = ("left_route", "center_route", "right_route")
CAMERAS = ("q_branch", "q_a", "q_b", "q_c")
SPECIALIST = {"q_a": 0, "q_b": 1, "q_c": 2}
STATE_DIM = 10
CLEARANCE_THRESHOLD_M = 0.004
RISK_THRESHOLD = 0.05
QUERY_COST = 0.05
STOP_UTILITY = -0.25
COLLISION_UTILITY = -4.0
SUCCESS_UTILITY = 1.0
ROBOT_ENVELOPE_M = 0.025
BRANCH_QUANTIZATION_M = 0.008
SPECIALIST_SIGMA_M = 0.0015


class ContinuousContractError(ValueError):
    pass


def _validate_particles(particles: np.ndarray) -> np.ndarray:
    value = np.asarray(particles, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != STATE_DIM or not np.isfinite(value).all():
        raise ContinuousContractError(f"particles must be finite Nx{STATE_DIM}")
    return value


def transform_unit_samples(unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != STATE_DIM or np.any((unit < 0.0) | (unit > 1.0)):
        raise ContinuousContractError("unit samples must be Nx10 in [0,1]")
    result = np.empty_like(unit)
    result[:, 0] = 2.0 * unit[:, 0] - 1.0
    for route_index in range(3):
        source = 1 + route_index * 3
        result[:, source] = 0.04 + 0.06 * unit[:, source]
        epsilon = -0.07 + 0.14 * unit[:, source + 1]
        result[:, source + 1] = np.clip(0.030 * result[:, 0] + epsilon, -0.10, 0.10)
        result[:, source + 2] = 0.025 + 0.030 * unit[:, source + 2]
    return result


def sample_prior(count: int, seed: int, *, qmc: bool = False) -> np.ndarray:
    if count <= 0:
        raise ContinuousContractError("count must be positive")
    if qmc:
        import torch

        engine = torch.quasirandom.SobolEngine(STATE_DIM, scramble=True, seed=int(seed))
        unit = engine.draw(int(count)).cpu().numpy().astype(np.float64)
    else:
        unit = np.random.default_rng(int(seed)).random((int(count), STATE_DIM))
    return transform_unit_samples(unit)


def route_clearances(particles: np.ndarray) -> np.ndarray:
    particles = _validate_particles(particles)
    output = np.empty((len(particles), 3), dtype=np.float64)
    for route_index in range(3):
        source = 1 + route_index * 3
        lateral = particles[:, source + 1]
        half_width = particles[:, source + 2]
        output[:, route_index] = np.abs(lateral) - half_width - ROBOT_ENVELOPE_M
    return output


def predicted_observation(particles: np.ndarray, camera: str) -> tuple[np.ndarray, np.ndarray]:
    clearances = route_clearances(particles)
    values = np.zeros_like(clearances)
    mask = np.zeros_like(clearances, dtype=bool)
    if camera == "q_branch":
        values[:] = np.round(clearances / BRANCH_QUANTIZATION_M) * BRANCH_QUANTIZATION_M
        mask[:] = True
    elif camera in SPECIALIST:
        index = SPECIALIST[camera]
        values[:, index] = clearances[:, index]
        mask[:, index] = True
    else:
        raise ContinuousContractError(f"unknown paid camera: {camera}")
    return values, mask


def make_observation(state: np.ndarray, camera: str, seed: int) -> dict:
    particles = _validate_particles(np.asarray(state, dtype=np.float64).reshape(1, -1))
    values, mask = predicted_observation(particles, camera)
    observed = values[0].copy()
    rng = np.random.default_rng(int(seed))
    if camera != "q_branch":
        observed[mask[0]] += rng.normal(0.0, SPECIALIST_SIGMA_M, int(mask[0].sum()))
    return {"camera_id": camera, "values": observed.tolist(), "mask": mask[0].tolist()}


def validate_observation(observation: dict) -> tuple[str, np.ndarray, np.ndarray]:
    if not isinstance(observation, dict) or set(observation) != {"camera_id", "values", "mask"}:
        raise ContinuousContractError("observation requires exactly camera_id, values, mask")
    camera = observation["camera_id"]
    if camera not in CAMERAS:
        raise ContinuousContractError("unregistered camera")
    values = np.asarray(observation["values"], dtype=np.float64)
    mask = np.asarray(observation["mask"], dtype=bool)
    if values.shape != (3,) or mask.shape != (3,) or not np.isfinite(values).all():
        raise ContinuousContractError("observation values/mask must be finite length three")
    _, expected = predicted_observation(np.zeros((1, STATE_DIM)), camera)
    if not np.array_equal(mask, expected[0]):
        raise ContinuousContractError("observation mask violates camera contract")
    return camera, values, mask


def log_likelihood(particles: np.ndarray, observation: dict) -> np.ndarray:
    particles = _validate_particles(particles)
    camera, observed, mask = validate_observation(observation)
    predicted, _ = predicted_observation(particles, camera)
    if camera == "q_branch":
        matches = np.all(np.isclose(predicted[:, mask], observed[mask], atol=1e-12), axis=1)
        return np.where(matches, 0.0, -math.log(1e9))
    residual = (predicted[:, mask] - observed[mask]) / SPECIALIST_SIGMA_M
    return -0.5 * np.square(residual).sum(axis=1) - int(mask.sum()) * math.log(SPECIALIST_SIGMA_M * math.sqrt(2.0 * math.pi))


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ContinuousContractError("log weights must be finite")
    shifted = values - values.max()
    weights = np.exp(shifted)
    total = weights.sum()
    if not math.isfinite(float(total)) or total <= 0.0:
        raise ContinuousContractError("zero or non-finite particle weight")
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or abs(float(weights.sum()) - 1.0) > 1e-9:
        raise ContinuousContractError("normalized finite weights required")
    return float(1.0 / np.square(weights).sum())


def systematic_resample(weights: np.ndarray, seed: int) -> np.ndarray:
    effective_sample_size(weights)
    count = len(weights)
    start = np.random.default_rng(int(seed)).uniform(0.0, 1.0 / count)
    positions = start + np.arange(count) / count
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def particle_hash(particles: np.ndarray, weights: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(particles, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()


def route_risks(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    particles = _validate_particles(particles)
    effective_sample_size(weights)
    if len(weights) != len(particles):
        raise ContinuousContractError("particle/weight count mismatch")
    unsafe = route_clearances(particles) < CLEARANCE_THRESHOLD_M
    return np.clip((unsafe * weights[:, None]).sum(axis=0), 0.0, 1.0)


def terminal_decision(risks: np.ndarray, route_lengths=(1.0, 1.0, 1.0)) -> dict:
    risks = np.asarray(risks, dtype=np.float64)
    if risks.shape != (3,) or not np.isfinite(risks).all() or np.any((risks < 0) | (risks > 1)):
        raise ContinuousContractError("three finite route risks in [0,1] required")
    candidates = []
    for index, route in enumerate(ROUTES):
        if risks[index] <= RISK_THRESHOLD:
            expected = (1.0 - risks[index]) * SUCCESS_UTILITY + risks[index] * COLLISION_UTILITY
            candidates.append((expected, -float(route_lengths[index]), -index, route))
    if not candidates:
        return {"kind": "terminal", "id": "stop", "expected_utility": STOP_UTILITY, "risks": risks.tolist()}
    value, _, _, route = max(candidates)
    return {"kind": "terminal", "id": route, "expected_utility": float(value), "risks": risks.tolist()}


def update_weights(particles: np.ndarray, weights: np.ndarray, observation: dict) -> np.ndarray:
    effective_sample_size(weights)
    if len(weights) != len(particles):
        raise ContinuousContractError("particle/weight count mismatch")
    return normalize_log_weights(np.log(np.maximum(weights, 1e-300)) + log_likelihood(particles, observation))


def expected_query_utility(particles: np.ndarray, weights: np.ndarray, camera: str) -> float:
    predicted, mask = predicted_observation(particles, camera)
    resolution = BRANCH_QUANTIZATION_M if camera == "q_branch" else SPECIALIST_SIGMA_M * 2.0
    keys = np.round(predicted[:, mask[0]] / resolution).astype(np.int64)
    if keys.ndim == 1:
        keys = keys[:, None]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    if len(unique) > 64:
        # Fixed deterministic coarsening prevents data-dependent compute growth.
        inverse = inverse % 64
    expected = -QUERY_COST
    for group in np.unique(inverse):
        selected = inverse == group
        probability = float(weights[selected].sum())
        if probability <= 0.0:
            continue
        child_weights = np.where(selected, weights, 0.0)
        child_weights /= child_weights.sum()
        child = terminal_decision(route_risks(particles, child_weights))
        expected += probability * child["expected_utility"]
    return float(expected)


def decide(particles: np.ndarray, weights: np.ndarray, available_cameras=CAMERAS, remaining_budget: int = 2) -> dict:
    terminal = terminal_decision(route_risks(particles, weights))
    if terminal["id"] != "stop" or remaining_budget <= 0:
        return terminal
    ranked = []
    for camera in available_cameras:
        value = expected_query_utility(particles, weights, camera)
        ranked.append((value, -CAMERAS.index(camera), camera))
    if not ranked:
        return terminal
    value, _, camera = max(ranked)
    if value <= STOP_UTILITY:
        return terminal
    return {"kind": "query", "id": camera, "expected_utility": float(value), "risks": terminal["risks"]}


def map_estimate(seed: int, observations: list[dict]) -> np.ndarray:
    candidates = sample_prior(16 * 32, seed, qmc=True)
    scores = np.zeros(len(candidates), dtype=np.float64)
    for observation in observations:
        scores += log_likelihood(candidates, observation)
    return candidates[int(np.argmax(scores))].copy()


def map_risks(state: np.ndarray) -> np.ndarray:
    clearances = route_clearances(np.asarray(state).reshape(1, -1))[0]
    z = (clearances - CLEARANCE_THRESHOLD_M) / SPECIALIST_SIGMA_M
    # Logistic approximation to the Gaussian lower-tail probability.
    return 1.0 / (1.0 + np.exp(np.clip(1.702 * z, -60.0, 60.0)))


@dataclass
class ParticleFilter:
    particles: np.ndarray
    weights: np.ndarray
    seed: int
    update_index: int = 0
    degeneration_count: int = 0

    @classmethod
    def create(cls, seed: int, count: int = 2048) -> "ParticleFilter":
        if count != 2048:
            raise ContinuousContractError("ordinary PF particle count is frozen at 2048")
        particles = sample_prior(count, seed)
        return cls(particles, np.full(count, 1.0 / count), int(seed))

    def update(self, observation: dict) -> dict:
        before = particle_hash(self.particles, self.weights)
        self.weights = update_weights(self.particles, self.weights, observation)
        ess_before = effective_sample_size(self.weights)
        resampled = ess_before < 1024.0
        if resampled:
            index = systematic_resample(self.weights, self.seed + 1009 * (self.update_index + 1))
            selected = self.particles[index]
            mean = selected.mean(axis=0)
            rng = np.random.default_rng(self.seed + 7919 * (self.update_index + 1))
            scale = selected.std(axis=0, ddof=0) * math.sqrt(1.0 - 0.98 ** 2)
            self.particles = np.clip(0.98 * selected + 0.02 * mean + rng.normal(size=selected.shape) * scale, _LOWER, _UPPER)
            self.weights = np.full(len(self.particles), 1.0 / len(self.particles))
        if not np.isfinite(self.particles).all():
            self.degeneration_count += 1
            raise ContinuousContractError("particle degeneration produced non-finite values")
        self.update_index += 1
        return {
            "before_hash": before,
            "after_hash": particle_hash(self.particles, self.weights),
            "ess_before_resample": ess_before,
            "resampled": resampled,
            "update_index": self.update_index,
            "degeneration_count": self.degeneration_count,
        }


_LOWER = np.array([-1.0, 0.04, -0.10, 0.025, 0.04, -0.10, 0.025, 0.04, -0.10, 0.025])
_UPPER = np.array([1.0, 0.10, 0.10, 0.055, 0.10, 0.10, 0.055, 0.10, 0.10, 0.055])


def reference_belief(seed: int, observations: list[dict], count: int = 2 ** 18) -> tuple[np.ndarray, np.ndarray]:
    if count not in {2 ** 17, 2 ** 18}:
        raise ContinuousContractError("reference count must be nested 2^17 or 2^18")
    particles = sample_prior(count, seed, qmc=True)
    log_weights = np.zeros(count, dtype=np.float64)
    for observation in observations:
        log_weights += log_likelihood(particles, observation)
    return particles, normalize_log_weights(log_weights)
