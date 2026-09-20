"""Exact six-state adaptive-branching contract for Phase 6A3."""

from functools import lru_cache
from itertools import permutations


STATES = ("001", "110", "010", "101", "011", "100")
ROUTES = ("left_route", "center_route", "right_route")
ROUTE_INDEX = {route: index for index, route in enumerate(ROUTES)}
CAMERAS = ("q_branch", "q_a", "q_b", "q_c")
FAMILIES = {
    "family_a": ("001", "110"),
    "family_b": ("010", "101"),
    "family_c": ("011", "100"),
}
SPECIALIST = {"family_a": "q_a", "family_b": "q_b", "family_c": "q_c"}
STATE_FAMILY = {state: family for family, states in FAMILIES.items() for state in states}
PRIOR = tuple(1.0 / len(STATES) for _ in STATES)


def _build_observation_table():
    table = {}
    for state in STATES:
        family = STATE_FAMILY[state]
        table[(state, "q_branch")] = family
        for candidate_family, camera in SPECIALIST.items():
            table[(state, camera)] = f"{candidate_family}:{state}" if family == candidate_family else "not_applicable"
    return table


OBSERVATION_TABLE = _build_observation_table()


def observation(state, camera):
    if state not in STATES or camera not in CAMERAS:
        raise ValueError((state, camera))
    return OBSERVATION_TABLE[(state, camera)]


def posterior(belief, camera, outcome):
    if len(belief) != len(STATES) or abs(sum(belief) - 1.0) > 1e-10:
        raise ValueError("normalized six-state belief required")
    kept = [probability if observation(state, camera) == outcome else 0.0 for state, probability in zip(STATES, belief)]
    evidence = sum(kept)
    if evidence <= 0:
        raise ValueError("zero-evidence observation")
    return tuple(value / evidence for value in kept)


def support(belief):
    return tuple(state for state, probability in zip(STATES, belief) if probability > 0.0)


def blocked_probability(belief, route):
    index = ROUTE_INDEX[route]
    return sum(probability for state, probability in zip(STATES, belief) if state[index] == "1")


def terminal_value(belief, action, collision_cost=4.0, stop_cost=0.25):
    if action == "stop":
        return -stop_cost
    blocked = blocked_probability(belief, action)
    return (1.0 - blocked) - collision_cost * blocked


def terminal_decision(belief, collision_cost=4.0, stop_cost=0.25):
    ranked = []
    for rank, action in enumerate((*ROUTES, "stop")):
        value = terminal_value(belief, action, collision_cost, stop_cost)
        ranked.append((value, -rank, action))
    value, _, action = max(ranked)
    return {"kind": "terminal", "id": action, "expected_value": value}


def plan(belief, cameras=CAMERAS, remaining=2, query_cost=0.05, collision_cost=4.0, stop_cost=0.25):
    """Return an exact value-of-information action for the registered contract."""
    if len(belief) != len(STATES) or abs(sum(belief) - 1.0) > 1e-10:
        raise ValueError("normalized six-state belief required")

    @lru_cache(maxsize=None)
    def solve(current, names, budget):
        terminal = terminal_decision(current, collision_cost, stop_cost)
        candidates = [(terminal["expected_value"], 1, 0, terminal)]
        if budget:
            for camera in names:
                outcomes = sorted({observation(state, camera) for state in STATES})
                expected = -query_cost
                next_names = tuple(name for name in names if name != camera)
                for outcome in outcomes:
                    probability = sum(p for state, p in zip(STATES, current) if observation(state, camera) == outcome)
                    if probability:
                        child = solve(posterior(current, camera, outcome), next_names, budget - 1)
                        expected += probability * child[0]
                candidates.append((expected, 0, -CAMERAS.index(camera), {
                    "kind": "query", "id": camera, "expected_value": expected,
                }))
        best = max(candidates, key=lambda row: row[:-1])
        return best[0], best[-1]

    return solve(tuple(float(value) for value in belief), tuple(cameras), int(remaining))[1]


def run_adaptive(state):
    belief = PRIOR
    available = list(CAMERAS)
    trace = []
    remaining = 2
    while True:
        decision = plan(belief, tuple(available), remaining)
        row = {"support": support(belief), "decision": decision}
        trace.append(row)
        if decision["kind"] == "terminal":
            break
        camera = decision["id"]
        outcome = observation(state, camera)
        row["observation"] = outcome
        belief = posterior(belief, camera, outcome)
        available.remove(camera)
        remaining -= 1
    return trace


def evaluate_fixed(sequence, query_cost=0.05, collision_cost=4.0, stop_cost=0.25):
    trials = []
    for state in STATES:
        belief = PRIOR
        trace = []
        for camera in sequence:
            decision = terminal_decision(belief, collision_cost, stop_cost)
            if decision["id"] != "stop":
                break
            outcome = observation(state, camera)
            trace.append({"support": support(belief), "camera": camera, "observation": outcome})
            belief = posterior(belief, camera, outcome)
        terminal = terminal_decision(belief, collision_cost, stop_cost)
        action = terminal["id"]
        collision = action in ROUTES and state[ROUTE_INDEX[action]] == "1"
        success = action in ROUTES and not collision
        utility = (1.0 if success else -collision_cost if collision else -stop_cost) - query_cost * len(trace)
        trials.append({
            "state": state, "sequence": tuple(sequence), "trace": trace, "terminal_action": action,
            "success": success, "collision": collision, "utility": utility,
        })
    return {
        "sequence": tuple(sequence),
        "trials": trials,
        "completion": sum(row["success"] for row in trials),
        "collision": sum(row["collision"] for row in trials),
        "stop": sum(row["terminal_action"] == "stop" for row in trials),
        "mean_queries": sum(len(row["trace"]) for row in trials) / len(trials),
        "mean_utility": sum(row["utility"] for row in trials) / len(trials),
    }


def all_fixed_results():
    return tuple(
        evaluate_fixed(sequence)
        for length in range(3)
        for sequence in permutations(CAMERAS, length)
    )
