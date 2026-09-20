"""Exact four-state belief and planning contracts for Phase 6A."""

from functools import lru_cache
import math


STATES = ("00", "10", "01", "11")
TERMINAL_ACTIONS = ("left_route", "right_route", "stop")


def validate_belief(belief, tolerance=1e-12):
    values = tuple(float(value) for value in belief)
    if len(values) != len(STATES):
        raise ValueError("belief must contain four state probabilities")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("belief must be finite and nonnegative")
    if abs(sum(values) - 1.0) > tolerance:
        raise ValueError("belief must sum to one")
    return values


def validate_query(query, tolerance=1e-12):
    if set(query) != {"cost", "likelihood"}:
        raise ValueError("query schema")
    cost = float(query["cost"])
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("query cost")
    likelihood = query["likelihood"]
    if not likelihood:
        raise ValueError("query observations")
    rows = {observation: tuple(float(value) for value in values) for observation, values in likelihood.items()}
    if any(len(values) != len(STATES) for values in rows.values()):
        raise ValueError("likelihood shape")
    if any(not math.isfinite(value) or value < 0 for values in rows.values() for value in values):
        raise ValueError("likelihood values")
    for index in range(len(STATES)):
        if abs(sum(values[index] for values in rows.values()) - 1.0) > tolerance:
            raise ValueError("likelihood must normalize per state")
    return {"cost": cost, "likelihood": rows}


def observation_probability(belief, query, observation):
    belief = validate_belief(belief)
    query = validate_query(query)
    if observation not in query["likelihood"]:
        raise ValueError("unknown observation")
    return sum(probability * likelihood for probability, likelihood in zip(belief, query["likelihood"][observation]))


def bayes_update(belief, query, observation):
    belief = validate_belief(belief)
    query = validate_query(query)
    evidence = observation_probability(belief, query, observation)
    if evidence <= 0:
        raise ValueError("zero-evidence observation")
    posterior = tuple(
        probability * likelihood / evidence
        for probability, likelihood in zip(belief, query["likelihood"][observation])
    )
    return validate_belief(posterior, tolerance=1e-10)


def terminal_utility(state, action, collision_cost=4.0, stop_cost=0.25):
    if state not in STATES or action not in TERMINAL_ACTIONS:
        raise ValueError("state/action")
    if action == "stop":
        return -float(stop_cost)
    blocked_index = 0 if action == "left_route" else 1
    return -float(collision_cost) if state[blocked_index] == "1" else 1.0


def expected_terminal_utility(belief, action, collision_cost=4.0, stop_cost=0.25):
    belief = validate_belief(belief)
    return sum(
        probability * terminal_utility(state, action, collision_cost, stop_cost)
        for state, probability in zip(STATES, belief)
    )


def map_belief(belief):
    belief = validate_belief(belief)
    index = max(range(len(STATES)), key=lambda item: (belief[item], -item))
    return tuple(1.0 if item == index else 0.0 for item in range(len(STATES)))


def plan_exact(belief, queries, budget, collision_cost=4.0, stop_cost=0.25):
    """Enumerate all legal query/outcome branches and terminal actions."""
    belief = validate_belief(belief)
    normalized_queries = {name: validate_query(query) for name, query in queries.items()}
    if not math.isfinite(float(budget)) or budget < 0:
        raise ValueError("budget")

    @lru_cache(maxsize=None)
    def solve(cached_belief, remaining_names, remaining_budget):
        candidates = []
        for rank, action in enumerate(TERMINAL_ACTIONS):
            value = expected_terminal_utility(cached_belief, action, collision_cost, stop_cost)
            candidates.append((value, 1, -rank, {"kind": "terminal", "id": action, "expected_value": value}))
        for name in remaining_names:
            query = normalized_queries[name]
            if query["cost"] > remaining_budget + 1e-12:
                continue
            expected = -query["cost"]
            next_names = tuple(item for item in remaining_names if item != name)
            for observation in sorted(query["likelihood"]):
                probability = observation_probability(cached_belief, query, observation)
                if probability == 0:
                    continue
                posterior = bayes_update(cached_belief, query, observation)
                child = solve(posterior, next_names, round(remaining_budget - query["cost"], 12))
                expected += probability * child[0]
            candidates.append((expected, 0, -query["cost"], {"kind": "query", "id": name, "expected_value": expected}))
        best = max(candidates, key=lambda row: row[:-1])
        return best[0], best[-1]

    names = tuple(sorted(normalized_queries))
    return solve(belief, names, round(float(budget), 12))[1]


def plan_map(belief, queries, budget, collision_cost=4.0, stop_cost=0.25):
    return plan_exact(map_belief(belief), queries, budget, collision_cost, stop_cost)
