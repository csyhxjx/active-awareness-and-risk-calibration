"""Exact eight-state belief planner for the Phase 6A2 demo."""

from functools import lru_cache
from itertools import product


STATES = tuple("".join(bits) for bits in product("01", repeat=3))
ROUTES = ("left_route", "center_route", "right_route")
ROUTE_INDEX = {route: index for index, route in enumerate(ROUTES)}


def oracle_outcome(state, visible_routes):
    parts = []
    for route in ROUTES:
        if route in visible_routes:
            parts.append(f"{route}:{'blocked' if state[ROUTE_INDEX[route]] == '1' else 'clear'}")
    return "|".join(parts) if parts else "unobserved"


def posterior(belief, visible_routes, observation):
    kept = [probability if oracle_outcome(state, visible_routes) == observation else 0.0 for state, probability in zip(STATES, belief)]
    evidence = sum(kept)
    if evidence <= 0:
        raise ValueError("zero-evidence observation")
    return tuple(value / evidence for value in kept)


def blocked_probability(belief, route):
    index = ROUTE_INDEX[route]
    return sum(probability for state, probability in zip(STATES, belief) if state[index] == "1")


def terminal_value(belief, action, collision_cost=4.0, stop_cost=0.25):
    if action == "stop":
        return -stop_cost
    blocked = blocked_probability(belief, action)
    return (1.0 - blocked) - collision_cost * blocked


def plan(belief, queries, remaining, query_cost=0.05, collision_cost=4.0, stop_cost=0.25):
    """Return the exact value-of-information action under a query-count budget."""
    if len(belief) != 8 or abs(sum(belief) - 1.0) > 1e-10:
        raise ValueError("eight-state normalized belief required")

    @lru_cache(maxsize=None)
    def solve(current, names, budget):
        candidates = []
        for rank, action in enumerate((*ROUTES, "stop")):
            value = terminal_value(current, action, collision_cost, stop_cost)
            candidates.append((value, 1, -rank, {"kind": "terminal", "id": action, "expected_value": value}))
        if budget:
            for name in names:
                visible = tuple(queries[name])
                outcomes = sorted({oracle_outcome(state, visible) for state in STATES})
                expected = -query_cost
                next_names = tuple(item for item in names if item != name)
                for outcome in outcomes:
                    probability = sum(p for state, p in zip(STATES, current) if oracle_outcome(state, visible) == outcome)
                    if probability:
                        child = solve(posterior(current, visible, outcome), next_names, budget - 1)
                        expected += probability * child[0]
                candidates.append((expected, 0, 0, {"kind": "query", "id": name, "expected_value": expected}))
        best = max(candidates, key=lambda row: row[:-1])
        return best[0], best[-1]

    return solve(tuple(float(value) for value in belief), tuple(sorted(queries)), int(remaining))[1]
