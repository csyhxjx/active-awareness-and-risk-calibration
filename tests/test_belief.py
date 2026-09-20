import unittest

from guard.active_vision.belief import (
    bayes_update,
    map_belief,
    plan_exact,
    plan_map,
    validate_query,
)


LEFT_SENSOR = {
    "cost": 0.05,
    "likelihood": {
        "clear": (1.0, 0.0, 1.0, 0.0),
        "blocked": (0.0, 1.0, 0.0, 1.0),
    },
}
RIGHT_SENSOR = {
    "cost": 0.04,
    "likelihood": {
        "clear": (1.0, 1.0, 0.0, 0.0),
        "blocked": (0.0, 0.0, 1.0, 1.0),
    },
}


class ExactBeliefTest(unittest.TestCase):
    def test_bayes_update_and_contradiction(self):
        posterior = bayes_update((0.4, 0.35, 0.2, 0.05), LEFT_SENSOR, "blocked")
        for actual, expected in zip(posterior, (0.0, 0.875, 0.0, 0.125)):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            bayes_update((1.0, 0.0, 0.0, 0.0), LEFT_SENSOR, "blocked")

    def test_query_likelihood_must_normalize(self):
        invalid = {"cost": 1, "likelihood": {"x": (0.5, 1.0, 1.0, 1.0)}}
        with self.assertRaises(ValueError):
            validate_query(invalid)

    def test_map_tie_order(self):
        self.assertEqual(map_belief((0.5, 0.5, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0))

    def test_uninformative_query_is_not_bought(self):
        query = {"cost": 0.01, "likelihood": {"same": (1.0, 1.0, 1.0, 1.0)}}
        decision = plan_exact((1.0, 0.0, 0.0, 0.0), {"q": query}, 1.0)
        self.assertEqual(decision, {"kind": "terminal", "id": "left_route", "expected_value": 1.0})

    def test_exact_queries_and_switches_while_map_executes(self):
        prior = (0.4, 0.35, 0.2, 0.05)
        exact = plan_exact(prior, {"inspect_left": LEFT_SENSOR}, 0.05)
        point = plan_map(prior, {"inspect_left": LEFT_SENSOR}, 0.05)
        self.assertEqual(exact["kind"], "query")
        self.assertEqual(point["id"], "left_route")
        posterior = bayes_update(prior, LEFT_SENSOR, "blocked")
        after = plan_exact(posterior, {}, 0.0)
        self.assertEqual(after["id"], "right_route")

    def test_query_ties_use_lower_cost_then_id(self):
        prior = (0.4, 0.35, 0.2, 0.05)
        expensive = {**LEFT_SENSOR, "cost": 0.06}
        decision = plan_exact(prior, {"z": expensive, "b": LEFT_SENSOR, "a": LEFT_SENSOR}, 0.06)
        self.assertEqual(decision["id"], "a")

    def test_complementary_updates_are_order_replayable(self):
        prior = (0.25, 0.25, 0.25, 0.25)
        left_then_right = bayes_update(bayes_update(prior, LEFT_SENSOR, "blocked"), RIGHT_SENSOR, "clear")
        right_then_left = bayes_update(bayes_update(prior, RIGHT_SENSOR, "clear"), LEFT_SENSOR, "blocked")
        self.assertEqual(left_then_right, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(left_then_right, right_then_left)

    def test_query_over_budget_is_not_available(self):
        decision = plan_exact((0.4, 0.35, 0.2, 0.05), {"inspect_left": LEFT_SENSOR}, 0.049)
        self.assertEqual(decision["kind"], "terminal")


if __name__ == "__main__":
    unittest.main()
