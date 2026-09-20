import unittest

from guard.active_vision.belief_scene import DEMO_VISIBILITY
from guard.active_vision.belief_three_route import blocked_probability, oracle_outcome, plan, posterior


class ThreeRouteBeliefTest(unittest.TestCase):
    def test_oracle_update(self):
        belief = tuple([1 / 8] * 8)
        updated = posterior(belief, ("left_route",), "left_route:blocked")
        self.assertEqual(blocked_probability(updated, "left_route"), 1.0)
        self.assertEqual(blocked_probability(updated, "right_route"), 0.5)

    def test_unobserved_does_not_update(self):
        belief = tuple([1 / 8] * 8)
        self.assertEqual(posterior(belief, (), "unobserved"), belief)

    def test_registered_adaptive_trace(self):
        belief = tuple([1 / 8] * 8)
        first = plan(belief, DEMO_VISIBILITY, 2)
        self.assertEqual(first["id"], "q_left")
        belief = posterior(belief, DEMO_VISIBILITY["q_left"], oracle_outcome("100", DEMO_VISIBILITY["q_left"]))
        remaining = {key: value for key, value in DEMO_VISIBILITY.items() if key != "q_left"}
        second = plan(belief, remaining, 1)
        self.assertEqual(second["id"], "q_right")
        belief = posterior(belief, DEMO_VISIBILITY["q_right"], oracle_outcome("100", DEMO_VISIBILITY["q_right"]))
        terminal = plan(belief, {}, 0)
        self.assertEqual(terminal["id"], "right_route")


if __name__ == "__main__": unittest.main()
