import unittest

from guard.active_vision.belief_branching import (
    CAMERAS,
    FAMILIES,
    OBSERVATION_TABLE,
    PRIOR,
    SPECIALIST,
    STATES,
    all_fixed_results,
    observation,
    plan,
    posterior,
    run_adaptive,
    support,
)
from guard.active_vision.belief_branch_scene import cue_colors


class BranchingBeliefContractTest(unittest.TestCase):
    def test_prior_and_complete_observation_table(self):
        self.assertEqual(PRIOR, (1 / 6,) * 6)
        self.assertEqual(set(OBSERVATION_TABLE), {(state, camera) for state in STATES for camera in CAMERAS})

    def test_first_query_posteriors(self):
        for family, states in FAMILIES.items():
            updated = posterior(PRIOR, "q_branch", family)
            self.assertEqual(support(updated), states)
            self.assertEqual([value for value in updated if value], [0.5, 0.5])
            self.assertEqual(plan(updated, (SPECIALIST[family],), 1)["id"], SPECIALIST[family])

    def test_specialist_posteriors_and_terminal_actions(self):
        expected = {
            "001": "left_route", "110": "right_route", "010": "left_route",
            "101": "center_route", "011": "left_route", "100": "center_route",
        }
        for state in STATES:
            family = observation(state, "q_branch")
            belief = posterior(PRIOR, "q_branch", family)
            camera = SPECIALIST[family]
            belief = posterior(belief, camera, observation(state, camera))
            self.assertEqual(support(belief), (state,))
            self.assertEqual(plan(belief, (), 0)["id"], expected[state])

    def test_adaptive_tree_uses_conditional_second_camera(self):
        for state in STATES:
            trace = run_adaptive(state)
            self.assertEqual([row["decision"]["id"] for row in trace[:2]], ["q_branch", SPECIALIST[observation(state, "q_branch")]])
            self.assertNotEqual(trace[-1]["decision"]["id"], "stop")

    def test_adaptive_strictly_beats_every_fixed_sequence(self):
        adaptive_values = [run_adaptive(state)[0]["decision"]["expected_value"] for state in STATES]
        for value in adaptive_values:
            self.assertAlmostEqual(value, 0.9)
        fixed = all_fixed_results()
        best = max(row["mean_utility"] for row in fixed)
        self.assertAlmostEqual(best, 0.5)
        self.assertTrue(all(row["completion"] < 6 for row in fixed))
        self.assertEqual(max(row["completion"] for row in fixed), 4)

    def test_physical_cue_colors_follow_observation_partition(self):
        for camera in CAMERAS:
            by_observation = {}
            for state in STATES:
                outcome = observation(state, camera)
                color = tuple(cue_colors(state)[camera])
                by_observation.setdefault(outcome, set()).add(color)
            self.assertTrue(all(len(colors) == 1 for colors in by_observation.values()))
            self.assertEqual(len({next(iter(colors)) for colors in by_observation.values()}), len(by_observation))


if __name__ == "__main__":
    unittest.main()
