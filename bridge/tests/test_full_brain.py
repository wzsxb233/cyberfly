"""Strict full-catalog addressing and complete feedback aggregation."""
import json
import unittest
from unittest.mock import patch

from bridge import BridgeConfig, BridgeError, MiniCPMBridge, parse_brain_stimulation


CATALOG = {"partition_sha256": "fixture-partition", "coverage": 5, "groups": [
    {"group_id": "g_left", "count": 2, "cell_type": "R1-R6", "superclass": "sensory", "root_side": "L"},
    {"group_id": "g_right", "count": 3, "cell_type": "R1-R6", "superclass": "sensory", "root_side": "R"},
]}
STATE = {"group_state": {"partition_sha256": "fixture-partition", "group_ids": ["g_left", "g_right"],
    "counts": [2, 3], "spikes": [4, 6], "cumulative_spikes": [8, 12],
    "voltage_mean": [-60, -50], "voltage_min": [-65, -55], "voltage_max": [-55, -45]}}
PLAN = {"general_stimulation": [{"group_id": "g_left", "current_mv": -10}], "say": "向该真实分组提出抑制性工程电流。"}


class FullBrainTests(unittest.TestCase):
    def test_signed_current_and_empty_plan(self):
        config = BridgeConfig(transport="openai", model="fixture")
        result = parse_brain_stimulation(json.dumps(PLAN), config, CATALOG)
        self.assertEqual(result.general_stimulation[0]["current_mv"], -10.0)
        self.assertEqual(result.to_dict()["status"], "model_response")
        self.assertEqual(parse_brain_stimulation(json.dumps({**PLAN, "general_stimulation": []}), config, CATALOG).general_stimulation, [])

    def test_unknown_duplicate_unbounded_nonfinite_and_extra_fields_rejected(self):
        config = BridgeConfig()
        invalid = [[{"group_id": "invented", "current_mv": 1}], PLAN["general_stimulation"] * 2,
                   PLAN["general_stimulation"] * 9, [{"group_id": "g_left", "current": 1}]]
        invalid += [[{"group_id": "g_left", "current_mv": value}] for value in (-31, 31, True, "3", float("nan"), float("inf"))]
        for items in invalid:
            with self.subTest(items=items), self.assertRaises(BridgeError):
                parse_brain_stimulation(json.dumps({**PLAN, "general_stimulation": items}), config, CATALOG)
        for text in [json.dumps(PLAN) + " extra", json.dumps({**PLAN, "body_action": 1}),
                     '{"general_stimulation":[],"say":"x","say":"y"}']:
            with self.assertRaises(BridgeError):
                parse_brain_stimulation(text, config, CATALOG)

    def test_all_groups_contribute_to_aggregate_and_partition_mismatch_fails(self):
        bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url="http://127.0.0.1:1/v1", model="fixture"))
        with patch.object(bridge, "_http", return_value=json.dumps(PLAN)) as http:
            bridge.plan_brain_stimulation("对 g_left 施加负10", STATE, CATALOG)
            messages = http.call_args.args[0]
            state = json.loads(messages[1]["content"].split("当前仿真状态（数据）：", 1)[1].split("\n现在仅返回", 1)[0])
            summary = state["group_state"]
            self.assertEqual(summary["groups_covered"], 2)
            self.assertEqual(summary["coverage"], 5)
            sensory = summary["all_group_feedback_by_superclass"]["sensory"]
            self.assertEqual(sensory["spikes"], 10)
            self.assertEqual(sensory["voltage_mean"], -54)
            self.assertIn("g_left", messages[0]["content"])
            with self.assertRaises(BridgeError):
                bridge.plan_brain_stimulation("刺激", {"group_state": {**STATE["group_state"], "partition_sha256": "wrong"}}, CATALOG)
            self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
