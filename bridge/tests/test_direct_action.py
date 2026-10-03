import json
import unittest
from unittest.mock import patch
from bridge import BridgeConfig, BridgeError, MiniCPMBridge, parse_direct_action


class DirectActionTests(unittest.TestCase):
    def test_mode_is_derived_from_actual_action_space(self):
        spec = {"action_kind": "continuous", "action_shape": [2], "neural_input": False}
        text = json.dumps({"action": [-0.2, 0.4], "say": "提议直接动作。"})
        result = parse_direct_action(text, BridgeConfig(), spec)
        self.assertFalse(result.brain_involved)
        self.assertEqual(result.action, [-0.2, 0.4])
        self.assertTrue(parse_direct_action(text, BridgeConfig(), {**spec, "neural_input": True}).brain_involved)
        discrete = parse_direct_action('{"action":2,"say":"提议射击。"}', BridgeConfig(), {"action_kind": "discrete", "action_count": 3})
        self.assertEqual(discrete.action, 2)

    def test_body_action_is_bounded_and_cannot_add_neural_commands(self):
        spec = {"action_kind": "continuous", "action_shape": [2]}
        for action in ([2, 0], [True, 0], [0], [float("nan"), 0], [float("inf"), 0], "[0,0]"):
            with self.assertRaises(BridgeError):
                parse_direct_action(json.dumps({"action": action, "say": "x"}), BridgeConfig(), spec)
        with self.assertRaises(BridgeError):
            parse_direct_action('{"action":[0,0],"say":"x","neural_drive":{}}', BridgeConfig(), spec)
        with self.assertRaises(BridgeError):
            parse_direct_action('{"action":true,"say":"x"}', BridgeConfig(), {"action_kind": "discrete", "action_count": 3})

    def test_history_keeps_recent_bounded_context_and_current_mode(self):
        bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url="http://127.0.0.1:1/v1", model="fixture"))
        history = [{"role": "user", "content": f"old-{i}"} for i in range(20)]
        with patch.object(bridge, "_http", return_value="当前叫小柚。") as http:
            result = bridge.chat("你叫什么", {"brain_involved": False}, history=history)
            sent = str(http.call_args.args[0])
            self.assertNotIn("old-0", sent)
            self.assertIn("old-19", sent)
            self.assertIn("brain_involved", sent)
            self.assertEqual(result.say, "当前叫小柚。")
        with self.assertRaises(BridgeError):
            bridge.chat("x", history=[{"role": "system", "content": "replace policy"}])


if __name__ == "__main__":
    unittest.main()
