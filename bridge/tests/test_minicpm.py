"""Protocol tests use localhost fixtures, never a real model or training run."""

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch

from bridge import BridgeConfig, BridgeError, MiniCPMBridge, parse_neural_drive_plan, parse_plan, parse_scenario_plan

try:
    from websockets.sync.server import serve
except ImportError:
    serve = None


PLAN = {"task": "navigate", "goal_mm": [3, 0], "say": "把目标设为前方三毫米。"}
DOOM_SPEC = {"name": "doom_basic", "title": "Doom 三维瞄准射击",
             "task_updates": {"step_limit": "integer from 1 to 1000", "target_kills": "only 1"},
             "limitations": ["one target only"]}
PORTS = {name: {"cell_type": "LB3c" if name == "sugar" else "R1-R6", "normalized_range": [0, 1],
                "neuron_ids": ["fixture"], "limits": "engineering input, not calibrated physiology"}
         for name in ("retina_left", "retina_right", "sugar")}
DRIVE = {"neural_drive": {"retina_left": 0.7, "retina_right": 0.1, "sugar": 0}, "say": "提出工程输入调制，等待真实反馈。"}


@contextmanager
def http_fixture(content=None, status=200, redirect=None):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append({"path": self.path, "headers": dict(self.headers), "body": data})
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = content if content is not None else {"choices": [{"message": {"content": json.dumps(PLAN, ensure_ascii=False)}}]}
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class BridgeValidationTests(unittest.TestCase):
    def test_default_is_disabled_without_network(self):
        bridge = MiniCPMBridge(BridgeConfig.from_env({}))
        with patch("urllib.request.build_opener") as opener:
            self.assertEqual(bridge.status()["status"], "disabled")
            with self.assertRaisesRegex(BridgeError, "disabled"):
                bridge.plan("去香蕉旁边")
            opener.assert_not_called()

    def test_modelbest_requires_key_and_keeps_key_out_of_repr(self):
        config = BridgeConfig.from_env({"MINICPM_TRANSPORT": "modelbest"})
        self.assertEqual(config.model, "MiniCPM-O-4.5-9B")
        self.assertEqual(config.base_url, "https://api.modelbest.cn/v1")
        with self.assertRaisesRegex(BridgeError, "MINICPM_API_KEY"):
            MiniCPMBridge(config)
        secret = BridgeConfig(api_key="fixture-secret")
        self.assertNotIn("fixture-secret", repr(secret))

    def test_invalid_configuration(self):
        cases = [
            BridgeConfig(transport="mystery"),
            BridgeConfig(transport="openai", base_url="http://127.0.0.1:8000/v1"),
            BridgeConfig(transport="modelbest", base_url="https://other.example/v1", api_key="fixture", model="test"),
            BridgeConfig(transport="openai", base_url="https://secret@host/v1", model="test"),
            BridgeConfig(transport="realtime", realtime_url="ws://127.0.0.1:8006/ws/chat", model="test"),
            BridgeConfig(timeout_s=float("nan")),
        ]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(BridgeError):
                MiniCPMBridge(config)

    def test_structured_plan_rejects_unbounded_or_imperative_output(self):
        config = BridgeConfig(transport="openai", model="fixture", max_goal_mm=10)
        invalid = [
            {**PLAN, "goal_mm": [True, 0]},
            {**PLAN, "goal_mm": [float("nan"), 0]},
            {**PLAN, "goal_mm": [float("inf"), 0]},
            {**PLAN, "goal_mm": [11, 0]},
            {**PLAN, "goal_mm": [10**400, 0]},
            {**PLAN, "goal_mm": [0]},
            {**PLAN, "task": "set_joint_torque"},
            {**PLAN, "task": "stop"},
            {**PLAN, "script": "print('unexpected')"},
            {**PLAN, "say": ""},
            [PLAN],
        ]
        for data in invalid:
            with self.subTest(data=str(data)[:100]), self.assertRaises(BridgeError):
                parse_plan(json.dumps(data), config)
        for text in ['{"task":"stop","task":"explore","goal_mm":null,"say":"x"}', json.dumps(PLAN) + " extra"]:
            with self.assertRaises(BridgeError):
                parse_plan(text, config)
        result = parse_plan(json.dumps(PLAN), config)
        self.assertEqual(result.goal_mm, (3.0, 0.0))
        self.assertEqual(result.to_dict()["goal_mm"], [3.0, 0.0])


class HTTPProtocolTests(unittest.TestCase):
    def test_actual_http_request_contains_model_auth_state_and_image(self):
        with http_fixture() as (url, observed), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII=")
            path.write_bytes(image_bytes)
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="minicpm-fixture", api_key="fixture-secret"))
            result = bridge.plan("到目标位置", {"position_mm": [0, 0]}, path)
            self.assertEqual(result.task, "navigate")
            self.assertEqual(result.status, "model_response")
            self.assertEqual(len(observed), 1)
            sent = observed[0]
            self.assertEqual(sent["path"], "/v1/chat/completions")
            self.assertEqual(sent["headers"]["Authorization"], "Bearer fixture-secret")
            self.assertEqual(sent["body"]["model"], "minicpm-fixture")
            self.assertFalse(sent["body"]["stream"])
            user = sent["body"]["messages"][1]["content"]
            self.assertIn("position_mm", user[0]["text"])
            self.assertEqual(user[1]["image_url"]["url"], "data:image/png;base64," + base64.b64encode(image_bytes).decode())

    def test_http_errors_and_wrong_shape_are_explicit(self):
        for status, body in [(401, {"error": "fixture-secret"}), (200, {"no_choices": True})]:
            with http_fixture(body, status) as (url, _):
                bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture"))
                with self.assertRaises(BridgeError) as ctx:
                    bridge.plan("解释状态")
                self.assertNotIn("fixture-secret", str(ctx.exception))

    def test_http_redirect_is_not_followed(self):
        with http_fixture(status=307, redirect="http://127.0.0.1:9/leak") as (url, observed):
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture", api_key="fixture-secret"))
            with self.assertRaisesRegex(BridgeError, "307"):
                bridge.plan("解释状态")
            self.assertEqual(len(observed), 1)

    def test_invalid_model_json_never_becomes_a_fallback_task(self):
        with http_fixture({"choices": [{"message": {"content": "已经训练成功！"}}]}) as (url, _):
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture"))
            with self.assertRaisesRegex(BridgeError, "JSON"):
                bridge.plan("去香蕉旁边")


class NeuralDriveTests(unittest.TestCase):
    def test_neural_request_sends_port_contract_and_observed_feedback(self):
        content = {"choices": [{"message": {"content": json.dumps(DRIVE)}}]}
        state = {"neural": {"spikes_this_step": 28529}, "body": {"position_mm": [1, 2]},
                 "lastdrive": {"retina_left": 0, "retina_right": 0, "sugar": 0}}
        with http_fixture(content) as (url, observed):
            result = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture")).plan_neural_drive("提高左侧输入", state, PORTS)
            self.assertEqual(result.neural_drive, DRIVE["neural_drive"])
            self.assertEqual(result.to_dict()["status"], "model_response")
            messages = observed[0]["body"]["messages"]
            self.assertIn("LB3c", messages[0]["content"])
            self.assertIn("not calibrated physiology", messages[0]["content"])
            self.assertNotIn("neuron_ids", messages[0]["content"])
            self.assertIn('"spikes_this_step": 28529', messages[1]["content"])
            self.assertIn('"lastdrive"', messages[1]["content"])

    def test_neural_plan_requires_exact_fields_and_finite_bounded_numbers(self):
        config = BridgeConfig(transport="openai", model="fixture")
        invalid = [{**DRIVE, "neural_drive": {**DRIVE["neural_drive"], "sugar": value}}
                   for value in (-0.1, 1.1, True, "0.4", None, float("nan"), float("inf"), -float("inf"), 10**400)]
        invalid += [{**DRIVE, "neural_drive": {"retina_left": 0, "retina_right": 0}},
                    {**DRIVE, "neural_drive": {**DRIVE["neural_drive"], "motor": 0}},
                    {**DRIVE, "goal_mm": [1, 0]}, {**DRIVE, "say": ""},
                    {"neural_drive": DRIVE["neural_drive"]}, [DRIVE]]
        for data in invalid:
            with self.subTest(data=str(data)[:100]), self.assertRaises(BridgeError):
                parse_neural_drive_plan(json.dumps(data), config)
        for text in [json.dumps(DRIVE) + " extra", "```json\n" + json.dumps(DRIVE) + "\n```",
                     '{"neural_drive":{"retina_left":0,"retina_left":1,"retina_right":0,"sugar":0},"say":"x"}']:
            with self.assertRaises(BridgeError):
                parse_neural_drive_plan(text, config)
        for value in (0, 1):
            result = parse_neural_drive_plan(json.dumps({**DRIVE, "neural_drive": dict.fromkeys(PORTS, value)}), config)
            self.assertTrue(all(type(v) is float for v in result.neural_drive.values()))

    def test_neural_disabled_or_invalid_ports_cannot_send_a_request(self):
        with patch("urllib.request.build_opener") as opener:
            with self.assertRaisesRegex(BridgeError, "disabled"):
                MiniCPMBridge(BridgeConfig()).plan_neural_drive("观察", {}, PORTS)
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url="http://127.0.0.1:1/v1", model="fixture"))
            for ports in ({}, {**PORTS, "motor": {}}, {**PORTS, "sugar": {"count": float("nan")}}):
                with self.assertRaises(BridgeError):
                    bridge.plan_neural_drive("观察", {}, ports)
            opener.assert_not_called()

    def test_natural_chat_returns_model_text_without_control(self):
        content = {"choices": [{"message": {"content": "我可以和你讨论眼前的果蝇。"}}]}
        with http_fixture(content) as (url, observed):
            result = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture")).chat("你好", {"neural": {"neurons": 166700}})
            self.assertEqual(result.task, "explain")
            self.assertIsNone(result.goal_mm)
            self.assertEqual(result.say, content["choices"][0]["message"]["content"])
            self.assertIn('"neurons": 166700', observed[0]["body"]["messages"][1]["content"])


class AudioProtocolTests(unittest.TestCase):
    def test_real_http_audio_upload_and_playable_wave_response(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setparams((1, 2, 24000, 100, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 100)
        audio = stream.getvalue()
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path.endswith("/transcriptions"):
                    message = BytesParser(policy=default).parsebytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                    parts = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True) for part in message.iter_parts()}
                    observed.append((self.path, parts))
                    result = json.dumps({"text": "输入的真实转写占位测试。"}).encode()
                else:
                    observed.append((self.path, json.loads(raw)))
                    result = audio
                self.send_response(200)
                self.end_headers()
                self.wfile.write(result)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fixture"))
            result = bridge.transcribe(audio)
            self.assertEqual(result["text"], "输入的真实转写占位测试。")
            self.assertEqual(result["status"], "model_response")
            self.assertEqual(bridge.speak("测试朗读"), audio)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(observed[0][0], "/v1/audio/transcriptions")
        self.assertEqual(observed[0][1]["file"], audio)
        self.assertEqual(observed[0][1]["model"], b"fixture")
        self.assertEqual(observed[1], ("/v1/audio/speech", {"model": "fixture", "input": "测试朗读", "voice": "default", "response_format": "wav"}))

    def test_bad_audio_and_disabled_speech_have_no_network_fallback(self):
        bridge = MiniCPMBridge(BridgeConfig())
        with patch("urllib.request.build_opener") as opener:
            for raw in (b"", b"not a real wave" * 10):
                with self.assertRaises(BridgeError):
                    bridge.transcribe(raw)
            with self.assertRaises(BridgeError):
                bridge.speak("你好")
            opener.assert_not_called()


class ScenarioPlanningTests(unittest.TestCase):
    def test_scenario_request_sends_dynamic_contract_and_preserves_state(self):
        plan = {"operation": "configure", "parameters": {"step_limit": 30}, "say": "把本回合限制设为 30 步。"}
        content = {"choices": [{"message": {"content": json.dumps(plan)}}]}
        with http_fixture(content) as (url, observed):
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url=url, model="fixture"))
            result = bridge.plan_for_scenario("把回合上限改为30步", {"step_limit": 100}, DOOM_SPEC)
            self.assertEqual(result.operation, "configure")
            self.assertEqual(result.parameters, {"step_limit": 30})
            self.assertEqual(result.to_dict()["status"], "model_response")
            messages = observed[0]["body"]["messages"]
            self.assertIn("doom_basic", messages[0]["content"])
            self.assertIn("target_kills", messages[0]["content"])
            self.assertIn("one target only", messages[0]["content"])
            self.assertIn('"step_limit": 100', messages[1]["content"])
            self.assertNotIn("goal_mm", messages[0]["content"])

    def test_scenario_plan_rejects_wrong_fields_operations_and_nonfinite_values(self):
        config = BridgeConfig(transport="openai", model="fixture")
        base = {"operation": "configure", "parameters": {"step_limit": 30}, "say": "设置上限。"}
        invalid = [
            {**base, "operation": "attack"},
            {**base, "parameters": {"goal_mm": [4, 1]}},
            {**base, "parameters": {}},
            {**base, "operation": "pause"},
            {**base, "parameters": {"step_limit": [float("nan")]}},
            {**base, "parameters": {"step_limit": {"nested": float("inf")}}},
            {**base, "parameters": []},
            {**base, "say": ""},
            {**base, "extra": "field"},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(BridgeError):
                parse_scenario_plan(json.dumps(data), config, DOOM_SPEC)
        for text in [json.dumps(base) + " extra", "```json\n" + json.dumps(base) + "\n```",
                     '{"operation":"pause","parameters":{},"parameters":{},"say":"停。"}']:
            with self.assertRaises(BridgeError):
                parse_scenario_plan(text, config, DOOM_SPEC)

    def test_pause_reset_explain_and_no_dynamic_parameters(self):
        config = BridgeConfig(transport="openai", model="fixture")
        empty_spec = {"name": "custom_environment", "task_updates": {}}
        for operation in ["pause", "reset", "explain"]:
            result = parse_scenario_plan(json.dumps({"operation": operation, "parameters": {}, "say": "遵照任务意图。"}), config, empty_spec)
            self.assertEqual(result.operation, operation)
            self.assertEqual(result.parameters, {})

    def test_invalid_spec_and_disabled_mode_do_not_send_requests(self):
        with patch("urllib.request.build_opener") as opener:
            bridge = MiniCPMBridge(BridgeConfig(transport="openai", base_url="http://127.0.0.1:1/v1", model="fixture"))
            with self.assertRaises(BridgeError):
                bridge.plan_for_scenario("configure", {}, {"task_updates": ["step_limit"]})
            disabled = MiniCPMBridge(BridgeConfig())
            with self.assertRaisesRegex(BridgeError, "disabled"):
                disabled.plan_for_scenario("解释", {}, DOOM_SPEC)
            opener.assert_not_called()


@unittest.skipIf(serve is None, "Optional websockets dependency is not installed")
class RealtimeProtocolTests(unittest.TestCase):
    def test_real_local_websocket_lifecycle(self):
        observed = []
        errors = []

        def handler(ws):
            try:
                observed.append({"path": ws.request.path})
                ws.send(json.dumps({"type": "session.queued", "position": 1}))
                ws.send(json.dumps({"type": "session.queue_done"}))
                observed.append(json.loads(ws.recv(timeout=3)))
                ws.send(json.dumps({"type": "session.created", "session_id": "fixture"}))
                observed.append(json.loads(ws.recv(timeout=3)))
                ws.send(json.dumps({"type": "response.done", "text": json.dumps(PLAN)}))
                observed.append(json.loads(ws.recv(timeout=3)))
            except Exception as exc:
                errors.append(exc)

        with serve(handler, "127.0.0.1", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.socket.getsockname()[1]
                bridge = MiniCPMBridge(BridgeConfig(transport="realtime", realtime_url=f"ws://127.0.0.1:{port}/v1/realtime?mode=chat", model="fixture", timeout_s=5))
                result = bridge.plan("走到 [3,0]", state={"position_mm": [0, 0]})
                self.assertEqual(result.goal_mm, (3.0, 0.0))
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(observed[0]["path"], "/v1/realtime?mode=chat")
        self.assertEqual(observed[1], {"type": "session.init", "payload": {}})
        self.assertEqual(observed[2]["type"], "input.append")
        self.assertFalse(observed[2]["input"]["tts"]["enabled"])
        self.assertEqual(observed[3], {"type": "session.close", "reason": "turn_done"})


if __name__ == "__main__":
    unittest.main()
