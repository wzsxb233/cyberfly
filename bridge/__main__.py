"""Run from the cyberfly directory: python -m bridge status|plan ..."""

import argparse
import json
import sys

from . import BridgeError, MiniCPMBridge


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniCPM 高层任务接口；默认不联网，不执行仿真动作。")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="只检查配置，不联网")
    plan = commands.add_parser("plan", help="向显式配置的实际模型服务请求一个任务")
    plan.add_argument("text")
    plan.add_argument("--state-json", default="{}")
    plan.add_argument("--image", help="显式发送的本地截图路径")
    args = parser.parse_args()
    try:
        bridge = MiniCPMBridge()
        if args.command == "status":
            result = bridge.status()
        else:
            state = json.loads(args.state_json)
            if not isinstance(state, dict):
                raise BridgeError("--state-json 必须为 JSON 对象。")
            result = bridge.plan(args.text, state=state, image_path=args.image).to_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BridgeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "task_executed": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
