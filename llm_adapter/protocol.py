"""Literal-only command parsing and explicit validation for smartbai tools."""

import ast
import json
import math
import re

ACTION_TOOLS = {"schedule_maintenance", "set_speed", "set_dispatch", "set_cooling"}
READ_TOOLS = {
    "get_line_state",
    "get_station",
    "get_sensor_history",
    "predict_failure",
    "get_quality",
}
KNOWN_TOOLS = READ_TOOLS | ACTION_TOOLS | {"run_whatif", "submit_decision"}


class ProtocolError(ValueError):
    pass


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("Non-finite numbers are not allowed")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ProtocolError("Object keys must be strings")
        for v in value.values():
            _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ProtocolError("Only JSON-compatible literal arguments are allowed")


def _object_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ProtocolError("Duplicate JSON key: " + key)
        out[key] = value
    return out


def parse_transport(raw):
    if not isinstance(raw, str) or len(raw) > 200000:
        raise ProtocolError("Expected a bounded JSON response string")
    try:
        obj = json.loads(
            raw.strip(),
            object_pairs_hook=_object_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(
                ProtocolError("Invalid JSON constant: " + v)
            ),
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON: " + str(exc)) from exc
    if not isinstance(obj, dict) or set(obj) != {"reason", "command"}:
        raise ProtocolError("Respond with exactly two JSON string keys: reason and command")
    if not isinstance(obj["reason"], str) or not isinstance(obj["command"], str):
        raise ProtocolError("reason and command must be strings")
    return obj


def parse_response(raw):
    obj = parse_transport(raw)
    try:
        node = ast.parse(obj["command"].strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ProtocolError("Expected one named tool call")
        if node.args or any(k.arg is None for k in node.keywords):
            raise ProtocolError("Use explicit keyword arguments; no positional args or **kwargs")
        if len({k.arg for k in node.keywords}) != len(node.keywords):
            raise ProtocolError("Duplicate keyword arguments")
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        _finite(args)
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError("Invalid literal tool call: " + str(exc)) from exc
    return obj["reason"], node.func.id, args


def number(value, name, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(name + " must be a finite number")
    if (
        positive
        and value <= 0
        or minimum is not None
        and value < minimum
        or maximum is not None
        and value > maximum
    ):
        raise ProtocolError(name + " is outside the allowed range")


def station(value):
    if not isinstance(value, str) or not re.fullmatch(r"S\d+", value):
        raise ProtocolError("station_id must be a station identifier such as S32")


def validate_action(action):
    if not isinstance(action, dict) or action.get("action") not in ACTION_TOOLS:
        raise ProtocolError("Unknown candidate action")
    kind = action["action"]
    validate_arguments(kind, {k: v for k, v in action.items() if k != "action"})


def validate_arguments(tool, args):
    required = {
        "get_line_state": set(),
        "get_station": {"station_id"},
        "get_sensor_history": {"station_id", "signal"},
        "predict_failure": {"station_id", "horizon_h"},
        "get_quality": set(),
        "run_whatif": {"actions"},
        "schedule_maintenance": {"station_id"},
        "set_speed": {"station_id", "pct"},
        "set_dispatch": {"weights"},
        "set_cooling": {"level"},
        "submit_decision": {"report"},
    }
    optional = {
        "get_sensor_history": {"hours"},
        "get_quality": {"hours"},
        "run_whatif": {"hours", "reps"},
        "schedule_maintenance": {"in_minutes"},
        "set_dispatch": {"hours"},
    }
    if tool not in required:
        raise ProtocolError("Tool is outside the adapter's supported registry")
    if not required[tool] <= set(args) or set(args) - required[tool] - optional.get(tool, set()):
        raise ProtocolError("Missing or unexpected parameters for " + tool)
    _finite(args)
    if "station_id" in args:
        station(args["station_id"])
    if tool == "get_sensor_history" and args["signal"] not in {
        "motor_temp",
        "vibration",
        "queue",
        "status",
    }:
        raise ProtocolError("Unknown sensor signal")
    if "hours" in args and args["hours"] is not None:
        number(
            args["hours"],
            "hours",
            maximum=(
                24
                if tool == "get_quality"
                else (48 if tool in {"get_sensor_history", "run_whatif"} else None)
            ),
            positive=True,
        )
    if tool == "predict_failure":
        number(args["horizon_h"], "horizon_h", positive=True)
    if tool == "schedule_maintenance" and "in_minutes" in args:
        number(args["in_minutes"], "in_minutes", minimum=0)
    if tool == "set_speed":
        number(args["pct"], "pct", minimum=30, maximum=150)
    if tool == "set_cooling" and args["level"] not in {"normal", "high"}:
        raise ProtocolError("Invalid cooling level")
    if tool == "set_dispatch":
        weights = args["weights"]
        if not isinstance(weights, dict) or not weights:
            raise ProtocolError("weights must be a nonempty object")
        for key, value in weights.items():
            station(key)
            number(value, "weight", minimum=0)
        if not any(weights.values()):
            raise ProtocolError("At least one dispatch weight must be positive")
    if tool == "run_whatif":
        if not isinstance(args["actions"], list):
            raise ProtocolError("actions must be a list")
        for action in args["actions"]:
            validate_action(action)
        if "reps" in args:
            if type(args["reps"]) is not int or not 1 <= args["reps"] <= 10:
                raise ProtocolError("reps must be an integer from 1 to 10")


def validate_report(report, mode, observations):
    main_key = "candidates" if mode == "certify" else "actions"
    if not isinstance(report, dict) or set(report) != {main_key, "rationale", "claims"}:
        raise ProtocolError("Report requires exactly " + main_key + ", rationale, claims")
    if not isinstance(report["rationale"], str) or not report["rationale"].strip():
        raise ProtocolError("rationale must be a nonempty string")
    if not isinstance(report[main_key], list):
        raise ProtocolError(main_key + " must be a list")
    if mode == "certify":
        if len(report[main_key]) > 5:
            raise ProtocolError("At most five candidates are allowed")
        for candidate in report[main_key]:
            if not isinstance(candidate, list) or not candidate:
                raise ProtocolError(
                    "Each candidate must be a nonempty action list; use candidates=[] if proposing none"
                )
            for action in candidate:
                validate_action(action)
    elif any(not isinstance(a, str) for a in report["actions"]):
        raise ProtocolError(
            "decision actions are text summaries; issue actual actions through tools"
        )
    if not isinstance(report["claims"], list):
        raise ProtocolError("claims must be a list")
    for claim in report["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"text", "value", "observation_id"}:
            raise ProtocolError("Each claim requires text, value, observation_id")
        if not isinstance(claim["text"], str) or not claim["text"].strip():
            raise ProtocolError("Claim text is empty")
        if not isinstance(claim["value"], (str, int, float)) or isinstance(claim["value"], bool):
            raise ProtocolError("Claim value must be a string or finite number")
        oid = claim["observation_id"]
        if not isinstance(oid, str) or oid not in observations:
            raise ProtocolError("Claim observation_id was not observed successfully in this case")


def _numbers(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers(v)]
    if isinstance(obj, str):
        try:
            return [float(obj)]
        except ValueError:
            return []
    return []


def value_matches(value, result):
    """Match eval/score.py's diagnostic, not a semantic correctness certificate."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return any(
            abs(n - value) <= (0.01 if abs(n) < 1 else 0.02 * abs(n)) for n in _numbers(result)
        )
    if isinstance(value, str):
        try:
            return value_matches(float(re.sub(r"[^0-9eE+.-]", "", value)), result)
        except ValueError:
            return value.lower() in json.dumps(result).lower()
    return False


def evidence_flags(report, observations):
    claims = report.get("claims", [])
    unsupported = mismatch = 0
    for claim in claims:
        obs = observations.get(claim.get("observation_id"))
        if obs is None:
            unsupported += 1
        elif not value_matches(claim["value"], obs["result"]):
            mismatch += 1
    return {
        "claims": len(claims),
        "unsupported": unsupported,
        "mismatch": mismatch,
        "scope": "eval/score.py value-in-observation diagnostic; claim semantics C/R are not automatically scored",
    }
