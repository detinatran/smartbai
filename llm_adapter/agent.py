"""Model-independent tool loop and Qwen3.5 entry points for smartbai runners."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from .prompts import system_prompt, correction_feedback
from .protocol import (
    ACTION_TOOLS,
    READ_TOOLS,
    KNOWN_TOOLS,
    ProtocolError,
    parse_transport,
    parse_response,
    validate_arguments,
    validate_report,
    evidence_flags,
)

PROTOCOL_VERSION = "smartbai_json_agent_v1"


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "decision"
    max_turns: int = 12

    def __post_init__(self):
        if self.mode not in {"decision", "certify"}:
            raise ValueError("mode must be decision or certify")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("max_turns must be positive")


class AgentRunError(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def code_hashes():
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("agent.py", "prompts.py", "protocol.py")
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalize_observation(data):
    """Represent unavailable nonfinite server numbers as null, with an audit trail.

    The upstream server emits NaN for some undefined KPIs at reset. JSON has no
    NaN value; never turn it into zero or let it become numerical evidence.
    """
    missing = []

    def visit(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            missing.append({"path": path, "original": repr(value)})
            return None
        if isinstance(value, dict):
            return {k: visit(v, path + "/" + str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, path + "/" + str(i)) for i, v in enumerate(value)]
        return value

    return visit(data, ""), missing


def discover_tools(client, mode, schema=None):
    task_info = client.task()
    if (
        not isinstance(task_info, dict)
        or "error" in task_info
        or not isinstance(task_info.get("tools"), list)
    ):
        raise ProtocolError("Cannot discover the current case's allowed tool list")
    if schema is None:
        if not hasattr(client, "_req"):
            raise ProtocolError("Provide tools_schema for non-HTTP clients")
        schema = client._req("GET", "/tools")
    if not isinstance(schema, dict) or "error" in schema:
        raise ProtocolError("Cannot read tool schemas from /tools")
    permitted = READ_TOOLS | {"submit_decision"} if mode == "certify" else KNOWN_TOOLS
    enabled = set(task_info["tools"]) & set(schema) & permitted
    tools = {name: deepcopy(schema[name]) for name in sorted(enabled)}
    if "submit_decision" not in tools:
        raise ProtocolError("submit_decision is unavailable")
    for name, spec in tools.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("params"), dict):
            raise ProtocolError("Invalid params schema for " + name)
    tools["submit_decision"]["params"]["report"] = (
        "object with exactly candidates (<=5 action lists), rationale, claims"
        if mode == "certify"
        else "object with exactly actions (text summaries), rationale, claims"
    )
    if "run_whatif" in tools:
        tools["run_whatif"]["params"][
            "actions"
        ] = "list of action objects: schedule_maintenance, set_speed, set_dispatch, set_cooling"
    return task_info, tools


def run_case(
    client,
    task,
    generate,
    *,
    config=None,
    trace_dir="out/smartbai_agent/traces",
    backend_identity=None,
    tools_schema=None,
    progress=None,
):
    """Run one attempt. The caller owns case reset and completed-case resume.

    Never reuse a cached report after the caller resets the server: actions and
    observation IDs belong to that live attempt. Every retry gets a fresh trace.
    ``generate`` is injected; importing this module never loads model weights.
    """
    config = config or AgentConfig()
    if not isinstance(task, str):
        raise TypeError("task must be text")
    info, tools = discover_tools(client, config.mode, tools_schema)
    if info.get("task") != task:
        raise ProtocolError("Task does not match the current live server case")
    case_id = str(info.get("case_id") or "case_" + hashlib.sha256(task.encode()).hexdigest()[:10])
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", case_id)[:90]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    path = Path(trace_dir) / config.mode / (safe_id + "_" + stamp + ".json")
    prompt = system_prompt(config.mode, tools)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": task}]
    record = {
        "case_id": case_id,
        "mode": config.mode,
        "protocol_version": PROTOCOL_VERSION,
        "max_turns": config.max_turns,
        "backend_identity": backend_identity or {"backend": "injected_unspecified"},
        "code_sha256": code_hashes(),
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tools": tools,
        "messages": messages,
        "trace": [],
        "trace_path": str(path),
        "finished": False,
        "report": None,
        "error": None,
        "phase": "ready",
        "evidence": None,
    }
    observations = {}
    start = time.perf_counter()
    save_json(path, record)
    try:
        for turn in range(1, config.max_turns + 1):
            record["phase"] = "generating"
            save_json(path, record)
            if progress is not None:
                progress(f"{case_id} turn {turn}/{config.max_turns}: generating")
            generated = generate(messages)
            if not isinstance(generated, dict) or not isinstance(generated.get("raw_output"), str):
                raise TypeError("generate(messages) must return a dict with string raw_output")
            # Reject non-JSON metadata before writing a corrupted trace.
            json.dumps(generated, allow_nan=False)
            entry = {
                "turn": turn,
                **generated,
                "format_valid": False,
                "command_valid": False,
                "tool_error": None,
                "observation": None,
            }
            record["trace"].append(entry)
            raw = generated["raw_output"]
            messages.append({"role": "assistant", "content": raw})
            try:
                parse_transport(raw)
                entry["format_valid"] = True
                reason, tool, args = parse_response(raw)
                entry.update(reason=reason, tool=tool, arguments=args, command_valid=True)
                if tool not in tools:
                    raise ProtocolError("Tool not allowed in this case/mode: " + tool)
                if set(args) - set(tools[tool]["params"]):
                    raise ProtocolError("Parameters not in server schema")
                validate_arguments(tool, args)
                if tool == "submit_decision":
                    validate_report(args["report"], config.mode, observations)
                    entry["evidence"] = evidence_flags(args["report"], observations)
                # Persist intent before a potentially mutating call. Transport errors abort
                # the attempt: outcome could be uncertain, so no automatic action retry.
                record["phase"] = "tool_pending"
                save_json(path, record)
                if progress is not None:
                    progress(f"{case_id} turn {turn}: calling {tool}")
                try:
                    response = client.call(tool, **args)
                except Exception as exc:
                    raise RuntimeError(
                        "Tool transport failed; outcome may be uncertain, so the case is stopped without retry: "
                        + str(exc)
                    ) from exc
                if not isinstance(response, dict):
                    raise RuntimeError("Tool server did not return an object")
                response, nonfinite = normalize_observation(response)
                entry["nonfinite_server_values"] = nonfinite
                entry["observation"] = response
                if type(response.get("http_status", 200)) is not int:
                    raise RuntimeError("Invalid HTTP status in tool response; case stopped")
                if "error" in response or response.get("http_status", 200) >= 400:
                    entry["tool_error"] = str(response.get("error", "HTTP tool error"))
                    feedback = {
                        "tool": tool,
                        "error": entry["tool_error"],
                        "http_status": response.get("http_status"),
                    }
                else:
                    if (
                        not isinstance(response.get("observation_id"), str)
                        or "result" not in response
                    ):
                        raise RuntimeError(
                            "Tool response lacks observation_id/result; stop because action outcome may be uncertain"
                        )
                    observations[response["observation_id"]] = response
                    feedback = {"tool": tool, "observation": response}
                    if nonfinite:
                        feedback["unavailable_values"] = nonfinite
                        feedback["notice"] = (
                            "Nonfinite server values are null (unavailable), not zero or evidence of success."
                        )
                    if tool == "submit_decision":
                        if response["result"].get("received") is not True:
                            raise RuntimeError("Submission was not acknowledged")
                        record.update(
                            finished=True,
                            report=args["report"],
                            evidence=entry["evidence"],
                            phase="finished",
                        )
                        break
            except (ProtocolError, SyntaxError, ValueError, TypeError) as exc:
                # Parser/report-validation failures are explicit model feedback, never
                # silently repaired text or fabricated successful tool observations.
                entry["tool_error"] = type(exc).__name__ + ": " + str(exc)
                feedback = correction_feedback(exc, config.mode, entry["format_valid"])
            messages.append(
                {
                    "role": "user",
                    "content": "Tool feedback: "
                    + json.dumps(feedback, ensure_ascii=False, allow_nan=False),
                }
            )
            record["phase"] = "ready"
            save_json(path, record)
        if not record["finished"]:
            record.update(error="max_turns_reached", phase="failed")
    except BaseException as exc:
        record.update(
            error=type(exc).__name__ + ": " + str(exc),
            phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            save_json(path, record)
            raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - start
        record["model_input_tokens"] = sum(t.get("input_tokens", 0) for t in record["trace"])
        record["model_output_tokens"] = sum(t.get("output_tokens", 0) for t in record["trace"])
        record["model_latency_seconds"] = sum(t.get("latency_seconds", 0) for t in record["trace"])
        save_json(path, record)
    if not record["finished"]:
        raise AgentRunError(record["error"], record)
    return record


def _qwen_agent(client, task, mode):
    from dataclasses import asdict
    from .backend_qwen35 import get_backend

    backend = get_backend()
    trace_root = Path(os.environ.get("SMARTBAI_TRACE_DIR", "out/smartbai_agent/traces"))
    config = AgentConfig(mode=mode, max_turns=int(os.environ.get("SMARTBAI_MAX_TURNS", "12")))
    try:
        return run_case(
            client,
            task,
            backend.generate,
            config=config,
            trace_dir=trace_root,
            backend_identity={"backend": "qwen35", "config": asdict(backend.config)},
        )
    finally:
        if backend.loaded:
            backend.write_metadata(trace_root)


def agent(client, task):
    """Default entry point requested in eval/README.md: direct decision mode."""
    return _qwen_agent(client, task, "decision")


def decision_agent(client, task):
    return _qwen_agent(client, task, "decision")


def certify_agent(client, task):
    """Entry point for eval/certify/proposers.py; read-only candidate proposer."""
    return _qwen_agent(client, task, "certify")
