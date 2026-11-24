"""Development-only local HTTP harness. No oracle or calibration/test evaluation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading

from .agent import (
    AgentConfig,
    AgentRunError,
    code_hashes,
    run_case,
    save_json,
    normalize_observation,
)
from .protocol import ACTION_TOOLS


class ScriptedSmoke:
    """Explicit test fixture, never presented as a real LLM or policy baseline."""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, messages):
        if messages[-1]["role"] != "user" or not messages[-1]["content"].startswith(
            "Tool feedback: "
        ):
            tool, args = "get_line_state", {}
        else:
            feedback = json.loads(messages[-1]["content"].removeprefix("Tool feedback: "))
            obs = feedback["observation"]
            report = {
                "rationale": "Scripted transport check only; this is not an LLM decision.",
                "claims": [
                    {
                        "text": "Outdoor temperature (C)",
                        "value": obs["result"]["outdoor_c"],
                        "observation_id": obs["observation_id"],
                    }
                ],
            }
            if self.mode == "certify":
                report["candidates"] = [
                    [
                        {
                            "action": "set_speed",
                            "station_id": obs["result"]["stations"][0]["station_id"],
                            "pct": 100,
                        }
                    ]
                ]
            else:
                report["actions"] = []
            tool, args = "submit_decision", {"report": report}
        command = tool + "(" + ", ".join(k + "=" + repr(v) for k, v in args.items()) + ")"
        return {
            "raw_output": json.dumps({"reason": "Scripted plumbing fixture", "command": command}),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "hit_generation_limit": False,
            "rendered_prompt_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
        }


def run_development(
    *,
    repo,
    artifacts,
    out,
    mode,
    backend_name,
    limit=1,
    max_turns=12,
    no_whatif=False,
    nf4=False,
    resume=False,
    max_new_tokens=384
):
    """Run only Certify dev specs, even in direct-decision plumbing mode.

    Finished cases are skipped before reset on resume. Failed/interrupted cases
    start a fresh server state/attempt, with previous trace files preserved.
    """
    repo, artifacts, out = Path(repo).resolve(), Path(artifacts).resolve(), Path(out).resolve()
    if not 1 <= limit <= 36:
        raise ValueError("Development harness limit must be from 1 to 36")
    if backend_name not in {"scripted", "qwen35"}:
        raise ValueError("Unknown backend")
    if nf4 and backend_name != "qwen35":
        raise ValueError("NF4 applies only to Qwen3.5")
    config = AgentConfig(mode, max_turns)
    paths = {
        "config": artifacts / "inputs/twin_v2_config.json",
        "climate": artifacts / "inputs/dubai_hourly_2015_2025_era5.csv",
        "model": artifacts / "inputs/failure_model.json",
        "cases": artifacts / "inputs/cases_certify_v2.json",
    }
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
    data = json.loads(paths["cases"].read_text(encoding="utf-8"))
    if data["version"] != "v2":
        raise ValueError("Expected Certify v2 inputs")
    selected = [c for c in data["cases"] if c["split"] == "dev"][:limit]
    if len(selected) != limit:
        raise ValueError("Insufficient development cases")
    backend = None
    if backend_name == "qwen35":
        from .backend_qwen35 import Qwen35Config, get_backend

        backend = get_backend(Qwen35Config(load_in_4bit=nf4, max_new_tokens=max_new_tokens))
        identity = {"backend": "qwen35", "config": asdict(backend.config)}
        generate = backend.generate
    else:
        identity = {"backend": "scripted_test_fixture", "not_an_llm": True}
        generate = ScriptedSmoke(mode)
    fingerprint = {
        "protocol_version": "smartbai_json_agent_v1",
        "mode": mode,
        "max_turns": max_turns,
        "no_whatif": no_whatif,
        "backend": identity,
        "inputs_sha256": hashes,
        "agent_code_sha256": code_hashes(),
        "backend_sha256": hashlib.sha256(
            (Path(__file__).parent / "backend_qwen35.py").read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": "dev",
    }
    manifest_path = out / "run_manifest.json"
    result_path = out / "results.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError(
                "Run exists. Use --resume with identical settings or choose a new --out."
            )
        if previous["fingerprint"] != fingerprint:
            raise RuntimeError("Run configuration changed; use a new --out.")
    else:
        if any(out.iterdir()):
            raise RuntimeError("Output folder is not empty; choose a fresh run directory")
        save_json(
            manifest_path,
            {
                "fingerprint": fingerprint,
                "scope": "development/plumbing only, not official decision30 scoring or Certify policy evaluation",
            },
        )
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["finished"]:
                done.add(row["case_id"])
    pending = [s for s in selected if s["case_id"] not in done]
    if not pending:
        return {"out": str(out), "skipped_completed": len(selected), "attempted": 0, "finished": 0}

    # Admin reset lives only in this harness; the model-facing run_case cannot read hidden state.
    sys.path.insert(0, str(repo / "sim"))
    sys.path.insert(0, str(repo / "eval"))
    from http.server import ThreadingHTTPServer
    from line_sim import load_climate
    from tool_server import Twin, make_handler
    from agents import ToolClient
    from score import evidence

    twin = Twin(
        json.loads(paths["config"].read_text(encoding="utf-8")),
        load_climate(paths["climate"]),
        json.loads(paths["model"].read_text(encoding="utf-8")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(twin))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    completed = 0
    try:
        client = ToolClient("http://127.0.0.1:" + str(server.server_port))
        for spec in pending:
            disabled = (
                sorted(ACTION_TOOLS | {"run_whatif"})
                if mode == "certify"
                else (["run_whatif"] if no_whatif else [])
            )
            reset = client.admin("reset", dict(spec, disabled_tools=disabled))
            if not reset.get("ok"):
                raise RuntimeError("Server reset failed")
            try:
                record = run_case(
                    client,
                    client.task()["task"],
                    generate,
                    config=config,
                    trace_dir=out / "traces",
                    backend_identity=identity,
                    progress=lambda message: print(message, flush=True),
                )
            except AgentRunError as exc:
                record = exc.record
            case = client.admin("case")
            case_path = out / "server_cases" / (Path(record["trace_path"]).stem + ".json")
            # Save observed public results and actions, not the hidden injection/spec.
            public_case, nonfinite = normalize_observation(
                {k: case[k] for k in ("case_id", "actions", "observations", "decision")}
            )
            public_case["nonfinite_server_values"] = nonfinite
            save_json(case_path, public_case)
            row = {
                "case_id": spec["case_id"],
                "mode": mode,
                "backend": backend_name,
                "finished": record["finished"],
                "error": record["error"],
                "trace_path": record["trace_path"],
                "server_case_path": str(case_path),
                "evidence": record["evidence"],
                "tool_calls": len(case["observations"]),
                "report": record["report"],
                "official_evidence_flags": evidence(case),
                "unverified_action_flag_applicable": mode == "decision" and not no_whatif,
                "scope": (
                    "development only; scripted is a fixture, not LLM output"
                    if backend_name == "scripted"
                    else "development only; no oracle or calibrated policy score"
                ),
            }
            if mode == "certify":
                row["candidates"] = (record["report"] or {}).get("candidates", [])
            with result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
            completed += int(record["finished"])
            print(
                spec["case_id"],
                "finished=",
                record["finished"],
                "error=",
                record["error"],
                flush=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if backend is not None and backend.loaded:
            backend.write_metadata(out)
    return {
        "out": str(out),
        "skipped_completed": len(selected) - len(pending),
        "attempted": len(pending),
        "finished": completed,
    }
