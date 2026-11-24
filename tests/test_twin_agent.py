"""Protocol and loop checks, without HF downloads, GPU or raw Bosch data."""

import json
from pathlib import Path
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_adapter.agent import AgentConfig, AgentRunError, run_case
from llm_adapter.protocol import ProtocolError, parse_response, parse_transport, validate_arguments
from llm_adapter.prompts import final_transport_example, correction_feedback

SCHEMA = {
    "get_line_state": {"description": "state", "params": {}},
    "set_speed": {"description": "speed", "params": {"station_id": "string", "pct": "number"}},
    "run_whatif": {
        "description": "whatif",
        "params": {"actions": "list", "hours": "number", "reps": "int"},
    },
    "submit_decision": {"description": "report", "params": {"report": "object"}},
}


def output(command):
    return {
        "raw_output": json.dumps({"reason": "Test fixture", "command": command}),
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_seconds": 0.01,
        "hit_generation_limit": False,
        "rendered_prompt_sha256": "fixture",
    }


def report(mode="decision", value=37, oid="obs-00001"):
    result = {
        "rationale": "Fixture rationale",
        "claims": [{"text": "Outdoor temperature", "value": value, "observation_id": oid}],
    }
    result["candidates" if mode == "certify" else "actions"] = (
        [[{"action": "set_speed", "station_id": "S0", "pct": 100}]] if mode == "certify" else []
    )
    return result


def submit(value):
    return "submit_decision(report=" + repr(value) + ")"


class Client:
    def __init__(self):
        self.calls = []
        self.observations = []
        self.decision = None
        self.fail_once = False

    def task(self):
        return {"case_id": "dev_test_0", "task": "Test task", "tools": list(SCHEMA)}

    def _req(self, method, path):
        assert (method, path) == ("GET", "/tools")
        return SCHEMA

    def call(self, tool, **args):
        self.calls.append((tool, args))
        if self.fail_once:
            self.fail_once = False
            return {"error": "temporary test tool error", "http_status": 400}
        if tool == "get_line_state":
            result = {"outdoor_c": 37, "stations": [{"station_id": "S0"}]}
        elif tool == "submit_decision":
            self.decision = args["report"]
            result = {"ok": True, "received": True}
        else:
            result = {"ok": True}
        obs = {
            "observation_id": f"obs-{len(self.observations)+1:05d}",
            "result": result,
            "t_sim": 0,
            "tool": tool,
            "args": args,
        }
        self.observations.append(obs)
        return obs


class Script:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.messages = []

    def __call__(self, messages):
        self.messages.append(json.loads(json.dumps(messages)))
        value = next(self.commands)
        return value if isinstance(value, dict) else output(value)


class Tests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "out/agent-unit-tests" / uuid.uuid4().hex

    def run_agent(self, script, client=None, mode="decision", max_turns=12):
        return run_case(
            client or Client(),
            "Test task",
            script,
            config=AgentConfig(mode, max_turns),
            trace_dir=self.root,
        )

    def test_final_example_has_valid_outer_json_and_python_call(self):
        for mode in ("certify", "decision"):
            reason, tool, args = parse_response(final_transport_example(mode))
            self.assertEqual(tool, "submit_decision")
            self.assertIn("claims", args["report"])
            if mode == "certify":
                self.assertIsInstance(args["report"]["candidates"][0], list)

    def test_missing_outer_brace_feedback_keeps_raw_failure(self):
        raw = output(submit(report()))["raw_output"][:-1]
        script = Script([{"raw_output": raw}, "get_line_state()", submit(report())])
        record = self.run_agent(script)
        self.assertEqual(record["trace"][0]["raw_output"], raw)
        self.assertFalse(record["trace"][0]["format_valid"])
        fb = json.loads(script.messages[1][-1]["content"].removeprefix("Tool feedback: "))
        self.assertEqual(fb["validation_stage"], "outer_json")
        self.assertIn("json_error_position", fb)
        parse_response(fb["complete_syntax_example"])

    def test_flat_candidates_not_dispatched_and_schema_feedback_recovers(self):
        bad = report("certify")
        bad["candidates"] = bad["candidates"][0]
        client = Client()
        script = Script(["get_line_state()", submit(bad), submit(report("certify"))])
        record = self.run_agent(script, client, mode="certify")
        self.assertEqual([t[0] for t in client.calls], ["get_line_state", "submit_decision"])
        fb = json.loads(script.messages[2][-1]["content"].removeprefix("Tool feedback: "))
        self.assertEqual(fb["validation_stage"], "command_or_report")
        self.assertIn("[[action_dict", fb["candidates_hint"])
        self.assertTrue(record["finished"])

    def test_literal_parser_rejects_executable_expressions(self):
        for command in [
            "__import__('os').system('x')",
            "get_line_state(x=open('x'))",
            "obj.tool()",
            "get_line_state(**{})",
            "get_line_state(); set_speed()",
        ]:
            with self.assertRaises(ProtocolError):
                parse_response(output(command)["raw_output"])

    def test_duplicate_json_or_ast_keyword_and_nonfinite_rejected(self):
        for raw in [
            '{"reason":"a","reason":"b","command":"get_line_state()"}',
            output("set_speed(station_id='S0', pct=1e999)")["raw_output"],
            output("set_speed(station_id='S0', pct=50, pct=60)")["raw_output"],
        ]:
            with self.assertRaises(ProtocolError):
                parse_response(raw)

    def test_ranges_and_unknown_parameters(self):
        for tool, args in [
            ("set_speed", {"station_id": "S0", "pct": 151}),
            ("run_whatif", {"actions": [], "reps": 11}),
            ("get_line_state", {"unexpected": 1}),
        ]:
            with self.assertRaises(ProtocolError):
                validate_arguments(tool, args)

    def test_success_and_every_turn_saved(self):
        client = Client()
        record = self.run_agent(Script(["get_line_state()", submit(report())]), client)
        self.assertTrue(record["finished"])
        self.assertEqual(record["model_input_tokens"], 20)
        self.assertEqual(record["evidence"]["mismatch"], 0)
        self.assertEqual(len(json.loads(Path(record["trace_path"]).read_text())["trace"]), 2)
        self.assertEqual([t[0] for t in client.calls], ["get_line_state", "submit_decision"])

    def test_undefined_server_kpi_is_null_with_audit_not_zero(self):
        class UndefinedKpi(Client):
            def call(self, tool, **args):
                obs = super().call(tool, **args)
                if tool == "get_line_state":
                    obs["result"]["undefined_kpi"] = float("nan")
                return obs

        script = Script(["get_line_state()", submit(report())])
        record = self.run_agent(script, UndefinedKpi())
        first = record["trace"][0]
        self.assertIsNone(first["observation"]["result"]["undefined_kpi"])
        self.assertEqual(
            first["nonfinite_server_values"], [{"path": "/result/undefined_kpi", "original": "nan"}]
        )
        self.assertIn("unavailable_values", script.messages[1][-1]["content"])
        json.dumps(record, allow_nan=False)

    def test_bad_json_then_recovery(self):
        script = Script([{"raw_output": "not json"}, "get_line_state()", submit(report())])
        record = self.run_agent(script)
        self.assertFalse(record["trace"][0]["format_valid"])
        self.assertIn("error", script.messages[1][-1]["content"])

    def test_json_format_separate_from_bad_command(self):
        record = self.run_agent(Script(["bad.member()", "get_line_state()", submit(report())]))
        self.assertTrue(record["trace"][0]["format_valid"])
        self.assertFalse(record["trace"][0]["command_valid"])

    def test_forbidden_tool_never_dispatched(self):
        client = Client()
        record = self.run_agent(
            Script(["admin_reset()", "get_line_state()", submit(report())]), client
        )
        self.assertNotIn("admin_reset", [c[0] for c in client.calls])
        self.assertIn("not allowed", record["trace"][0]["tool_error"])

    def test_certify_blocks_actions_even_if_server_exposes_them(self):
        client = Client()
        record = self.run_agent(
            Script(
                [
                    "set_speed(station_id='S0', pct=100)",
                    "get_line_state()",
                    submit(report("certify")),
                ]
            ),
            client,
            mode="certify",
        )
        self.assertNotIn("set_speed", [c[0] for c in client.calls])
        self.assertNotIn("run_whatif", record["tools"])

    def test_decision_dispatches_whatif_and_action_before_submission(self):
        client = Client()
        self.run_agent(
            Script(
                [
                    "get_line_state()",
                    "run_whatif(actions=[{'action':'set_speed','station_id':'S0','pct':100}], hours=1, reps=1)",
                    "set_speed(station_id='S0', pct=100)",
                    submit(report()),
                ]
            ),
            client,
        )
        self.assertEqual(
            [c[0] for c in client.calls],
            ["get_line_state", "run_whatif", "set_speed", "submit_decision"],
        )

    def test_unknown_observation_id_is_report_feedback_not_submission(self):
        client = Client()
        record = self.run_agent(
            Script(["get_line_state()", submit(report(oid="invented")), submit(report())]), client
        )
        self.assertEqual([c[0] for c in client.calls].count("submit_decision"), 1)
        self.assertIn("not observed", record["trace"][1]["tool_error"])

    def test_value_mismatch_is_flagged_not_silently_repaired(self):
        client = Client()
        record = self.run_agent(Script(["get_line_state()", submit(report(value=999))]), client)
        self.assertEqual(record["evidence"]["mismatch"], 1)
        self.assertEqual(client.decision["claims"][0]["value"], 999)

    def test_too_many_candidates_rejected(self):
        bad = report("certify")
        bad["candidates"] *= 6
        record = self.run_agent(
            Script(["get_line_state()", submit(bad), submit(report("certify"))]), mode="certify"
        )
        self.assertIn("five", record["trace"][1]["tool_error"])

    def test_http_tool_error_is_feedback(self):
        client = Client()
        client.fail_once = True
        record = self.run_agent(
            Script(["get_line_state()", "get_line_state()", submit(report())]), client
        )
        self.assertEqual(record["trace"][0]["observation"]["http_status"], 400)

    def test_transport_failure_does_not_retry_action(self):
        client = Client()

        def uncertain(tool, **args):
            client.calls.append((tool, args))
            raise TimeoutError("test")

        client.call = uncertain
        with self.assertRaises(AgentRunError) as ctx:
            self.run_agent(Script(["set_speed(station_id='S0', pct=100)"]), client)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("uncertain", ctx.exception.record["error"])

    def test_max_turns_failure_persisted_and_no_fake_finish(self):
        client = Client()
        with self.assertRaises(AgentRunError) as ctx:
            self.run_agent(Script(["get_line_state()"]), client, max_turns=1)
        self.assertFalse(ctx.exception.record["finished"])
        self.assertIsNone(client.decision)
        self.assertEqual(ctx.exception.record["error"], "max_turns_reached")

    def test_backend_failure_saved(self):
        def broken(messages):
            raise RuntimeError("fake backend failed")

        with self.assertRaises(AgentRunError) as ctx:
            self.run_agent(broken)
        saved = json.loads(Path(ctx.exception.record["trace_path"]).read_text())
        self.assertIn("fake backend failed", saved["error"])

    def test_second_attempt_never_reuses_old_report_or_observations(self):
        a = self.run_agent(Script(["get_line_state()", submit(report())]))
        b = self.run_agent(Script(["get_line_state()", submit(report())]))
        self.assertNotEqual(a["trace_path"], b["trace_path"])

    def test_no_reference_labels_or_hidden_admin_state_in_prompt(self):
        script = Script(["get_line_state()", submit(report())])
        self.run_agent(script)
        prompt = script.messages[0][0]["content"]
        self.assertNotIn("/admin/", prompt)
        self.assertNotIn("oracle.json", prompt)


if __name__ == "__main__":
    unittest.main()
