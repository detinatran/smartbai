"""Mock-only extraction checks; do not download model weights or require CUDA."""

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
import uuid
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_adapter import backend_qwen35 as backend


class Batch(dict):
    def __init__(self, size):
        super().__init__(input_ids=types.SimpleNamespace(shape=(1, size)))
        self.device = None

    def to(self, device):
        self.device = device
        return self


def fake_libraries():
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.device_count.return_value = 2
    torch.cuda.mem_get_info.return_value = (int(14.46 * 2**30), int(15 * 2**30))
    torch.cuda.get_device_name.return_value = "Fake T4 (test fixture)"
    torch.float16 = "fake_float16"
    torch.version.cuda = "mock_cuda"
    torch.inference_mode.side_effect = lambda: nullcontext()
    hf = MagicMock()
    tokenizer = hf.AutoTokenizer.from_pretrained.return_value
    tokenizer.get_chat_template.return_value = "template with enable_thinking parameter"
    tokenizer.apply_chat_template.return_value = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tokenizer.return_value = Batch(2090)
    tokenizer.eos_token_id, tokenizer.pad_token_id = 123, 456
    tokenizer.convert_tokens_to_ids.return_value = 124
    tokenizer.convert_ids_to_tokens.return_value = "<|im_end|>"
    tokenizer.decode.return_value = '{"status":"ready"}'
    model = MagicMock()
    model.hf_device_map = {"a": 0, "b": 1}
    model.generation_config.eos_token_id = 123
    model.get_input_embeddings.return_value.weight.device = "cuda:0"
    model.generate.return_value.__getitem__.return_value.numel.return_value = 24
    model.generate.return_value.__getitem__.return_value.__getitem__.return_value.item.return_value = (
        124
    )
    hf.Qwen3_5ForConditionalGeneration.from_pretrained.return_value = (
        model,
        {"missing_keys": set(), "unexpected_keys": [], "mismatched_keys": []},
    )
    hf.GenerationConfig.side_effect = lambda **kw: types.SimpleNamespace(
        **kw, to_dict=lambda: dict(kw)
    )
    return torch, hf, tokenizer, model


class BackendTests(unittest.TestCase):
    def setUp(self):
        backend._default_backend = None
        self.torch, self.hf, self.tokenizer, self.model = fake_libraries()
        self.lib_patch = patch.object(backend, "_libraries", return_value=(self.torch, self.hf))
        self.lib_patch.start()
        self.addCleanup(self.lib_patch.stop)
        self.versions = patch.object(
            backend.importlib.metadata,
            "version",
            side_effect=lambda name: "5.21.1" if name == "transformers" else "mock",
        )
        self.versions.start()
        self.addCleanup(self.versions.stop)
        self.messages = [{"role": "user", "content": "A test-only prompt"}]

    def test_import_does_not_import_torch_or_transformers(self):
        script = (
            "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); "
            "import llm_adapter.backend_qwen35; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_repeated_calls_load_once_and_keep_original_settings(self):
        obj = backend.get_backend()
        self.assertFalse(obj.loaded)
        a = backend.generate(self.messages)
        b = backend.generate(self.messages)
        self.assertIs(obj, backend.get_backend())
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_called_once()
        self.hf.AutoTokenizer.from_pretrained.assert_called_once()
        self.hf.set_seed.assert_called_once_with(42)
        self.model.eval.assert_called_once()
        self.assertEqual(self.model.generate.call_count, 2)
        call = self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.call_args
        self.assertEqual(call.args, ("Qwen/Qwen3.5-9B",))
        self.assertEqual(call.kwargs["revision"], "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
        self.assertEqual(call.kwargs["dtype"], "fake_float16")
        self.assertEqual(call.kwargs["attn_implementation"], "sdpa")
        self.assertFalse(call.kwargs["token"])
        self.assertNotIn("quantization_config", call.kwargs)
        gen = obj._generation.to_dict()
        self.assertEqual(
            gen,
            dict(
                do_sample=False,
                num_beams=1,
                max_new_tokens=384,
                use_cache=True,
                prefill_chunk_size=256,
                eos_token_id=[123, 124],
                pad_token_id=456,
            ),
        )
        self.assertEqual(
            set(a),
            {
                "raw_output",
                "input_tokens",
                "output_tokens",
                "latency_seconds",
                "hit_generation_limit",
                "rendered_prompt_sha256",
                "generation_stop",
                "last_output_token_id",
            },
        )
        self.assertEqual(a["generation_stop"], "eos")
        self.assertEqual(a["input_tokens"], 2090)
        self.assertEqual(b["output_tokens"], 24)
        self.assertFalse(a["hit_generation_limit"])
        self.assertEqual(len(a["rendered_prompt_sha256"]), 64)
        self.tokenizer.assert_called_with(
            self.tokenizer.apply_chat_template.return_value,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.apply_chat_template.assert_called_with(
            self.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def test_context_limit_rejects_without_truncation_or_generation(self):
        self.tokenizer.return_value = Batch(16385)
        with self.assertRaisesRegex(RuntimeError, "No automatic"):
            backend.generate(self.messages)
        self.model.generate.assert_not_called()

    def test_generation_cap_is_reported(self):
        self.model.generate.return_value.__getitem__.return_value.numel.return_value = 384
        self.model.generate.return_value.__getitem__.return_value.__getitem__.return_value.item.return_value = (
            999
        )
        result = backend.generate(self.messages)
        self.assertTrue(result["hit_generation_limit"])
        self.assertEqual(result["generation_stop"], "max_new_tokens")

    def test_checkpoint_stop_ids_are_preserved_and_turn_end_is_added(self):
        self.model.generation_config.eos_token_id = [123, 125]
        obj = backend.get_backend().load()
        self.assertEqual(obj._generation.eos_token_id, [123, 125, 124])
        self.assertEqual(backend._chat_stop_ids(self.tokenizer, [124, 123]), [124, 123])

    def test_pinned_qwen_chat_eos_regression(self):
        # Observed run v1 used only 248044. The pinned tokenizer maps the end of
        # an assistant turn to 248046, so that token must reach model.generate.
        self.tokenizer.eos_token_id = 248044
        self.tokenizer.convert_tokens_to_ids.return_value = 248046
        self.model.generation_config.eos_token_id = 248044
        backend.generate(self.messages)
        config = self.model.generate.call_args.kwargs["generation_config"]
        self.assertEqual(config.eos_token_id, [248044, 248046])

    def test_prefill_memory_controls_and_extended_report_budget(self):
        obj = backend.get_backend(backend.Qwen35Config(max_new_tokens=1024))
        result = obj.generate(self.messages)
        call = self.model.generate.call_args.kwargs
        self.assertEqual(call["logits_to_keep"], 1)
        self.assertEqual(call["generation_config"].prefill_chunk_size, 256)
        self.assertEqual(call["generation_config"].max_new_tokens, 1024)
        self.assertEqual(result["input_tokens"], 2090)
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                backend.Qwen35Config(prefill_chunk_size=value)

    def test_unknown_turn_end_token_rejected_before_loading_weights(self):
        self.tokenizer.convert_ids_to_tokens.return_value = "<unk>"
        with self.assertRaisesRegex(RuntimeError, "im_end"):
            backend.get_backend().load()
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_not_called()

    def test_gpu0_runtime_headroom_has_explicit_placement_budget(self):
        obj = backend.get_backend().load()
        memory = self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.call_args.kwargs[
            "max_memory"
        ]
        free = self.torch.cuda.mem_get_info.return_value[0]
        self.assertEqual(memory, {0: int(free * 0.60), 1: int(free * 0.80)})
        self.assertEqual(obj.metadata()["runtime_fix"]["gpu0_weight_memory_fraction"], 0.60)
        for value in [0, -0.1, 0.81, True, float("nan")]:
            with self.assertRaises(ValueError):
                backend.Qwen35Config(gpu0_weight_memory_fraction=value)

    def test_missing_weights_fail_and_do_not_reload(self):
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.return_value = (
            self.model,
            {"missing_keys": ["bad.weight"]},
        )
        obj = backend.get_backend()
        with self.assertRaisesRegex(RuntimeError, "Missing/mismatched"):
            obj.load()
        with self.assertRaisesRegex(RuntimeError, "previous load failed"):
            obj.load()
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_called_once()
        self.model.generate.assert_not_called()

    def test_cpu_offload_is_rejected(self):
        self.model.hf_device_map = {"a": "cpu"}
        with self.assertRaisesRegex(RuntimeError, "offload"):
            backend.generate(self.messages)
        self.model.generate.assert_not_called()

    def test_insufficient_gpu_does_not_silently_quantize(self):
        self.torch.cuda.device_count.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "24 GiB"):
            backend.generate(self.messages)
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_not_called()

    def test_cpu_only_does_not_load_weights(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            backend.generate(self.messages)
        self.hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_not_called()

    def test_explicit_nf4_is_distinct_and_fixed(self):
        cfg = backend.Qwen35Config(load_in_4bit=True)
        obj = backend.get_backend(cfg).load()
        self.assertEqual(obj.metadata()["precision"], "NF4")
        self.hf.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="fake_float16",
        )
        with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
            backend.get_backend(backend.Qwen35Config())
        with self.assertRaises(FrozenInstanceError):
            cfg.seed = 10

    def test_metadata_is_serializable_and_has_no_aircraft_dataset(self):
        obj = backend.get_backend().load()
        meta = obj.metadata()
        json.dumps(meta, allow_nan=False)
        self.assertNotIn("dataset", meta)
        meta["config"]["seed"] = 99
        self.assertEqual(obj.metadata()["config"]["seed"], 42)
        directory = ROOT / "out/backend-tests" / uuid.uuid4().hex
        obj.write_metadata(directory)
        self.assertTrue((directory / "chat_template.jinja").exists())
        saved = json.loads((directory / "backend_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["model_sha"], obj.config.model_revision)

    def test_invalid_messages_do_not_load(self):
        for value in (
            [],
            "not messages",
            [{"role": "user", "content": []}],
            [{"role": "bad", "content": "x"}],
        ):
            with self.assertRaises(ValueError):
                backend.generate(value)
        self.hf.AutoTokenizer.from_pretrained.assert_not_called()

    def test_thinking_mode_cannot_change_baseline(self):
        with self.assertRaises(ValueError):
            backend.Qwen35Config(enable_thinking=True)


if __name__ == "__main__":
    unittest.main()
