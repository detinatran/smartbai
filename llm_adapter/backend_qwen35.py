"""Lazy Qwen3.5 inference with explicit precision, memory and decoding settings.

Weights are loaded once per backend instance. Importing this module does not
import the inference stack, download weights or connect to the twin.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

BACKEND_VERSION = "smartbai_qwen35_chunked_prefill_v3"


@dataclass(frozen=True)
class Qwen35Config:
    model_id: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    load_in_4bit: bool = False
    enable_thinking: bool = False
    seed: int = 42
    max_new_tokens: int = 384
    max_input_tokens: int = 16384
    gpu0_weight_memory_fraction: float = 0.60
    prefill_chunk_size: int | None = 256

    def __post_init__(self):
        if self.enable_thinking:
            raise ValueError(
                "This extracted baseline is non-thinking; use a separate backend/run for thinking."
            )
        if self.max_new_tokens <= 0 or self.max_input_tokens <= 0:
            raise ValueError("Token budgets must be positive.")
        if self.model_id != "Qwen/Qwen3.5-9B":
            raise ValueError(
                "This loader is scoped to Qwen/Qwen3.5-9B, not a generic model loader."
            )
        if self.prefill_chunk_size is not None and (
            type(self.prefill_chunk_size) is not int or self.prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        if (
            isinstance(self.gpu0_weight_memory_fraction, bool)
            or not isinstance(self.gpu0_weight_memory_fraction, (int, float))
            or not 0 < self.gpu0_weight_memory_fraction <= 0.80
        ):
            raise ValueError("gpu0_weight_memory_fraction must be positive and at most 0.80.")


def _libraries():
    """Lazy import only; never run pip or request credentials on import."""
    if sys.version_info < (3, 11):
        raise RuntimeError("Qwen3.5 backend requires Python >= 3.11.")
    # Set before importing the HF stack, as in the original notebook.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Install llm_adapter/requirements_qwen35.txt in the inference environment first."
        ) from exc
    version = importlib.metadata.version("transformers")
    if version != "5.21.1":
        raise RuntimeError(
            f"Expected transformers==5.21.1 from the original notebook; found {version}. "
            "Use a fresh inference kernel with requirements_qwen35.txt."
        )
    return torch, transformers


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    return value


def _chat_stop_ids(tokenizer, model_eos):
    """Keep checkpoint EOS and also stop at the end of an assistant turn."""
    turn_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        type(turn_end) is not int
        or turn_end < 0
        or tokenizer.convert_ids_to_tokens(turn_end) != "<|im_end|>"
    ):
        raise RuntimeError(
            "Tokenizer lacks a verified <|im_end|> token; refusing unbounded chat continuation."
        )
    values = (
        list(model_eos)
        if isinstance(model_eos, (list, tuple))
        else ([model_eos] if model_eos is not None else [])
    )
    if tokenizer.eos_token_id is not None:
        values.append(tokenizer.eos_token_id)
    values.append(turn_end)
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError("Invalid checkpoint/tokenizer EOS token ID")
    return list(dict.fromkeys(values))


class Qwen35Backend:
    """One reusable model instance; inject backend.generate into the agent loop.

    Create once, reuse for every case. Calls on an instance are serialized to avoid
    concurrent model generation and seed interference inside that instance.
    """

    def __init__(self, config=None):
        self._config = config if config is not None else Qwen35Config()
        if not isinstance(self._config, Qwen35Config):
            raise TypeError("config must be a Qwen35Config")
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded = False
        self._failed_load = False
        self._metadata = None

    @property
    def config(self):
        return self._config

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        """Idempotent: download/load only on the first successful call."""
        with self._load_lock:
            if self._loaded:
                return self
            if self._failed_load:
                raise RuntimeError(
                    "A previous load failed; fix the cause and restart the inference process/kernel "
                    "before retrying, to avoid overlapping partial model loads."
                )
            try:
                self._load()
            except Exception:
                self._failed_load = True
                raise
            self._loaded = True
            return self

    def _load(self):
        cfg = self.config
        torch, hf = _libraries()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; enable GPU before loading this backend.")
        gpu_info = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_GiB": round(torch.cuda.mem_get_info(i)[0] / 2**30, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        if not cfg.load_in_4bit and sum(g["free_GiB"] for g in gpu_info) < 24:
            raise RuntimeError(
                "Original FP16 configuration requires at least 24 GiB total free GPU memory "
                "(the previous run used two T4 GPUs). NF4 must be explicitly selected in a new run; "
                "there is no automatic quantization fallback."
            )
        if cfg.load_in_4bit:
            try:
                importlib.metadata.version("bitsandbytes")
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    "Explicit NF4 mode needs bitsandbytes>=0.45,<1 installed before loading."
                ) from exc

        tokenizer = hf.AutoTokenizer.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, token=False, trust_remote_code=False
        )
        template = tokenizer.get_chat_template()
        if "enable_thinking" not in template:
            raise RuntimeError("Unexpected chat template; inspect before inference.")
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with a JSON object."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if "</think>" not in probe.rsplit("<|im_start|>assistant", 1)[-1]:
            raise RuntimeError("Non-thinking prefix is not configured; inspect before inference.")

        # Verify before loading large weights. The tokenizer's general EOS alone
        # may be <|endoftext|>, which does not stop at the chat <|im_end|> token.
        _chat_stop_ids(tokenizer, None)

        kwargs = dict(
            revision=cfg.model_revision,
            token=False,
            trust_remote_code=False,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            ignore_mismatched_sizes=False,
            max_memory={
                i: int(
                    torch.cuda.mem_get_info(i)[0]
                    * (cfg.gpu0_weight_memory_fraction if i == 0 else 0.80)
                )
                for i in range(torch.cuda.device_count())
            },
        )
        if cfg.load_in_4bit:
            kwargs["quantization_config"] = hf.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        # Preserve the full checkpoint architecture. All generation inputs remain text-only.
        model, info = hf.Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.model_id, output_loading_info=True, **kwargs
        )
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError(
                "Missing/mismatched model weights; refusing inference with incomplete weights."
            )
        model.eval()
        device_map = {str(k): str(v) for k, v in model.hf_device_map.items()}
        if any(v in ("cpu", "disk") for v in device_map.values()):
            raise RuntimeError(
                "CPU/disk offload detected. Restart with suitable GPUs or explicit NF4 mode."
            )
        hf.set_seed(cfg.seed)
        eos_ids = _chat_stop_ids(tokenizer, model.generation_config.eos_token_id)
        generation = hf.GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=cfg.max_new_tokens,
            use_cache=True,
            prefill_chunk_size=cfg.prefill_chunk_size,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        packages = ["torch", "transformers", "accelerate", "huggingface_hub", "safetensors"]
        if cfg.load_in_4bit:
            packages.append("bitsandbytes")
        versions = {name: importlib.metadata.version(name) for name in packages}
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        self._generation, self._chat_template = generation, template
        self._metadata = {
            "backend": "llm_adapter.backend_qwen35",
            "backend_version": BACKEND_VERSION,
            "config": asdict(cfg),
            "model_id": cfg.model_id,
            "model_sha": cfg.model_revision,
            "model_class": type(model).__name__,
            "precision": "NF4" if cfg.load_in_4bit else "FP16",
            "input_modality": "text_only; full checkpoint architecture loaded",
            "enable_thinking": False,
            "seed": cfg.seed,
            "max_input_tokens": cfg.max_input_tokens,
            "generation": generation.to_dict(),
            "gpu": gpu_info,
            "device_map": device_map,
            "versions": versions,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "tokenizer_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "loading_info": _json_safe(info),
            "runtime_fix": {
                "stop_at_chat_turn_end": True,
                "gpu0_weight_memory_fraction": cfg.gpu0_weight_memory_fraction,
                "prefill_chunk_size": cfg.prefill_chunk_size,
                "logits_to_keep": 1,
                "other_gpu_weight_memory_fraction": 0.80,
                "note": "Weight-placement budgets reserve runtime headroom; they are not a guarantee against OOM.",
            },
            "origin": "Extracted from kaggle_qwen35_9b_sop_aircraft.ipynb; task/data/tool loop excluded",
        }

    def generate(self, messages):
        """Return the old six fields plus explicit termination diagnostics.

        Oversized contexts raise an error; logs are never silently truncated.
        Latency excludes initial loading/tokenization, matching the notebook.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list of text-only chat messages")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("Only text content is supported by this extracted backend")
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ValueError("Invalid message role")
        self.load()
        with self._inference_lock:
            torch = self._torch
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            n_input = batch["input_ids"].shape[-1]
            if n_input > self.config.max_input_tokens:
                raise RuntimeError(
                    "Input exceeds max_input_tokens; start a separate run/configuration. "
                    "No automatic context/log truncation is performed."
                )
            batch = batch.to(self._model.get_input_embeddings().weight.device)
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            start = time.perf_counter()
            with torch.inference_mode():
                ids = self._model.generate(
                    **batch, generation_config=self._generation, logits_to_keep=1
                )[0, n_input:]
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            n_output = int(ids.numel())
            last_id = int(ids[-1].item()) if n_output else None
            stopped_at_eos = last_id in self._generation.eos_token_id
            return {
                "raw_output": self._tokenizer.decode(ids, skip_special_tokens=True),
                "input_tokens": n_input,
                "output_tokens": n_output,
                "latency_seconds": time.perf_counter() - start,
                "hit_generation_limit": bool(n_output >= self.config.max_new_tokens),
                "generation_stop": (
                    "eos"
                    if stopped_at_eos
                    else ("max_new_tokens" if n_output >= self.config.max_new_tokens else "other")
                ),
                "last_output_token_id": last_id,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }

    __call__ = generate

    def metadata(self):
        """Read configuration/loaded metadata without triggering a download."""
        if not self.loaded:
            return {"loaded": False, "config": asdict(self.config)}
        return {"loaded": True, **deepcopy(self._metadata)}

    def write_metadata(self, directory):
        """After loading, save backend metadata and chat template for the run."""
        if not self.loaded:
            raise RuntimeError(
                "Load/generate first; metadata writing must not trigger a hidden model download."
            )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "backend_metadata.json").write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (directory / "chat_template.jinja").write_text(self._chat_template, encoding="utf-8")


_default_backend = None
_default_lock = threading.Lock()


def get_backend(config=None):
    """Return one shared instance per Python process; do not load weights yet.

    An explicit configuration change after instance creation is rejected, rather
    than silently loading another model or changing a running experiment.
    """
    global _default_backend
    with _default_lock:
        if _default_backend is None:
            _default_backend = Qwen35Backend(config)
        elif config is not None and config != _default_backend.config:
            raise RuntimeError(
                "Backend configuration is fixed for this process; restart for a new configuration."
            )
        return _default_backend


def load_backend(config=None):
    """Explicitly load and return the process-shared backend."""
    return get_backend(config).load()


def generate(messages):
    """Drop-in callback: repeated calls reuse the shared loaded model."""
    return get_backend().generate(messages)
