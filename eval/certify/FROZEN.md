# Frozen LLM proposer (CERTIFY_PROTOCOL.md, amendment 6)

The LLM proposer was fixed on 2026-11-23, before any v6 calibration or test case was generated or run. It must not change
for v6. Only the dev split was used to develop it.

## Model
- Qwen3.5-9B (base model under Apache 2.0), GGUF quantisation Q4_K_M by bartowski:
  `bartowski/Qwen_Qwen3.5-9B-GGUF`, file `Qwen_Qwen3.5-9B-Q4_K_M.gguf`, revision
  `182be2fd6c7bc44887d88a91cb03ff009cc9f549`, 6,169,341,984 bytes, SHA-256
  `d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43`.
- Open weights, run locally on one RTX 4090 (24 GB). No hosted API is called.

## Server
- llama.cpp release `b12628` (commit `8a97fa98a`), Windows CUDA 12.4 binaries
  `llama-b12628-bin-win-cuda-12.4-x64.zip` (SHA-256 `dfb2574d6d5ea69d79b1ff69331b9e85b76bb6920278d3f0193b1586276f3bbd`)
  and `cudart-llama-bin-win-cuda-12.4-x64.zip` (SHA-256 `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6`).
- Command: `llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 -np 4 -fa on --jinja --seed 0`
  (4 slots of 16,384 tokens; the model's own chat template, tool calls through the OpenAI-compatible endpoint).

## Agent (`eval/certify/llm_agent.py`)
- System prompt: the `SYSTEM` string of `llm_agent.py` at the commit that adds this file (dev version "try3").
- Tools: the five read tools of the tool server (`get_line_state`, `get_station`, `get_sensor_history`,
  `predict_failure`, `get_quality`) and `submit_decision`. Action tools and `run_whatif` are disabled (403).
- Decoding: temperature 0, top_p 1, seed 0, max_tokens 4096 per turn; thinking off (`enable_thinking=false`).
- At most 10 turns with read tools, then one turn with `submit_decision` only. A turn without a tool call gets one
  reminder. A server error 500 (a cut-off or malformed tool call) gets a request to call again with valid JSON and a
  shorter rationale; such turns count towards the 10. Tool results are compact JSON cut at 8,000 characters.
- Up to K = 5 candidates are kept, as for every proposer; invalid actions are dropped and counted.
- If no decision is submitted, the case gets no candidates (it abstains) and is reported as such.

## Proposals are stored
Proposals are collected once per split (`proposers.py --agent llm_agent:agent`, 3 shards) and stored as JSONL with the
full trace (turns, tokens, tool calls, errors). `run.py --llm-proposals` reads them; every result is reproducible from the
stored proposals. Regenerating the proposals may not give identical text: llama.cpp batches the parallel slots, and
floating-point results can depend on the batch.

## Development on dev (36 cases, 6 per family)
Three prompts were tried. G is the mean plant gain per case of the naive gate (execute j* if s > 0); descriptive only.

| Prompt | Change | Executed | Mean G | Harms |
|---|---|---|---|---|
| try1 | first version | 15/36 | 206 | 1 |
| try2 | costs of each action; "maintaining several healthy stations almost always loses value"; candidates of 1–3 actions | 15/36 | 92 | 0 |
| try3 (frozen) | try2 without the claim on maintenance; asks for a spread of candidates (one broad plan, narrower plans, plans of another kind); max_tokens 2048 → 4096; retry on error 500 | 12/25 * | 241 * | 0 * |

\* try3 was chosen when 25 of its 36 dev cases had been evaluated; on those 25 cases try1 had mean G 239 and try2 110.
The remaining 11 dev cases are reported with the v6 result and did not affect the choice.
