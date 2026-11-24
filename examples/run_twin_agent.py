"""Development smoke runner; uses only dev cases and does not require an oracle."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_adapter.development import run_development


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["scripted", "qwen35"], required=True)
    p.add_argument("--mode", choices=["decision", "certify"], required=True)
    p.add_argument("--artifacts", type=Path, default=ROOT / "experiments/input_rebuild/artifacts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--no-whatif", action="store_true")
    p.add_argument("--nf4", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    result = run_development(
        repo=ROOT,
        artifacts=args.artifacts,
        out=args.out,
        mode=args.mode,
        backend_name=args.backend,
        limit=args.limit,
        max_turns=args.max_turns,
        no_whatif=args.no_whatif,
        nf4=args.nf4,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["attempted"] != result["finished"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
