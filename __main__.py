# Allow `python -m dragon` to show basic help and available subcommands.
from __future__ import annotations

import sys

HELP = """
Dragon module runner

Subcommands (run with python -m <name>):
  - dragon.build_tokenizer   Build BPE tokenizer from config.tokenizer.files
  - dragon.train_pan_tadeusz Train model according to config.training
  - dragon.generate          Generate text using a trained checkpoint
  - dragon.model_inspect     Inspect model salience and export a graph
  - dragon.view_graph        Preview graph.json using NetworkX/Matplotlib

Examples:
  python -m dragon.build_tokenizer
  python -m dragon.train_pan_tadeusz
  python -m dragon.generate
  python -m dragon.model_inspect
  python -m dragon.view_graph --graph graph.json --out graph.png --html graph.html
""".strip()


def main() -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
