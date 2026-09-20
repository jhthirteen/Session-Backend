"""Interactive playground for the NLP visualizer.

Usage:
    .venv/bin/python src/data_tooling/test.py

Type a question (e.g. "how many points did jalen brunson average last year"),
get the answer + viz hint. Type 'quit'/'exit'/'q' or Ctrl-C to leave.
"""

import json
import sys

try:
    from .agent import run_query
except ImportError:  # allow `python src/data_tooling/test.py` direct runs
    from src.data_tooling.agent import run_query


def _print_response(resp) -> None:
    print("\n--- ANSWER ---")
    print(resp.answer_text)
    print("\n--- SPEC ---")
    print(f"intent={resp.spec.intent} players={resp.spec.players} "
          f"teams={resp.spec.teams} season={resp.spec.season} "
          f"metrics={resp.spec.metrics} last_n={resp.spec.last_n}")
    print("\n--- VIZ (for React) ---")
    print(resp.viz_hint.model_dump_json())
    print(f"\n--- DATA ({len(resp.data)} rows) ---")
    for row in resp.data[:10]:
        print(json.dumps(row, default=str)[:300])
    if len(resp.data) > 10:
        print(f"... and {len(resp.data) - 10} more rows")
    if resp.debug:
        print("\n--- TOOLS ---")
        for t in resp.debug:
            print(f"{t.tool} {t.args} -> {t.result_summary}")


def main() -> None:
    print("NBA NLP visualizer playground. Type 'quit' to exit.")
    while True:
        try:
            query = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("bye!")
            return
        try:
            resp = run_query(query)
        except Exception as e:  # never crash the REPL on a bad query
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        _print_response(resp)


if __name__ == "__main__":
    main()
