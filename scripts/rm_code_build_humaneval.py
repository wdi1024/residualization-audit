"""HumanEval replication of the MBPP 2x2 construction (rm_code_build.py).

Same recipe, second code dataset: per problem, {correct, buggy} x
{terse, verbose} with unit-test-verified labels; all candidates round-tripped
through ast.parse/unparse so code style cannot encode mutation status.
A candidate is the full function (signature + docstring + body); tests run
the problem's own `check(entry_point)` harness in a subprocess with timeout.

Writes ../cache/humaneval_rows.json: list of [prompt, code, y, qid, cell].
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "cache", "humaneval_rows.json")
TIMEOUT = 6

sys.path.insert(0, HERE)
from rm_code_build import Mutate, mutants, verbosify  # noqa: E402


def run_tests(code, test, entry):
    src = code + "\n" + test + f"\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=TIMEOUT)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    from datasets import load_dataset
    d = load_dataset("openai_humaneval", split="test")
    rows = []
    qid = 0
    stats = {"no_pass": 0, "no_fail_mutant": 0, "verbose_broke": 0, "ok": 0}
    for r in d:
        test, entry = r["test"], r["entry_point"]
        try:
            code = ast.unparse(ast.parse(r["prompt"] + r["canonical_solution"]))
        except Exception:
            continue
        if not run_tests(code, test, entry):
            stats["no_pass"] += 1
            continue
        bug = None
        for cand in mutants(code):
            if not run_tests(cand, test, entry):
                bug = cand
                break
        if bug is None:
            stats["no_fail_mutant"] += 1
            continue
        vc, vb = verbosify(code), verbosify(bug)
        if not run_tests(vc, test, entry) or run_tests(vb, test, entry):
            stats["verbose_broke"] += 1
            continue
        prompt = ("Complete the following Python function so that it "
                  "satisfies its docstring:\n" + r["prompt"])
        for c, y, cell in ((code, 1, "correct-terse"),
                           (vc, 1, "correct-verbose"),
                           (bug, 0, "buggy-terse"),
                           (vb, 0, "buggy-verbose")):
            rows.append([prompt, c, y, qid, cell])
        stats["ok"] += 1
        qid += 1
    json.dump(rows, open(OUT, "w"))
    print("stats:", json.dumps(stats), "n_rows:", len(rows))


if __name__ == "__main__":
    main()
