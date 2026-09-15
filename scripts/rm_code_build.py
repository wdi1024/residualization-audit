"""Card C, stage 1: build a code-preference audit setting from MBPP.

Per problem we construct four candidates spanning {correct, buggy} x
{terse, verbose}:
  1. canonical solution                      (correct, terse)
  2. canonical + docstring/comments          (correct, verbose)
  3. single-AST-mutation of the canonical    (buggy,  terse)
  4. mutant + docstring/comments             (buggy,  verbose)

Correctness labels are MECHANICAL: every candidate is executed against the
problem's own unit tests in a subprocess with a timeout; a variant counts as
correct iff all asserts pass, and a mutant is kept only if it compiles and
FAILS the tests. Verbose edits are comment/docstring-only, so they cannot
change semantics; we still re-run the tests on them (label verified, not
assumed). By construction the format channel (length, comments) varies
independently of correctness within each problem.

Writes ../cache/mbpp_rows.json: list of [prompt, code, y, qid, cell].
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(CACHE, "mbpp_rows.json")
N_PROBLEMS = 500
TIMEOUT = 6


class Mutate(ast.NodeTransformer):
    """Apply exactly one mutation, selected by a running counter."""
    SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
             ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.Add: ast.Sub, ast.Sub: ast.Add}

    def __init__(self, target):
        self.target = target
        self.count = 0
        self.done = False

    def _hit(self):
        self.count += 1
        if self.count - 1 == self.target and not self.done:
            self.done = True
            return True
        return False

    def visit_Compare(self, node):
        self.generic_visit(node)
        ops = []
        for op in node.ops:
            if type(op) in self.SWAPS and self._hit():
                ops.append(self.SWAPS[type(op)]())
            else:
                ops.append(op)
        node.ops = ops
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAPS and self._hit():
            node.op = self.SWAPS[type(node.op)]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) \
                and 0 <= node.value <= 100 and self._hit():
            node.value = node.value + 1
        return node


def mutants(code, max_variants=12):
    """Yield single-mutation rewrites of `code`."""
    for t in range(max_variants):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        m = Mutate(t)
        new = m.visit(tree)
        if not m.done:
            return
        try:
            yield ast.unparse(new)
        except Exception:
            continue


def body_indent(lines, i):
    """Indentation string of the first non-empty line after line i."""
    for ln in lines[i + 1:]:
        if ln.strip():
            return ln[:len(ln) - len(ln.lstrip())]
    return "    "


def verbosify(code):
    """Comment/docstring-only edit: cannot change semantics."""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    added_doc = False
    for i, ln in enumerate(lines):
        out.append(ln)
        st = ln.strip()
        if not added_doc and st.startswith("def ") and st.endswith(":"):
            indent = body_indent(lines, i)
            out.append(f'{indent}"""Compute and return the required result.')
            out.append(f"{indent}")
            out.append(f"{indent}This implementation follows the problem")
            out.append(f"{indent}statement step by step and handles the")
            out.append(f"{indent}standard input cases directly.")
            out.append(f'{indent}"""')
            out.append(f"{indent}# main computation below")
            added_doc = True
    out.append("")
    out.append("# end of solution; verified against the provided examples")
    return "\n".join(out)


def run_tests(code, setup, tests):
    src = code + "\n" + (setup or "") + "\n" + "\n".join(tests) + "\n"
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
    d = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rows = []
    qid = 0
    stats = {"no_pass": 0, "no_fail_mutant": 0, "verbose_broke": 0, "ok": 0}
    for r in d:
        if qid >= N_PROBLEMS:
            break
        tests = r["test_list"]
        setup = r.get("test_setup_code", "")
        prompt = r["text"]
        # normalize the canonical solution through the SAME parse/unparse
        # round-trip the mutants go through, so correct and buggy candidates
        # share one code style and format cannot encode mutation status
        try:
            code = ast.unparse(ast.parse(r["code"]))
        except Exception:
            continue
        if not run_tests(code, setup, tests):
            stats["no_pass"] += 1
            continue
        bug = None
        for cand in mutants(code):
            if not run_tests(cand, setup, tests):
                bug = cand
                break
        if bug is None:
            stats["no_fail_mutant"] += 1
            continue
        vc, vb = verbosify(code), verbosify(bug)
        # labels are verified by execution even for comment-only edits
        if not run_tests(vc, setup, tests) or run_tests(vb, setup, tests):
            stats["verbose_broke"] += 1
            continue
        for c, y, cell in ((code, 1, "correct-terse"),
                           (vc, 1, "correct-verbose"),
                           (bug, 0, "buggy-terse"),
                           (vb, 0, "buggy-verbose")):
            rows.append([prompt, c, y, qid, cell])
        stats["ok"] += 1
        qid += 1
        if qid % 50 == 0:
            print(f"{qid} problems built", flush=True)
    json.dump(rows, open(OUT, "w"))
    print("stats:", json.dumps(stats), "n_rows:", len(rows))


if __name__ == "__main__":
    main()
