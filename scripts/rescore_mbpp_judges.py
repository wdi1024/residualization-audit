"""Re-score the MBPP reward-model audit candidates with LLM judges.

The bridge probe (scripts/bridge_feasibility_probe.py) needs a scorer that has
BOTH a real construct signal (it actually tracks correctness) AND a format
loading (there is a surface channel to remove). The audited 2023-generation
reward models have the loading but almost no construct signal, so the probe's
absolute effects are small.

This script re-scores the same 1412 candidates with several judge models so the
probe can be re-run on a scorer that has both properties. Which model lands in
that regime is not predictable in advance — a judge that is too strong reads
format so weakly that there is nothing to remove (the paper's own WikiQA judge
probe measured R^2 = 0.031, below the gate) — so we sweep rather than guess.

Uses the Batch API (50% cheaper, no latency requirement) and structured outputs
so every reply parses. Submit with `submit`, collect with `collect`.
"""

import argparse
import json
import os
import sys
import time

import anthropic
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
STATE = os.path.join(CACHE, "..", "rescore_mbpp_judges_batches.json")

INSTRUCTION = (
    "You are grading a candidate Python solution.\n\n"
    "Problem:\n{problem}\n\n"
    "Candidate solution:\n```python\n{code}\n```\n\n"
    "Rate how likely this solution is correct, 0-100."
)

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}

# Thinking is on by default on Sonnet 5 and Opus 5; this is a single-integer
# grading task, so disable it. On Opus 5 disabling is only accepted at effort
# `high` or lower, hence the explicit low effort.
MODELS = {
    "haiku45": ("claude-haiku-4-5", {}),
    "sonnet5": ("claude-sonnet-5", {"thinking": {"type": "disabled"}}),
    "opus5": ("claude-opus-5", {"thinking": {"type": "disabled"},
                                "effort": "low"}),
}


def load_rows():
    return json.load(open(os.path.join(CACHE, "mbpp_rows.json")))


def build_requests(rows, model, extra):
    effort = extra.pop("effort", None)
    output_config = {"format": {"type": "json_schema", "schema": SCORE_SCHEMA}}
    if effort:
        output_config["effort"] = effort
    reqs = []
    for i, r in enumerate(rows):
        reqs.append(Request(
            custom_id=f"row-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=64,
                output_config=output_config,
                messages=[{"role": "user",
                           "content": INSTRUCTION.format(problem=r[0], code=r[1])}],
                **extra,
            ),
        ))
    return reqs


def submit(client):
    rows = load_rows()
    state = {}
    for key, (model, extra) in MODELS.items():
        batch = client.messages.batches.create(
            requests=build_requests(rows, model, dict(extra)))
        state[key] = {"batch_id": batch.id, "model": model, "n": len(rows)}
        print(f"{key:8s} {model:18s} -> {batch.id}")
    json.dump(state, open(STATE, "w"), indent=2)
    print(f"\nstate written to {STATE}")


def collect(client):
    state = json.load(open(STATE))
    for key, info in state.items():
        batch = client.messages.batches.retrieve(info["batch_id"])
        if batch.processing_status != "ended":
            print(f"{key}: {batch.processing_status} "
                  f"(processing {batch.request_counts.processing})")
            continue

        # Results arrive in arbitrary order — key by custom_id, never position.
        scores = np.full(info["n"], np.nan)
        errors = 0
        for result in client.messages.batches.results(info["batch_id"]):
            idx = int(result.custom_id.split("-")[1])
            if result.result.type != "succeeded":
                errors += 1
                continue
            msg = result.result.message
            if msg.stop_reason == "refusal":
                errors += 1
                continue
            text = next((b.text for b in msg.content if b.type == "text"), None)
            try:
                scores[idx] = float(json.loads(text)["score"])
            except (TypeError, ValueError, KeyError):
                # A handful of replies hit max_tokens mid-object and the JSON is
                # truncated. Leave them NaN here; `repair` re-runs just those.
                errors += 1

        # Re-run the stragglers one at a time with room to finish the object.
        missing = np.flatnonzero(np.isnan(scores))
        if len(missing):
            rows = load_rows()
            extra = dict(MODELS[key][1])
            eff = extra.pop("effort", None)
            oc = {"format": {"type": "json_schema", "schema": SCORE_SCHEMA}}
            if eff:
                oc["effort"] = eff
            for idx in missing:
                r = rows[idx]
                msg = client.messages.create(
                    model=info["model"], max_tokens=512, output_config=oc,
                    messages=[{"role": "user", "content": INSTRUCTION.format(
                        problem=r[0], code=r[1])}], **extra)
                text = next((b.text for b in msg.content if b.type == "text"), None)
                scores[idx] = float(json.loads(text)["score"])
            print(f"{key:8s} repaired {len(missing)} truncated repl{'y' if len(missing)==1 else 'ies'}")

        out = os.path.join(CACHE, f"mbpp_judge_{key}_s.npy")
        np.save(out, scores)
        miss = int(np.isnan(scores).sum())
        print(f"{key:8s} saved {out}  (batch errors {errors}, still missing {miss})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "collect"])
    args = ap.parse_args()
    client = anthropic.Anthropic()
    (submit if args.action == "submit" else collect)(client)
