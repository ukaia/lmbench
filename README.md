# lmbench

A tiny single-file TUI for benchmarking local LLMs served by
[LM Studio](https://lmstudio.ai) — or any OpenAI-compatible endpoint
(Ollama, llama.cpp server, vLLM, …).

Pick a model, press `r`, get comparable numbers. Every run is saved and feeds
a local leaderboard.

![lmbench screenshot](screenshot.png)

## Quickstart

Requires Python ≥ 3.10 and a running LM Studio server
(Developer tab → **Start server**, default `http://localhost:1234/v1`).

With [uv](https://docs.astral.sh/uv/) (dependencies resolve automatically from
the script header):

```bash
uv run lmbench.py
```

Without uv:

```bash
./run.sh
```

(creates a local venv with `textual` + `httpx` on first use), or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install "textual>=1.0" "httpx>=0.27"
.venv/bin/python lmbench.py
```

Point it elsewhere with the endpoint field or `LMSTUDIO_ENDPOINT`.

## What it measures

A fixed, deterministic suite (v2): warmup + 5 prompts — short QA, math
reasoning, code generation, summarization of a fixed passage, and long-form
writing — all at temperature 0, seed 42, fixed `max_tokens`. Identical prompts
every run, so results are comparable across models and over time. The warmup
absorbs model load time and is never scored. Token budgets (256–1024) are
sized so reasoning/thinking models have room to finish their chain of thought
and still answer; direct models simply stop early at EOS.

Per task and per run:

| metric | meaning |
|---|---|
| **gen tok/s** | decode speed: `(completion_tokens − 1) / time after first token` |
| **ttft** | time to first streamed token |
| **prompt tok/s** | prompt processing speed: `prompt_tokens / ttft` |
| **quality** | 0–100 output-quality score against reference answers (below) |

Token counts come from the server's `usage` field, falling back to counting
stream chunks (marked *estimated*) on servers that don't report usage.

## Quality score

Each scored task also gets a deterministic, formulaic quality score — no LLM
judge, so scores are reproducible and comparable across machines:

- **60% objective checks**, per task: the math task must reach the correct
  answer; the generated `is_prime` is actually executed against 15 fixed unit
  tests including edge cases (in a `python -I` subprocess with a 10 s
  timeout); the essay is checked for coverage of 8 key concepts plus
  three-paragraph structure; the summary for sentence count and topicality;
  the QA answer for the physically correct keywords.
- **40% n-gram similarity** (ROUGE-style unigram + bigram F1) to bundled
  reference answers written by Claude (Fable 5).

Reasoning models are handled explicitly: `<think>…</think>` blocks and the
`reasoning_content` channel are excluded from scoring (only the final answer
counts), the chain of thought is saved in the run JSON for inspection, and
`finish_reason` is recorded — so the per-task note distinguishes a wrong
answer from "truncated at budget". A model that solves the task in its chain
of thought but never emits a final answer within budget is scored from the
thinking text at half credit — solving without delivering beats being wrong,
but delivery matters.

Treat the number as a sanity signal, not an eval harness: a correct answer
phrased very differently from the reference scores mid-range on the
similarity component, and 5 fixed prompts is a small sample. It will reliably
flag a broken or badly quantized model; it won't rank two good ones
precisely.

The leaderboard only aggregates runs from the current suite version —
older-suite runs stay browsable in Results (with a suite column) but aren't
ranked; rerun a model to get it back on the board.

## Tabs

1. **Benchmark** — endpoint + connect, model list (embedding models hidden),
   live stats, progress, log
2. **Results** — every saved run, with a per-task breakdown for the
   highlighted row
3. **Leaderboard** — models ranked by average gen tok/s across all saved
   runs, with quality alongside

Runs are stored as JSON in `~/.lmbench/results/` (override with `LMBENCH_DIR`),
one self-describing file per run — including the full model outputs, so you
can inspect answers or re-score later.

## Keys

`c` connect · `r` run benchmark · `x` cancel · `d` delete highlighted run
(Results tab) · `q` quit

## Notes

- Before each run the app calls `lms unload --all` (LM Studio's CLI) so only
  the benchmarked model is resident — no memory contention from other loaded
  models skewing results. Localhost endpoints only; silently skipped when the
  CLI isn't installed.
- If JIT model loading is disabled in LM Studio, load the model there first;
  otherwise the first request loads it (absorbed by warmup).
- A model that streams no tokens (failed load, crashed runtime) shows up as a
  failed task rather than crashing the run.
- Benchmarks are single-request and sequential by design — this measures
  interactive single-user speed, not batched throughput.
