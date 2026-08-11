#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "textual>=1.0",
#   "httpx>=0.27",
# ]
# ///
"""lmbench — a tiny TUI for benchmarking local LLMs served by LM Studio.

Works against any OpenAI-compatible endpoint (LM Studio, Ollama, llama.cpp
server). Runs a fixed, deterministic prompt suite and records time-to-first-
token, generation speed and prompt-processing speed per task. Results are
saved as JSON and browsable in the Results and Leaderboard tabs.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

DEFAULT_ENDPOINT = os.environ.get("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1")
RESULTS_DIR = Path(os.environ.get("LMBENCH_DIR", str(Path.home() / ".lmbench"))) / "results"
SUITE_VERSION = 1

# Fixed passage used by the summarization task, so every run processes an
# identical prompt of a few hundred tokens (exercises prompt processing).
PASSAGE = (
    "For most of human history, the measurement of time was tied to the sky. "
    "Sundials divided the day by the movement of shadows, while water clocks "
    "marked the hours by the steady drip of liquid from one vessel into another. "
    "The invention of the mechanical clock in medieval Europe changed the "
    "relationship between people and time. Powered first by falling weights and "
    "later by coiled springs, these machines used an escapement, a device that "
    "converts continuous force into discrete, regular ticks, to divide time into "
    "equal, countable units. Towns built clock towers, and daily life began to "
    "organize itself around hours rather than daylight. The pendulum clock, "
    "introduced in the seventeenth century, improved accuracy from minutes per "
    "day to seconds, making precise astronomy and navigation practical. Portable "
    "timekeepers followed: the marine chronometer solved the problem of finding "
    "longitude at sea, and the pocket watch put personal timekeeping into "
    "everyday life. Each improvement in accuracy enabled new kinds of "
    "coordination, from railway schedules to telegraph networks and factory "
    "shifts, until standardized time zones stitched the world into a single "
    "temporal grid. The story of the clock is therefore not only a story about "
    "machinery; it is a story about how societies choose to structure attention, "
    "labor, and daily rhythm."
)


@dataclass(frozen=True)
class BenchTask:
    name: str
    prompt: str
    max_tokens: int


WARMUP = BenchTask("warmup", "Reply with exactly one word: ready", 16)

SCORED: list[BenchTask] = [
    BenchTask(
        "qa_short",
        "Answer in one sentence: why is the sky blue?",
        64,
    ),
    BenchTask(
        "reasoning",
        "Solve step by step, showing your work: A bookstore sells paperbacks "
        "for $8 and hardcovers for $19. Maya buys 7 books for $89 total. "
        "How many hardcovers did she buy?",
        256,
    ),
    BenchTask(
        "code_gen",
        "Write a Python function is_prime(n) with a docstring, then show three "
        "example calls with expected output. Output only code.",
        256,
    ),
    BenchTask(
        "summarize",
        f"Summarize the following passage in exactly two sentences.\n\n{PASSAGE}",
        96,
    ),
    BenchTask(
        "long_form",
        "Write a detailed three-paragraph explanation of how photosynthesis "
        "converts sunlight into chemical energy, covering the light-dependent "
        "reactions, the Calvin cycle, and why the process matters for life on "
        "Earth.",
        512,
    ),
]


QUALITY_VERSION = 1

# Reference answers to the scored prompts, written by Claude (Fable 5).
# Model outputs are compared against these with n-gram F1 similarity as part
# of the quality score.
REFERENCES: dict[str, str] = {
    "qa_short": (
        "The sky looks blue because gas molecules in the atmosphere scatter "
        "shorter wavelengths of sunlight much more strongly than longer ones "
        "(Rayleigh scattering), so scattered blue light reaches your eyes "
        "from every direction in the sky."
    ),
    "reasoning": (
        "Let p be the number of paperbacks and h the number of hardcovers.\n\n"
        "Step 1: Two equations. Count: p + h = 7. Cost: 8p + 19h = 89.\n"
        "Step 2: Substitute p = 7 - h into the cost equation: "
        "8(7 - h) + 19h = 89.\n"
        "Step 3: Expand: 56 - 8h + 19h = 89, so 56 + 11h = 89.\n"
        "Step 4: Solve: 11h = 33, so h = 3.\n"
        "Step 5: Check: 3 hardcovers = $57, 4 paperbacks = $32, total $89. "
        "Correct.\n\n"
        "Maya bought 3 hardcovers."
    ),
    "code_gen": (
        "def is_prime(n):\n"
        '    """Return True if n is a prime number, False otherwise.\n'
        "\n"
        "    A prime is an integer greater than 1 whose only positive\n"
        "    divisors are 1 and itself. Runs in O(sqrt(n)) time.\n"
        '    """\n'
        "    if n < 2:\n"
        "        return False\n"
        "    if n < 4:\n"
        "        return True\n"
        "    if n % 2 == 0:\n"
        "        return False\n"
        "    i = 3\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 2\n"
        "    return True\n"
        "\n"
        "\n"
        "print(is_prime(2))   # True\n"
        "print(is_prime(15))  # False\n"
        "print(is_prime(97))  # True\n"
    ),
    "summarize": (
        "Timekeeping evolved from sundials and water clocks into mechanical "
        "clocks whose escapements divided time into equal, countable units, "
        "with later inventions like the pendulum clock, marine chronometer, "
        "and pocket watch steadily improving accuracy and portability. These "
        "advances reorganized society itself, enabling navigation, railway "
        "schedules, and standardized time zones, and reshaping how people "
        "structure attention, labor, and daily life."
    ),
    "long_form": (
        "Photosynthesis starts with the light-dependent reactions, which "
        "take place in the thylakoid membranes of chloroplasts. There, "
        "pigments such as chlorophyll absorb sunlight, and that captured "
        "energy is used to split water molecules, releasing oxygen as a "
        "byproduct. The energized electrons travel down an electron "
        "transport chain, pumping protons across the membrane; the "
        "resulting gradient drives ATP synthase to produce ATP, while the "
        "electrons ultimately reduce NADP+ to NADPH. In this way light "
        "energy is converted into two portable forms of chemical energy.\n\n"
        "The second stage, the Calvin cycle, runs in the stroma and does "
        "not need light directly. Carbon dioxide from the air is fixed onto "
        "a five-carbon sugar by the enzyme rubisco, and the resulting "
        "molecules are reduced, using the ATP and NADPH made earlier, into "
        "three-carbon sugars such as G3P. Most of these are recycled to "
        "keep the cycle turning, but some are exported to build glucose and "
        "other carbohydrates that store energy in stable chemical bonds.\n\n"
        "The process matters because it is the entry point for nearly all "
        "energy in living systems. Plants, algae, and cyanobacteria form "
        "the base of food webs, so the sugars made by photosynthesis feed "
        "almost everything else, directly or indirectly. The oxygen "
        "released sustains aerobic respiration, and the constant drawdown "
        "of carbon dioxide shapes Earth's atmosphere and climate; even "
        "fossil fuels are the stored products of ancient photosynthesis."
    ),
}

_PRIME_CASES = [
    (0, False), (1, False), (2, True), (3, True), (4, False),
    (9, False), (17, True), (25, False), (97, True), (100, False),
]


@dataclass
class TaskResult:
    name: str
    ok: bool
    error: str = ""
    ttft_s: float = 0.0
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_s: float = 0.0
    tokens_estimated: bool = False
    output: str = ""
    quality: float | None = None
    quality_note: str = ""


class HTTPBenchError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


# --------------------------------------------------------------------------
# Benchmark engine (plain async functions; no UI dependencies)
# --------------------------------------------------------------------------


async def fetch_models(client: httpx.AsyncClient, endpoint: str) -> list[str]:
    resp = await client.get(f"{endpoint}/models")
    resp.raise_for_status()
    return [m.get("id", "?") for m in resp.json().get("data", [])]


async def _run_stream(
    client: httpx.AsyncClient, url: str, payload: dict, name: str
) -> TaskResult:
    t0 = time.perf_counter()
    ttft: float | None = None
    chunk_tokens = 0
    usage: dict | None = None
    pieces: list[str] = []

    async with client.stream("POST", url, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "replace")
            raise HTTPBenchError(resp.status_code, body[:200])
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunk_tokens += 1
                    if delta.get("content"):
                        pieces.append(piece)

    total = time.perf_counter() - t0
    if ttft is None:
        return TaskResult(name=name, ok=False, error="no tokens streamed", total_s=total)

    estimated = usage is None
    completion = (usage.get("completion_tokens") if usage else 0) or chunk_tokens
    prompt_toks = (usage.get("prompt_tokens") if usage else 0) or 0

    gen_time = total - ttft
    if completion > 1 and gen_time > 0:
        gen_tps = (completion - 1) / gen_time
    else:
        gen_tps = completion / total if total > 0 else 0.0
    prompt_tps = (prompt_toks / ttft) if (prompt_toks and ttft > 0) else 0.0

    return TaskResult(
        name=name,
        ok=True,
        ttft_s=ttft,
        gen_tps=gen_tps,
        prompt_tps=prompt_tps,
        prompt_tokens=int(prompt_toks),
        completion_tokens=int(completion),
        total_s=total,
        tokens_estimated=estimated,
        output="".join(pieces),
    )


async def stream_bench(
    client: httpx.AsyncClient, endpoint: str, model: str, task: BenchTask
) -> TaskResult:
    url = f"{endpoint}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": task.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        return await _run_stream(client, url, payload, task.name)
    except HTTPBenchError as exc:
        # Some servers reject stream_options with a 400; retry without it and
        # fall back to counting chunks as a token estimate.
        if exc.status == 400 and "stream_options" in payload:
            retry = {k: v for k, v in payload.items() if k != "stream_options"}
            return await _run_stream(client, url, retry, task.name)
        raise


async def run_bench_task(
    client: httpx.AsyncClient, endpoint: str, model: str, task: BenchTask
) -> TaskResult:
    try:
        return await stream_bench(client, endpoint, model, task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure becomes a task error
        return TaskResult(name=task.name, ok=False, error=str(exc)[:200])


# --------------------------------------------------------------------------
# Quality scoring: 60% objective per-task checks + 40% n-gram similarity to
# the bundled reference answers. Deterministic, no LLM judge.
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngram_f1(ref_tokens: list[str], hyp_tokens: list[str], n: int) -> float:
    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return 0.0
    ref_counts = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    hyp_counts = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
    overlap = sum((ref_counts & hyp_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


def similarity(ref: str, hyp: str) -> float:
    ref_tokens, hyp_tokens = _tokenize(ref), _tokenize(hyp)
    return 0.5 * _ngram_f1(ref_tokens, hyp_tokens, 1) + 0.5 * _ngram_f1(ref_tokens, hyp_tokens, 2)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    cut = text.lower().find("<think")
    if cut != -1:  # unclosed thinking block (ran out of tokens)
        text = text[:cut]
    return text.strip()


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.S)
    return "\n\n".join(b.strip() for b in blocks) if blocks else text.strip()


def _check_qa_short(text: str) -> float:
    t = text.lower()
    hits = sum(1 for k in ("scatter", "blue", "light") if k in t)
    bonus = 1 if ("rayleigh" in t or "wavelength" in t) else 0
    return (hits + bonus) / 4


def _check_reasoning(text: str) -> float:
    t = text.lower()
    answer = r"\b(?:3|three)\b"
    if re.search(answer, t[-300:]):
        return 1.0
    if re.search(answer, t):
        return 0.5
    return 0.0


def _check_summarize(text: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    count = len(sentences)
    base = {2: 1.0, 3: 0.6}.get(count, 0.3 if 1 <= count <= 5 else 0.0)
    topical = any(k in text.lower() for k in ("clock", "time"))
    return base if topical else base * 0.5


def _check_long_form(text: str) -> float:
    t = text.lower()
    concepts = [
        ("chlorophyll",),
        ("light-dependent", "light dependent", "light reactions"),
        ("calvin",),
        ("atp",),
        ("nadph",),
        ("carbon dioxide", "co2", "co₂"),
        ("glucose", "sugar"),
        ("oxygen",),
    ]
    hits = sum(1 for alts in concepts if any(a in t for a in alts))
    return hits / len(concepts)


async def _check_code_gen(text: str) -> float:
    """Run the generated is_prime against fixed unit tests in a subprocess."""
    code = _extract_code(text)
    if "def is_prime" not in code:
        return 0.0
    harness = (
        f"{code}\n\n"
        f"_cases = {_PRIME_CASES!r}\n"
        "_passed = 0\n"
        "for _n, _want in _cases:\n"
        "    try:\n"
        "        if bool(is_prime(_n)) == _want:\n"
        "            _passed += 1\n"
        "    except Exception:\n"
        "        pass\n"
        "print('LMBENCH_PASS', _passed, len(_cases))\n"
    )
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", harness,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
        return 0.0
    except Exception:
        return 0.0
    for line in reversed(out.decode("utf-8", "replace").splitlines()):
        if line.startswith("LMBENCH_PASS"):
            _, passed, total = line.split()
            return int(passed) / int(total)
    return 0.0


_CHECKS = {
    "qa_short": _check_qa_short,
    "reasoning": _check_reasoning,
    "summarize": _check_summarize,
    "long_form": _check_long_form,
}


async def score_output(name: str, raw_text: str) -> tuple[float | None, str]:
    """Return (quality 0-100, note) for a task output, or (None, '') if unscored."""
    ref = REFERENCES.get(name)
    if ref is None:
        return None, ""
    text = _strip_thinking(raw_text)
    if not text:
        return 0.0, "no visible answer"
    if name == "code_gen":
        check = await _check_code_gen(text)
        sim = similarity(ref, _extract_code(text))
    else:
        check = _CHECKS[name](text)
        sim = similarity(ref, text)
    quality = round(100 * (0.6 * check + 0.4 * sim), 1)
    return quality, f"check {check:.2f} · sim {sim:.2f}"


def summarize(results: list[TaskResult]) -> dict:
    ok = [r for r in results if r.ok]
    qualities = [r.quality for r in ok if r.quality is not None]
    return {
        "tasks_total": len(results),
        "tasks_ok": len(ok),
        "avg_gen_tps": round(mean([r.gen_tps for r in ok]), 2) if ok else 0.0,
        "avg_ttft_s": round(mean([r.ttft_s for r in ok]), 3) if ok else 0.0,
        "avg_prompt_tps": round(mean([r.prompt_tps for r in ok if r.prompt_tps > 0] or [0.0]), 1),
        "avg_quality": round(mean(qualities), 1) if qualities else None,
        "total_completion_tokens": sum(r.completion_tokens for r in ok),
    }


def save_run(endpoint: str, model: str, results: list[TaskResult], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model)[:60]
    path = RESULTS_DIR / f"{ts:%Y%m%d-%H%M%S}_{slug}.json"
    doc = {
        "version": SUITE_VERSION,
        "quality_version": QUALITY_VERSION,
        "timestamp": ts.isoformat(timespec="seconds"),
        "endpoint": endpoint,
        "model": model,
        "tasks": [asdict(r) for r in results],
        "summary": summary,
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def load_runs() -> list[dict]:
    runs: list[dict] = []
    if not RESULTS_DIR.exists():
        return runs
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        doc["_path"] = str(path)
        runs.append(doc)
    return runs


def board_rows(runs: list[dict]) -> list[tuple]:
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        summary = run.get("summary") or {}
        if summary.get("tasks_ok", 0) == 0:
            continue
        by_model.setdefault(run.get("model", "?"), []).append(run)
    rows = []
    for model, model_runs in by_model.items():
        gens = [r["summary"]["avg_gen_tps"] for r in model_runs]
        ttfts = [r["summary"]["avg_ttft_s"] for r in model_runs]
        ptps = [r["summary"].get("avg_prompt_tps", 0.0) for r in model_runs]
        quals = [q for r in model_runs if (q := r["summary"].get("avg_quality")) is not None]
        avg_quality = mean(quals) if quals else None
        last = max(r.get("timestamp", "") for r in model_runs)
        rows.append(
            (model, len(model_runs), mean(gens), max(gens), avg_quality, mean(ttfts), mean(ptps), last)
        )
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows


def rank_label(index: int) -> str:
    return {0: "🥇", 1: "🥈", 2: "🥉"}.get(index, f"{index + 1}.")


def fmt_quality(quality: float | None) -> str:
    return "—" if quality is None else f"{quality:.0f}"


# --------------------------------------------------------------------------
# TUI
# --------------------------------------------------------------------------


class LMBench(App):
    TITLE = "lmbench"
    SUB_TITLE = "LM Studio local model benchmark"

    CSS = """
    TabbedContent { height: 1fr; }
    #endpoint-row { height: 3; margin: 0 1; }
    #endpoint-input { width: 1fr; }
    #connect-btn { margin-left: 1; }
    #status-line { height: 1; padding: 0 2; color: $text-muted; }
    #bench-body { height: 1fr; margin: 0 1; }
    #models-pane { width: 44; min-width: 28; margin-right: 1; }
    #models-table { height: 1fr; }
    #run-pane { width: 1fr; }
    #run-controls { height: 3; }
    #cancel-btn { margin-left: 1; }
    #live-stats { height: 5; border: round $panel; padding: 0 1; margin: 1 0; }
    #bench-log { height: 1fr; border: round $panel; }
    #runs-table { height: 1fr; margin: 0 1; }
    #detail-table { height: 14; margin: 0 1 1 1; }
    #board-table { height: 1fr; margin: 0 1; }
    #board-note { height: 1; padding: 0 2; color: $text-muted; }
    Label { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "connect", "Connect"),
        Binding("r", "run_bench", "Run bench"),
        Binding("x", "cancel_bench", "Cancel"),
        Binding("d", "delete_run", "Delete run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.selected_model: str | None = None
        self._bench_running = False
        self._runs: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-bench"):
            with TabPane("Benchmark", id="tab-bench"):
                with Horizontal(id="endpoint-row"):
                    yield Input(value=DEFAULT_ENDPOINT, id="endpoint-input")
                    yield Button("Connect", id="connect-btn", variant="primary")
                yield Static("not connected", id="status-line")
                with Horizontal(id="bench-body"):
                    with Vertical(id="models-pane"):
                        yield Label("Models (↑/↓ to pick)")
                        yield DataTable(id="models-table")
                    with Vertical(id="run-pane"):
                        with Horizontal(id="run-controls"):
                            yield Button("Run benchmark", id="run-btn", variant="success")
                            yield Button("Cancel", id="cancel-btn", variant="error", disabled=True)
                        yield ProgressBar(id="bench-progress", show_eta=False)
                        yield Static("idle", id="live-stats")
                        yield RichLog(id="bench-log", markup=True, wrap=True)
            with TabPane("Results", id="tab-results"):
                yield Label("Saved runs (d deletes highlighted)")
                yield DataTable(id="runs-table")
                yield Label("Run detail")
                yield DataTable(id="detail-table")
            with TabPane("Leaderboard", id="tab-board"):
                yield DataTable(id="board-table")
                yield Static(
                    "Ranked by average generation tok/s · quality = 0-100 vs "
                    "reference answers (60% objective checks, 40% n-gram similarity)",
                    id="board-note",
                )
        yield Footer()

    def on_mount(self) -> None:
        models_table = self.query_one("#models-table", DataTable)
        models_table.cursor_type = "row"
        models_table.zebra_stripes = True
        models_table.add_columns("model")

        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.cursor_type = "row"
        runs_table.zebra_stripes = True
        runs_table.add_columns(
            "date", "model", "gen tok/s", "quality", "ttft ms", "prompt tok/s", "tasks"
        )

        detail_table = self.query_one("#detail-table", DataTable)
        detail_table.cursor_type = "row"
        detail_table.zebra_stripes = True
        detail_table.add_columns(
            "task", "quality", "ttft ms", "gen tok/s", "prompt tok/s", "in→out", "time s", "note"
        )

        board_table = self.query_one("#board-table", DataTable)
        board_table.cursor_type = "row"
        board_table.zebra_stripes = True
        board_table.add_columns(
            "rank", "model", "runs", "avg tok/s", "best tok/s", "quality", "avg ttft ms",
            "avg prompt tok/s", "last run",
        )

        self.refresh_results_table()
        self.refresh_board_table()
        self.action_connect()

    # ---------------------------------------------------------------- helpers

    def _endpoint(self) -> str:
        raw = self.query_one("#endpoint-input", Input).value.strip().rstrip("/")
        if raw and "://" not in raw:
            raw = f"http://{raw}"
        return raw or DEFAULT_ENDPOINT

    def _log(self) -> RichLog:
        return self.query_one("#bench-log", RichLog)

    def _set_status(self, markup: str) -> None:
        self.query_one("#status-line", Static).update(markup)

    def _set_live(self, markup: str) -> None:
        self.query_one("#live-stats", Static).update(markup)

    def _set_running(self, running: bool) -> None:
        self._bench_running = running
        self.query_one("#run-btn", Button).disabled = running
        self.query_one("#cancel-btn", Button).disabled = not running

    # ---------------------------------------------------------------- actions

    def action_connect(self) -> None:
        self.connect_worker(self._endpoint())

    def action_run_bench(self) -> None:
        if self._bench_running:
            self.notify("benchmark already running", severity="warning")
            return
        if not self.selected_model:
            self.notify("no model selected — connect first, then pick one", severity="warning")
            return
        self.bench_worker(self._endpoint(), self.selected_model)

    def action_cancel_bench(self) -> None:
        if self._bench_running:
            self.workers.cancel_group(self, "bench")

    def action_delete_run(self) -> None:
        if self.query_one(TabbedContent).active != "tab-results":
            return
        table = self.query_one("#runs-table", DataTable)
        if table.row_count == 0:
            return
        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        if row_key is None or row_key.value is None:
            return
        path = Path(str(row_key.value))
        try:
            path.unlink()
        except OSError as exc:
            self.notify(f"delete failed: {exc}", severity="error")
            return
        self.notify(f"deleted {path.name}")
        self.refresh_results_table()
        self.refresh_board_table()

    # ---------------------------------------------------------------- events

    @on(Button.Pressed, "#connect-btn")
    def _connect_pressed(self) -> None:
        self.action_connect()

    @on(Button.Pressed, "#run-btn")
    def _run_pressed(self) -> None:
        self.action_run_bench()

    @on(Button.Pressed, "#cancel-btn")
    def _cancel_pressed(self) -> None:
        self.action_cancel_bench()

    @on(Input.Submitted, "#endpoint-input")
    def _endpoint_submitted(self) -> None:
        self.action_connect()

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-results":
            self.refresh_results_table()
        elif event.pane.id == "tab-board":
            self.refresh_board_table()

    @on(DataTable.RowHighlighted, "#models-table")
    def _model_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.selected_model = str(event.row_key.value)

    @on(DataTable.RowHighlighted, "#runs-table")
    def _run_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.show_detail(str(event.row_key.value))

    # ---------------------------------------------------------------- workers

    @work(exclusive=True, group="connect")
    async def connect_worker(self, endpoint: str) -> None:
        self._set_status(f"connecting to {endpoint} …")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                models = await fetch_models(client, endpoint)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"[red]✗ {endpoint} — {str(exc)[:120]}[/red]")
            return

        chat_models = [m for m in models if "embed" not in m.lower()]
        skipped = len(models) - len(chat_models)

        table = self.query_one("#models-table", DataTable)
        table.clear()
        for model in chat_models:
            table.add_row(model, key=model)
        if chat_models:
            self.selected_model = chat_models[0]
        else:
            self.selected_model = None

        note = f" ({skipped} embedding model{'s' if skipped != 1 else ''} hidden)" if skipped else ""
        self._set_status(
            f"[green]● connected[/green] {endpoint} — {len(chat_models)} chat models{note}"
        )

    @work(exclusive=True, group="bench")
    async def bench_worker(self, endpoint: str, model: str) -> None:
        log = self._log()
        progress = self.query_one("#bench-progress", ProgressBar)
        total_steps = len(SCORED) + 1
        progress.update(total=total_steps, progress=0)
        self._set_running(True)
        results: list[TaskResult] = []
        started = time.perf_counter()

        timeout = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=60.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                log.write(f"[b]{model}[/b] — suite v{SUITE_VERSION}, temp 0, seed 42")
                self._set_live(f"[b]{model}[/b]\nwarmup (loads model if needed) …")
                warm = await run_bench_task(client, endpoint, model, WARMUP)
                progress.advance(1)
                if not warm.ok:
                    log.write(f"[red]warmup failed: {warm.error}[/red]")
                    self._set_live("[red]warmup failed — see log[/red]")
                    return
                log.write(f"  warmup ok ({warm.total_s:.1f}s, not scored)")

                for index, task in enumerate(SCORED, start=1):
                    self._set_live(
                        f"[b]{model}[/b]\n"
                        f"task {index}/{len(SCORED)}: {task.name} "
                        f"(max {task.max_tokens} tok) …"
                    )
                    result = await run_bench_task(client, endpoint, model, task)
                    if result.ok:
                        result.quality, result.quality_note = await score_output(
                            task.name, result.output
                        )
                    results.append(result)
                    progress.advance(1)
                    if result.ok:
                        est = " [dim](tokens estimated)[/dim]" if result.tokens_estimated else ""
                        qual = (
                            f"qual {result.quality:3.0f} · " if result.quality is not None else ""
                        )
                        log.write(
                            f"  [b]{result.name:<10}[/b] "
                            f"{result.gen_tps:6.1f} tok/s · "
                            f"{qual}"
                            f"ttft {result.ttft_s * 1000:5.0f} ms · "
                            f"prompt {result.prompt_tps:6.0f} tok/s · "
                            f"{result.prompt_tokens}→{result.completion_tokens}{est}"
                        )
                    else:
                        log.write(f"  [red]{result.name} failed: {result.error}[/red]")

            summary = summarize(results)
            elapsed = time.perf_counter() - started
            path = save_run(endpoint, model, results, summary)
            log.write(
                f"[green]done in {elapsed:.0f}s — avg {summary['avg_gen_tps']:.1f} tok/s, "
                f"qual {fmt_quality(summary['avg_quality'])}, "
                f"avg ttft {summary['avg_ttft_s'] * 1000:.0f} ms — saved {path.name}[/green]"
            )
            self._set_live(
                f"[b]{model}[/b]\n"
                f"[green]avg {summary['avg_gen_tps']:.1f} tok/s · "
                f"qual {fmt_quality(summary['avg_quality'])}/100 · "
                f"ttft {summary['avg_ttft_s'] * 1000:.0f} ms · "
                f"prompt {summary['avg_prompt_tps']:.0f} tok/s[/green]\n"
                f"{summary['tasks_ok']}/{summary['tasks_total']} tasks ok · saved"
            )
            self.refresh_results_table()
            self.refresh_board_table()
        except asyncio.CancelledError:
            log.write("[yellow]benchmark cancelled — nothing saved[/yellow]")
            self._set_live("[yellow]cancelled[/yellow]")
            raise
        finally:
            self._set_running(False)

    # ---------------------------------------------------------------- tables

    def refresh_results_table(self) -> None:
        runs = load_runs()
        self._runs = {run["_path"]: run for run in runs}
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for run in runs:
            summary = run.get("summary") or {}
            table.add_row(
                run.get("timestamp", "?")[:16].replace("T", " "),
                run.get("model", "?"),
                f"{summary.get('avg_gen_tps', 0):.1f}",
                fmt_quality(summary.get("avg_quality")),
                f"{summary.get('avg_ttft_s', 0) * 1000:.0f}",
                f"{summary.get('avg_prompt_tps', 0):.0f}",
                f"{summary.get('tasks_ok', 0)}/{summary.get('tasks_total', 0)}",
                key=run["_path"],
            )
        if runs:
            self.show_detail(runs[0]["_path"])
        else:
            self.query_one("#detail-table", DataTable).clear()

    def show_detail(self, path: str) -> None:
        run = self._runs.get(path)
        table = self.query_one("#detail-table", DataTable)
        table.clear()
        if run is None:
            return
        for task in run.get("tasks", []):
            if task.get("ok"):
                notes = []
                if task.get("quality_note"):
                    notes.append(task["quality_note"])
                if task.get("tokens_estimated"):
                    notes.append("tokens estimated")
                table.add_row(
                    task.get("name", "?"),
                    fmt_quality(task.get("quality")),
                    f"{task.get('ttft_s', 0) * 1000:.0f}",
                    f"{task.get('gen_tps', 0):.1f}",
                    f"{task.get('prompt_tps', 0):.0f}",
                    f"{task.get('prompt_tokens', 0)}→{task.get('completion_tokens', 0)}",
                    f"{task.get('total_s', 0):.1f}",
                    " · ".join(notes),
                )
            else:
                table.add_row(
                    task.get("name", "?"), "—", "—", "—", "—", "—",
                    f"{task.get('total_s', 0):.1f}",
                    f"failed: {task.get('error', '?')[:40]}",
                )

    def refresh_board_table(self) -> None:
        table = self.query_one("#board-table", DataTable)
        table.clear()
        for index, row in enumerate(board_rows(load_runs())):
            model, run_count, avg_gen, best_gen, avg_quality, avg_ttft, avg_ptps, last = row
            table.add_row(
                rank_label(index),
                model,
                str(run_count),
                f"{avg_gen:.1f}",
                f"{best_gen:.1f}",
                fmt_quality(avg_quality),
                f"{avg_ttft * 1000:.0f}",
                f"{avg_ptps:.0f}",
                last[:16].replace("T", " "),
            )


if __name__ == "__main__":
    LMBench().run()
