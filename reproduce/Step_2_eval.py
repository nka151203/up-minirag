"""LLM-as-judge evaluation for MiniRAG QA output.

Reads the CSV produced by ``Step_1_QA.py`` (columns ``Question``,
``Gold Answer``, ``minirag``), asks an OpenAI-compatible judge whether each
answer is correct / erroneous, and writes a per-question CSV plus a summary.

The judge prompt is adapted from NDAKG's ``compair_ans.py`` so that results on
the VNH3045 dataset are comparable, while this script itself only depends on
the standard MiniRAG environment.

Usage:
    python3 reproduce/Step_2_eval.py \
        --input  ./outputs/NDAKG-VNH3045-gpt41mini/qa_output.csv \
        --output ./outputs/NDAKG-VNH3045-gpt41mini/qa_eval.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_async

load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
)

JUDGE_PROMPT = """You are an expert judge evaluating a Question Answering system.
Given a Question, an Expected Answer (Gold), and a System Response (RAG), evaluate the System Response based on two metrics:

1. Accuracy (acc): True if the System Response is semantically equivalent to or correctly answers the Expected Answer. For instance, expected "bottle", response "water bottle" is True.
2. Error (err): True if the System Response provides incorrect information without recognizing its mistake. For example, if it confidently gives a wrong answer, this is True.

Return your evaluation as a valid JSON object with the keys "acc" (boolean) and "err" (boolean). Do not include any other text.

Question: {question}
Expected Answer: {gold}
System Response: {rag}"""


def _parse_json(content: str) -> dict:
    content = content.strip()
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fall back to the first {...} block in the response.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _as_bool(value):
    """Interpret common JSON/string truth values; return None if unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0", ""}:
            return False
    return None


async def judge_one(
    client: AsyncOpenAI,
    model: str,
    question: str,
    gold: str,
    rag: str,
    temperature: float | None = None,
    json_mode: bool = False,
) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, rag=rag)
    try:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        result = _parse_json(content)
        acc = _as_bool(result.get("acc"))
        err = _as_bool(result.get("err"))
        if acc is None or err is None:
            raise ValueError(f"judge did not return boolean acc/err: {content[:200]!r}")
        return {
            "acc": acc,
            "err": err,
            "judge_ok": True,
            "judge_raw": content,
        }
    except Exception as exc:  # keep the run going, but mark the row as a judge failure
        return {
            "acc": "",
            "err": "",
            "judge_ok": False,
            "judge_raw": f"ERROR: {exc}",
        }


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row.get("Question", "") for row in read_rows(path)}


async def run(args) -> None:
    rows = read_rows(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Judge model: {args.model}")
    print(f"Judge base URL: {args.base_url}")

    gold_col = args.gold_column
    answer_col = args.answer_column
    missing = [c for c in ("Question", gold_col, answer_col) if c not in rows[0]]
    if missing:
        raise SystemExit(f"Missing column(s) in {args.input}: {missing}")

    out_path = Path(args.output)
    if args.force and out_path.exists():
        out_path.unlink()

    done = already_done(out_path)
    todo = [row for row in rows if row["Question"] not in done]
    if not todo:
        print(f"All {len(rows)} rows already evaluated in {out_path}")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(row: dict) -> dict:
        async with semaphore:
            result = await judge_one(
                client,
                args.model,
                row["Question"],
                row.get(gold_col, ""),
                row.get(answer_col, ""),
                temperature=args.temperature,
                json_mode=args.json_mode,
            )
        return {**row, **result}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["acc", "err", "judge_ok", "judge_raw"]
    write_header = not out_path.exists()

    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        tasks = [worker(row) for row in todo]
        for coro in tqdm_async(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
            writer.writerow(await coro)
            handle.flush()

    results = read_rows(out_path)
    total = len(results)
    judged = [r for r in results if _as_bool(r.get("judge_ok")) is True]
    failed = [r for r in results if _as_bool(r.get("judge_ok")) is not True]
    n = len(judged)
    acc = err = 0

    print(f"Total rows: {total}")
    if n:
        acc = sum(1 for r in judged if _as_bool(r.get("acc")) is True)
        err = sum(1 for r in judged if _as_bool(r.get("err")) is True)
        print(f"Accuracy (acc): {acc}/{n} ({acc / n * 100:.2f}%)")
        print(f"Error rate (err): {err}/{n} ({err / n * 100:.2f}%)")
    if failed:
        print(
            f"Judge failures: {len(failed)}/{total} "
            "(not counted as wrong answers)"
        )
        for row in failed[:3]:
            print("  -", str(row.get("judge_raw"))[:300])
    print(f"Per-question results: {out_path}")

    meta = {
        "judge_model": args.model,
        "judge_base_url": args.base_url,
        "input": str(args.input),
        "output": str(out_path),
        "rows": total,
        "judged": n,
        "acc": acc if n else None,
        "err": err if n else None,
        "judge_failures": len(failed),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Run metadata: {meta_path}")


def get_args():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for MiniRAG QA output")
    parser.add_argument("--input", required=True, help="CSV produced by Step_1_QA.py")
    parser.add_argument("--output", required=True, help="Per-question evaluation CSV")
    parser.add_argument(
        "--gold-column", default="Gold Answer", help="Gold answer column name"
    )
    parser.add_argument(
        "--answer-column", default="minirag", help="System answer column name"
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional; omitted by default because some models only accept their default",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Send response_format=json_object (not supported by every endpoint)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate all rows and replace the output file",
    )
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Missing OPENAI_API_BASE / OPENAI_API_KEY (set .env or pass flags)")
    return args


if __name__ == "__main__":
    asyncio.run(run(get_args()))
