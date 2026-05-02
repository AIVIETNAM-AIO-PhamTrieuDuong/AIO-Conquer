"""
Debug script for parse_response.

Usage:
    python scripts/debug_parser.py
    python scripts/debug_parser.py --raw "your raw llm output here"
    python scripts/debug_parser.py --file path/to/raw.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from app.validation.parser import _extract_json, parse_response  # noqa: E402

SAMPLES = {
    "plain_json": '{"answer": "Yes", "explanation": "Because", "cot": null, "premises": null, "fol": null, "confidence": 0.9}',
    "fenced_json": '```json\n{"answer": "Yes", "explanation": "Because", "cot": null, "premises": null, "fol": null, "confidence": 0.9}\n```',
    "fenced_no_lang": '```\n{"answer": "Yes", "explanation": null, "cot": null, "premises": null, "fol": null, "confidence": null}\n```',
    "json_with_prefix": 'Here is the answer:\n{"answer": "Yes", "explanation": null, "cot": null, "premises": null, "fol": null, "confidence": null}',
    "broken_json": '{"answer": "Yes", explanation missing quotes, "cot": null}',
    "nested_json_in_explanation": '{"answer": "?", "explanation": "{\\n  \\"answer\\": \\"real answer\\"\\n}", "cot": null, "premises": null, "fol": null, "confidence": null}',
    "truncated_json": '{\n  "answer": "Bộ dữ liệu có 5000 hàng",\n  "explanation": "Chi tiết phân tích...",\n  "cot": [\n    "Bước 1: xác định shape",\n    "Bước 2: đếm outl',
}


def run_sample(name: str, raw: str) -> None:
    print(f"\n{'='*60}")
    print(f"SAMPLE: {name}")
    print(f"RAW ({len(raw)} chars):\n{raw[:300]}")
    print("-" * 40)

    extracted = _extract_json(raw)
    print(f"EXTRACTED:\n{extracted[:300]}")
    print("-" * 40)

    result = parse_response(raw)
    print(f"answer     : {result.answer!r}")
    print(f"explanation: {result.explanation!r}")
    print(f"cot        : {result.cot}")
    print(f"premises   : {result.premises}")
    print(f"confidence : {result.confidence}")

    if result.answer == "?":
        print("STATUS: PARSE FAILED")
    else:
        print("STATUS: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", help="Raw LLM output string to test")
    parser.add_argument("--file", help="Path to file containing raw LLM output")
    args = parser.parse_args()

    if args.raw:
        run_sample("cli_input", args.raw)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            run_sample(args.file, f.read())
    else:
        print("Running all built-in samples...")
        for name, raw in SAMPLES.items():
            run_sample(name, raw)


if __name__ == "__main__":
    main()
