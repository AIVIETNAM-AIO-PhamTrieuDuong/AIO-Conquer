"""
Locust load test — blasts /ask with questions from MobileGameChurn_QA_Benchmark.csv
and appends the LLM responses into a new column.

Usage:
    locust -f scripts/locustfile.py --headless -u 1 -r 1 --host http://localhost:8000

Results are written back to the same CSV (new column: llm_answer).
"""

import csv
import threading
from pathlib import Path

from locust import HttpUser, task, between, events

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent.parent / "app" / "data" / "SuperStore_QA_Benchmark.csv"
RESULT_COLUMN = "llm_answer"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_rows: list[dict] = []
_index: int = 0


def _load_csv() -> None:
    global _rows, _index
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _rows = list(reader)
        for row in _rows:
            row.setdefault(RESULT_COLUMN, "")
    _index = 0


def _save_csv() -> None:
    if not _rows:
        return
    fieldnames = list(_rows[0].keys())
    if RESULT_COLUMN not in fieldnames:
        fieldnames.append(RESULT_COLUMN)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_rows)


# ---------------------------------------------------------------------------
# Locust events
# ---------------------------------------------------------------------------
@events.init.add_listener
def on_init(environment, **_kwargs):
    _load_csv()
    print(f"[locust] Loaded {len(_rows)} questions from {CSV_PATH}")


@events.quitting.add_listener
def on_quit(environment, **_kwargs):
    _save_csv()
    answered = sum(1 for r in _rows if r.get(RESULT_COLUMN))
    print(f"[locust] Done — {answered}/{len(_rows)} answers saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# User — single user, sequential
# ---------------------------------------------------------------------------
class AskUser(HttpUser):
    wait_time = between(0, 0)   # no wait between requests

    @task
    def ask_question(self):
        global _index

        if _index >= len(_rows):
            self.environment.runner.quit()
            return

        row = _rows[_index]
        _index += 1

        if row.get(RESULT_COLUMN):   # skip already answered
            return

        with self.client.post(
            "/ask",
            json={"question": row["Q"]},
            catch_response=True,
            name="/ask",
            timeout=120,
        ) as resp:
            if resp.status_code == 200:
                payload = resp.json()
                row[RESULT_COLUMN] = payload.get("answer") or str(payload)
                print(f"[{_index}/{len(_rows)}] {row['Q'][:60]}")
                resp.success()
            else:
                # mark failure but don't retry — move on to next row
                print(f"[{_index}/{len(_rows)}] FAILED {resp.status_code}: {row['Q'][:60]}")
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

