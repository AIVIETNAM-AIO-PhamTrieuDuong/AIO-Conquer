from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 20
QUESTION = "What is the region that contributes the most to the net profit?"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "app" / "data" / "SuperStore" / "SuperStores.csv"
OUTPUT_DIR = REPO_ROOT / "tests"


def main() -> None:
    """Run the SuperStore EDA upload, poll active status, and ask QA flow."""
    with httpx.Client(timeout=120.0) as client:
        job_id = start_eda_job(client)
        active_payload = wait_for_active_eda(client)
        answer_payload = ask_question(client, job_id)

    output_path = write_timestamped_response(
        {
            "job_id": job_id,
            "active_eda": active_payload,
            "question": QUESTION,
            "ask_response": answer_payload,
        }
    )
    print(f"Completed SuperStore QA flow. Response written to {output_path}")


def start_eda_job(client: httpx.Client) -> str:
    """Upload the SuperStore CSV to start an EDA analysis job."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    with DATASET_PATH.open("rb") as file_obj:
        response = client.post(
            f"{BASE_URL}/eda/analyze",
            files={
                "file": (
                    DATASET_PATH.name,
                    file_obj,
                    "text/csv",
                )
            },
        )
    response.raise_for_status()

    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job_id in EDA response: {payload}")
    return str(job_id)


def wait_for_active_eda(client: httpx.Client) -> dict[str, Any]:
    """Poll /dev/eda/active until the status field is no longer pending."""
    while True:
        response = client.get(f"{BASE_URL}/dev/eda/active")
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "pending":
            return payload

        time.sleep(POLL_INTERVAL_SECONDS)


def ask_question(client: httpx.Client, job_id: str) -> dict[str, Any]:
    """Ask the requested question using the EDA job_id as the thread_id."""
    response = client.post(
        f"{BASE_URL}/ask",
        json={"question": QUESTION, "thread_id": job_id},
    )
    response.raise_for_status()
    return response.json()


def write_timestamped_response(payload: dict[str, Any]) -> Path:
    """Write the collected flow response to a timestamp-named JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    main()
