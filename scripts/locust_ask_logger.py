"""
Locust load test cho endpoint POST /ask.

Nhiệm vụ:
  - Đọc một file CSV chứa câu hỏi (mỗi dòng 1 câu hỏi).
  - Bắn từng câu hỏi vào /ask (mỗi câu gửi đúng 1 lần, chia đều cho các user).
  - Log TẤT CẢ response (kèm metadata) ra MỘT file JSON khi test kết thúc.
    Đồng thời ghi dần ra file .jsonl để an toàn nếu test bị kill giữa chừng.

Cách chạy:
    locust -f scripts/locust_ask_logger.py --headless -u 5 -r 1 \
           --host http://localhost:8000

Cấu hình: sửa trực tiếp 2 hằng số CSV_PATH và QUESTION_COLUMN bên dưới.
Output (JSON + JSONL) được ghi ra ĐÚNG thư mục chứa file CSV.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Config — sửa trực tiếp ở đây
# ---------------------------------------------------------------------------
# Đường dẫn file CSV câu hỏi.
CSV_PATH = Path(r"C:\workspace\AIO-Conquer\app\data\EmployeeAttrition\EmployeeAttrition_QA_Benchmark.csv")

# Tên cột chứa câu hỏi trong CSV.
QUESTION_COLUMN = "Q"

# Mỗi câu hỏi dùng 1 thread_id riêng (uuid) để độc lập, không dính bộ nhớ
# hội thoại của nhau. Đổi thành False nếu muốn dùng chung thread "default".
UNIQUE_THREADS = True

# Timeout mỗi request (giây).
REQUEST_TIMEOUT = 180.0

# Giới hạn số câu hỏi (để debug nhanh). 0 = chạy hết.
MAX_QUESTIONS = 0

# Output ghi ra ĐÚNG thư mục chứa file CSV, tên theo file CSV.
# Ví dụ: .../SuperStore/ask_SuperStore_QA_Benchmark_<timestamp>.json (+ .jsonl)
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON = CSV_PATH.parent / f"ask_{CSV_PATH.stem}_{_TIMESTAMP}.json"
OUTPUT_JSONL = OUTPUT_JSON.with_suffix(".jsonl")  # bản ghi-dần an toàn

# ---------------------------------------------------------------------------
# Shared state (an toàn đa luồng)
# ---------------------------------------------------------------------------
_task_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
_results: list[dict[str, Any]] = []
_results_lock = threading.Lock()
_jsonl_lock = threading.Lock()
_total_questions = 0
_quit_lock = threading.Lock()
_quitting = False


def _load_questions() -> None:
    """Nạp câu hỏi từ CSV vào hàng đợi dùng chung."""
    global _total_questions

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV câu hỏi: {CSV_PATH}")

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if QUESTION_COLUMN not in fieldnames:
            raise ValueError(
                f"Không thấy cột '{QUESTION_COLUMN}' trong CSV. Các cột hiện có: {fieldnames}"
            )

        count = 0
        for i, row in enumerate(reader):
            question = (row.get(QUESTION_COLUMN) or "").strip()
            if not question:
                continue

            thread_id = f"loadtest-{uuid.uuid4().hex[:12]}" if UNIQUE_THREADS else "default"

            _task_queue.put({"index": i, "question": question, "thread_id": thread_id})
            count += 1
            if MAX_QUESTIONS and count >= MAX_QUESTIONS:
                break

    _total_questions = count
    # đảm bảo thư mục output tồn tại + reset file jsonl của lần chạy này
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.write_text("", encoding="utf-8")


def _record(entry: dict[str, Any]) -> None:
    """Lưu một kết quả vào bộ nhớ + ghi dần ra .jsonl (an toàn đa luồng)."""
    with _results_lock:
        _results.append(entry)
    line = json.dumps(entry, ensure_ascii=False)
    with _jsonl_lock:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _extract_answer(state: dict[str, Any]) -> Optional[str]:
    """Lấy câu trả lời gọn từ GraphState trả về (field 'response' kiểu QAResponse)."""
    response = state.get("response")
    if isinstance(response, dict):
        return response.get("answer")
    return None


def _stop_runner(environment) -> None:
    """Dừng test khi đã gửi hết câu hỏi (chỉ gọi quit() một lần)."""
    global _quitting
    with _quit_lock:
        if _quitting:
            return
        _quitting = True
    if environment.runner is not None:
        environment.runner.quit()


# ---------------------------------------------------------------------------
# Locust events
# ---------------------------------------------------------------------------
@events.init.add_listener
def on_init(environment, **_kwargs):
    _load_questions()
    print(
        f"[locust] Đã nạp {_total_questions} câu hỏi từ {CSV_PATH}\n"
        f"[locust] Output JSON : {OUTPUT_JSON}\n"
        f"[locust] Output JSONL: {OUTPUT_JSONL} (ghi dần, an toàn nếu bị kill)"
    )


@events.quitting.add_listener
def on_quit(environment, **_kwargs):
    """Khi test kết thúc: dump toàn bộ kết quả ra 1 file JSON."""
    with _results_lock:
        # giữ thứ tự ổn định theo index câu hỏi trong CSV
        ordered = sorted(_results, key=lambda r: r.get("index", 0))
        payload = {
            "meta": {
                "host": getattr(environment, "host", None),
                "csv_path": str(CSV_PATH),
                "total_questions": _total_questions,
                "answered": len(ordered),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "results": ordered,
        }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[locust] Hoàn tất — đã ghi {len(ordered)}/{_total_questions} "
        f"response vào {OUTPUT_JSON}"
    )


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class AskUser(HttpUser):
    """Mỗi user lấy câu hỏi từ hàng đợi và gọi /ask cho tới khi hết câu hỏi."""

    # Không chờ giữa các request — bắn liên tục. Đổi thành between(1, 3) nếu
    # muốn mô phỏng người dùng thật.
    wait_time = between(0, 0)

    @task
    def ask_question(self):
        try:
            item = _task_queue.get_nowait()
        except queue.Empty:
            _stop_runner(self.environment)
            return

        index = item["index"]
        question = item["question"]
        thread_id = item["thread_id"]
        started_at = datetime.now(timezone.utc).isoformat()

        with self.client.post(
            "/ask",
            json={"question": question, "thread_id": thread_id},
            catch_response=True,
            name="/ask",
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            entry: dict[str, Any] = {
                "index": index,
                "question": question,
                "thread_id": thread_id,
                "status_code": resp.status_code,
                "elapsed_ms": resp.elapsed.total_seconds() * 1000 if resp.elapsed else None,
                "started_at": started_at,
            }

            if resp.status_code == 200:
                try:
                    state = resp.json()
                except json.JSONDecodeError:
                    entry["error"] = "Response không phải JSON hợp lệ"
                    entry["raw_text"] = resp.text[:2000]
                    resp.failure(entry["error"])
                else:
                    entry["answer"] = _extract_answer(state)
                    entry["warnings"] = state.get("warnings")
                    entry["run_id"] = state.get("run_id")
                    # Lưu nguyên response (toàn bộ GraphState) để không mất dữ liệu.
                    entry["response_state"] = state
                    resp.success()
                    print(f"[{index}] OK ({entry['elapsed_ms']:.0f} ms): {question[:60]}")
            else:
                entry["error"] = f"HTTP {resp.status_code}"
                entry["raw_text"] = resp.text[:2000]
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                print(f"[{index}] FAIL {resp.status_code}: {question[:60]}")

            _record(entry)
