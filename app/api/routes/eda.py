from __future__ import annotations

import os
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from app.api.schemas import EDAJobResponse, EDAResult
from app.core.eda_pipeline import run_eda
from app.memory.redis_client import memory

SESSION_ID = "default"

router = APIRouter(prefix="/eda", tags=["eda"])

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@router.post("/analyze", response_model=EDAJobResponse)
async def analyze_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> EDAJobResponse:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    job_id = str(uuid.uuid4())

    # Save upload to temp file
    tmp_dir = os.path.join(tempfile.gettempdir(), "eda", "uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, f"{job_id}{ext}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    await memory.set_eda_status(job_id, "pending")
    await memory.set_active_eda(SESSION_ID, job_id)

    background_tasks.add_task(run_eda, job_id, file_path)

    return EDAJobResponse(job_id=job_id, status="pending")


@router.get("/result/{job_id}", response_model=EDAResult)
async def get_result(job_id: str) -> EDAResult:
    status = await memory.get_eda_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    if status.startswith("error:"):
        raise HTTPException(status_code=500, detail=status)

    if status == "pending":
        return EDAResult(job_id=job_id, status="pending")

    result = await memory.get_eda_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return EDAResult(job_id=job_id, status="done", **result)
