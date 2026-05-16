from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import chainlit as cl
import httpx

try:
    from config import (
        API_BASE_URL,
        REQUEST_TIMEOUT,
        EDA_POLL_INTERVAL_SECONDS,
        EDA_MAX_WAIT_SECONDS,
        SUPPORTED_EXTENSIONS,
    )
except ModuleNotFoundError:
    from app.ui.config import (
        API_BASE_URL,
        REQUEST_TIMEOUT,
        EDA_POLL_INTERVAL_SECONDS,
        EDA_MAX_WAIT_SECONDS,
        SUPPORTED_EXTENSIONS,
    )

class BackendError(RuntimeError):
    pass


SUMMARY_TOGGLE_MESSAGE = "aio:toggle-summary"
SUMMARY_READY_MESSAGE = "aio:summary-ready"
SUMMARY_DISABLED_MESSAGE = "aio:summary-disabled"
SUMMARY_VISIBLE_MESSAGE = "aio:summary-visible"
SUMMARY_HIDDEN_MESSAGE = "aio:summary-hidden"


def _file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def _is_supported_file(file_name: str) -> bool:
    return _file_extension(file_name) in SUPPORTED_EXTENSIONS


def _shape_label(result: dict[str, Any]) -> str:
    shape = result.get("shape") or {}
    rows = shape.get("rows")
    cols = shape.get("cols")
    if rows is None or cols is None:
        return "processed"
    return f"processed ({rows:,} rows x {cols:,} columns)"


def _format_answer(payload: dict[str, Any]) -> str:
    answer = payload.get("answer") or "No answer returned by the backend."
    explanation = payload.get("explanation")
    confidence = payload.get("confidence")
    premises = payload.get("premises") or []

    parts = [answer.strip()]
    if explanation:
        parts.append(f"**Explanation**\n{explanation.strip()}")
    if premises:
        premise_lines = "\n".join(f"- {premise}" for premise in premises)
        parts.append(f"**Premises**\n{premise_lines}")
    if confidence is not None:
        parts.append(f"**Confidence:** {confidence:.2f}")
    return "\n\n".join(parts)


async def _post_file_to_backend(file_path: str, file_name: str) -> str:
    if not _is_supported_file(file_name):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise BackendError(f"Unsupported file type. Upload one of: {supported}.")

    mime_type = "text/csv" if _file_extension(file_name) == ".csv" else "application/octet-stream"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            with open(file_path, "rb") as file_obj:
                response = await client.post(
                    f"{API_BASE_URL}/eda/analyze",
                    files={"file": (file_name, file_obj, mime_type)},
                )
    except httpx.RequestError as exc:
        raise BackendError(f"Could not reach FastAPI backend at {API_BASE_URL}: {exc}") from exc

    if response.status_code >= 400:
        raise BackendError(response.text)

    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise BackendError("Backend did not return a job_id for the uploaded file.")
    return job_id


async def _poll_eda_result(job_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + EDA_MAX_WAIT_SECONDS

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        while True:
            try:
                response = await client.get(f"{API_BASE_URL}/eda/result/{job_id}")
            except httpx.RequestError as exc:
                raise BackendError(f"Could not poll FastAPI backend at {API_BASE_URL}: {exc}") from exc

            if response.status_code >= 400:
                raise BackendError(response.text)

            payload = response.json()
            if payload.get("status") == "done":
                return payload

            if asyncio.get_running_loop().time() >= deadline:
                raise BackendError(
                    f"Timed out waiting for EDA job {job_id} after {EDA_MAX_WAIT_SECONDS} seconds."
                )

            await asyncio.sleep(EDA_POLL_INTERVAL_SECONDS)


async def _get_eda_result(job_id: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(f"{API_BASE_URL}/eda/result/{job_id}")
    except httpx.RequestError as exc:
        raise BackendError(f"Could not fetch FastAPI backend at {API_BASE_URL}: {exc}") from exc

    if response.status_code >= 400:
        raise BackendError(response.text)

    payload = response.json()
    if payload.get("status") != "done":
        raise BackendError("The uploaded file is still being processed.")
    return payload


async def _hide_summary_sidebar() -> None:
    await cl.ElementSidebar.set_elements([])
    cl.user_session.set("summary_visible", False)
    await cl.send_window_message(SUMMARY_HIDDEN_MESSAGE)


async def _show_summary_sidebar(job_id: str) -> None:
    result = await _get_eda_result(job_id)
    summary_md = result.get("summary_md")
    if not summary_md:
        raise BackendError("No summary markdown is available for this file.")

    file_name = cl.user_session.get("active_file_name") or "Processed file"
    await cl.ElementSidebar.set_title(f"Summary: {file_name}")
    await cl.ElementSidebar.set_elements(
        [
            cl.Text(
                name="summary_md",
                content=summary_md,
                display="side",
            )
        ]
    )
    cl.user_session.set("summary_visible", True)
    await cl.send_window_message(SUMMARY_VISIBLE_MESSAGE)


async def _toggle_summary_sidebar() -> None:
    job_id = cl.user_session.get("summary_job_id")
    if not job_id or not cl.user_session.get("summary_ready"):
        await cl.Message(content="Upload and process a file before viewing its summary.").send()
        return

    if cl.user_session.get("summary_visible"):
        await _hide_summary_sidebar()
        return

    try:
        await _show_summary_sidebar(job_id)
    except BackendError as exc:
        await cl.Message(content=f"Could not load summary: `{exc}`").send()


async def _analyze_file(file_path: str, file_name: str) -> dict[str, Any]:
    job_id = await _post_file_to_backend(file_path, file_name)
    cl.user_session.set("eda_job_id", job_id)
    return await _poll_eda_result(job_id)


async def _ask_backend(question: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.post(f"{API_BASE_URL}/ask", json={"question": question})
    except httpx.RequestError as exc:
        raise BackendError(f"Could not reach FastAPI backend at {API_BASE_URL}: {exc}") from exc

    if response.status_code >= 400:
        raise BackendError(response.text)
    return response.json()


async def _handle_upload(file_path: str, file_name: str) -> None:
    status = cl.Message(content=f"Processing `{file_name}`...")
    await status.send()
    await cl.send_window_message(SUMMARY_DISABLED_MESSAGE)
    await _hide_summary_sidebar()
    cl.user_session.set("summary_ready", False)
    cl.user_session.set("summary_job_id", None)

    result = await _analyze_file(file_path, file_name)
    job_id = cl.user_session.get("eda_job_id")
    cl.user_session.set("active_file_name", file_name)
    cl.user_session.set("summary_job_id", job_id)
    cl.user_session.set("summary_ready", True)

    status.content = f"`{file_name}` is {_shape_label(result)}. Ask a question about it."
    await status.update()
    await cl.send_window_message(SUMMARY_READY_MESSAGE)


async def _ask_for_initial_file() -> None:
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload a CSV or Excel file to begin.",
            accept={
                "text/csv": [".csv"],
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
                "application/vnd.ms-excel": [".xls"],
            },
            max_files=1,
            max_size_mb=100,
            timeout=180,
            raise_on_timeout=False,
        ).send()

    uploaded = files[0]
    await _handle_upload(uploaded.path, uploaded.name)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("eda_job_id", None)
    cl.user_session.set("active_file_name", None)
    cl.user_session.set("summary_ready", False)
    cl.user_session.set("summary_job_id", None)
    cl.user_session.set("summary_visible", False)
    await cl.send_window_message(SUMMARY_DISABLED_MESSAGE)

    try:
        await _ask_for_initial_file()
    except BackendError as exc:
        await cl.Message(content=f"File processing failed: `{exc}`").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    file_elements = [
        element
        for element in (message.elements or [])
        if getattr(element, "path", None)
    ]
    unsupported_files = [
        getattr(element, "name", "uploaded file")
        for element in file_elements
        if not _is_supported_file(getattr(element, "name", ""))
    ]
    if unsupported_files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        await cl.Message(
            content=f"`{unsupported_files[0]}` is not supported. Upload one of: {supported}."
        ).send()
        return

    if file_elements:
        uploaded = file_elements[0]
        try:
            await _handle_upload(uploaded.path, uploaded.name)
        except BackendError as exc:
            await cl.Message(content=f"File processing failed: `{exc}`").send()
            return

        if not message.content.strip():
            return

    if not cl.user_session.get("eda_job_id"):
        await _ask_for_initial_file()
        return

    question = message.content.strip()
    if not question:
        await cl.Message(content="Ask a question about the processed file.").send()
        return

    active_file_name = cl.user_session.get("active_file_name")
    status = cl.Message(content=f"Querying `{active_file_name}`...")
    await status.send()

    try:
        payload = await _ask_backend(question)
    except BackendError as exc:
        status.content = f"Backend request failed: `{exc}`"
        await status.update()
        return

    status.content = _format_answer(payload)
    await status.update()


@cl.on_window_message
async def on_window_message(message: str) -> None:
    if message == SUMMARY_TOGGLE_MESSAGE:
        await _toggle_summary_sidebar()
