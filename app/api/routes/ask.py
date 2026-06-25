from fastapi import APIRouter
from app.api.schemas import AskRequest
from app.core.pipeline import SESSION_ID, run_qa_pipeline
from app.graph.schema import GraphState
from app.memory.eda_store import eda_store

router = APIRouter()


@router.post("/ask", response_model=GraphState)
async def ask(request: AskRequest) -> GraphState:
    """Run QA in the active EDA job thread when no thread is provided."""
    # TODO: validate max question length to prevent oversized prompts
    active_job_id = await eda_store.get_active_eda(SESSION_ID)
    thread_id = request.thread_id
    if active_job_id and thread_id == SESSION_ID:
        thread_id = active_job_id
    if active_job_id:
        await eda_store.set_active_eda(thread_id, active_job_id)
    if thread_id != request.thread_id:
        request = request.model_copy(update={"thread_id": thread_id})
    return await run_qa_pipeline(request)
