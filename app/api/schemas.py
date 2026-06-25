from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Represent a user question and optional LangGraph conversation thread."""

    question: str
    thread_id: str = "default"


class QAResponse(BaseModel):
    answer: str
    explanation: str
    fol: Optional[str] = None
    cot: Optional[List[str]] = None
    premises: Optional[List[str]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class EDAJobResponse(BaseModel):
    job_id: str
    status: str = "pending"


class EDAResult(BaseModel):
    job_id: str
    status: str
    summary_md: Optional[str] = None
    num_stats: Optional[Dict[str, Any]] = None
    cat_stats: Optional[Dict[str, Any]] = None
    profile_text: Optional[str] = None
    shape: Optional[Dict[str, int]] = None
    cleaned_file_path: Optional[str] = None
