from pydantic import BaseModel
from typing import Dict, Any, Optional


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str
    content: Optional[str] = None