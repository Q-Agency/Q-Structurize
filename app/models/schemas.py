from pydantic import BaseModel
from typing import Dict, Any


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str
    data: Dict[str, Any]