from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


class TableMode(str, Enum):
    """Table extraction mode options."""
    FAST = "fast"
    ACCURATE = "accurate"


class AcceleratorDevice(str, Enum):
    """Accelerator device options."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class PipelineOptions(BaseModel):
    """
    Configurable options for Docling PDF pipeline.
    
    These options allow fine-tuning of the parsing process for optimal
    performance on the 72-core Xeon 6960P CPU.
    """
    enable_ocr: bool = Field(
        default=False,
        description="Enable OCR for scanned documents and images"
    )
    ocr_languages: List[str] = Field(
        default=["en"],
        description="OCR languages (e.g., ['en', 'es', 'de'])"
    )
    table_mode: TableMode = Field(
        default=TableMode.FAST,
        description="Table extraction mode: 'fast' or 'accurate'"
    )
    do_table_structure: bool = Field(
        default=True,
        description="Enable table structure extraction"
    )
    do_cell_matching: bool = Field(
        default=True,
        description="Enable cell matching for better table accuracy"
    )
    num_threads: int = Field(
        default=8,
        ge=1,
        le=144,
        description="Number of threads for processing (1-144, optimized for 72-core Xeon)"
    )
    accelerator_device: AcceleratorDevice = Field(
        default=AcceleratorDevice.CPU,
        description="Accelerator device: 'cpu', 'cuda', or 'auto'"
    )
    # Enrichment options (advanced features)
    do_code_enrichment: bool = Field(
        default=False,
        description="Enable code block language detection and parsing"
    )
    do_formula_enrichment: bool = Field(
        default=False,
        description="Enable formula analysis and LaTeX extraction"
    )
    do_picture_classification: bool = Field(
        default=False,
        description="Enable image classification (charts, diagrams, logos, etc.)"
    )
    do_picture_description: bool = Field(
        default=False,
        description="Enable AI-powered image description generation (requires VLM)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "enable_ocr": True,
                "ocr_languages": ["en"],
                "table_mode": "fast",
                "do_table_structure": True,
                "do_cell_matching": True,
                "num_threads": 16,
                "accelerator_device": "cpu",
                "do_code_enrichment": False,
                "do_formula_enrichment": False,
                "do_picture_classification": False,
                "do_picture_description": False
            }
        }


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str
    content: Optional[str] = None