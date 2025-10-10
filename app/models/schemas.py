from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
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
    # ThreadedPdfPipeline batching parameters
    layout_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Batch size for layout detection (1-32, higher = more throughput, more memory)"
    )
    ocr_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Batch size for OCR processing (1-32, higher = more throughput, more memory)"
    )
    table_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Batch size for table extraction (1-32, higher = more throughput, more memory)"
    )
    queue_max_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum queue size for backpressure control (10-1000)"
    )
    batch_timeout_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Timeout for batch processing in seconds (0.1-30.0)"
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
                "do_picture_description": False,
                "layout_batch_size": 4,
                "ocr_batch_size": 4,
                "table_batch_size": 4,
                "queue_max_size": 100,
                "batch_timeout_seconds": 2.0
            }
        }


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    content_type: str = Field(
        description="Type of content in the chunk: 'text', 'table', 'list', or 'heading'"
    )
    heading_path: Optional[str] = Field(
        default=None,
        description="Hierarchical path of headings (e.g., 'Chapter 1 > Section 1.1')"
    )
    pages: Optional[List[int]] = Field(
        default=None,
        description="List of page numbers where this chunk appears"
    )
    captions: Optional[List[str]] = Field(
        default=None,
        description="Captions for tables or figures in this chunk"
    )
    has_table_structure: Optional[bool] = Field(
        default=None,
        description="Whether this chunk contains table structure"
    )
    doc_items_count: Optional[int] = Field(
        default=None,
        description="Number of document items in this chunk"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_type": "text",
                "heading_path": "Chapter 1 > Introduction",
                "pages": [1, 2],
                "captions": ["Figure 1: Overview diagram"],
                "doc_items_count": 5
            }
        }


class ChunkData(BaseModel):
    """A single document chunk with text and metadata."""
    section_title: Optional[str] = Field(
        default=None,
        description="The section title (most specific heading) for this chunk"
    )
    text: str = Field(
        description="The chunk text with search prefix and contextualization"
    )
    chunk_index: int = Field(
        description="Index of this chunk in the document (0-based)"
    )
    metadata: ChunkMetadata = Field(
        description="Rich metadata about the chunk content and location"
    )
    full_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complete Docling metadata from model_dump() (included when include_full_metadata=True)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "section_title": "Introduction",
                "text": "search_document: Introduction\n\nThis document presents...",
                "chunk_index": 0,
                "metadata": {
                    "content_type": "text",
                    "heading_path": "Chapter 1 > Introduction",
                    "pages": [1],
                    "doc_items_count": 3
                }
            }
        }


class ChunkingData(BaseModel):
    """Data container for chunking results."""
    chunks: List[ChunkData] = Field(
        description="List of document chunks with metadata"
    )


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str
    data: Optional[ChunkingData] = Field(
        default=None,
        description="Chunking data (only when chunking is enabled)"
    )
    content: Optional[str] = Field(
        default=None,
        description="Parsed content in markdown format (only when chunking disabled)"
    )