"""
Pipeline configuration options for Docling StandardPdfPipeline with ThreadedPdfPipelineOptions.

This module contains all available configuration options, examples,
and usage documentation for the PDF parsing pipeline with batching support.
"""

PIPELINE_OPTIONS_CONFIG = {
    "description": "Available pipeline configuration options for Docling StandardPdfPipeline with ThreadedPdfPipelineOptions (batched processing)",
    "options": {
        "enable_ocr": {
            "type": "boolean",
            "default": False,
            "description": "Enable OCR for scanned documents and images",
            "notes": "Increases processing time but essential for scanned PDFs"
        },
        "ocr_languages": {
            "type": "array[string]",
            "default": ["en"],
            "description": "OCR language codes (ISO 639-1)",
            "examples": [["en"], ["es"], ["en", "de"], ["fr", "it"]],
            "notes": "Multiple languages can be specified. Common codes: en, es, de, fr, it, pt, ru, zh, ja, ko"
        },
        "table_mode": {
            "type": "string",
            "default": "fast",
            "valid_values": ["fast", "accurate"],
            "description": "Table extraction mode",
            "notes": "'fast' is recommended for most documents. 'accurate' is slower but better for complex tables"
        },
        "do_table_structure": {
            "type": "boolean",
            "default": True,
            "description": "Enable table structure extraction",
            "notes": "Set to false to disable table processing entirely"
        },
        "do_cell_matching": {
            "type": "boolean",
            "default": True,
            "description": "Enable cell matching for better table accuracy",
            "notes": "Improves table cell boundary detection"
        },
        "num_threads": {
            "type": "integer",
            "default": 8,
            "min": 1,
            "max": 144,
            "description": "Number of threads for processing",
            "notes": "Optimized for 72-core Xeon 6960P. Higher values = faster processing but more CPU/memory usage",
            "recommendations": {
                "light_load": "8-16 threads",
                "balanced": "16-32 threads",
                "high_performance": "32-64 threads",
                "maximum": "64-144 threads (only if system resources available)"
            }
        },
        "accelerator_device": {
            "type": "string",
            "default": "cpu",
            "valid_values": ["cpu", "cuda", "auto"],
            "description": "Accelerator device selection",
            "notes": "'cpu' for CPU-only, 'cuda' for GPU, 'auto' for automatic detection"
        },
        "do_code_enrichment": {
            "type": "boolean",
            "default": False,
            "description": "Enable code block language detection and parsing",
            "notes": "Detects programming languages in code blocks. Increases processing time.",
            "use_cases": ["Technical documentation", "Programming tutorials", "Code repositories"]
        },
        "do_formula_enrichment": {
            "type": "boolean",
            "default": False,
            "description": "Enable formula analysis and LaTeX extraction",
            "notes": "Extracts mathematical formulas in LaTeX format. Useful for scientific papers. Increases processing time.",
            "use_cases": ["Research papers", "Academic documents", "Mathematical textbooks"]
        },
        "do_picture_classification": {
            "type": "boolean",
            "default": False,
            "description": "Enable image classification",
            "notes": "Classifies images into types: charts, diagrams, logos, signatures, etc. Increases processing time.",
            "use_cases": ["Business reports", "Presentations", "Mixed content documents"],
            "categories": ["Charts", "Flow diagrams", "Logos", "Signatures", "Natural images"]
        },
        "do_picture_description": {
            "type": "boolean",
            "default": False,
            "description": "Enable AI-powered image description generation",
            "notes": "⚠️ Requires Vision-Language Model (VLM). Significantly increases processing time and resource usage.",
            "use_cases": ["Accessibility", "Content understanding", "Visual analysis"],
            "requirements": "VLM model must be configured"
        },
        "layout_batch_size": {
            "type": "integer",
            "default": 4,
            "min": 1,
            "max": 32,
            "description": "Batch size for layout detection processing",
            "notes": "Higher values = more throughput but more memory usage. Processes multiple pages in parallel.",
            "recommendations": {
                "low_memory": "1-4",
                "balanced": "4-8",
                "high_throughput": "8-16",
                "maximum": "16-32 (requires significant memory)"
            }
        },
        "ocr_batch_size": {
            "type": "integer",
            "default": 4,
            "min": 1,
            "max": 32,
            "description": "Batch size for OCR processing",
            "notes": "Higher values = more throughput but more memory usage. Only applies when OCR is enabled.",
            "recommendations": {
                "low_memory": "1-4",
                "balanced": "4-8",
                "high_throughput": "8-16",
                "maximum": "16-32 (requires significant memory)"
            }
        },
        "table_batch_size": {
            "type": "integer",
            "default": 4,
            "min": 1,
            "max": 32,
            "description": "Batch size for table extraction processing",
            "notes": "Higher values = more throughput but more memory usage. Processes multiple tables in parallel.",
            "recommendations": {
                "low_memory": "1-4",
                "balanced": "4-8",
                "high_throughput": "8-16",
                "maximum": "16-32 (requires significant memory)"
            }
        },
        "queue_max_size": {
            "type": "integer",
            "default": 100,
            "min": 10,
            "max": 1000,
            "description": "Maximum queue size for backpressure control",
            "notes": "Prevents memory overflow on large documents by limiting pending operations. Higher = more buffering.",
            "recommendations": {
                "small_documents": "50-100",
                "medium_documents": "100-300",
                "large_documents": "300-500",
                "very_large": "500-1000"
            }
        },
        "batch_timeout_seconds": {
            "type": "float",
            "default": 2.0,
            "min": 0.1,
            "max": 30.0,
            "description": "Timeout for batch processing in seconds",
            "notes": "Time to wait for a batch to fill before processing. Lower = more responsive, higher = better batching efficiency.",
            "recommendations": {
                "low_latency": "0.1-1.0",
                "balanced": "1.0-3.0",
                "high_throughput": "3.0-10.0",
                "maximum_batching": "10.0-30.0"
            }
        }
    },
    "example_configurations": {
        "default": {
            "description": "Default configuration - fast, no OCR",
            "config": {}
        },
        "scanned_document": {
            "description": "For scanned PDFs with English text",
            "config": {
                "enable_ocr": True,
                "ocr_languages": ["en"],
                "num_threads": 16
            }
        },
        "multilingual": {
            "description": "For documents with multiple languages",
            "config": {
                "enable_ocr": True,
                "ocr_languages": ["en", "es", "de"],
                "num_threads": 16
            }
        },
        "high_accuracy_tables": {
            "description": "For documents with complex tables",
            "config": {
                "table_mode": "accurate",
                "do_cell_matching": True,
                "num_threads": 24
            }
        },
        "high_performance": {
            "description": "Maximum performance for 72-core Xeon",
            "config": {
                "num_threads": 64,
                "table_mode": "fast"
            }
        },
        "complete_extraction": {
            "description": "Full extraction with OCR and accurate tables",
            "config": {
                "enable_ocr": True,
                "ocr_languages": ["en"],
                "table_mode": "accurate",
                "do_cell_matching": True,
                "num_threads": 32
            }
        },
        "scientific_paper": {
            "description": "For research papers with formulas and code",
            "config": {
                "enable_ocr": False,
                "table_mode": "accurate",
                "do_formula_enrichment": True,
                "do_code_enrichment": True,
                "do_picture_classification": True,
                "num_threads": 24
            }
        },
        "technical_documentation": {
            "description": "For technical docs with code samples and diagrams",
            "config": {
                "do_code_enrichment": True,
                "do_picture_classification": True,
                "table_mode": "accurate",
                "num_threads": 16
            }
        },
        "business_report": {
            "description": "For business reports with charts and tables",
            "config": {
                "table_mode": "accurate",
                "do_picture_classification": True,
                "num_threads": 16
            }
        },
        "high_throughput_batching": {
            "description": "Maximum throughput with aggressive batching (2x 72-core Xeon)",
            "config": {
                "num_threads": 64,
                "layout_batch_size": 16,
                "table_batch_size": 16,
                "ocr_batch_size": 16,
                "queue_max_size": 500,
                "batch_timeout_seconds": 5.0
            }
        },
        "low_latency": {
            "description": "Low latency processing with minimal batching",
            "config": {
                "num_threads": 16,
                "layout_batch_size": 1,
                "table_batch_size": 1,
                "ocr_batch_size": 1,
                "queue_max_size": 50,
                "batch_timeout_seconds": 0.5
            }
        },
        "balanced_batching": {
            "description": "Balanced configuration for good throughput and latency",
            "config": {
                "num_threads": 32,
                "layout_batch_size": 8,
                "table_batch_size": 8,
                "ocr_batch_size": 8,
                "queue_max_size": 200,
                "batch_timeout_seconds": 2.0
            }
        },
        "large_document_processing": {
            "description": "Optimized for large multi-page documents",
            "config": {
                "num_threads": 64,
                "layout_batch_size": 16,
                "table_batch_size": 12,
                "ocr_batch_size": 12,
                "queue_max_size": 1000,
                "batch_timeout_seconds": 3.0,
                "table_mode": "fast"
            }
        }
    },
    "chunking_options": {
        "enable_chunking": {
            "type": "boolean",
            "default": False,
            "description": "Enable hybrid chunking for RAG and semantic search",
            "notes": "When enabled, returns structured chunks instead of full markdown"
        },
        "max_tokens_per_chunk": {
            "type": "integer",
            "default": 512,
            "min": 128,
            "max": 2048,
            "description": "Maximum tokens per chunk (128-2048)",
            "notes": "Should match your embedding model's token limit"
        },
        "merge_peers": {
            "type": "boolean",
            "default": True,
            "description": "Merge undersized successive chunks with same headings",
            "notes": "Helps avoid very small chunks"
        },
        "embedding_model": {
            "type": "string",
            "default": None,
            "description": "HuggingFace embedding model name for tokenization",
            "examples": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "intfloat/e5-small-v2"
            ],
            "notes": "If not specified, uses HybridChunker's built-in tokenizer. Specify a model to match your embedding pipeline's tokenizer for accurate token counting.",
            "use_cases": [
                "Match tokenizer to your embedding model for accurate chunking",
                "Ensure chunks fit within your model's context window",
                "Use multilingual tokenizers for non-English documents"
            ]
        },
        "include_markdown": {
            "type": "boolean",
            "default": False,
            "description": "Include full markdown content when chunking is enabled",
            "notes": "Adds complete document markdown to response"
        },
        "include_full_metadata": {
            "type": "boolean",
            "default": False,
            "description": "Include complete Docling metadata in addition to curated metadata",
            "notes": "Adds full_metadata field with complete Docling metadata dump"
        }
    },
    "usage": {
        "note": "Pass options as individual form fields",
        "example_curl": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F "enable_ocr=true" \\
  -F "num_threads=16"
''',
        "example_curl_advanced": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F "enable_ocr=true" \\
  -F "ocr_languages=en,es,de" \\
  -F "table_mode=accurate" \\
  -F "do_cell_matching=true" \\
  -F "num_threads=32"
''',
        "example_curl_batching": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F "num_threads=64" \\
  -F "layout_batch_size=16" \\
  -F "table_batch_size=16" \\
  -F "queue_max_size=500" \\
  -F "batch_timeout_seconds=3.0"
''',
        "example_curl_chunking": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F "enable_chunking=true" \\
  -F "max_tokens_per_chunk=512" \\
  -F "merge_peers=true" \\
  -F "embedding_model=sentence-transformers/all-MiniLM-L6-v2"
''',
        "example_curl_chunking_custom": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F "enable_chunking=true" \\
  -F "max_tokens_per_chunk=768" \\
  -F "embedding_model=BAAI/bge-small-en-v1.5" \\
  -F "include_full_metadata=true"
''',
        "example_python": '''import requests

response = requests.post(
    "http://localhost:8000/parse/file",
    files={"file": open("document.pdf", "rb")},
    data={
        "enable_ocr": "true",
        "num_threads": "16",
        "table_mode": "accurate"
    }
)
''',
        "example_python_chunking": '''import requests

# With custom embedding model tokenizer
response = requests.post(
    "http://localhost:8000/parse/file",
    files={"file": open("document.pdf", "rb")},
    data={
        "enable_chunking": "true",
        "max_tokens_per_chunk": "512",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "merge_peers": "true"
    }
)

# Using default tokenizer
response = requests.post(
    "http://localhost:8000/parse/file",
    files={"file": open("document.pdf", "rb")},
    data={
        "enable_chunking": "true",
        "max_tokens_per_chunk": "512"
    }
)
'''
    }
}

