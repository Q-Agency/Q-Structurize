"""
Pipeline configuration options for Docling StandardPdfPipeline.

This module contains all available configuration options, examples,
and usage documentation for the PDF parsing pipeline.
"""

PIPELINE_OPTIONS_CONFIG = {
    "description": "Available pipeline configuration options for Docling StandardPdfPipeline",
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
        }
    },
    "usage": {
        "note": "Pass options as JSON string in the 'pipeline_options' form field",
        "example_curl": '''curl -X POST "http://localhost:8000/parse/file" \\
  -F "file=@document.pdf" \\
  -F 'pipeline_options={"enable_ocr": true, "num_threads": 16}'
''',
        "example_python": '''import requests

response = requests.post(
    "http://localhost:8000/parse/file",
    files={"file": open("document.pdf", "rb")},
    data={
        "pipeline_options": '{"enable_ocr": true, "num_threads": 16, "table_mode": "accurate"}'
    }
)
'''
    }
}

