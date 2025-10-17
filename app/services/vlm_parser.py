"""
VLM (Vision Language Model) parser service for end-to-end PDF parsing.
"""

import os
import tempfile
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Try to import docling VLM components
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.settings import settings

    # 🔑 VLM model option types (this is the important part to satisfy Pydantic)
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import (
        InlineVlmOptions,
        InferenceFramework,
        ResponseFormat,
        TransformersModelType,
        # Optional: TransformersPromptStyle,  # if you ever need RAW templates
    )
    from docling.datamodel.accelerator_options import AcceleratorDevice

    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    VlmPipeline = None
    VlmPipelineOptions = None
    InlineVlmOptions = None
    InferenceFramework = None
    ResponseFormat = None
    TransformersModelType = None
    AcceleratorDevice = None
    settings = None
    logger.error(f"Failed to import docling VLM components: {e}")


class VlmParser:
    """
    VLM-powered PDF parser using Docling's VlmPipeline with Granite-Docling.
    """

    def __init__(self):
        """Initialize the VLM parser with explicit Granite-Docling configuration."""
        self.mode = "vlm-pipeline"

        if not VLM_AVAILABLE:
            logger.error("Docling VLM components are not available")
            self.converter = None
            return

        # Threads
        num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))

        # Enable pipeline profiling if available
        if settings:
            settings.debug.profile_pipeline_timings = True
            logger.info("🔍 Docling VLM pipeline profiling enabled")

        logger.info("============================================================")
        logger.info("🚀 Initializing VLM DocumentConverter (ONE-TIME SETUP)")
        logger.info("⚙️  Configuration:")
        logger.info("   🤖 Model: Granite-Docling (explicit)")
        logger.info("   🎯 Backend: Transformers (CUDA)")
        logger.info(f"   🧵 Threads: {num_threads} (OMP_NUM_THREADS)")

        init_start = time.time()

        try:
            # ---- Explicit Granite-Docling config (this fixes the Pydantic error) ----
            model_id = os.getenv("DOCLING_VLM_MODEL", "ibm-granite/granite-docling-258M")
            logger.info(f"🔧 Using explicit VLM model_id = {model_id}")

            # Choose device from env, default to CUDA
            device_env = os.getenv("DOCLING_ACCELERATOR_DEVICE", "cuda").lower()
            device = (
                AcceleratorDevice.CUDA
                if device_env.startswith("cuda")
                else AcceleratorDevice.CPU
            )

            # Hugging Face trust_remote_code (safe here; Docling examples rely on it)
            trust_remote_code = True

            # Granite-Docling outputs **DocTags**; let Docling convert that to Markdown/HTML
            vlm_options = InlineVlmOptions(
                repo_id=model_id,
                prompt="Convert this page to docling.",  # canonical Granite-Docling instruction
                response_format=ResponseFormat.DOCTAGS,  # key: model returns DocTags
                inference_framework=InferenceFramework.TRANSFORMERS,
                transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                temperature=0.0,
                scale=2.0,
                trust_remote_code=trust_remote_code,
            )

            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=False,  # local Transformers, not an API call
            )
            # Set accelerator/device explicitly
            pipeline_options.accelerator_options.device = device

            # If you want FlashAttention2 on supported CUDA cards, uncomment:
            # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

            # (Optional) Respect artifacts path cache if provided
            artifacts_dir = os.getenv("DOCLING_ARTIFACTS_PATH")
            if artifacts_dir:
                pipeline_options.artifacts_path = artifacts_dir

            # Build the converter with our explicit VLM config
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=pipeline_options,
                    ),
                }
            )

            # Pre-initialize (loads Granite-Docling into memory/VRAM)
            logger.info("📦 Pre-loading Granite-Docling VLM model...")
            self.converter.initialize_pipeline(InputFormat.PDF)

            init_time = time.time() - init_start
            logger.info(f"✅ VLM converter initialized in {init_time:.2f} seconds")
            logger.info("📝 VLM model is now cached in memory for fast inference")
            logger.info("============================================================")

        except Exception as e:
            logger.error(f"❌ Failed to initialize VLM converter: {str(e)}", exc_info=True)
            self.converter = None
            raise

    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using VLM (Vision Language Model).

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Dictionary with parsing results
        """
        if not VLM_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "VLM parser is not available",
                "content": None,
            }

        try:
            logger.info("============================================================")
            logger.info("🤖 Starting VLM PDF parsing with pre-initialized converter")
            logger.info("------------------------------------------------------------")

            file_write_start = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                file_write_time = time.time() - file_write_start
                logger.info(f"⏱️  Temp file write: {file_write_time:.3f}s")

                logger.info("⏳ Processing document with VLM...")
                processing_start = time.time()

                result = self.converter.convert(source=tmp_file.name)

                conversion_time = time.time() - processing_start
                logger.info(f"   ✅ VLM conversion complete: {conversion_time:.3f}s")

                # Export to markdown from the DoclingDocument
                logger.info("   └─ Exporting to markdown")
                export_start = time.time()
                document = result.document
                markdown_content = document.export_to_markdown()
                export_time = time.time() - export_start
                logger.info(f"   ✅ Export complete: {export_time:.3f}s")

                total_time = time.time() - processing_start

                logger.info("📊 VLM Performance Breakdown:")
                logger.info(
                    f"   ├─ File I/O:        {file_write_time:.3f}s ({file_write_time/total_time*100:.1f}%)"
                )
                logger.info(
                    f"   ├─ VLM Conversion:  {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)"
                )
                logger.info(
                    f"   ├─ Markdown Export: {export_time:.3f}s ({export_time/total_time*100:.1f}%)"
                )
                logger.info(f"   └─ TOTAL:           {total_time:.3f}s")
                logger.info("============================================================")

                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass

                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None,
                    "processing_time": total_time,
                    "timings": {
                        "file_write": file_write_time,
                        "vlm_conversion": conversion_time,
                        "markdown_export": export_time,
                        "total": total_time,
                    },
                }

        except Exception as e:
            logger.error(f"❌ Error during VLM PDF parsing: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "content": None}

    def is_available(self) -> bool:
        """Check if VLM parsing is available."""
        return VLM_AVAILABLE and self.converter is not None

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the VLM parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if VLM_AVAILABLE else None,
            "pipeline": "VlmPipeline (Granite-Docling / Transformers)",
            "description": "Vision Language Model for end-to-end PDF parsing",
            "performance_mode": "optimized_with_preinitialization",
            "model": os.getenv("DOCLING_VLM_MODEL", "ibm-granite/granite-docling-258M"),
            "backend": "Transformers (CUDA)",
            "features": {
                "vision_based": True,
                "complex_layouts": True,
                "images_and_figures": True,
                "chunking_support": False,
                "optimization_support": False,
            },
            "performance": {
                "initialization": "One-time at startup",
                "per_request": "Fast inference with pre-loaded model",
            },
        }
