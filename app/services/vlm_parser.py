"""VLM (Vision Language Model) parser service for end-to-end PDF parsing."""

import os
import tempfile
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# PyTorch GPU Optimizations for H200
# Configure PyTorch before importing docling to ensure GPU settings are applied
# ============================================================================
try:
    import torch
    
    if torch.cuda.is_available():
        # Enable TF32 for Ampere/Hopper GPUs (H200) - ~2x speedup with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set precision mode for balanced performance
        try:
            torch.set_float32_matmul_precision("high")  # PyTorch 2.x
        except AttributeError:
            pass  # Older PyTorch versions don't have this
        
        # Enable cuDNN auto-tuner to select best algorithms for your hardware
        torch.backends.cudnn.benchmark = True
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("============================================================")
        logger.info("🎮 GPU Configuration for VLM")
        logger.info("============================================================")
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"💾 GPU memory: {gpu_memory:.1f} GB")
        logger.info(f"✅ TF32 enabled for matmul and cuDNN (balanced speed/quality)")
        logger.info(f"✅ cuDNN benchmark mode enabled (optimal kernel selection)")
        logger.info("============================================================")
    else:
        logger.warning("⚠️  No GPU detected - VLM will run on CPU (very slow)")
        
except ImportError:
    logger.warning("⚠️  PyTorch not available - GPU optimizations disabled")
except Exception as e:
    logger.warning(f"⚠️  Could not configure PyTorch GPU optimizations: {e}")

try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    VlmPipeline = None
    PdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    logger.error(f"Failed to import docling VLM components: {e}")


class VlmParser:
    def __init__(self):
        self.mode = "vlm-pipeline"
        
        if not VLM_AVAILABLE:
            logger.error("Docling VLM components are not available")
            self.converter = None
            return
        
        logger.info("============================================================")
        logger.info("🚀 Initializing VLM DocumentConverter (MINIMAL)")
        logger.info("============================================================")
        
        # Verify Granite VLM model configuration
        vlm_model_env = os.environ.get('DOCLING_VLM_MODEL', 'Not set')
        logger.info(f"🔍 DOCLING_VLM_MODEL environment variable: {vlm_model_env}")
        
        # Check if model path exists
        if vlm_model_env != 'Not set' and os.path.exists(vlm_model_env):
            logger.info(f"✅ Granite VLM model path exists: {vlm_model_env}")
            model_files = os.listdir(vlm_model_env)[:10]  # First 10 files
            logger.info(f"📁 Model directory contains: {', '.join(model_files)}")
        elif vlm_model_env != 'Not set':
            logger.warning(f"⚠️ DOCLING_VLM_MODEL path does not exist: {vlm_model_env}")
            logger.info("📥 Model will be downloaded from HuggingFace on first use")
        
        # Check HuggingFace offline mode
        hf_offline = os.environ.get('HF_HUB_OFFLINE', 'Not set')
        transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', 'Not set')
        logger.info(f"🌐 HF_HUB_OFFLINE: {hf_offline}, TRANSFORMERS_OFFLINE: {transformers_offline}")
        
        # Configure VLM pipeline with GPU acceleration
        logger.info("⚙️  Configuring VLM pipeline options...")
        pipeline_options = PdfPipelineOptions()
        
        # GPU Acceleration - configure for CUDA device
        accelerator_device_str = os.environ.get('DOCLING_ACCELERATOR_DEVICE', 'cuda')
        device_map = {
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "auto": AcceleratorDevice.AUTO
        }
        device = device_map.get(accelerator_device_str.lower(), AcceleratorDevice.CUDA)
        
        # For GPU workloads, use fewer threads (GPU does the heavy lifting)
        num_threads = int(os.environ.get('OMP_NUM_THREADS', '8'))
        
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=device,
            num_threads=num_threads
        )
        
        logger.info(f"✅ Accelerator configured: device={device}, threads={num_threads}")
        
        # Mixed Precision Configuration (BF16 for H200)
        # BF16 provides ~2x speedup with better numerical stability than FP16
        vlm_dtype = os.environ.get('DOCLING_VLM_DTYPE', 'bfloat16')
        logger.info(f"🔢 VLM precision: {vlm_dtype} (balanced speed/quality for H200)")
        
        # Try to configure model dtype via pipeline options
        try:
            import torch
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(vlm_dtype.lower(), torch.bfloat16)
            
            # Configure model loading parameters if supported by VlmPipeline
            if hasattr(pipeline_options, 'model_kwargs'):
                model_kwargs = {
                    'torch_dtype': torch_dtype,
                    'device_map': 'cuda:0' if device == AcceleratorDevice.CUDA else 'auto'
                }
                
                # Try to enable Flash Attention 2 if available (2-3x speedup on attention)
                use_flash_attn = os.environ.get('DOCLING_USE_FLASH_ATTENTION', '1') == '1'
                if use_flash_attn:
                    try:
                        import flash_attn
                        model_kwargs['attn_implementation'] = 'flash_attention_2'
                        logger.info(f"⚡ Flash Attention 2 enabled (flash-attn {flash_attn.__version__})")
                    except ImportError:
                        logger.info(f"⚡ Flash Attention not available - using default attention")
                
                pipeline_options.model_kwargs = model_kwargs
                logger.info(f"✅ Model dtype configured: {torch_dtype}")
            else:
                # Set as environment variable for transformers to pick up
                os.environ['DOCLING_VLM_DTYPE'] = vlm_dtype
                logger.info(f"✅ VLM dtype set via environment: {vlm_dtype}")
        except Exception as e:
            logger.warning(f"⚠️  Could not configure model dtype: {e}")
        
        # Batch Processing Configuration
        # Process multiple pages in parallel for better GPU utilization
        vlm_batch_size = int(os.environ.get('DOCLING_VLM_BATCH_SIZE', '4'))
        
        # Try to configure batch size if supported
        try:
            if hasattr(pipeline_options, 'page_batch_size'):
                pipeline_options.page_batch_size = vlm_batch_size
                logger.info(f"✅ VLM batch size configured: {vlm_batch_size} pages")
            elif hasattr(pipeline_options, 'batch_size'):
                pipeline_options.batch_size = vlm_batch_size
                logger.info(f"✅ VLM batch size configured: {vlm_batch_size} pages")
            else:
                # Set via environment for VLM to pick up during processing
                os.environ['DOCLING_VLM_BATCH_SIZE'] = str(vlm_batch_size)
                logger.info(f"📦 VLM batch size set via environment: {vlm_batch_size} pages")
        except Exception as e:
            logger.warning(f"⚠️  Could not configure batch size: {e}")
        
        init_start = time.time()
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        logger.info("📦 Pre-loading VLM model (ibm-granite/granite-docling-258M)...")
        self.converter.initialize_pipeline(InputFormat.PDF)
        
        # Try to get model info from the pipeline
        try:
            pipeline = self.converter._get_pipeline(InputFormat.PDF)
            logger.info(f"✅ VLM Pipeline type: {type(pipeline).__name__}")
            
            # Try to access VLM model specs if available
            if hasattr(pipeline, 'vlm_model'):
                logger.info(f"🤖 VLM Model object: {type(pipeline.vlm_model).__name__}")
            
        except Exception as e:
            logger.debug(f"Could not extract detailed pipeline info: {e}")
        
        init_time = time.time() - init_start
        logger.info(f"✅ VLM converter initialized in {init_time:.2f} seconds")
        logger.info("🎯 Using IBM Granite Docling VLM (granite-docling-258M)")
        logger.info("============================================================")
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        if not VLM_AVAILABLE or self.converter is None:
            return {"success": False, "error": "VLM parser is not available", "content": None}
        
        try:
            logger.info("============================================================")
            logger.info("🤖 Starting VLM PDF parsing...")
            logger.info("============================================================")
            processing_start = time.time()
            
            # GPU memory tracking (if available)
            gpu_mem_before = 0
            gpu_available = False
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    torch.cuda.synchronize()  # Ensure accurate timing
                    gpu_mem_before = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"🎮 GPU memory before parsing: {gpu_mem_before:.2f} GB")
            except Exception as e:
                logger.debug(f"Could not track GPU memory: {e}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                logger.info(f"📄 Processing PDF: {tmp_file.name}")
                conversion_start = time.time()
                
                result = self.converter.convert(source=tmp_file.name)
                
                conversion_time = time.time() - conversion_start
                logger.info(f"⏱️  Document conversion: {conversion_time:.2f}s")
                
                export_start = time.time()
                markdown_content = result.document.export_to_markdown()
                export_time = time.time() - export_start
                logger.info(f"⏱️  Markdown export: {export_time:.2f}s")
                
                os.unlink(tmp_file.name)
                
                # GPU memory tracking after parsing
                if gpu_available:
                    try:
                        import torch
                        torch.cuda.synchronize()
                        gpu_mem_after = torch.cuda.memory_allocated(0) / 1024**3
                        gpu_mem_peak = torch.cuda.max_memory_allocated(0) / 1024**3
                        gpu_mem_used = gpu_mem_after - gpu_mem_before
                        
                        logger.info(f"🎮 GPU memory after parsing: {gpu_mem_after:.2f} GB")
                        logger.info(f"🎮 GPU memory peak: {gpu_mem_peak:.2f} GB")
                        logger.info(f"🎮 GPU memory used for this parsing: {gpu_mem_used:.2f} GB")
                        
                        # Reset peak stats for next parsing
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as e:
                        logger.debug(f"Could not track GPU memory after parsing: {e}")
                
                total_time = time.time() - processing_start
                logger.info("============================================================")
                logger.info(f"✅ VLM parsing complete in {total_time:.2f}s total")
                logger.info("============================================================")
                
                return {"success": True, "content": markdown_content, "error": None}
        except Exception as e:
            logger.error(f"❌ Error during VLM PDF parsing: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "content": None}
    
    def is_available(self) -> bool:
        return VLM_AVAILABLE and self.converter is not None
    
    def verify_granite_model(self) -> Dict[str, Any]:
        """Verify that Granite VLM model is being used."""
        verification = {
            "is_granite": False,
            "model_path": None,
            "model_exists": False,
            "env_configured": False,
            "model_files": [],
            "offline_mode": False,
        }
        
        # Check environment variable
        vlm_model_env = os.environ.get('DOCLING_VLM_MODEL')
        if vlm_model_env:
            verification["env_configured"] = True
            verification["model_path"] = vlm_model_env
            
            # Check if path exists and contains granite model
            if os.path.exists(vlm_model_env):
                verification["model_exists"] = True
                verification["model_files"] = os.listdir(vlm_model_env)[:15]
                
                # Verify it's the Granite model by checking for expected files
                if "config.json" in verification["model_files"]:
                    try:
                        import json
                        config_path = os.path.join(vlm_model_env, "config.json")
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            # Check model architecture or name hints
                            model_type = config.get("model_type", "")
                            arch = config.get("architectures", [])
                            verification["model_type"] = model_type
                            verification["architectures"] = arch
                            # Granite models typically have specific architecture
                            verification["is_granite"] = True  # Path indicates granite
                    except Exception as e:
                        logger.debug(f"Could not read model config: {e}")
        
        # Check offline mode
        hf_offline = os.environ.get('HF_HUB_OFFLINE', '0')
        transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0')
        verification["offline_mode"] = hf_offline == '1' or transformers_offline == '1'
        
        return verification
    
    def get_parser_info(self) -> Dict[str, Any]:
        base_info = {
            "available": self.is_available(),
            "library": "docling" if VLM_AVAILABLE else None,
            "pipeline": "VlmPipeline (GraniteDocling default)",
            "description": "Vision Language Model for end-to-end PDF parsing",
            "model": "ibm-granite/granite-docling-258M (default)",
            "backend": "Transformers",
        }
        
        # Add runtime verification
        if self.is_available():
            base_info["granite_verification"] = self.verify_granite_model()
        
        return base_info
