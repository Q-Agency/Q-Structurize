"""
Configuration module for Q-Structurize API.

This module contains configuration constants and schema definitions
used throughout the application.
"""

from .pipeline_config import PIPELINE_OPTIONS_CONFIG
from .openapi_schema import get_custom_openapi

__all__ = [
    "PIPELINE_OPTIONS_CONFIG",
    "get_custom_openapi"
]

