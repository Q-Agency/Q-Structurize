"""
Services package for Q-Structurize.

Provides PDF parsing, chunking, and table serialization services.
"""

from app.services.table_serializer import (
    extract_tables_from_document,
    serialize_table_item,
    format_table_as_keyvalue,
)

from app.services.document_inspector import (
    inspect_document_structure,
    inspect_table_data,
)

__all__ = [
    'extract_tables_from_document',
    'serialize_table_item',
    'format_table_as_keyvalue',
    'inspect_document_structure',
    'inspect_table_data',
]