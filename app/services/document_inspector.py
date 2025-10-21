"""
Debug utility to inspect DoclingDocument structure.

This module helps understand how Docling stores tables and other elements
in the document structure.
"""

import logging
from typing import Any
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


def inspect_document_structure(document: DoclingDocument) -> None:
    """
    Print detailed information about DoclingDocument structure.
    
    Useful for understanding how to access tables, figures, and other elements.
    
    Args:
        document: DoclingDocument to inspect
    """
    print("=" * 80)
    print("DOCLING DOCUMENT STRUCTURE INSPECTION")
    print("=" * 80)
    
    # Document-level attributes
    print("\n📄 Document Attributes:")
    doc_attrs = [attr for attr in dir(document) if not attr.startswith('_')]
    print(f"   Available: {doc_attrs[:20]}")
    
    # Body structure
    if hasattr(document, 'body'):
        print("\n📦 Document Body:")
        print(f"   Type: {type(document.body)}")
        body_attrs = [attr for attr in dir(document.body) if not attr.startswith('_')]
        print(f"   Attributes: {body_attrs[:15]}")
    
    # Iterate through items
    if hasattr(document, 'body') and hasattr(document.body, 'iterate_items'):
        print("\n🔍 Iterating Document Items:")
        item_count = 0
        table_count = 0
        
        for item, level in document.body.iterate_items():
            item_count += 1
            
            if hasattr(item, 'label'):
                label = item.label
                
                if label == 'table':
                    table_count += 1
                    print(f"\n   📊 Table #{table_count} (Item #{item_count}, Level {level}):")
                    print(f"      Label: {label}")
                    
                    # Table attributes
                    item_attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                    print(f"      Attributes: {item_attrs[:20]}")
                    
                    # Table data
                    if hasattr(item, 'data'):
                        print(f"      Has data: {item.data is not None}")
                        if item.data:
                            print(f"      Data type: {type(item.data)}")
                            data_attrs = [attr for attr in dir(item.data) if not attr.startswith('_')]
                            print(f"      Data attributes: {data_attrs[:15]}")
                            
                            # Grid structure
                            if hasattr(item.data, 'grid'):
                                print(f"      Has grid: {item.data.grid is not None}")
                                if item.data.grid:
                                    grid_attrs = [attr for attr in dir(item.data.grid) if not attr.startswith('_')]
                                    print(f"      Grid attributes: {grid_attrs[:15]}")
                    
                    # Captions
                    if hasattr(item, 'captions'):
                        print(f"      Captions: {item.captions}")
                    
                    # Text representation
                    if hasattr(item, 'text'):
                        text_preview = item.text[:100] if item.text else None
                        print(f"      Text preview: {text_preview}...")
        
        print(f"\n📊 Summary:")
        print(f"   Total items: {item_count}")
        print(f"   Tables found: {table_count}")
    
    print("\n" + "=" * 80)


def inspect_table_data(item: Any) -> None:
    """
    Deep inspection of a single table item.
    
    Args:
        item: Table item from document.body.iterate_items()
    """
    print("\n" + "=" * 80)
    print("DETAILED TABLE INSPECTION")
    print("=" * 80)
    
    if not hasattr(item, 'label') or item.label != 'table':
        print("❌ Not a table item")
        return
    
    print("\n📊 Table Item:")
    print(f"   Label: {item.label}")
    
    # All attributes
    attrs = [attr for attr in dir(item) if not attr.startswith('_')]
    print(f"\n   All attributes ({len(attrs)}):")
    for attr in attrs:
        try:
            value = getattr(item, attr)
            if not callable(value):
                value_str = str(value)[:50] if value else 'None'
                print(f"      {attr}: {value_str}")
        except Exception as e:
            print(f"      {attr}: <error: {e}>")
    
    # Data structure
    if hasattr(item, 'data') and item.data:
        print("\n   📦 Table Data:")
        print(f"      Type: {type(item.data)}")
        
        data_attrs = [attr for attr in dir(item.data) if not attr.startswith('_')]
        print(f"\n      Data attributes ({len(data_attrs)}):")
        for attr in data_attrs:
            try:
                value = getattr(item.data, attr)
                if not callable(value):
                    value_str = str(value)[:50] if value else 'None'
                    print(f"         {attr}: {value_str}")
            except Exception as e:
                print(f"         {attr}: <error: {e}>")
        
        # Grid
        if hasattr(item.data, 'grid') and item.data.grid:
            print("\n      🔲 Grid Structure:")
            print(f"         Type: {type(item.data.grid)}")
            
            grid_attrs = [attr for attr in dir(item.data.grid) if not attr.startswith('_')]
            print(f"\n         Grid attributes ({len(grid_attrs)}):")
            for attr in grid_attrs:
                try:
                    value = getattr(item.data.grid, attr)
                    if not callable(value):
                        value_str = str(value)[:50] if value else 'None'
                        print(f"            {attr}: {value_str}")
                except Exception as e:
                    print(f"            {attr}: <error: {e}>")
    
    print("\n" + "=" * 80)

