"""
Test script to inspect and extract tables from DoclingDocument.

This demonstrates the new approach that extracts tables directly from
document structure instead of parsing text.

Usage:
    python test_document_tables.py <pdf_file>
"""

import sys
from app.services.docling_parser import DoclingParser
from app.services.document_inspector import inspect_document_structure
from app.services.table_serializer import extract_tables_from_document


def test_table_extraction(pdf_path: str):
    """
    Test table extraction from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
    """
    print("=" * 80)
    print("TABLE EXTRACTION TEST")
    print("=" * 80)
    
    # Parse PDF
    print(f"\n📄 Parsing PDF: {pdf_path}")
    parser = DoclingParser()
    
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    print(f"   File size: {len(pdf_content):,} bytes")
    
    # Get DoclingDocument
    print("\n⏳ Converting to DoclingDocument...")
    document = parser.parse_pdf_to_document(pdf_content)
    print("✅ Document created")
    
    # Inspect document structure
    print("\n" + "=" * 80)
    print("DOCUMENT STRUCTURE INSPECTION")
    print("=" * 80)
    inspect_document_structure(document)
    
    # Extract tables
    print("\n" + "=" * 80)
    print("EXTRACTING TABLES")
    print("=" * 80)
    
    tables = extract_tables_from_document(document)
    
    print(f"\n✅ Extracted {len(tables)} tables\n")
    
    # Display each table
    for i, table_info in enumerate(tables, 1):
        print("-" * 80)
        print(f"📊 Table {i}")
        print("-" * 80)
        
        if table_info.get('caption'):
            print(f"Caption: {table_info['caption']}")
        
        print(f"Headers: {table_info.get('headers')}")
        print(f"Rows: {table_info.get('num_rows')}")
        print(f"Level: {table_info.get('level')}")
        
        print(f"\n🔄 Serialized for embedding:")
        print("-" * 40)
        print(table_info['serialized_text'])
        print("-" * 40)
        print()
    
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    
    return tables


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_document_tables.py <pdf_file>")
        print("\nThis script will:")
        print("  1. Parse the PDF to DoclingDocument")
        print("  2. Inspect the document structure")
        print("  3. Extract tables directly from document")
        print("  4. Show serialized table output")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_table_extraction(pdf_path)

