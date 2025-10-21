"""
Quick test to demonstrate the text-based table parser.
This shows how the parser handles Docling's prose-style table exports.
"""

from app.services.table_serializer import _parse_docling_text_table, format_table_as_keyvalue

# Example text from your output
example_text = """
Bidders confirm receipt with intend to Bid, Due Date = Immediately. 
Bidders confirm receipt with intend to Bid, Location/Instructions = • Attachment 5 - Acknowledgement and Intent to Bid Send the filled, signed& stamped by Email. 
Final Bidder questions, Due Date = 31 st March, 23. 
Final Bidder questions, Location/Instructions = • Bidder questions are to be submitted using Attachment 3 - Bidder Question and Answer Sheet.
Proposal Submission, Due Date = 6 th April 23. 
Proposal Submission, Location/Instructions = All required sections must reach the ET contact thru portal by 16:00 P.M.
"""

print("=" * 80)
print("TESTING TEXT-BASED TABLE PARSER")
print("=" * 80)

print("\n📄 Input text (first 200 chars):")
print(example_text[:200] + "...")

print("\n🔍 Parsing...")
parsed = _parse_docling_text_table(example_text)

if parsed:
    print("\n✅ Successfully parsed!")
    print(f"\n📊 Headers: {parsed['headers']}")
    print(f"📊 Rows: {len(parsed['rows'])}")
    
    print("\n🔄 Formatted as key-value pairs:")
    print("-" * 80)
    formatted = format_table_as_keyvalue(parsed)
    print(formatted)
    print("-" * 80)
    
    print("\n✨ This is the format that will be used for embeddings!")
else:
    print("\n❌ Parser failed - might need adjustment for this specific format")
    print("Check the debug logs when running with serialize_tables=True")

print("\n" + "=" * 80)

