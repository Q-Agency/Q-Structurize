"""Simple test script for Docling VLM functionality."""

import requests
import os

def test_docling_parsing():
    """Test the Docling VLM parsing endpoint."""
    
    # API endpoint
    url = "http://localhost:8000/parse/file"
    
    # Test with a sample PDF file
    test_pdf_path = "test.txt"  # Replace with actual PDF file path
    
    if not os.path.exists(test_pdf_path):
        print("Please provide a PDF file to test with")
        print("You can download a sample PDF from: https://arxiv.org/pdf/2501.17887")
        return
    
    # Test parameters
    files = {
        'file': ('test.pdf', open(test_pdf_path, 'rb'), 'application/pdf')
    }
    
    data = {
        'max_tokens_per_chunk': 512,
        'optimize_pdf': True,
        'use_vlm': True
    }
    
    try:
        print("Testing Docling VLM parsing...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Parsing successful!")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            if result.get('content'):
                print(f"Content preview: {result['content'][:200]}...")
        else:
            print(f"❌ Parsing failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    finally:
        if 'files' in locals():
            files['file'][1].close()

if __name__ == "__main__":
    test_docling_parsing()
