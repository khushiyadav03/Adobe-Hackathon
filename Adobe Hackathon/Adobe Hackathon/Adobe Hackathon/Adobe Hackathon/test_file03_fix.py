#!/usr/bin/env python3
"""
Test script to verify file03.pdf fix
"""

import json
from src.adobe_optimized_pipeline import AdobeOptimizedPipeline

def test_file03_fix():
    """Test the file03.pdf fix"""
    print("🔧 TESTING FILE03.PDF FIX")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AdobeOptimizedPipeline()
    
    # Test file03.pdf
    pdf_path = "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file03.pdf"
    
    try:
        # Generate output
        result = pipeline.generate_round1a_output(pdf_path)
        
        print(f"✅ Processing completed")
        print(f"📄 Title: {result.get('title', 'Not found')}")
        print(f"📋 Headings found: {len(result.get('headings', []))}")
        
        # Save output
        output_file = "test_file03_fixed_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Output saved to: {output_file}")
        
        # Show first few headings
        if result.get('headings'):
            print("\n📋 FIRST 10 HEADINGS:")
            for i, heading in enumerate(result['headings'][:10], 1):
                print(f"  {i:2d}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_file03_fix() 