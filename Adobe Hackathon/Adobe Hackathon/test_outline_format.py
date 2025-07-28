#!/usr/bin/env python3
import json
import sys
import os

# Add current directory to path
sys.path.append('.')

from src.adobe_optimized_pipeline import AdobeOptimizedPipeline

def main():
    print("Testing file03.pdf with correct outline format...")
    
    pipeline = AdobeOptimizedPipeline()
    pdf_path = "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file03.pdf"
    
    if os.path.exists(pdf_path):
        result = pipeline.generate_round1a_output(pdf_path)
        
        # Save to new file
        with open("corrected_file03_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Title: {result.get('title', 'Not found')}")
        print(f"Outline count: {len(result.get('outline', []))}")
        print("Output saved to corrected_file03_output.json")
        
        # Show JSON structure
        print("\nJSON Structure:")
        print(f"  - title: {type(result.get('title'))}")
        print(f"  - outline: {type(result.get('outline'))} (array with {len(result.get('outline', []))} items)")
        
        # Show first 3 outline items
        if result.get('outline'):
            print("\nFirst 3 outline items:")
            for i, item in enumerate(result.get('outline', [])[:3], 1):
                print(f"  {i}. [{item['level']}] {item['text']} (Page {item['page']})")
    else:
        print(f"PDF not found: {pdf_path}")

if __name__ == "__main__":
    main() 