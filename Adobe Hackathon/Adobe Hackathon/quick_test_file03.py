#!/usr/bin/env python3
import json
import sys
import os

# Add current directory to path
sys.path.append('.')

from src.adobe_optimized_pipeline import AdobeOptimizedPipeline

def main():
    print("Testing file03.pdf with fixed patterns...")
    
    pipeline = AdobeOptimizedPipeline()
    pdf_path = "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file03.pdf"
    
    if os.path.exists(pdf_path):
        result = pipeline.generate_round1a_output(pdf_path)
        
        # Save to new file
        with open("fixed_file03_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Title: {result.get('title', 'Not found')}")
        print(f"Headings count: {len(result.get('headings', []))}")
        print("Output saved to fixed_file03_output.json")
        
        # Show first 5 headings
        for i, heading in enumerate(result.get('headings', [])[:5], 1):
            print(f"{i}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
    else:
        print(f"PDF not found: {pdf_path}")

if __name__ == "__main__":
    main() 