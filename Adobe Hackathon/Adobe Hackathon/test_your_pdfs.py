#!/usr/bin/env python3
"""
Simple script to test your own PDFs for Adobe Hackathon
"""

import os
import json
import time
from pathlib import Path
from src.adobe_optimized_pipeline import AdobeOptimizedPipeline

def test_single_pdf_round1a(pdf_path):
    """Test a single PDF for Round 1A"""
    print(f"\n🎯 ROUND 1A: {os.path.basename(pdf_path)}")
    print("=" * 50)
    
    try:
        pipeline = AdobeOptimizedPipeline()
        start_time = time.time()
        result = pipeline.generate_round1a_output(pdf_path)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing time: {processing_time:.3f}s")
        print(f"📄 Title: {result.get('title', 'Not found')}")
        print(f"📋 Headings found: {len(result.get('headings', []))}")
        
        if result.get('headings'):
            print("\n📋 EXTRACTED HEADINGS:")
            for i, heading in enumerate(result['headings'], 1):
                print(f"  {i:2d}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
        
        # Save output
        output_file = f"my_round1a_{Path(pdf_path).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_multiple_pdfs_round1b(pdf_paths, persona, job):
    """Test multiple PDFs for Round 1B"""
    print(f"\n🎯 ROUND 1B: {len(pdf_paths)} documents")
    print("=" * 50)
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")
    
    try:
        pipeline = AdobeOptimizedPipeline()
        start_time = time.time()
        result = pipeline.generate_round1b_output(persona, job, pdf_paths)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing time: {processing_time:.3f}s")
        
        if result.get('ranked_sections'):
            print(f"\n📊 TOP RANKED SECTIONS ({len(result['ranked_sections'])} found):")
            for i, section in enumerate(result['ranked_sections'][:10], 1):
                print(f"  {i:2d}. [{section['relevance_score']:.2f}] {section['document_title']}")
                print(f"      [{section['section']['level']}] {section['section']['text']}")
                print()
        
        # Save output
        output_file = "my_round1b_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    print("🎯 ADOBE HACKATHON - TEST YOUR OWN PDFS")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Test Round 1A (Single PDF)")
        print("2. Test Round 1B (Multiple PDFs)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            pdf_path = input("Enter PDF path: ").strip()
            if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
                test_single_pdf_round1a(pdf_path)
            else:
                print("❌ Invalid PDF path")
        
        elif choice == "2":
            print("Enter PDF paths (separated by commas):")
            pdf_input = input("PDF paths: ").strip()
            pdf_paths = [p.strip() for p in pdf_input.split(',')]
            
            valid_paths = []
            for path in pdf_paths:
                if os.path.exists(path) and path.lower().endswith('.pdf'):
                    valid_paths.append(path)
                else:
                    print(f"❌ Invalid path: {path}")
            
            if valid_paths:
                persona = input("Enter persona description: ").strip()
                job = input("Enter job description: ").strip()
                
                if persona and job:
                    test_multiple_pdfs_round1b(valid_paths, persona, job)
                else:
                    print("❌ Persona and job description required")
            else:
                print("❌ No valid PDF files found")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 