#!/usr/bin/env python3
"""
Adobe Hackathon - Document Intelligence System
Main processing script for Round 1A and Round 1B
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from src.adobe_optimized_pipeline import AdobeOptimizedPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_round1a():
    """Process Round 1A - Extract headings from PDFs"""
    logger.info("Starting Round 1A processing...")
    
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = AdobeOptimizedPipeline()
    
    # Process all PDFs in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            start_time = time.time()
            
            # Generate Round 1A output
            result = pipeline.generate_round1a_output(str(pdf_file))
            
            # Create output filename
            output_filename = pdf_file.stem + ".json"
            output_path = output_dir / output_filename
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            logger.info(f"Completed {pdf_file.name} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            # Create empty result on error
            result = {"title": "Error", "headings": []}
            output_filename = pdf_file.stem + ".json"
            output_path = output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info("Round 1A processing completed!")

def process_round1b():
    """Process Round 1B - Persona-driven document intelligence"""
    logger.info("Starting Round 1B processing...")
    
    # This would be implemented based on Round 1B requirements
    # For now, we focus on Round 1A
    logger.info("Round 1B processing not implemented in this version")

def main():
    """Main processing function"""
    logger.info("Adobe Hackathon Document Intelligence System")
    logger.info("Processing PDFs for Round 1A...")
    
    # Check if we're in Docker environment
    if os.path.exists("/app/input"):
        process_round1a()
    else:
        logger.info("Not in Docker environment, skipping processing")
        logger.info("This script is designed to run in Docker container")

if __name__ == "__main__":
    main()
