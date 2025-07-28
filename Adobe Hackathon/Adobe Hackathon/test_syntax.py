#!/usr/bin/env python3
"""
Test script to check syntax of the pipeline file
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

print("🔍 Testing syntax...")

try:
    # Test syntax
    with open('src/adobe_optimized_pipeline.py', 'r') as f:
        code = f.read()
    
    # Compile to check syntax
    compile(code, 'src/adobe_optimized_pipeline.py', 'exec')
    print("✅ Syntax is correct!")
    
    # Test imports
    print("🔍 Testing imports...")
    from src.adobe_optimized_pipeline import AdobeOptimizedPipeline
    print("✅ Import successful!")
    
    # Test initialization
    print("🔍 Testing initialization...")
    pipeline = AdobeOptimizedPipeline()
    print("✅ Initialization successful!")
    
    print("\n🎉 All tests passed! The file is ready for use.")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}") 