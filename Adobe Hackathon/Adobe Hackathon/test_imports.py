#!/usr/bin/env python3
"""
Test script to identify import errors in the pipeline
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

print("🔍 Testing imports...")

try:
    print("Testing modular_heading_extractor...")
    from src.modular_heading_extractor import ModularHeadingExtractor
    print("✅ ModularHeadingExtractor imported successfully")
except Exception as e:
    print(f"❌ Error importing ModularHeadingExtractor: {e}")

try:
    print("Testing modular_persona_analyzer...")
    from src.modular_persona_analyzer import EnhancedModularPersonaAnalyzer
    print("✅ EnhancedModularPersonaAnalyzer imported successfully")
except Exception as e:
    print(f"❌ Error importing EnhancedModularPersonaAnalyzer: {e}")

try:
    print("Testing adobe_optimized_pipeline...")
    from src.adobe_optimized_pipeline import AdobeOptimizedPipeline
    print("✅ AdobeOptimizedPipeline imported successfully")
except Exception as e:
    print(f"❌ Error importing AdobeOptimizedPipeline: {e}")

print("\n🔍 Testing pipeline initialization...")

try:
    pipeline = AdobeOptimizedPipeline()
    print("✅ Pipeline initialized successfully")
except Exception as e:
    print(f"❌ Error initializing pipeline: {e}")

print("\n✅ Import test completed!") 