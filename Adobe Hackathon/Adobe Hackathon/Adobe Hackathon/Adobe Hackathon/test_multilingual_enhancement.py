#!/usr/bin/env python3
"""
Test Multilingual Enhancement for Adobe Hackathon
Comprehensive testing of multilingual capabilities
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict

# Import enhanced pipeline
from src.enhanced_multilingual_pipeline import EnhancedMultilingualPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualEnhancementTester:
    """Test multilingual enhancement capabilities"""
    
    def __init__(self):
        self.pipeline = EnhancedMultilingualPipeline()
        self.test_results = {}
    
    def test_language_detection(self):
        """Test language detection capabilities"""
        print("\n🔍 TESTING LANGUAGE DETECTION")
        print("=" * 50)
        
        test_cases = [
            ("第1章 はじめに", "ja"),
            ("第一章 引言", "zh"),
            ("제1장 서론", "ko"),
            ("الفصل الأول مقدمة", "ar"),
            ("अध्याय 1 परिचय", "hi"),
            ("Capítulo 1 Introducción", "es"),
            ("Chapitre 1 Introduction", "fr"),
            ("Kapitel 1 Einführung", "de"),
            ("Chapter 1 Introduction", "en"),
            ("Capitolo 1 Introduzione", "it"),
            ("Capítulo 1 Introdução", "pt"),
            ("Hoofdstuk 1 Inleiding", "nl"),
            ("Kapitel 1 Inledning", "sv"),
            ("Kapitel 1 Innledning", "no"),
            ("Kapitel 1 Indledning", "da"),
            ("Rozdział 1 Wprowadzenie", "pl"),
            ("Kapitola 1 Úvod", "cs"),
            ("Kapitola 1 Úvod", "sk"),
            ("Fejezet 1 Bevezetés", "hu"),
            ("Capitol 1 Introducere", "ro"),
            ("Глава 1 Въведение", "bg"),
            ("Poglavlje 1 Uvod", "hr"),
            ("Poglavje 1 Uvod", "sl"),
            ("บทที่ 1 บทนำ", "th"),
            ("Κεφάλαιο 1 Εισαγωγή", "el"),
            ("פרק 1 מבוא", "he")
        ]
        
        correct_detections = 0
        results = {}
        
        for text, expected_lang in test_cases:
            detected_lang = self.pipeline.language_detector.detect_language(text)
            is_correct = detected_lang == expected_lang
            status = "✅" if is_correct else "❌"
            
            print(f"{status} '{text}' -> Expected: {expected_lang}, Detected: {detected_lang}")
            
            if is_correct:
                correct_detections += 1
            
            results[text] = {
                'expected': expected_lang,
                'detected': detected_lang,
                'correct': is_correct
            }
        
        accuracy = (correct_detections / len(test_cases)) * 100
        print(f"\n📊 Language Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{len(test_cases)})")
        
        self.test_results['language_detection'] = {
            'accuracy': accuracy,
            'results': results
        }
        
        return accuracy
    
    def test_heading_patterns(self):
        """Test multilingual heading pattern recognition"""
        print("\n📋 TESTING MULTILINGUAL HEADING PATTERNS")
        print("=" * 50)
        
        test_cases = [
            # English
            ("Chapter 1: Introduction", "en", "Title"),
            ("1.1 Background", "en", "H1"),
            ("1.1.1 Research Methods", "en", "H2"),
            ("Appendix A: Data", "en", "H1"),
            
            # Japanese
            ("第1章 はじめに", "ja", "Title"),
            ("1.1 背景", "ja", "H1"),
            ("1.1.1 研究方法", "ja", "H2"),
            ("付録A データ", "ja", "H1"),
            
            # Chinese
            ("第一章 引言", "zh", "Title"),
            ("1.1 背景", "zh", "H1"),
            ("1.1.1 研究方法", "zh", "H2"),
            ("附录A 数据", "zh", "H1"),
            
            # Korean
            ("제1장 서론", "ko", "Title"),
            ("1.1 배경", "ko", "H1"),
            ("1.1.1 연구 방법", "ko", "H2"),
            ("부록A 데이터", "ko", "H1"),
            
            # Arabic
            ("الفصل الأول مقدمة", "ar", "Title"),
            ("1.1 خلفية", "ar", "H1"),
            ("1.1.1 طرق البحث", "ar", "H2"),
            ("ملحق أ البيانات", "ar", "H1"),
            
            # Hindi
            ("अध्याय 1 परिचय", "hi", "Title"),
            ("1.1 पृष्ठभूमि", "hi", "H1"),
            ("1.1.1 शोध विधियां", "hi", "H2"),
            ("परिशिष्ट ए डेटा", "hi", "H1"),
            
            # Spanish
            ("Capítulo 1: Introducción", "es", "Title"),
            ("1.1 Antecedentes", "es", "H1"),
            ("1.1.1 Métodos de Investigación", "es", "H2"),
            ("Apéndice A: Datos", "es", "H1"),
            
            # French
            ("Chapitre 1: Introduction", "fr", "Title"),
            ("1.1 Contexte", "fr", "H1"),
            ("1.1.1 Méthodes de Recherche", "fr", "H2"),
            ("Annexe A: Données", "fr", "H1"),
            
            # German
            ("Kapitel 1: Einführung", "de", "Title"),
            ("1.1 Hintergrund", "de", "H1"),
            ("1.1.1 Forschungsmethoden", "de", "H2"),
            ("Anhang A: Daten", "de", "H1")
        ]
        
        correct_patterns = 0
        results = {}
        
        for text, language, expected_level in test_cases:
            # Test pattern recognition
            is_heading = self.pipeline.heading_patterns.is_heading(text, language)
            detected_level = self.pipeline.heading_patterns.extract_heading_level(text, language)
            
            is_correct = detected_level == expected_level
            status = "✅" if is_correct else "❌"
            
            print(f"{status} [{language.upper()}] '{text}' -> Expected: {expected_level}, Detected: {detected_level}")
            
            if is_correct:
                correct_patterns += 1
            
            results[f"{language}_{text}"] = {
                'language': language,
                'text': text,
                'expected_level': expected_level,
                'detected_level': detected_level,
                'is_heading': is_heading,
                'correct': is_correct
            }
        
        accuracy = (correct_patterns / len(test_cases)) * 100
        print(f"\n📊 Heading Pattern Accuracy: {accuracy:.1f}% ({correct_patterns}/{len(test_cases)})")
        
        self.test_results['heading_patterns'] = {
            'accuracy': accuracy,
            'results': results
        }
        
        return accuracy
    
    def test_text_normalization(self):
        """Test multilingual text normalization"""
        print("\n📝 TESTING TEXT NORMALIZATION")
        print("=" * 50)
        
        test_cases = [
            ("第１章　はじめに", "ja", "第1章 はじめに"),
            ("第一章　引言", "zh", "第一章 引言"),
            ("제１장　서론", "ko", "제1장 서론"),
            ("الفصل　الأول　مقدمة", "ar", "الفصل الأول مقدمة"),
            ("अध्याय　१　परिचय", "hi", "अध्याय 1 परिचय"),
            ("Capítulo　1:　Introducción", "es", "Capítulo 1: Introducción"),
            ("Chapitre　1:　Introduction", "fr", "Chapitre 1: Introduction"),
            ("Kapitel　1:　Einführung", "de", "Kapitel 1: Einführung")
        ]
        
        correct_normalizations = 0
        results = {}
        
        for original, language, expected in test_cases:
            normalized = self.pipeline.text_normalizer.normalize(original, language)
            is_correct = normalized == expected
            status = "✅" if is_correct else "❌"
            
            print(f"{status} [{language.upper()}] '{original}' -> '{normalized}'")
            
            if is_correct:
                correct_normalizations += 1
            
            results[f"{language}_{original}"] = {
                'language': language,
                'original': original,
                'expected': expected,
                'normalized': normalized,
                'correct': is_correct
            }
        
        accuracy = (correct_normalizations / len(test_cases)) * 100
        print(f"\n📊 Text Normalization Accuracy: {accuracy:.1f}% ({correct_normalizations}/{len(test_cases)})")
        
        self.test_results['text_normalization'] = {
            'accuracy': accuracy,
            'results': results
        }
        
        return accuracy
    
    def test_feature_extraction(self):
        """Test multilingual feature extraction"""
        print("\n🔧 TESTING FEATURE EXTRACTION")
        print("=" * 50)
        
        test_cases = [
            ("第1章 はじめに", "ja", 16.0, "Arial", True, False, 1),
            ("第一章 引言", "zh", 16.0, "SimSun", True, False, 1),
            ("제1장 서론", "ko", 16.0, "Malgun Gothic", True, False, 1),
            ("الفصل الأول مقدمة", "ar", 16.0, "Arial", True, False, 1),
            ("अध्याय 1 परिचय", "hi", 16.0, "Arial", True, False, 1),
            ("Capítulo 1: Introducción", "es", 16.0, "Arial", True, False, 1),
            ("Chapitre 1: Introduction", "fr", 16.0, "Arial", True, False, 1),
            ("Kapitel 1: Einführung", "de", 16.0, "Arial", True, False, 1)
        ]
        
        successful_extractions = 0
        results = {}
        
        for text, language, font_size, font_name, is_bold, is_italic, page_num in test_cases:
            try:
                features = self.pipeline.feature_extractor.extract_features(
                    text, font_size, font_name, is_bold, is_italic, page_num
                )
                
                # Check if language was detected correctly
                detected_lang = features.get('language', 'unknown')
                is_correct = detected_lang == language
                status = "✅" if is_correct else "❌"
                
                print(f"{status} [{language.upper()}] '{text}' -> Language: {detected_lang}")
                
                if is_correct:
                    successful_extractions += 1
                
                results[f"{language}_{text}"] = {
                    'language': language,
                    'text': text,
                    'detected_language': detected_lang,
                    'features': features,
                    'correct': is_correct
                }
                
            except Exception as e:
                print(f"❌ Error extracting features for '{text}': {e}")
                results[f"{language}_{text}"] = {
                    'language': language,
                    'text': text,
                    'error': str(e),
                    'correct': False
                }
        
        accuracy = (successful_extractions / len(test_cases)) * 100
        print(f"\n📊 Feature Extraction Success Rate: {accuracy:.1f}% ({successful_extractions}/{len(test_cases)})")
        
        self.test_results['feature_extraction'] = {
            'accuracy': accuracy,
            'results': results
        }
        
        return accuracy
    
    def test_multilingual_support_info(self):
        """Test multilingual support information"""
        print("\n🌍 TESTING MULTILINGUAL SUPPORT INFO")
        print("=" * 50)
        
        support_info = self.pipeline.get_multilingual_support_info()
        
        print(f"📊 Supported Languages: {support_info['supported_languages']}")
        print(f"🎯 Overall Accuracy: {support_info['overall_accuracy']:.1%}")
        print(f"📈 Multilingual Accuracy Range: {support_info['multilingual_accuracy_range']['min']:.1%} - {support_info['multilingual_accuracy_range']['max']:.1%}")
        print(f"📊 Average Multilingual Accuracy: {support_info['multilingual_accuracy_range']['average']:.1%}")
        
        print("\n📋 Language Details:")
        for lang, info in support_info['language_details'].items():
            print(f"  {lang.upper()}: {info['accuracy']:.1%} accuracy ({info['support_level']} support)")
        
        self.test_results['multilingual_support'] = support_info
        
        return support_info
    
    def test_with_sample_pdfs(self):
        """Test with existing Adobe sample PDFs"""
        print("\n📄 TESTING WITH SAMPLE PDFS")
        print("=" * 50)
        
        # Test with existing Adobe PDFs
        sample_pdfs = [
            "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file01.pdf",
            "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file02.pdf",
            "test_pdfs/Adobe-India-Hackathon25-main/Challenge_1a/sample_dataset/pdfs/file03.pdf"
        ]
        
        results = {}
        
        for pdf_path in sample_pdfs:
            if os.path.exists(pdf_path):
                print(f"\n📄 Testing: {os.path.basename(pdf_path)}")
                
                try:
                    # Test Round 1A
                    start_time = time.time()
                    result = self.pipeline.generate_multilingual_round1a_output(pdf_path)
                    processing_time = time.time() - start_time
                    
                    print(f"  ✅ Processing time: {processing_time:.3f}s")
                    print(f"  🌍 Language detected: {result.get('language_detected', 'unknown')}")
                    print(f"  📋 Headings found: {len(result.get('headings', []))}")
                    print(f"  📊 Estimated accuracy: {result.get('multilingual_features', {}).get('estimated_accuracy', 0):.1%}")
                    
                    results[pdf_path] = {
                        'success': True,
                        'processing_time': processing_time,
                        'language_detected': result.get('language_detected', 'unknown'),
                        'headings_count': len(result.get('headings', [])),
                        'estimated_accuracy': result.get('multilingual_features', {}).get('estimated_accuracy', 0)
                    }
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    results[pdf_path] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                print(f"  ❌ PDF not found: {pdf_path}")
        
        self.test_results['sample_pdf_testing'] = results
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive multilingual testing"""
        print("🌍 MULTILINGUAL ENHANCEMENT COMPREHENSIVE TESTING")
        print("=" * 60)
        
        # Run all tests
        language_detection_accuracy = self.test_language_detection()
        heading_pattern_accuracy = self.test_heading_patterns()
        text_normalization_accuracy = self.test_text_normalization()
        feature_extraction_accuracy = self.test_feature_extraction()
        support_info = self.test_multilingual_support_info()
        sample_pdf_results = self.test_with_sample_pdfs()
        
        # Calculate overall accuracy
        test_accuracies = [
            language_detection_accuracy,
            heading_pattern_accuracy,
            text_normalization_accuracy,
            feature_extraction_accuracy
        ]
        
        overall_accuracy = sum(test_accuracies) / len(test_accuracies)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏆 MULTILINGUAL ENHANCEMENT TEST SUMMARY")
        print("=" * 60)
        
        print(f"🔍 Language Detection: {language_detection_accuracy:.1f}%")
        print(f"📋 Heading Patterns: {heading_pattern_accuracy:.1f}%")
        print(f"📝 Text Normalization: {text_normalization_accuracy:.1f}%")
        print(f"🔧 Feature Extraction: {feature_extraction_accuracy:.1f}%")
        print(f"📊 Overall Test Accuracy: {overall_accuracy:.1f}%")
        print(f"🌍 Supported Languages: {support_info['supported_languages']}")
        print(f"🎯 Maintained English Accuracy: {support_info['overall_accuracy']:.1%}")
        
        # Save results
        with open('multilingual_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test results saved to: multilingual_test_results.json")
        
        return self.test_results

def main():
    """Main testing function"""
    tester = MultilingualEnhancementTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("✅ MULTILINGUAL ENHANCEMENT TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 