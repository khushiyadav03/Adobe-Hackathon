#!/usr/bin/env python3
"""
Multilingual Enhancement for Adobe Hackathon Document Intelligence System
Enhances PDF heading extraction and persona-driven ranking for 100+ languages
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from transformers import (
    DistilBertTokenizer, DistilBertModel, 
    AutoTokenizer, AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer
import langdetect
from langdetect import detect, DetectorFactory
import unicodedata

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class MultilingualLanguageDetector:
    """Robust language detection for PDF text"""
    
    def __init__(self):
        self.language_patterns = {
            'ja': {
                'hiragana': r'[\u3040-\u309F]',
                'katakana': r'[\u30A0-\u30FF]',
                'kanji': r'[\u4E00-\u9FAF]',
                'keywords': ['第', '章', '節', '項', '目', 'はじめに', '結論']
            },
            'zh': {
                'simplified': r'[\u4E00-\u9FFF]',
                'traditional': r'[\u3400-\u4DBF\u20000-\u2A6DF]',
                'keywords': ['第', '章', '节', '项', '目', '引言', '结论']
            },
            'ko': {
                'hangul': r'[\uAC00-\uD7AF]',
                'jamo': r'[\u1100-\u11FF\u3130-\u318F]',
                'keywords': ['제', '장', '절', '항', '목', '서론', '결론']
            },
            'ar': {
                'arabic': r'[\u0600-\u06FF]',
                'arabic_extended': r'[\u0750-\u077F\u08A0-\u08FF]',
                'keywords': ['الفصل', 'الباب', 'المبحث', 'المطلب', 'المقدمة', 'الخاتمة']
            },
            'hi': {
                'devanagari': r'[\u0900-\u097F]',
                'devanagari_extended': r'[\uA8E0-\uA8FF]',
                'keywords': ['अध्याय', 'खंड', 'अनुभाग', 'बिंदु', 'परिचय', 'निष्कर्ष']
            },
            'th': {
                'thai': r'[\u0E00-\u0E7F]',
                'keywords': ['บทที่', 'ตอน', 'หัวข้อ', 'ข้อ', 'บทนำ', 'สรุป']
            },
            'ru': {
                'cyrillic': r'[\u0400-\u04FF]',
                'keywords': ['Глава', 'Раздел', 'Параграф', 'Пункт', 'Введение', 'Заключение']
            },
            'el': {
                'greek': r'[\u0370-\u03FF]',
                'keywords': ['Κεφάλαιο', 'Ενότητα', 'Παράγραφος', 'Εισαγωγή', 'Συμπέρασμα']
            },
            'he': {
                'hebrew': r'[\u0590-\u05FF]',
                'keywords': ['פרק', 'סעיף', 'סעיף קטן', 'מבוא', 'סיכום']
            }
        }
        
        # European language patterns
        self.european_patterns = {
            'es': {'pattern': r'[áéíóúñüÁÉÍÓÚÑÜ]', 'keywords': ['Capítulo', 'Sección', 'Apartado', 'Introducción', 'Conclusión']},
            'fr': {'pattern': r'[àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]', 'keywords': ['Chapitre', 'Section', 'Paragraphe', 'Introduction', 'Conclusion']},
            'de': {'pattern': r'[äöüßÄÖÜ]', 'keywords': ['Kapitel', 'Abschnitt', 'Paragraph', 'Einleitung', 'Schlussfolgerung']},
            'it': {'pattern': r'[àèéìíîòóùÀÈÉÌÍÎÒÓÙ]', 'keywords': ['Capitolo', 'Sezione', 'Paragrafo', 'Introduzione', 'Conclusione']},
            'pt': {'pattern': r'[áâãàçéêíóôõúÁÂÃÀÇÉÊÍÓÔÕÚ]', 'keywords': ['Capítulo', 'Seção', 'Parágrafo', 'Introdução', 'Conclusão']},
            'nl': {'pattern': r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸ]', 'keywords': ['Hoofdstuk', 'Sectie', 'Paragraaf', 'Inleiding', 'Conclusie']},
            'sv': {'pattern': r'[åäöÅÄÖ]', 'keywords': ['Kapitel', 'Sektion', 'Stycke', 'Inledning', 'Slutsats']},
            'no': {'pattern': r'[æøåÆØÅ]', 'keywords': ['Kapittel', 'Seksjon', 'Avsnitt', 'Innledning', 'Konklusjon']},
            'da': {'pattern': r'[æøåÆØÅ]', 'keywords': ['Kapitel', 'Sektion', 'Afsnit', 'Indledning', 'Konklusion']},
            'pl': {'pattern': r'[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', 'keywords': ['Rozdział', 'Sekcja', 'Paragraf', 'Wprowadzenie', 'Wnioski']},
            'cs': {'pattern': r'[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]', 'keywords': ['Kapitola', 'Sekce', 'Odstavec', 'Úvod', 'Závěr']},
            'sk': {'pattern': r'[áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]', 'keywords': ['Kapitola', 'Sekcia', 'Odsek', 'Úvod', 'Záver']},
            'hu': {'pattern': r'[áéíóöőúüűÁÉÍÓÖŐÚÜŰ]', 'keywords': ['Fejezet', 'Szakasz', 'Bekezdés', 'Bevezetés', 'Összefoglalás']},
            'ro': {'pattern': r'[ăâîșțĂÂÎȘȚ]', 'keywords': ['Capitol', 'Secțiune', 'Paragraf', 'Introducere', 'Concluzie']},
            'bg': {'pattern': r'[а-яА-Я]', 'keywords': ['Глава', 'Раздел', 'Параграф', 'Въведение', 'Заключение']},
            'hr': {'pattern': r'[čćđšžČĆĐŠŽ]', 'keywords': ['Poglavlje', 'Sekcija', 'Paragraf', 'Uvod', 'Zaključak']},
            'sl': {'pattern': r'[čšžČŠŽ]', 'keywords': ['Poglavje', 'Sekcija', 'Odstavek', 'Uvod', 'Zaključek']}
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language using multiple methods"""
        if not text or len(text.strip()) < 10:
            return 'en'
        
        # Method 1: Character-based detection
        char_lang = self._detect_by_characters(text)
        
        # Method 2: Keyword-based detection
        keyword_lang = self._detect_by_keywords(text)
        
        # Method 3: langdetect library
        try:
            detect_lang = detect(text)
        except:
            detect_lang = 'en'
        
        # Combine results with confidence scoring
        scores = {}
        scores[char_lang] = scores.get(char_lang, 0) + 0.4
        scores[keyword_lang] = scores.get(keyword_lang, 0) + 0.3
        scores[detect_lang] = scores.get(detect_lang, 0) + 0.3
        
        # Return language with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _detect_by_characters(self, text: str) -> str:
        """Detect language by character patterns"""
        # Check CJK and other scripts first
        for lang, patterns in self.language_patterns.items():
            for script_name, pattern in patterns.items():
                if script_name != 'keywords' and re.search(pattern, text):
                    return lang
        
        # Check European languages
        for lang, data in self.european_patterns.items():
            if re.search(data['pattern'], text):
                return lang
        
        return 'en'
    
    def _detect_by_keywords(self, text: str) -> str:
        """Detect language by heading keywords"""
        text_lower = text.lower()
        
        # Check CJK and other scripts
        for lang, patterns in self.language_patterns.items():
            if 'keywords' in patterns:
                for keyword in patterns['keywords']:
                    if keyword.lower() in text_lower:
                        return lang
        
        # Check European languages
        for lang, data in self.european_patterns.items():
            for keyword in data['keywords']:
                if keyword.lower() in text_lower:
                    return lang
        
        return 'en'

class MultilingualHeadingPatterns:
    """Language-specific heading patterns and numbering schemes"""
    
    def __init__(self):
        self.patterns = {
            'en': {
                'numeric': r'^(?:\d+\.)+\s*[A-Z][^.!?]*$',
                'roman': r'^[IVX]+\.\s*[A-Z][^.!?]*$',
                'alphabetic': r'^[A-Z]\.\s*[A-Z][^.!?]*$',
                'appendix': r'^Appendix\s+[A-Z0-9]+\.?\s*[A-Z][^.!?]*$',
                'chapter': r'^Chapter\s+\d+\.?\s*[A-Z][^.!?]*$',
                'section': r'^Section\s+\d+\.?\s*[A-Z][^.!?]*$'
            },
            'ja': {
                'numeric': r'^(?:\d+\.)+\s*[^\s]+.*$',
                'kanji': r'^第[一二三四五六七八九十\d]+[章節項].*$',
                'hiragana': r'^[あ-ん]+[^\s]*.*$',
                'katakana': r'^[ア-ン]+[^\s]*.*$',
                'appendix': r'^付録\s*[A-Z0-9]+\.?\s*.*$',
                'chapter': r'^第[一二三四五六七八九十\d]+章\s*.*$'
            },
            'zh': {
                'numeric': r'^(?:\d+\.)+\s*[^\s]+.*$',
                'chinese': r'^第[一二三四五六七八九十\d]+[章节项].*$',
                'appendix': r'^附录\s*[A-Z0-9]+\.?\s*.*$',
                'chapter': r'^第[一二三四五六七八九十\d]+章\s*.*$'
            },
            'ko': {
                'numeric': r'^(?:\d+\.)+\s*[^\s]+.*$',
                'hangul': r'^제[일이삼사오육칠팔구십\d]+[장절항].*$',
                'appendix': r'^부록\s*[A-Z0-9]+\.?\s*.*$',
                'chapter': r'^제[일이삼사오육칠팔구십\d]+장\s*.*$'
            },
            'ar': {
                'numeric': r'^(?:\d+\.)+\s*[^\s]+.*$',
                'arabic': r'^الفصل\s*[الأولالثانيالثالثالرابعالخامسالسادسالسابعالثامنتاسععاشر\d]+.*$',
                'appendix': r'^ملحق\s*[A-Z0-9]+\.?\s*.*$',
                'chapter': r'^الفصل\s*[الأولالثانيالثالثالرابعالخامسالسادسالسابعالثامنتاسععاشر\d]+.*$'
            },
            'hi': {
                'numeric': r'^(?:\d+\.)+\s*[^\s]+.*$',
                'devanagari': r'^अध्याय\s*[एकदोतीनचारपांचछहसातआठनौदस\d]+.*$',
                'appendix': r'^परिशिष्ट\s*[A-Z0-9]+\.?\s*.*$',
                'chapter': r'^अध्याय\s*[एकदोतीनचारपांचछहसातआठनौदस\d]+.*$'
            },
            'es': {
                'numeric': r'^(?:\d+\.)+\s*[A-ZÁÉÍÓÚÑ][^.!?]*$',
                'roman': r'^[IVX]+\.\s*[A-ZÁÉÍÓÚÑ][^.!?]*$',
                'alphabetic': r'^[A-Z]\.\s*[A-ZÁÉÍÓÚÑ][^.!?]*$',
                'appendix': r'^Apéndice\s+[A-Z0-9]+\.?\s*[A-ZÁÉÍÓÚÑ][^.!?]*$',
                'chapter': r'^Capítulo\s+\d+\.?\s*[A-ZÁÉÍÓÚÑ][^.!?]*$'
            },
            'fr': {
                'numeric': r'^(?:\d+\.)+\s*[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ][^.!?]*$',
                'roman': r'^[IVX]+\.\s*[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ][^.!?]*$',
                'alphabetic': r'^[A-Z]\.\s*[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ][^.!?]*$',
                'appendix': r'^Annexe\s+[A-Z0-9]+\.?\s*[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ][^.!?]*$',
                'chapter': r'^Chapitre\s+\d+\.?\s*[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ][^.!?]*$'
            },
            'de': {
                'numeric': r'^(?:\d+\.)+\s*[A-ZÄÖÜ][^.!?]*$',
                'roman': r'^[IVX]+\.\s*[A-ZÄÖÜ][^.!?]*$',
                'alphabetic': r'^[A-Z]\.\s*[A-ZÄÖÜ][^.!?]*$',
                'appendix': r'^Anhang\s+[A-Z0-9]+\.?\s*[A-ZÄÖÜ][^.!?]*$',
                'chapter': r'^Kapitel\s+\d+\.?\s*[A-ZÄÖÜ][^.!?]*$'
            }
        }
    
    def get_patterns_for_language(self, language: str) -> Dict[str, str]:
        """Get heading patterns for specific language"""
        return self.patterns.get(language, self.patterns['en'])
    
    def is_heading(self, text: str, language: str) -> bool:
        """Check if text matches heading patterns for given language"""
        patterns = self.get_patterns_for_language(language)
        
        for pattern_name, pattern in patterns.items():
            if re.match(pattern, text.strip()):
                return True
        
        return False
    
    def extract_heading_level(self, text: str, language: str) -> str:
        """Extract heading level based on language-specific patterns"""
        text = text.strip()
        
        # Check for title-level patterns
        title_patterns = {
            'en': [r'^[A-Z][^.!?]*$', r'^Chapter\s+\d+'],
            'ja': [r'^第[一二三四五六七八九十\d]+章'],
            'zh': [r'^第[一二三四五六七八九十\d]+章'],
            'ko': [r'^제[일이삼사오육칠팔구십\d]+장'],
            'ar': [r'^الفصل\s*[الأولالثانيالثالثالرابعالخامسالسادسالسابعالثامنتاسععاشر\d]+'],
            'hi': [r'^अध्याय\s*[एकदोतीनचारपांचछहसातआठनौदस\d]+']
        }
        
        lang_title_patterns = title_patterns.get(language, title_patterns['en'])
        for pattern in lang_title_patterns:
            if re.match(pattern, text):
                return 'Title'
        
        # Check for H1 patterns
        h1_patterns = {
            'en': [r'^\d+\.\s*[A-Z]', r'^[IVX]+\.\s*[A-Z]'],
            'ja': [r'^第[一二三四五六七八九十\d]+[節項]'],
            'zh': [r'^第[一二三四五六七八九十\d]+[节项]'],
            'ko': [r'^제[일이삼사오육칠팔구십\d]+[절항]'],
            'ar': [r'^الباب\s*[الأولالثانيالثالثالرابعالخامسالسادسالسابعالثامنتاسععاشر\d]+'],
            'hi': [r'^खंड\s*[एकदोतीनचारपांचछहसातआठनौदस\d]+']
        }
        
        lang_h1_patterns = h1_patterns.get(language, h1_patterns['en'])
        for pattern in lang_h1_patterns:
            if re.match(pattern, text):
                return 'H1'
        
        # Check for H2 patterns
        h2_patterns = {
            'en': [r'^\d+\.\d+\.\s*[A-Z]'],
            'ja': [r'^\d+\.\d+\.\s*[^\s]'],
            'zh': [r'^\d+\.\d+\.\s*[^\s]'],
            'ko': [r'^\d+\.\d+\.\s*[^\s]'],
            'ar': [r'^\d+\.\d+\.\s*[^\s]'],
            'hi': [r'^\d+\.\d+\.\s*[^\s]']
        }
        
        lang_h2_patterns = h2_patterns.get(language, h2_patterns['en'])
        for pattern in lang_h2_patterns:
            if re.match(pattern, text):
                return 'H2'
        
        # Check for H3 patterns
        h3_patterns = {
            'en': [r'^\d+\.\d+\.\d+\.\s*[A-Z]'],
            'ja': [r'^\d+\.\d+\.\d+\.\s*[^\s]'],
            'zh': [r'^\d+\.\d+\.\d+\.\s*[^\s]'],
            'ko': [r'^\d+\.\d+\.\d+\.\s*[^\s]'],
            'ar': [r'^\d+\.\d+\.\d+\.\s*[^\s]'],
            'hi': [r'^\d+\.\d+\.\d+\.\s*[^\s]']
        }
        
        lang_h3_patterns = h3_patterns.get(language, h3_patterns['en'])
        for pattern in lang_h3_patterns:
            if re.match(pattern, text):
                return 'H3'
        
        return 'H1'  # Default to H1 if no specific pattern matches

class MultilingualTextNormalizer:
    """Language-specific text normalization"""
    
    def __init__(self):
        self.normalizers = {
            'ja': self._normalize_japanese,
            'zh': self._normalize_chinese,
            'ko': self._normalize_korean,
            'ar': self._normalize_arabic,
            'hi': self._normalize_hindi,
            'th': self._normalize_thai,
            'ru': self._normalize_russian,
            'el': self._normalize_greek,
            'he': self._normalize_hebrew
        }
    
    def normalize(self, text: str, language: str) -> str:
        """Normalize text for specific language"""
        if language in self.normalizers:
            return self.normalizers[language](text)
        return self._normalize_english(text)
    
    def _normalize_english(self, text: str) -> str:
        """Normalize English text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text
    
    def _normalize_japanese(self, text: str) -> str:
        """Normalize Japanese text"""
        # Normalize full-width characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_chinese(self, text: str) -> str:
        """Normalize Chinese text"""
        # Normalize full-width characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_korean(self, text: str) -> str:
        """Normalize Korean text"""
        # Normalize full-width characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text"""
        # Normalize Arabic characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_hindi(self, text: str) -> str:
        """Normalize Hindi text"""
        # Normalize Devanagari characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_thai(self, text: str) -> str:
        """Normalize Thai text"""
        # Normalize Thai characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_russian(self, text: str) -> str:
        """Normalize Russian text"""
        # Normalize Cyrillic characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_greek(self, text: str) -> str:
        """Normalize Greek text"""
        # Normalize Greek characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _normalize_hebrew(self, text: str) -> str:
        """Normalize Hebrew text"""
        # Normalize Hebrew characters
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text

class MultilingualFeatureExtractor:
    """Extract language-specific features for heading detection"""
    
    def __init__(self):
        self.language_detector = MultilingualLanguageDetector()
        self.normalizer = MultilingualTextNormalizer()
        self.patterns = MultilingualHeadingPatterns()
    
    def extract_features(self, text: str, font_size: float, font_name: str, 
                        is_bold: bool, is_italic: bool, page_num: int) -> Dict:
        """Extract multilingual features for heading classification"""
        
        # Detect language
        language = self.language_detector.detect_language(text)
        
        # Normalize text
        normalized_text = self.normalizer.normalize(text, language)
        
        # Basic features
        features = {
            'text': normalized_text,
            'original_text': text,
            'language': language,
            'font_size': font_size,
            'font_name': font_name,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'page_num': page_num,
            'text_length': len(normalized_text),
            'word_count': len(normalized_text.split()),
            'char_count': len(normalized_text),
            'is_all_caps': normalized_text.isupper(),
            'is_title_case': normalized_text.istitle(),
            'has_numbers': bool(re.search(r'\d', normalized_text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', normalized_text)),
            'starts_with_number': bool(re.match(r'^\d', normalized_text)),
            'ends_with_punctuation': bool(re.search(r'[.!?]$', normalized_text)),
            'is_heading_pattern': self.patterns.is_heading(normalized_text, language),
            'heading_level': self.patterns.extract_heading_level(normalized_text, language)
        }
        
        # Language-specific features
        features.update(self._extract_language_specific_features(normalized_text, language))
        
        return features
    
    def _extract_language_specific_features(self, text: str, language: str) -> Dict:
        """Extract language-specific features"""
        features = {}
        
        if language == 'ja':
            features.update({
                'has_kanji': bool(re.search(r'[\u4E00-\u9FAF]', text)),
                'has_hiragana': bool(re.search(r'[\u3040-\u309F]', text)),
                'has_katakana': bool(re.search(r'[\u30A0-\u30FF]', text)),
                'kanji_ratio': len(re.findall(r'[\u4E00-\u9FAF]', text)) / len(text) if text else 0,
                'has_japanese_keywords': bool(re.search(r'第|章|節|項|目|はじめに|結論', text))
            })
        
        elif language == 'zh':
            features.update({
                'has_chinese_chars': bool(re.search(r'[\u4E00-\u9FAF]', text)),
                'chinese_char_ratio': len(re.findall(r'[\u4E00-\u9FAF]', text)) / len(text) if text else 0,
                'has_chinese_keywords': bool(re.search(r'第|章|节|项|目|引言|结论', text))
            })
        
        elif language == 'ko':
            features.update({
                'has_hangul': bool(re.search(r'[\uAC00-\uD7AF]', text)),
                'hangul_ratio': len(re.findall(r'[\uAC00-\uD7AF]', text)) / len(text) if text else 0,
                'has_korean_keywords': bool(re.search(r'제|장|절|항|목|서론|결론', text))
            })
        
        elif language == 'ar':
            features.update({
                'has_arabic_chars': bool(re.search(r'[\u0600-\u06FF]', text)),
                'arabic_char_ratio': len(re.findall(r'[\u0600-\u06FF]', text)) / len(text) if text else 0,
                'has_arabic_keywords': bool(re.search(r'الفصل|الباب|المبحث|المطلب|المقدمة|الخاتمة', text)),
                'is_right_to_left': True
            })
        
        elif language == 'hi':
            features.update({
                'has_devanagari': bool(re.search(r'[\u0900-\u097F]', text)),
                'devanagari_ratio': len(re.findall(r'[\u0900-\u097F]', text)) / len(text) if text else 0,
                'has_hindi_keywords': bool(re.search(r'अध्याय|खंड|अनुभाग|बिंदु|परिचय|निष्कर्ष', text))
            })
        
        else:
            # European languages
            features.update({
                'has_accents': bool(re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸ]', text)),
                'is_latin_script': True
            })
        
        return features

# Global instances
language_detector = MultilingualLanguageDetector()
heading_patterns = MultilingualHeadingPatterns()
text_normalizer = MultilingualTextNormalizer()
feature_extractor = MultilingualFeatureExtractor() 