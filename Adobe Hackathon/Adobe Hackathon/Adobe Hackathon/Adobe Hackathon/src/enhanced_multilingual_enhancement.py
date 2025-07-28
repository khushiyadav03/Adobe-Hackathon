#!/usr/bin/env python3
"""
Enhanced Multilingual Enhancement Module
Improved language detection, feature extraction, and heading patterns
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedLanguageDetector:
    """Enhanced language detection with improved accuracy"""
    
    def __init__(self):
        # Enhanced character patterns for better language detection
        self.language_patterns = {
            # CJK Languages
            'ja': {
                'hiragana': r'[\u3040-\u309F]',
                'katakana': r'[\u30A0-\u30FF]',
                'kanji': r'[\u4E00-\u9FAF]',
                'keywords': ['はじめに', '背景', '研究方法', '付録', '章', '節', '目次', '参考文献']
            },
            'zh': {
                'simplified': r'[\u4E00-\u9FAF]',
                'keywords': ['引言', '背景', '研究方法', '附录', '章', '节', '目录', '参考文献', '摘要', '结论']
            },
            'ko': {
                'hangul': r'[\uAC00-\uD7AF]',
                'keywords': ['서론', '배경', '연구방법', '부록', '장', '절', '목차', '참고문헌', '초록', '결론']
            },
            # Middle Eastern Languages
            'ar': {
                'arabic': r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',
                'keywords': ['مقدمة', 'خلفية', 'طرق البحث', 'ملحق', 'فصل', 'قسم', 'فهرس', 'مراجع']
            },
            'he': {
                'hebrew': r'[\u0590-\u05FF\uFB1D-\uFB4F]',
                'keywords': ['מבוא', 'רקע', 'שיטות מחקר', 'נספח', 'פרק', 'סעיף', 'תוכן עניינים', 'ביבליוגרפיה']
            },
            # South Asian Languages
            'hi': {
                'devanagari': r'[\u0900-\u097F]',
                'keywords': ['परिचय', 'पृष्ठभूमि', 'शोध विधियां', 'परिशिष्ट', 'अध्याय', 'खंड', 'सूची', 'संदर्भ']
            },
            'th': {
                'thai': r'[\u0E00-\u0E7F]',
                'keywords': ['บทนำ', 'ภูมิหลัง', 'วิธีการวิจัย', 'ภาคผนวก', 'บท', 'ส่วน', 'สารบัญ', 'บรรณานุกรม']
            },
            # European Languages with enhanced patterns
            'en': {
                'keywords': ['introduction', 'background', 'methods', 'appendix', 'chapter', 'section', 'table of contents', 'references', 'abstract', 'conclusion']
            },
            'es': {
                'keywords': ['introducción', 'antecedentes', 'métodos', 'apéndice', 'capítulo', 'sección', 'índice', 'referencias', 'resumen', 'conclusión']
            },
            'fr': {
                'keywords': ['introduction', 'contexte', 'méthodes', 'annexe', 'chapitre', 'section', 'table des matières', 'références', 'résumé', 'conclusion']
            },
            'de': {
                'keywords': ['einführung', 'hintergrund', 'methoden', 'anhang', 'kapitel', 'abschnitt', 'inhaltsverzeichnis', 'referenzen', 'zusammenfassung', 'schlussfolgerung']
            },
            'it': {
                'keywords': ['introduzione', 'contesto', 'metodi', 'appendice', 'capitolo', 'sezione', 'indice', 'riferimenti', 'riassunto', 'conclusione']
            },
            'pt': {
                'keywords': ['introdução', 'antecedentes', 'métodos', 'apêndice', 'capítulo', 'seção', 'índice', 'referências', 'resumo', 'conclusão']
            },
            'nl': {
                'keywords': ['inleiding', 'achtergrond', 'methoden', 'bijlage', 'hoofdstuk', 'sectie', 'inhoudsopgave', 'referenties', 'samenvatting', 'conclusie']
            },
            'sv': {
                'keywords': ['inledning', 'bakgrund', 'metoder', 'bilaga', 'kapitel', 'sektion', 'innehållsförteckning', 'referenser', 'sammanfattning', 'slutsats']
            },
            'no': {
                'keywords': ['innledning', 'bakgrunn', 'metoder', 'vedlegg', 'kapittel', 'seksjon', 'innholdsfortegnelse', 'referanser', 'sammendrag', 'konklusjon']
            },
            'da': {
                'keywords': ['indledning', 'baggrund', 'metoder', 'bilag', 'kapitel', 'sektion', 'indholdsfortegnelse', 'referencer', 'resumé', 'konklusion']
            },
            'pl': {
                'keywords': ['wprowadzenie', 'tło', 'metody', 'załącznik', 'rozdział', 'sekcja', 'spis treści', 'odniesienia', 'streszczenie', 'wnioski']
            },
            'cs': {
                'keywords': ['úvod', 'pozadí', 'metody', 'příloha', 'kapitola', 'sekce', 'obsah', 'reference', 'shrnutí', 'závěr']
            },
            'sk': {
                'keywords': ['úvod', 'pozadie', 'metódy', 'príloha', 'kapitola', 'sekcia', 'obsah', 'referencie', 'zhrnutie', 'záver']
            },
            'hu': {
                'keywords': ['bevezetés', 'háttér', 'módszerek', 'függelék', 'fejezet', 'szekció', 'tartalomjegyzék', 'hivatkozások', 'összefoglalás', 'következtetés']
            },
            'ro': {
                'keywords': ['introducere', 'fundal', 'metode', 'anexă', 'capitol', 'secțiune', 'cuprins', 'referințe', 'rezumat', 'concluzie']
            },
            'bg': {
                'keywords': ['въведение', 'фон', 'методи', 'приложение', 'глава', 'секция', 'съдържание', 'референции', 'резюме', 'заключение']
            },
            'hr': {
                'keywords': ['uvod', 'pozadina', 'metode', 'dodatak', 'poglavlje', 'sekcija', 'sadržaj', 'reference', 'sažetak', 'zaključak']
            },
            'sl': {
                'keywords': ['uvod', 'ozadje', 'metode', 'dodatek', 'poglavje', 'sekcija', 'kazalo', 'reference', 'povzetek', 'zaključek']
            },
            'ru': {
                'keywords': ['введение', 'фон', 'методы', 'приложение', 'глава', 'раздел', 'содержание', 'ссылки', 'резюме', 'заключение']
            },
            'el': {
                'keywords': ['εισαγωγή', 'φόντο', 'μέθοδοι', 'παράρτημα', 'κεφάλαιο', 'ενότητα', 'περιεχόμενα', 'αναφορές', 'περίληψη', 'συμπέρασμα']
            }
        }
        
        # Language-specific character ratios for better detection
        self.character_ratios = {
            'ja': {'hiragana': 0.1, 'katakana': 0.1, 'kanji': 0.3},
            'zh': {'simplified': 0.4},
            'ko': {'hangul': 0.4},
            'ar': {'arabic': 0.3},
            'he': {'hebrew': 0.3},
            'hi': {'devanagari': 0.3},
            'th': {'thai': 0.3}
        }
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection with multiple strategies"""
        if not text or len(text.strip()) < 3:
            return 'en'
        
        text = text.strip()
        
        # Strategy 1: Character-based detection for non-Latin scripts
        for lang, patterns in self.language_patterns.items():
            if lang in ['ja', 'zh', 'ko', 'ar', 'he', 'hi', 'th']:
                if self._check_character_patterns(text, patterns):
                    return lang
        
        # Strategy 2: Keyword-based detection with confidence scoring
        lang_scores = {}
        for lang, patterns in self.language_patterns.items():
            if 'keywords' in patterns:
                score = self._calculate_keyword_score(text.lower(), patterns['keywords'])
                if score > 0:
                    lang_scores[lang] = score
        
        # Strategy 3: Character ratio analysis for ambiguous cases
        if lang_scores:
            # Boost scores for languages with specific character patterns
            for lang in lang_scores:
                if lang in self.character_ratios:
                    ratio_boost = self._calculate_character_ratio(text, lang)
                    lang_scores[lang] *= (1 + ratio_boost)
            
            # Return language with highest score
            best_lang = max(lang_scores.items(), key=lambda x: x[1])
            if best_lang[1] > 0.1:  # Minimum confidence threshold
                return best_lang[0]
        
        # Strategy 4: Fallback to langdetect for remaining cases
        try:
            from langdetect import detect, LangDetectException
            detected = detect(text)
            # Map langdetect codes to our codes
            lang_mapping = {
                'zh-cn': 'zh', 'zh-tw': 'zh', 'zh': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'he': 'he',
                'hi': 'hi', 'th': 'th', 'ru': 'ru', 'el': 'el'
            }
            return lang_mapping.get(detected, detected)
        except (LangDetectException, ImportError):
            pass
        
        return 'en'
    
    def _check_character_patterns(self, text: str, patterns: Dict) -> bool:
        """Check if text matches character patterns for a language"""
        for pattern_name, pattern in patterns.items():
            if pattern_name != 'keywords':
                matches = re.findall(pattern, text)
                if matches:
                    ratio = len(''.join(matches)) / len(text)
                    if ratio > 0.2:  # At least 20% of characters match
                        return True
        return False
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword-based language score"""
        score = 0.0
        for keyword in keywords:
            if keyword in text:
                score += 1.0
                # Bonus for longer keywords
                score += len(keyword) * 0.1
        return score
    
    def _calculate_character_ratio(self, text: str, language: str) -> float:
        """Calculate character ratio for specific language"""
        if language not in self.character_ratios:
            return 0.0
        
        total_ratio = 0.0
        for char_type, min_ratio in self.character_ratios[language].items():
            if char_type in self.language_patterns[language]:
                pattern = self.language_patterns[language][char_type]
                matches = re.findall(pattern, text)
                if matches:
                    ratio = len(''.join(matches)) / len(text)
                    if ratio >= min_ratio:
                        total_ratio += ratio
        
        return total_ratio

class EnhancedMultilingualTextNormalizer:
    """Enhanced text normalization with language-specific rules"""
    
    def __init__(self):
        self.normalization_rules = {
            'ja': {
                'fullwidth_digits': {chr(i): str(i - 0xFF10) for i in range(0xFF10, 0xFF1A)},
                'fullwidth_letters': {chr(i): chr(i - 0xFF20) for i in range(0xFF21, 0xFF3B)},
                'fullwidth_space': {'　': ' '}
            },
            'zh': {
                'fullwidth_digits': {chr(i): str(i - 0xFF10) for i in range(0xFF10, 0xFF1A)},
                'fullwidth_letters': {chr(i): chr(i - 0xFF20) for i in range(0xFF21, 0xFF3B)},
                'fullwidth_space': {'　': ' '}
            },
            'ko': {
                'fullwidth_digits': {chr(i): str(i - 0xFF10) for i in range(0xFF10, 0xFF1A)},
                'fullwidth_letters': {chr(i): chr(i - 0xFF20) for i in range(0xFF21, 0xFF3B)},
                'fullwidth_space': {'　': ' '}
            },
            'hi': {
                'devanagari_digits': {
                    '१': '1', '२': '2', '३': '3', '४': '4', '५': '5',
                    '६': '6', '७': '7', '८': '8', '९': '9', '०': '0'
                }
            }
        }
    
    def normalize(self, text: str, language: str = None) -> str:
        """Enhanced text normalization"""
        if not text:
            return text
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Language-specific normalization
        if language and language in self.normalization_rules:
            text = self._apply_language_specific_rules(text, language)
        
        # General whitespace normalization
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _apply_language_specific_rules(self, text: str, language: str) -> str:
        """Apply language-specific normalization rules"""
        rules = self.normalization_rules[language]
        
        for rule_type, replacements in rules.items():
            for old_char, new_char in replacements.items():
                text = text.replace(old_char, new_char)
        
        return text

class EnhancedMultilingualHeadingPatterns:
    """Enhanced heading patterns with improved accuracy"""
    
    def __init__(self):
        # Enhanced patterns for better heading detection
        self.patterns = {
            'en': {
                'title': [
                    r'^Chapter\s+\d+[:\s]',
                    r'^Introduction$',
                    r'^Abstract$',
                    r'^Conclusion$',
                    r'^References?$',
                    r'^Bibliography$',
                    r'^Table of Contents$',
                    r'^Index$'
                ],
                'h1': [
                    r'^\d+\.\s+[A-Z]',
                    r'^[A-Z][A-Z\s]+$',
                    r'^Appendix\s+[A-Z]',
                    r'^Section\s+\d+',
                    r'^Part\s+\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s+[A-Z]',
                    r'^[A-Z][a-z]+[A-Z][a-z\s]+$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s+[A-Z]',
                    r'^[a-z]+[A-Z][a-z\s]+$'
                ]
            },
            'ja': {
                'title': [
                    r'^第\d+章\s*',
                    r'^はじめに$',
                    r'^概要$',
                    r'^結論$',
                    r'^参考文献$',
                    r'^目次$',
                    r'^索引$'
                ],
                'h1': [
                    r'^\d+\.\s*[^\d]',
                    r'^[あ-んア-ン一-龯]{2,}$',
                    r'^付録[A-Z]',
                    r'^節\d+',
                    r'^部\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s*[^\d]',
                    r'^[あ-んア-ン一-龯]{3,}$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s*[^\d]',
                    r'^[あ-んア-ン一-龯]{4,}$'
                ]
            },
            'zh': {
                'title': [
                    r'^第[一二三四五六七八九十\d]+章\s*',
                    r'^引言$',
                    r'^摘要$',
                    r'^结论$',
                    r'^参考文献$',
                    r'^目录$',
                    r'^索引$'
                ],
                'h1': [
                    r'^\d+\.\s*[^\d]',
                    r'^[一-龯]{2,}$',
                    r'^附录[A-Z]',
                    r'^节\d+',
                    r'^部\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s*[^\d]',
                    r'^[一-龯]{3,}$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s*[^\d]',
                    r'^[一-龯]{4,}$'
                ]
            },
            'ko': {
                'title': [
                    r'^제\d+장\s*',
                    r'^서론$',
                    r'^초록$',
                    r'^결론$',
                    r'^참고문헌$',
                    r'^목차$',
                    r'^색인$'
                ],
                'h1': [
                    r'^\d+\.\s*[^\d]',
                    r'^[가-힣]{2,}$',
                    r'^부록[A-Z]',
                    r'^절\d+',
                    r'^부\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s*[^\d]',
                    r'^[가-힣]{3,}$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s*[^\d]',
                    r'^[가-힣]{4,}$'
                ]
            },
            'ar': {
                'title': [
                    r'^الفصل\s+\d+\s*',
                    r'^مقدمة$',
                    r'^ملخص$',
                    r'^خاتمة$',
                    r'^مراجع$',
                    r'^فهرس$',
                    r'^دليل$'
                ],
                'h1': [
                    r'^\d+\.\s*[^\d]',
                    r'^[ء-ي]{2,}$',
                    r'^ملحق\s*[ء-ي]',
                    r'^قسم\s+\d+',
                    r'^جزء\s+\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s*[^\d]',
                    r'^[ء-ي]{3,}$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s*[^\d]',
                    r'^[ء-ي]{4,}$'
                ]
            },
            'hi': {
                'title': [
                    r'^अध्याय\s+\d+\s*',
                    r'^परिचय$',
                    r'^सारांश$',
                    r'^निष्कर्ष$',
                    r'^संदर्भ$',
                    r'^सूची$',
                    r'^अनुक्रमणिका$'
                ],
                'h1': [
                    r'^\d+\.\s*[^\d]',
                    r'^[अ-ह]{2,}$',
                    r'^परिशिष्ट\s*[अ-ह]',
                    r'^खंड\s+\d+',
                    r'^भाग\s+\d+'
                ],
                'h2': [
                    r'^\d+\.\d+\s*[^\d]',
                    r'^[अ-ह]{3,}$'
                ],
                'h3': [
                    r'^\d+\.\d+\.\d+\s*[^\d]',
                    r'^[अ-ह]{4,}$'
                ]
            }
        }
        
        # Add European language patterns
        european_languages = ['es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'no', 'da', 'pl', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sl', 'ru', 'el']
        for lang in european_languages:
            self.patterns[lang] = self._create_european_patterns(lang)
    
    def _create_european_patterns(self, language: str) -> Dict:
        """Create patterns for European languages"""
        chapter_words = {
            'es': 'Capítulo', 'fr': 'Chapitre', 'de': 'Kapitel', 'it': 'Capitolo',
            'pt': 'Capítulo', 'nl': 'Hoofdstuk', 'sv': 'Kapitel', 'no': 'Kapitel',
            'da': 'Kapitel', 'pl': 'Rozdział', 'cs': 'Kapitola', 'sk': 'Kapitola',
            'hu': 'Fejezet', 'ro': 'Capitol', 'bg': 'Глава', 'hr': 'Poglavlje',
            'sl': 'Poglavje', 'ru': 'Глава', 'el': 'Κεφάλαιο'
        }
        
        chapter_word = chapter_words.get(language, 'Chapter')
        
        return {
            'title': [
                rf'^{chapter_word}\s+\d+[:\s]',
                r'^Introduction$',
                r'^Abstract$',
                r'^Conclusion$',
                r'^References?$',
                r'^Bibliography$',
                r'^Table of Contents$',
                r'^Index$'
            ],
            'h1': [
                r'^\d+\.\s+[A-Z]',
                r'^[A-Z][A-Z\s]+$',
                r'^Appendix\s+[A-Z]',
                r'^Section\s+\d+',
                r'^Part\s+\d+'
            ],
            'h2': [
                r'^\d+\.\d+\s+[A-Z]',
                r'^[A-Z][a-z]+[A-Z][a-z\s]+$'
            ],
            'h3': [
                r'^\d+\.\d+\.\d+\s+[A-Z]',
                r'^[a-z]+[A-Z][a-z\s]+$'
            ]
        }
    
    def is_heading(self, text: str, language: str = 'en') -> bool:
        """Enhanced heading detection"""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        patterns = self.patterns.get(language, self.patterns['en'])
        
        # Check all heading levels
        for level in ['title', 'h1', 'h2', 'h3']:
            for pattern in patterns[level]:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        return False
    
    def extract_heading_level(self, text: str, language: str = 'en') -> str:
        """Enhanced heading level extraction"""
        if not text or len(text.strip()) < 2:
            return 'Text'
        
        text = text.strip()
        patterns = self.patterns.get(language, self.patterns['en'])
        
        # Check in order of specificity (most specific first)
        for level in ['h3', 'h2', 'h1', 'title']:
            for pattern in patterns[level]:
                if re.match(pattern, text, re.IGNORECASE):
                    return level.upper()
        
        return 'Text'

class EnhancedMultilingualFeatureExtractor:
    """Enhanced feature extraction with improved language detection"""
    
    def __init__(self):
        self.language_detector = EnhancedLanguageDetector()
        self.text_normalizer = EnhancedMultilingualTextNormalizer()
    
    def extract_features(self, text: str, font_size: float = 16.0, font_name: str = "Arial", 
                        is_bold: bool = False, is_italic: bool = False, page_num: int = 1) -> Dict[str, Any]:
        """Enhanced feature extraction with language-specific features"""
        
        # Normalize text
        normalized_text = self.text_normalizer.normalize(text)
        
        # Detect language
        detected_language = self.language_detector.detect_language(normalized_text)
        
        # Base features
        features = {
            'text': normalized_text,
            'original_text': text,
            'language': detected_language,
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
            'ends_with_punctuation': bool(re.search(r'[.!?;:]$', normalized_text)),
            'is_heading_pattern': False,
            'heading_level': 'Text',
            'has_accents': bool(re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', normalized_text, re.IGNORECASE))
        }
        
        # Language-specific features
        features.update(self._extract_language_specific_features(normalized_text, detected_language))
        
        # Heading pattern detection
        heading_patterns = EnhancedMultilingualHeadingPatterns()
        features['is_heading_pattern'] = heading_patterns.is_heading(normalized_text, detected_language)
        features['heading_level'] = heading_patterns.extract_heading_level(normalized_text, detected_language)
        
        return features
    
    def _extract_language_specific_features(self, text: str, language: str) -> Dict[str, Any]:
        """Extract language-specific features"""
        features = {}
        
        if language == 'ja':
            features.update(self._extract_japanese_features(text))
        elif language == 'zh':
            features.update(self._extract_chinese_features(text))
        elif language == 'ko':
            features.update(self._extract_korean_features(text))
        elif language == 'ar':
            features.update(self._extract_arabic_features(text))
        elif language == 'he':
            features.update(self._extract_hebrew_features(text))
        elif language == 'hi':
            features.update(self._extract_hindi_features(text))
        elif language == 'th':
            features.update(self._extract_thai_features(text))
        else:
            features['is_latin_script'] = True
        
        return features
    
    def _extract_japanese_features(self, text: str) -> Dict[str, Any]:
        """Extract Japanese-specific features"""
        hiragana_chars = re.findall(r'[\u3040-\u309F]', text)
        katakana_chars = re.findall(r'[\u30A0-\u30FF]', text)
        kanji_chars = re.findall(r'[\u4E00-\u9FAF]', text)
        
        return {
            'has_hiragana': len(hiragana_chars) > 0,
            'hiragana_ratio': len(hiragana_chars) / len(text) if text else 0,
            'has_katakana': len(katakana_chars) > 0,
            'katakana_ratio': len(katakana_chars) / len(text) if text else 0,
            'has_kanji': len(kanji_chars) > 0,
            'kanji_ratio': len(kanji_chars) / len(text) if text else 0,
            'is_japanese_script': True
        }
    
    def _extract_chinese_features(self, text: str) -> Dict[str, Any]:
        """Extract Chinese-specific features"""
        chinese_chars = re.findall(r'[\u4E00-\u9FAF]', text)
        
        return {
            'has_chinese_chars': len(chinese_chars) > 0,
            'chinese_char_ratio': len(chinese_chars) / len(text) if text else 0,
            'is_chinese_script': True
        }
    
    def _extract_korean_features(self, text: str) -> Dict[str, Any]:
        """Extract Korean-specific features"""
        hangul_chars = re.findall(r'[\uAC00-\uD7AF]', text)
        
        return {
            'has_hangul': len(hangul_chars) > 0,
            'hangul_ratio': len(hangul_chars) / len(text) if text else 0,
            'is_korean_script': True
        }
    
    def _extract_arabic_features(self, text: str) -> Dict[str, Any]:
        """Extract Arabic-specific features"""
        arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text)
        
        return {
            'has_arabic_chars': len(arabic_chars) > 0,
            'arabic_char_ratio': len(arabic_chars) / len(text) if text else 0,
            'is_right_to_left': True,
            'is_arabic_script': True
        }
    
    def _extract_hebrew_features(self, text: str) -> Dict[str, Any]:
        """Extract Hebrew-specific features"""
        hebrew_chars = re.findall(r'[\u0590-\u05FF\uFB1D-\uFB4F]', text)
        
        return {
            'has_hebrew_chars': len(hebrew_chars) > 0,
            'hebrew_char_ratio': len(hebrew_chars) / len(text) if text else 0,
            'is_right_to_left': True,
            'is_hebrew_script': True
        }
    
    def _extract_hindi_features(self, text: str) -> Dict[str, Any]:
        """Extract Hindi-specific features"""
        devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
        
        return {
            'has_devanagari_chars': len(devanagari_chars) > 0,
            'devanagari_char_ratio': len(devanagari_chars) / len(text) if text else 0,
            'is_hindi_script': True
        }
    
    def _extract_thai_features(self, text: str) -> Dict[str, Any]:
        """Extract Thai-specific features"""
        thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
        
        return {
            'has_thai_chars': len(thai_chars) > 0,
            'thai_char_ratio': len(thai_chars) / len(text) if text else 0,
            'is_thai_script': True
        } 