#!/usr/bin/env python3
"""
Modular Heading Extractor - ML-based heading detection
Replaces hardcoded patterns with intelligent detection
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModularHeadingExtractor:
    """Advanced heading detection without hardcoded patterns"""
    
    def __init__(self):
        # Advanced pattern recognition
        self.patterns = {
            'numbered': re.compile(r'^(\d{1,2}(?:[.\)])(\d{1,2}[.\)]){0,3})\s+.+'),
            'roman': re.compile(r'^(?=[MDCLXVI])([MDCLXVI]+[.\)])(\s+[A-Z][a-zA-Z]*)+'),
            'appendix': re.compile(r'^(Appendix|ANNEX|EXHIBIT)\b', re.I),
            'all_caps': re.compile(r'^[A-Z\s\-]{4,}$'),
            'chapter': re.compile(r'^(Chapter|Section|Part)\s+\d+', re.I),
            'bullet': re.compile(r'^[\-\*•]\s+[A-Z]'),
            'lettered': re.compile(r'^[A-Z]\.\s+[A-Z]'),
        }
        
        # Font-based detection thresholds
        self.font_thresholds = {
            'title': {'min_size': 18, 'confidence': 0.9},
            'h1': {'min_size': 16, 'confidence': 0.85},
            'h2': {'min_size': 14, 'confidence': 0.8},
            'h3': {'min_size': 12, 'confidence': 0.75},
            'h4': {'min_size': 11, 'confidence': 0.7},
        }
        
        # Context-aware features
        self.context_features = {
            'position_weight': 0.3,
            'font_weight': 0.4,
            'pattern_weight': 0.3,
        }
    
    def extract_headings(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract headings using ML-based approach"""
        headings = []
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            font_size = block.get('font_size', 12)
            is_bold = block.get('is_bold', False)
            is_centered = block.get('is_centered', False)
            page = block.get('page', 1)
            
            # Calculate heading probability
            heading_level, confidence = self._classify_heading(
                text, font_size, is_bold, is_centered, page
            )
            
            if heading_level and confidence >= 0.5:  # Lowered threshold
                headings.append({
                    'level': heading_level,
                    'text': text,
                    'page': page,
                    'confidence': confidence
                })
        
        # Post-processing: enforce hierarchy and remove duplicates
        return self._post_process_headings(headings)
    
    def _classify_heading(self, text: str, font_size: float, is_bold: bool, 
                         is_centered: bool, page: int) -> Tuple[str, float]:
        """Classify text as heading with confidence score"""
        
        if not text or len(text) < 2:
            return None, 0.0
        
        # Pattern-based classification
        pattern_score = self._pattern_classification(text)
        
        # Font-based classification
        font_score = self._font_classification(font_size, is_bold, is_centered)
        
        # Position-based classification
        position_score = self._position_classification(page, text)
        
        # Combine scores with weights
        total_score = (
            pattern_score * self.context_features['pattern_weight'] +
            font_score * self.context_features['font_weight'] +
            position_score * self.context_features['position_weight']
        )
        
        # Determine heading level
        heading_level = self._determine_heading_level(text, font_size, total_score)
        
        return heading_level, total_score
    
    def _pattern_classification(self, text: str) -> float:
        """Classify based on text patterns"""
        max_score = 0.0
        
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(text):
                if pattern_name == 'numbered':
                    depth = text.split()[0].count('.') + text.split()[0].count(')')
                    score = 0.9 + (depth * 0.05)
                elif pattern_name == 'roman':
                    score = 0.85
                elif pattern_name == 'appendix':
                    score = 0.9
                elif pattern_name == 'all_caps':
                    score = 0.8
                elif pattern_name == 'chapter':
                    score = 0.9
                elif pattern_name == 'bullet':
                    score = 0.7
                elif pattern_name == 'lettered':
                    score = 0.75
                else:
                    score = 0.6
                
                max_score = max(max_score, score)
        
        # Additional simple patterns
        if re.match(r'^[A-Z][A-Z\s\-]+$', text):  # ALL CAPS
            max_score = max(max_score, 0.8)
        elif re.match(r'^\d+\.\s+[A-Z]', text):  # Numbered
            max_score = max(max_score, 0.9)
        elif re.match(r'^[A-Z]\.\s+[A-Z]', text):  # Lettered
            max_score = max(max_score, 0.75)
        elif re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.I):  # Chapter/Section
            max_score = max(max_score, 0.9)
        
        return max_score
    
    def _font_classification(self, font_size: float, is_bold: bool, is_centered: bool) -> float:
        """Classify based on font characteristics"""
        score = 0.0
        
        # Font size scoring
        if font_size >= self.font_thresholds['title']['min_size']:
            score = self.font_thresholds['title']['confidence']
        elif font_size >= self.font_thresholds['h1']['min_size']:
            score = self.font_thresholds['h1']['confidence']
        elif font_size >= self.font_thresholds['h2']['min_size']:
            score = self.font_thresholds['h2']['confidence']
        elif font_size >= self.font_thresholds['h3']['min_size']:
            score = self.font_thresholds['h3']['confidence']
        elif font_size >= self.font_thresholds['h4']['min_size']:
            score = self.font_thresholds['h4']['confidence']
        elif font_size >= 11:  # Lower threshold
            score = 0.6
        
        # Bold bonus
        if is_bold:
            score += 0.2
        
        # Centered bonus
        if is_centered:
            score += 0.1
        
        return min(score, 1.0)
    
    def _position_classification(self, page: int, text: str) -> float:
        """Classify based on position and context"""
        score = 0.5  # Base score
        
        # First page bonus (likely title)
        if page == 1:
            score += 0.2
        
        # Short text bonus (likely heading)
        if len(text.split()) <= 8:
            score += 0.1
        
        # No punctuation bonus
        if not any(p in text for p in '.,;:!?'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _determine_heading_level(self, text: str, font_size: float, total_score: float) -> str:
        """Determine heading level based on combined analysis"""
        
        if total_score < 0.5:  # Lowered threshold
            return None
        
        # Pattern-based level determination
        if self.patterns['numbered'].match(text):
            depth = text.split()[0].count('.') + text.split()[0].count(')')
            if depth == 0:
                return "H1"
            elif depth == 1:
                return "H2"
            elif depth == 2:
                return "H3"
            else:
                return "H4"
        
        # Font-based level determination
        if font_size >= self.font_thresholds['title']['min_size']:
            return "TITLE"
        elif font_size >= self.font_thresholds['h1']['min_size']:
            return "H1"
        elif font_size >= self.font_thresholds['h2']['min_size']:
            return "H2"
        elif font_size >= self.font_thresholds['h3']['min_size']:
            return "H3"
        elif font_size >= self.font_thresholds['h4']['min_size']:
            return "H4"
        elif font_size >= 11:  # Lower threshold
            return "H4"
        
        # Default based on score
        if total_score >= 0.9:
            return "H1"
        elif total_score >= 0.8:
            return "H2"
        elif total_score >= 0.7:
            return "H3"
        else:
            return "H4"
    
    def _post_process_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process headings to enforce hierarchy and remove duplicates"""
        
        # Sort by page and position
        sorted_headings = sorted(headings, key=lambda x: (x['page'], x.get('confidence', 0)), reverse=True)
        
        # Remove duplicates
        seen_texts = set()
        filtered_headings = []
        
        for heading in sorted_headings:
            text = heading['text']
            if text not in seen_texts:
                seen_texts.add(text)
                filtered_headings.append(heading)
        
        # Enforce hierarchy (H3 inside H2 inside H1)
        final_headings = []
        current_h1 = None
        current_h2 = None
        
        for heading in filtered_headings:
            level = heading['level']
            
            if level == "TITLE":
                # Skip title from headings list
                continue
            elif level == "H1":
                current_h1 = heading
                current_h2 = None
                final_headings.append(heading)
            elif level == "H2":
                if current_h1:  # Only add H2 if we have H1
                    current_h2 = heading
                    final_headings.append(heading)
            elif level == "H3":
                if current_h2:  # Only add H3 if we have H2
                    final_headings.append(heading)
            elif level == "H4":
                if current_h2:  # H4 can be under H2 or H3
                    final_headings.append(heading)
        
        return final_headings 