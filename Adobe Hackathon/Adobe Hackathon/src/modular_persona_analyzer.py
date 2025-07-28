#!/usr/bin/env python3
"""
Enhanced Modular Persona Analyzer
Robust persona-driven document intelligence for any domain
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import re
from collections import Counter

logger = logging.getLogger(__name__)

class EnhancedModularPersonaAnalyzer:
    """Enhanced persona analyzer for any domain and document type"""
    
    def __init__(self):
        self.domain_keywords = {
            'academic': ['research', 'study', 'analysis', 'methodology', 'literature', 'thesis', 'dissertation'],
            'business': ['business', 'corporate', 'financial', 'market', 'strategy', 'revenue', 'investment'],
            'technical': ['technical', 'engineering', 'development', 'software', 'system', 'architecture'],
            'medical': ['medical', 'health', 'clinical', 'patient', 'treatment', 'diagnosis'],
            'legal': ['legal', 'law', 'contract', 'regulation', 'compliance', 'policy'],
            'educational': ['education', 'learning', 'teaching', 'curriculum', 'student', 'course']
        }
        
        self.persona_types = {
            'researcher': ['research', 'phd', 'academic', 'scientist', 'scholar'],
            'student': ['student', 'undergraduate', 'graduate', 'learner'],
            'analyst': ['analyst', 'investment', 'financial', 'business'],
            'executive': ['executive', 'manager', 'director', 'ceo', 'cfo'],
            'professional': ['professional', 'consultant', 'specialist', 'expert']
        }
        
        self.job_types = {
            'literature_review': ['review', 'literature', 'survey', 'overview'],
            'analysis': ['analyze', 'analysis', 'examine', 'evaluate'],
            'summary': ['summarize', 'summary', 'overview', 'brief'],
            'study': ['study', 'learn', 'prepare', 'understand'],
            'research': ['research', 'investigate', 'explore', 'discover']
        }
        
        logger.info("Enhanced Modular Persona Analyzer initialized")

    def analyze_documents(self, persona: str, job_description: str, 
                         documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced document analysis for any persona and job"""
        
        logger.info(f"Analyzing {len(documents)} documents for persona: {persona}")
        
        # Enhanced persona analysis
        persona_analysis = self._analyze_enhanced_persona(persona, job_description)
        
        # Process documents
        processed_docs = []
        for doc in documents:
            processed_doc = self._process_document(doc, persona_analysis)
            processed_docs.append(processed_doc)
        
        # Extract relevant sections
        extracted_sections = self._extract_relevant_sections(processed_docs, persona_analysis)
        
        # Analyze subsections
        subsection_analysis = self._analyze_subsections(processed_docs, persona_analysis)
        
        # Generate output
        output = {
            'processing_timestamp': datetime.now().isoformat(),
            'input_documents': [doc.get('title', 'Unknown') for doc in documents],
            'persona': persona,
            'job_to_be_done': job_description,
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }
        
        logger.info(f"Analysis completed. Found {len(extracted_sections)} relevant sections")
        return output

    def _analyze_enhanced_persona(self, persona: str, job_description: str) -> Dict[str, Any]:
        """Enhanced persona analysis for any domain"""
        
        persona_lower = persona.lower()
        job_lower = job_description.lower()
        
        # Determine domain
        domain = self._determine_domain(persona_lower, job_lower)
        
        # Classify persona type
        persona_type = self._classify_persona(persona_lower)
        
        # Classify job type
        job_type = self._classify_job(job_lower)
        
        # Extract key requirements
        key_requirements = self._extract_key_requirements(job_description)
        
        # Determine focus areas
        focus_areas = self._determine_focus_areas(persona_lower, job_lower, domain)
        
        return {
            'domain': domain,
            'persona_type': persona_type,
            'job_type': job_type,
            'key_requirements': key_requirements,
            'focus_areas': focus_areas,
            'priority_keywords': self._extract_priority_keywords(job_description)
        }

    def _determine_domain(self, persona: str, job: str) -> str:
        """Determine the domain based on persona and job description"""
        
        text = persona + " " + job
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return 'general'

    def _classify_persona(self, persona: str) -> str:
        """Classify the persona type"""
        
        for persona_type, keywords in self.persona_types.items():
            if any(keyword in persona for keyword in keywords):
                return persona_type
        
        return 'general'

    def _classify_job(self, job: str) -> str:
        """Classify the job type"""
        
        for job_type, keywords in self.job_types.items():
            if any(keyword in job for keyword in keywords):
                return job_type
        
        return 'general'

    def _extract_key_requirements(self, job_description: str) -> List[str]:
        """Extract key requirements from job description"""
        
        requirements = []
        words = job_description.lower().split()
        
        # Important action words
        action_words = [
            'analyze', 'review', 'summarize', 'compare', 'evaluate', 'assess',
            'identify', 'examine', 'investigate', 'explore', 'understand', 'learn'
        ]
        
        # Important content words
        content_words = [
            'trends', 'patterns', 'insights', 'findings', 'conclusions',
            'recommendations', 'methodology', 'results', 'performance'
        ]
        
        for word in words:
            if word in action_words or word in content_words:
                requirements.append(word)
        
        return list(set(requirements))

    def _determine_focus_areas(self, persona: str, job: str, domain: str) -> List[str]:
        """Determine focus areas based on persona, job, and domain"""
        
        focus_areas = []
        text = persona + " " + job
        
        # Domain-specific focus areas
        if domain == 'academic':
            focus_areas.extend(['research_methodology', 'literature_review', 'findings'])
        elif domain == 'business':
            focus_areas.extend(['financial_analysis', 'market_trends', 'strategy'])
        elif domain == 'technical':
            focus_areas.extend(['technical_details', 'implementation', 'architecture'])
        elif domain == 'medical':
            focus_areas.extend(['clinical_data', 'treatment_plans', 'patient_outcomes'])
        
        # General focus areas based on job type
        if 'analysis' in job:
            focus_areas.append('data_analysis')
        if 'summary' in job:
            focus_areas.append('executive_summary')
        if 'trends' in job:
            focus_areas.append('trend_analysis')
        if 'comparison' in job:
            focus_areas.append('comparative_analysis')
        
        return list(set(focus_areas))

    def _extract_priority_keywords(self, job_description: str) -> List[str]:
        """Extract priority keywords from job description"""
        
        # Extract nouns and important phrases
        words = job_description.lower().split()
        keywords = []
        
        for word in words:
            if len(word) > 3 and word not in ['with', 'from', 'that', 'this', 'they', 'have']:
                keywords.append(word)
        
        return keywords[:10]  # Top 10 keywords

    def _process_document(self, document: Dict[str, Any], persona_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document for analysis"""
        
        processed_doc = {
            'title': document.get('title', ''),
            'outline': document.get('outline', []),
            'path': document.get('path', ''),
            'sections': [],
            'relevance_scores': {}
        }
        
        # Process each heading in the outline
        for heading in document.get('outline', []):
            section = self._process_section(heading, persona_analysis)
            processed_doc['sections'].append(section)
        
        return processed_doc

    def _process_section(self, heading: Dict[str, Any], persona_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single section/heading"""
        
        section_text = heading.get('text', '')
        level = heading.get('level', 'H1')
        page = heading.get('page', 1)
        
        # Calculate relevance score
        relevance_score = self._calculate_section_relevance(section_text, level, persona_analysis)
        
        return {
            'text': section_text,
            'level': level,
            'page': page,
            'relevance_score': relevance_score,
            'domain_match': self._check_domain_match(section_text, persona_analysis),
            'keyword_matches': self._find_keyword_matches(section_text, persona_analysis)
        }

    def _calculate_section_relevance(self, section_text: str, level: str, 
                                   persona_analysis: Dict[str, Any]) -> float:
        """Calculate relevance score for a section"""
        
        score = 0.0
        text_lower = section_text.lower()
        
        # Domain matching
        domain = persona_analysis.get('domain', 'general')
        if domain in text_lower:
            score += 0.3
        
        # Priority keywords matching
        priority_keywords = persona_analysis.get('priority_keywords', [])
        keyword_matches = sum(1 for keyword in priority_keywords if keyword in text_lower)
        score += min(keyword_matches * 0.1, 0.3)
        
        # Focus areas matching
        focus_areas = persona_analysis.get('focus_areas', [])
        for area in focus_areas:
            if area.replace('_', ' ') in text_lower:
                score += 0.2
        
        # Level importance
        if level == 'H1':
            score += 0.2
        elif level == 'H2':
            score += 0.1
        
        # Content-based scoring
        if any(word in text_lower for word in ['introduction', 'conclusion', 'summary']):
            score += 0.1
        
        return min(score, 1.0)

    def _check_domain_match(self, section_text: str, persona_analysis: Dict[str, Any]) -> bool:
        """Check if section matches the domain"""
        
        domain = persona_analysis.get('domain', 'general')
        text_lower = section_text.lower()
        
        if domain == 'general':
            return True
        
        domain_keywords = self.domain_keywords.get(domain, [])
        return any(keyword in text_lower for keyword in domain_keywords)

    def _find_keyword_matches(self, section_text: str, persona_analysis: Dict[str, Any]) -> List[str]:
        """Find keyword matches in section text"""
        
        priority_keywords = persona_analysis.get('priority_keywords', [])
        text_lower = section_text.lower()
        
        matches = []
        for keyword in priority_keywords:
            if keyword in text_lower:
                matches.append(keyword)
        
        return matches

    def _extract_relevant_sections(self, processed_docs: List[Dict[str, Any]], 
                                 persona_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant sections based on persona analysis"""
        
        relevant_sections = []
        
        for doc in processed_docs:
            document_name = os.path.basename(doc.get('path', 'Unknown'))
            
            for section in doc.get('sections', []):
                if section['relevance_score'] > 0.3:  # Threshold for relevance
                    relevant_sections.append({
                        'document': document_name,
                        'page_number': section['page'],
                        'section_title': section['text'],
                        'importance_rank': section['relevance_score']
                    })
        
        # Sort by importance rank
        relevant_sections.sort(key=lambda x: x['importance_rank'], reverse=True)
        
        return relevant_sections

    def _analyze_subsections(self, processed_docs: List[Dict[str, Any]], 
                           persona_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze subsections with enhanced granularity"""
        
        subsection_analyses = []
        
        for doc in processed_docs:
            document_name = os.path.basename(doc.get('path', 'Unknown'))
            sections = doc.get('sections', [])
            
            # Group sections by hierarchy
            current_main_section = None
            subsections = []
            
            for section in sections:
                level = section['level']
                
                if level == 'H1':
                    # Save previous section analysis
                    if current_main_section and subsections:
                        subsection_analyses.append({
                            'document': document_name,
                            'section': current_main_section['text'],
                            'subsections': [sub['text'] for sub in subsections],
                            'refined_text': self._generate_refined_text(current_main_section, subsections),
                            'page_number': current_main_section['page']
                        })
                    
                    # Start new main section
                    current_main_section = section
                    subsections = []
                elif level in ['H2', 'H3'] and current_main_section:
                    subsections.append(section)
            
            # Add final section analysis
            if current_main_section and subsections:
                subsection_analyses.append({
                    'document': document_name,
                    'section': current_main_section['text'],
                    'subsections': [sub['text'] for sub in subsections],
                    'refined_text': self._generate_refined_text(current_main_section, subsections),
                    'page_number': current_main_section['page']
                })
        
        return subsection_analyses

    def _generate_refined_text(self, main_section: Dict[str, Any], 
                             subsections: List[Dict[str, Any]]) -> str:
        """Generate refined text for a section and its subsections"""
        
        refined_text = f"{main_section['text']}\n\n"
        
        for subsection in subsections:
            refined_text += f"• {subsection['text']}\n"
        
        return refined_text.strip() 