#!/usr/bin/env python3
"""
Training Scripts for Multilingual Models
Train multilingual heading extraction and persona ranking models
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel

# Import our modules
from src.multilingual_enhancement import (
    MultilingualLanguageDetector, MultilingualHeadingPatterns,
    MultilingualTextNormalizer, MultilingualFeatureExtractor
)
from src.multilingual_models import (
    MultilingualModelManager, MultilingualHeadingClassifier,
    MultilingualPersonaRanker
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualDataGenerator:
    """Generate multilingual training data"""
    
    def __init__(self):
        self.language_detector = MultilingualLanguageDetector()
        self.heading_patterns = MultilingualHeadingPatterns()
        self.text_normalizer = MultilingualTextNormalizer()
        self.feature_extractor = MultilingualFeatureExtractor()
    
    def generate_multilingual_heading_data(self, output_path: str = "multilingual_heading_data.csv"):
        """Generate multilingual heading classification data"""
        
        # English heading examples
        english_headings = [
            ("Introduction", "H1"),
            ("Chapter 1: Background", "Title"),
            ("1.1 Research Objectives", "H1"),
            ("1.2 Methodology", "H1"),
            ("1.2.1 Data Collection", "H2"),
            ("1.2.2 Analysis Methods", "H2"),
            ("2. Literature Review", "H1"),
            ("2.1 Previous Studies", "H2"),
            ("2.1.1 Early Research", "H3"),
            ("2.1.2 Recent Developments", "H3"),
            ("3. Results", "H1"),
            ("3.1 Statistical Analysis", "H2"),
            ("4. Discussion", "H1"),
            ("5. Conclusion", "H1"),
            ("Appendix A: Data Tables", "H1"),
            ("References", "H1"),
            ("Abstract", "H1"),
            ("Executive Summary", "H1"),
            ("Table of Contents", "H1"),
            ("Acknowledgments", "H1")
        ]
        
        # Japanese heading examples
        japanese_headings = [
            ("はじめに", "H1"),
            ("第1章 背景", "Title"),
            ("1.1 研究目的", "H1"),
            ("1.2 方法論", "H1"),
            ("1.2.1 データ収集", "H2"),
            ("1.2.2 分析方法", "H2"),
            ("2章 文献レビュー", "H1"),
            ("2.1 先行研究", "H2"),
            ("2.1.1 初期研究", "H3"),
            ("2.1.2 最近の展開", "H3"),
            ("3章 結果", "H1"),
            ("3.1 統計分析", "H2"),
            ("4章 考察", "H1"),
            ("5章 結論", "H1"),
            ("付録A データ表", "H1"),
            ("参考文献", "H1"),
            ("要約", "H1"),
            ("エグゼクティブサマリー", "H1"),
            ("目次", "H1"),
            ("謝辞", "H1")
        ]
        
        # Chinese heading examples
        chinese_headings = [
            ("引言", "H1"),
            ("第一章 背景", "Title"),
            ("1.1 研究目标", "H1"),
            ("1.2 方法论", "H1"),
            ("1.2.1 数据收集", "H2"),
            ("1.2.2 分析方法", "H2"),
            ("第二章 文献综述", "H1"),
            ("2.1 先前研究", "H2"),
            ("2.1.1 早期研究", "H3"),
            ("2.1.2 最新发展", "H3"),
            ("第三章 结果", "H1"),
            ("3.1 统计分析", "H2"),
            ("第四章 讨论", "H1"),
            ("第五章 结论", "H1"),
            ("附录A 数据表", "H1"),
            ("参考文献", "H1"),
            ("摘要", "H1"),
            ("执行摘要", "H1"),
            ("目录", "H1"),
            ("致谢", "H1")
        ]
        
        # Korean heading examples
        korean_headings = [
            ("서론", "H1"),
            ("제1장 배경", "Title"),
            ("1.1 연구 목적", "H1"),
            ("1.2 방법론", "H1"),
            ("1.2.1 데이터 수집", "H2"),
            ("1.2.2 분석 방법", "H2"),
            ("제2장 문헌 고찰", "H1"),
            ("2.1 선행 연구", "H2"),
            ("2.1.1 초기 연구", "H3"),
            ("2.1.2 최근 발전", "H3"),
            ("제3장 결과", "H1"),
            ("3.1 통계 분석", "H2"),
            ("제4장 논의", "H1"),
            ("제5장 결론", "H1"),
            ("부록A 데이터 표", "H1"),
            ("참고문헌", "H1"),
            ("초록", "H1"),
            ("실행 요약", "H1"),
            ("목차", "H1"),
            ("감사의 글", "H1")
        ]
        
        # Arabic heading examples
        arabic_headings = [
            ("مقدمة", "H1"),
            ("الفصل الأول: الخلفية", "Title"),
            ("1.1 أهداف البحث", "H1"),
            ("1.2 المنهجية", "H1"),
            ("1.2.1 جمع البيانات", "H2"),
            ("1.2.2 طرق التحليل", "H2"),
            ("الفصل الثاني: مراجعة الأدبيات", "H1"),
            ("2.1 الدراسات السابقة", "H2"),
            ("2.1.1 الأبحاث المبكرة", "H3"),
            ("2.1.2 التطورات الحديثة", "H3"),
            ("الفصل الثالث: النتائج", "H1"),
            ("3.1 التحليل الإحصائي", "H2"),
            ("الفصل الرابع: المناقشة", "H1"),
            ("الفصل الخامس: الخاتمة", "H1"),
            ("الملحق أ: جداول البيانات", "H1"),
            ("المراجع", "H1"),
            ("الملخص", "H1"),
            ("الملخص التنفيذي", "H1"),
            ("جدول المحتويات", "H1"),
            ("شكر وتقدير", "H1")
        ]
        
        # Hindi heading examples
        hindi_headings = [
            ("परिचय", "H1"),
            ("अध्याय 1: पृष्ठभूमि", "Title"),
            ("1.1 शोध उद्देश्य", "H1"),
            ("1.2 पद्धति", "H1"),
            ("1.2.1 डेटा संग्रह", "H2"),
            ("1.2.2 विश्लेषण विधियां", "H2"),
            ("अध्याय 2: साहित्य समीक्षा", "H1"),
            ("2.1 पूर्व अध्ययन", "H2"),
            ("2.1.1 प्रारंभिक शोध", "H3"),
            ("2.1.2 हाल के विकास", "H3"),
            ("अध्याय 3: परिणाम", "H1"),
            ("3.1 सांख्यिकीय विश्लेषण", "H2"),
            ("अध्याय 4: चर्चा", "H1"),
            ("अध्याय 5: निष्कर्ष", "H1"),
            ("परिशिष्ट ए: डेटा तालिकाएं", "H1"),
            ("संदर्भ", "H1"),
            ("सारांश", "H1"),
            ("कार्यकारी सारांश", "H1"),
            ("विषय सूची", "H1"),
            ("आभार", "H1")
        ]
        
        # Spanish heading examples
        spanish_headings = [
            ("Introducción", "H1"),
            ("Capítulo 1: Antecedentes", "Title"),
            ("1.1 Objetivos de la Investigación", "H1"),
            ("1.2 Metodología", "H1"),
            ("1.2.1 Recolección de Datos", "H2"),
            ("1.2.2 Métodos de Análisis", "H2"),
            ("Capítulo 2: Revisión de Literatura", "H1"),
            ("2.1 Estudios Previos", "H2"),
            ("2.1.1 Investigación Temprana", "H3"),
            ("2.1.2 Desarrollos Recientes", "H3"),
            ("Capítulo 3: Resultados", "H1"),
            ("3.1 Análisis Estadístico", "H2"),
            ("Capítulo 4: Discusión", "H1"),
            ("Capítulo 5: Conclusiones", "H1"),
            ("Apéndice A: Tablas de Datos", "H1"),
            ("Referencias", "H1"),
            ("Resumen", "H1"),
            ("Resumen Ejecutivo", "H1"),
            ("Tabla de Contenidos", "H1"),
            ("Agradecimientos", "H1")
        ]
        
        # French heading examples
        french_headings = [
            ("Introduction", "H1"),
            ("Chapitre 1: Contexte", "Title"),
            ("1.1 Objectifs de la Recherche", "H1"),
            ("1.2 Méthodologie", "H1"),
            ("1.2.1 Collecte de Données", "H2"),
            ("1.2.2 Méthodes d'Analyse", "H2"),
            ("Chapitre 2: Revue de Littérature", "H1"),
            ("2.1 Études Précédentes", "H2"),
            ("2.1.1 Recherche Précoce", "H3"),
            ("2.1.2 Développements Récents", "H3"),
            ("Chapitre 3: Résultats", "H1"),
            ("3.1 Analyse Statistique", "H2"),
            ("Chapitre 4: Discussion", "H1"),
            ("Chapitre 5: Conclusion", "H1"),
            ("Annexe A: Tableaux de Données", "H1"),
            ("Références", "H1"),
            ("Résumé", "H1"),
            ("Résumé Exécutif", "H1"),
            ("Table des Matières", "H1"),
            ("Remerciements", "H1")
        ]
        
        # German heading examples
        german_headings = [
            ("Einleitung", "H1"),
            ("Kapitel 1: Hintergrund", "Title"),
            ("1.1 Forschungsziele", "H1"),
            ("1.2 Methodik", "H1"),
            ("1.2.1 Datenerhebung", "H2"),
            ("1.2.2 Analysemethoden", "H2"),
            ("Kapitel 2: Literaturübersicht", "H1"),
            ("2.1 Frühere Studien", "H2"),
            ("2.1.1 Frühe Forschung", "H3"),
            ("2.1.2 Aktuelle Entwicklungen", "H3"),
            ("Kapitel 3: Ergebnisse", "H1"),
            ("3.1 Statistische Analyse", "H2"),
            ("Kapitel 4: Diskussion", "H1"),
            ("Kapitel 5: Schlussfolgerung", "H1"),
            ("Anhang A: Datentabellen", "H1"),
            ("Literaturverzeichnis", "H1"),
            ("Zusammenfassung", "H1"),
            ("Management Summary", "H1"),
            ("Inhaltsverzeichnis", "H1"),
            ("Danksagung", "H1")
        ]
        
        # Combine all languages
        all_headings = {
            'en': english_headings,
            'ja': japanese_headings,
            'zh': chinese_headings,
            'ko': korean_headings,
            'ar': arabic_headings,
            'hi': hindi_headings,
            'es': spanish_headings,
            'fr': french_headings,
            'de': german_headings
        }
        
        # Create dataset
        data = []
        for language, headings in all_headings.items():
            for text, label in headings:
                # Add positive examples
                data.append({
                    'text': text,
                    'label': label,
                    'language': language,
                    'is_heading': 1
                })
                
                # Add negative examples (non-heading text)
                data.append({
                    'text': f"This is a regular paragraph about {text.lower()} that contains normal text content.",
                    'label': 'H3',  # Default label for non-headings
                    'language': language,
                    'is_heading': 0
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated multilingual heading data: {len(df)} samples")
        
        return output_path
    
    def generate_multilingual_persona_data(self, output_path: str = "multilingual_persona_data.csv"):
        """Generate multilingual persona ranking data"""
        
        # Define personas and sections for different languages
        persona_section_pairs = {
            'en': [
                ("Travel enthusiast planning a trip to Europe", "Best cities to visit in Europe", 0.9),
                ("Travel enthusiast planning a trip to Europe", "European cuisine recommendations", 0.8),
                ("Travel enthusiast planning a trip to Europe", "European history overview", 0.7),
                ("Business analyst researching market trends", "Market analysis methodology", 0.9),
                ("Business analyst researching market trends", "Statistical analysis techniques", 0.8),
                ("Business analyst researching market trends", "Travel recommendations", 0.2),
                ("Student studying computer science", "Programming fundamentals", 0.9),
                ("Student studying computer science", "Algorithm complexity analysis", 0.8),
                ("Student studying computer science", "Cooking recipes", 0.1),
            ],
            'ja': [
                ("ヨーロッパ旅行を計画している旅行愛好家", "ヨーロッパの主要都市ガイド", 0.9),
                ("ヨーロッパ旅行を計画している旅行愛好家", "ヨーロッパ料理のレコメンデーション", 0.8),
                ("ヨーロッパ旅行を計画している旅行愛好家", "ヨーロッパの歴史概要", 0.7),
                ("市場動向を調査しているビジネスアナリスト", "市場分析手法", 0.9),
                ("市場動向を調査しているビジネスアナリスト", "統計分析技術", 0.8),
                ("市場動向を調査しているビジネスアナリスト", "旅行レコメンデーション", 0.2),
                ("コンピュータサイエンスを学ぶ学生", "プログラミング基礎", 0.9),
                ("コンピュータサイエンスを学ぶ学生", "アルゴリズム複雑性分析", 0.8),
                ("コンピュータサイエンスを学ぶ学生", "料理レシピ", 0.1),
            ],
            'zh': [
                ("计划欧洲旅行的旅行爱好者", "欧洲主要城市指南", 0.9),
                ("计划欧洲旅行的旅行爱好者", "欧洲美食推荐", 0.8),
                ("计划欧洲旅行的旅行爱好者", "欧洲历史概述", 0.7),
                ("研究市场趋势的商业分析师", "市场分析方法", 0.9),
                ("研究市场趋势的商业分析师", "统计分析技术", 0.8),
                ("研究市场趋势的商业分析师", "旅行推荐", 0.2),
                ("学习计算机科学的学生", "编程基础", 0.9),
                ("学习计算机科学的学生", "算法复杂度分析", 0.8),
                ("学习计算机科学的学生", "烹饪食谱", 0.1),
            ],
            'ko': [
                ("유럽 여행을 계획하는 여행 애호가", "유럽 주요 도시 가이드", 0.9),
                ("유럽 여행을 계획하는 여행 애호가", "유럽 요리 추천", 0.8),
                ("유럽 여행을 계획하는 여행 애호가", "유럽 역사 개요", 0.7),
                ("시장 동향을 조사하는 비즈니스 애널리스트", "시장 분석 방법론", 0.9),
                ("시장 동향을 조사하는 비즈니스 애널리스트", "통계 분석 기술", 0.8),
                ("시장 동향을 조사하는 비즈니스 애널리스트", "여행 추천", 0.2),
                ("컴퓨터 과학을 공부하는 학생", "프로그래밍 기초", 0.9),
                ("컴퓨터 과학을 공부하는 학생", "알고리즘 복잡성 분석", 0.8),
                ("컴퓨터 과학을 공부하는 학생", "요리 레시피", 0.1),
            ],
            'ar': [
                ("عشاق السفر الذين يخططون لرحلة إلى أوروبا", "دليل المدن الرئيسية في أوروبا", 0.9),
                ("عشاق السفر الذين يخططون لرحلة إلى أوروبا", "توصيات المطبخ الأوروبي", 0.8),
                ("عشاق السفر الذين يخططون لرحلة إلى أوروبا", "نظرة عامة على التاريخ الأوروبي", 0.7),
                ("محلل أعمال يبحث في اتجاهات السوق", "منهجية تحليل السوق", 0.9),
                ("محلل أعمال يبحث في اتجاهات السوق", "تقنيات التحليل الإحصائي", 0.8),
                ("محلل أعمال يبحث في اتجاهات السوق", "توصيات السفر", 0.2),
                ("طالب يدرس علوم الحاسوب", "أساسيات البرمجة", 0.9),
                ("طالب يدرس علوم الحاسوب", "تحليل تعقيد الخوارزميات", 0.8),
                ("طالب يدرس علوم الحاسوب", "وصفات الطبخ", 0.1),
            ],
            'hi': [
                ("यूरोप की यात्रा की योजना बना रहे यात्रा प्रेमी", "यूरोप के प्रमुख शहरों की गाइड", 0.9),
                ("यूरोप की यात्रा की योजना बना रहे यात्रा प्रेमी", "यूरोपीय व्यंजन सिफारिशें", 0.8),
                ("यूरोप की यात्रा की योजना बना रहे यात्रा प्रेमी", "यूरोपीय इतिहास का अवलोकन", 0.7),
                ("बाजार के रुझानों पर शोध कर रहे व्यवसाय विश्लेषक", "बाजार विश्लेषण पद्धति", 0.9),
                ("बाजार के रुझानों पर शोध कर रहे व्यवसाय विश्लेषक", "सांख्यिकीय विश्लेषण तकनीक", 0.8),
                ("बाजार के रुझानों पर शोध कर रहे व्यवसाय विश्लेषक", "यात्रा सिफारिशें", 0.2),
                ("कंप्यूटर विज्ञान पढ़ रहे छात्र", "प्रोग्रामिंग मूल बातें", 0.9),
                ("कंप्यूटर विज्ञान पढ़ रहे छात्र", "एल्गोरिथम जटिलता विश्लेषण", 0.8),
                ("कंप्यूटर विज्ञान पढ़ रहे छात्र", "खाना पकाने की रेसिपी", 0.1),
            ],
            'es': [
                ("Entusiasta de viajes que planea un viaje a Europa", "Guía de las mejores ciudades de Europa", 0.9),
                ("Entusiasta de viajes que planea un viaje a Europa", "Recomendaciones de cocina europea", 0.8),
                ("Entusiasta de viajes que planea un viaje a Europa", "Resumen de la historia europea", 0.7),
                ("Analista de negocios investigando tendencias del mercado", "Metodología de análisis de mercado", 0.9),
                ("Analista de negocios investigando tendencias del mercado", "Técnicas de análisis estadístico", 0.8),
                ("Analista de negocios investigando tendencias del mercado", "Recomendaciones de viaje", 0.2),
                ("Estudiante estudiando ciencias de la computación", "Fundamentos de programación", 0.9),
                ("Estudiante estudiando ciencias de la computación", "Análisis de complejidad de algoritmos", 0.8),
                ("Estudiante estudiando ciencias de la computación", "Recetas de cocina", 0.1),
            ],
            'fr': [
                ("Passionné de voyage planifiant un voyage en Europe", "Guide des meilleures villes d'Europe", 0.9),
                ("Passionné de voyage planifiant un voyage en Europe", "Recommandations de cuisine européenne", 0.8),
                ("Passionné de voyage planifiant un voyage en Europe", "Aperçu de l'histoire européenne", 0.7),
                ("Analyste commercial recherchant les tendances du marché", "Méthodologie d'analyse de marché", 0.9),
                ("Analyste commercial recherchant les tendances du marché", "Techniques d'analyse statistique", 0.8),
                ("Analyste commercial recherchant les tendances du marché", "Recommandations de voyage", 0.2),
                ("Étudiant étudiant l'informatique", "Fondamentaux de la programmation", 0.9),
                ("Étudiant étudiant l'informatique", "Analyse de la complexité des algorithmes", 0.8),
                ("Étudiant étudiant l'informatique", "Recettes de cuisine", 0.1),
            ],
            'de': [
                ("Reisebegeisterter, der eine Europareise plant", "Reiseführer zu den besten Städten Europas", 0.9),
                ("Reisebegeisterter, der eine Europareise plant", "Empfehlungen für europäische Küche", 0.8),
                ("Reisebegeisterter, der eine Europareise plant", "Überblick über die europäische Geschichte", 0.7),
                ("Business Analyst, der Markttrends erforscht", "Methodik der Marktanalyse", 0.9),
                ("Business Analyst, der Markttrends erforscht", "Statistische Analysetechniken", 0.8),
                ("Business Analyst, der Markttrends erforscht", "Reiseempfehlungen", 0.2),
                ("Student, der Informatik studiert", "Programmiergrundlagen", 0.9),
                ("Student, der Informatik studiert", "Algorithmus-Komplexitätsanalyse", 0.8),
                ("Student, der Informatik studiert", "Kochrezepte", 0.1),
            ]
        }
        
        # Create dataset
        data = []
        for language, pairs in persona_section_pairs.items():
            for persona, section, relevance in pairs:
                data.append({
                    'persona': persona,
                    'section': section,
                    'relevance_score': relevance,
                    'language': language
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated multilingual persona data: {len(df)} samples")
        
        return output_path

def main():
    """Main training function"""
    logger.info("Starting multilingual model training...")
    
    # Create data generator
    data_generator = MultilingualDataGenerator()
    
    # Generate training data
    logger.info("Generating multilingual training data...")
    heading_data_path = data_generator.generate_multilingual_heading_data()
    persona_data_path = data_generator.generate_multilingual_persona_data()
    
    # Initialize model manager
    model_manager = MultilingualModelManager()
    
    # Train heading classifier
    logger.info("Training multilingual heading classifier...")
    model_manager.train_heading_classifier(heading_data_path)
    
    # Train persona ranker
    logger.info("Training multilingual persona ranker...")
    model_manager.train_persona_ranker(persona_data_path)
    
    logger.info("Multilingual model training completed!")

if __name__ == "__main__":
    main() 