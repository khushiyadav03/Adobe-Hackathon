#!/usr/bin/env python3
"""
Multilingual Model Training and Inference for Adobe Hackathon
Handles multilingual transformer models for heading extraction and persona ranking
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from transformers import (
    DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    pipeline
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MultilingualHeadingDataset(Dataset):
    """Dataset for multilingual heading classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, language_labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_labels = language_labels or ['en'] * len(texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        language = self.language_labels[idx]
        
        # Add language token for multilingual models
        if language != 'en':
            text = f"[{language.upper()}] {text}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultilingualPersonaDataset(Dataset):
    """Dataset for multilingual persona-driven ranking"""
    
    def __init__(self, personas, sections, labels, tokenizer, max_length=512, language_labels=None):
        self.personas = personas
        self.sections = sections
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_labels = language_labels or ['en'] * len(personas)
    
    def __len__(self):
        return len(self.personas)
    
    def __getitem__(self, idx):
        persona = str(self.personas[idx])
        section = str(self.sections[idx])
        label = self.labels[idx]
        language = self.language_labels[idx]
        
        # Add language token for multilingual models
        if language != 'en':
            persona = f"[{language.upper()}] {persona}"
            section = f"[{language.upper()}] {section}"
        
        # Combine persona and section for training
        combined_text = f"Persona: {persona} Section: {section}"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class MultilingualHeadingClassifier:
    """Multilingual heading classification model"""
    
    def __init__(self, model_name='distilbert-base-multilingual-cased', num_labels=4):
        self.model_name = model_name
        self.num_labels = num_labels  # Title, H1, H2, H3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        self.model.to(self.device)
        self.label_map = {'Title': 0, 'H1': 1, 'H2': 2, 'H3': 3}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def train(self, train_texts, train_labels, val_texts, val_labels, 
              language_labels=None, batch_size=16, epochs=3, learning_rate=2e-5):
        """Train the multilingual heading classifier"""
        
        # Create datasets
        train_dataset = MultilingualHeadingDataset(
            train_texts, train_labels, self.tokenizer, language_labels=language_labels
        )
        val_dataset = MultilingualHeadingDataset(
            val_texts, val_labels, self.tokenizer, language_labels=language_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./multilingual_heading_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        self.model.save_pretrained("./models/multilingual_heading_model")
        self.tokenizer.save_pretrained("./models/multilingual_heading_model")
        
        logger.info("Multilingual heading classifier training completed")
    
    def predict(self, texts, language_labels=None):
        """Predict heading levels for given texts"""
        self.model.eval()
        predictions = []
        
        # Add language tokens if provided
        if language_labels:
            processed_texts = []
            for text, lang in zip(texts, language_labels):
                if lang != 'en':
                    processed_texts.append(f"[{lang.upper()}] {text}")
                else:
                    processed_texts.append(text)
        else:
            processed_texts = texts
        
        # Batch processing
        batch_size = 32
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i+batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                pred_labels = torch.argmax(logits, dim=1)
            
            # Convert to labels
            for pred in pred_labels.cpu().numpy():
                predictions.append(self.reverse_label_map[pred])
        
        return predictions
    
    def _compute_metrics(self, pred):
        """Compute metrics for evaluation"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

class MultilingualPersonaRanker:
    """Multilingual persona-driven ranking model"""
    
    def __init__(self, model_name='microsoft/Multilingual-MiniLM-L12-H384'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
    
    def train(self, personas, sections, labels, language_labels=None, 
              batch_size=16, epochs=3, learning_rate=2e-5):
        """Train the multilingual persona ranker"""
        
        # Create training examples
        train_examples = []
        for persona, section, label, lang in zip(personas, sections, labels, language_labels or ['en'] * len(personas)):
            # Add language token
            if lang != 'en':
                persona = f"[{lang.upper()}] {persona}"
                section = f"[{lang.upper()}] {section}"
            
            train_examples.append(InputExample(
                texts=[persona, section],
                label=float(label)
            ))
        
        # Split into train/validation
        train_examples, val_examples = train_test_split(train_examples, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=batch_size)
        
        # Loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Training
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )
        
        # Save the model
        self.model.save("./models/multilingual_persona_ranker")
        
        logger.info("Multilingual persona ranker training completed")
    
    def encode(self, texts, language_labels=None):
        """Encode texts to embeddings"""
        # Add language tokens if provided
        if language_labels:
            processed_texts = []
            for text, lang in zip(texts, language_labels):
                if lang != 'en':
                    processed_texts.append(f"[{lang.upper()}] {text}")
                else:
                    processed_texts.append(text)
        else:
            processed_texts = texts
        
        # Encode
        embeddings = self.model.encode(processed_texts, convert_to_tensor=True)
        return embeddings
    
    def compute_similarity(self, persona_embeddings, section_embeddings):
        """Compute cosine similarity between persona and section embeddings"""
        similarities = F.cosine_similarity(persona_embeddings, section_embeddings)
        return similarities.cpu().numpy()

class MultilingualModelManager:
    """Manager for multilingual models"""
    
    def __init__(self):
        self.heading_classifier = None
        self.persona_ranker = None
        self.language_detector = None
        
        # Model paths
        self.heading_model_path = "./models/multilingual_heading_model"
        self.persona_model_path = "./models/multilingual_persona_ranker"
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists(self.heading_model_path):
                self.heading_classifier = MultilingualHeadingClassifier()
                self.heading_classifier.model = DistilBertForSequenceClassification.from_pretrained(self.heading_model_path)
                self.heading_classifier.tokenizer = DistilBertTokenizer.from_pretrained(self.heading_model_path)
                self.heading_classifier.model.to(self.heading_classifier.device)
                logger.info("Loaded multilingual heading classifier")
        except Exception as e:
            logger.warning(f"Could not load heading classifier: {e}")
        
        try:
            if os.path.exists(self.persona_model_path):
                self.persona_ranker = MultilingualPersonaRanker()
                self.persona_ranker.model = SentenceTransformer(self.persona_model_path)
                self.persona_ranker.model.to(self.persona_ranker.device)
                logger.info("Loaded multilingual persona ranker")
        except Exception as e:
            logger.warning(f"Could not load persona ranker: {e}")
    
    def train_heading_classifier(self, data_path):
        """Train the heading classifier with multilingual data"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare data
        texts = df['text'].tolist()
        labels = [self.heading_classifier.label_map[label] for label in df['label'].tolist()]
        language_labels = df.get('language', ['en'] * len(texts)).tolist()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels, train_langs, val_langs = train_test_split(
            texts, labels, language_labels, test_size=0.2, random_state=42
        )
        
        # Initialize and train
        self.heading_classifier = MultilingualHeadingClassifier()
        self.heading_classifier.train(
            train_texts, train_labels, val_texts, val_labels,
            language_labels=train_langs, epochs=3
        )
    
    def train_persona_ranker(self, data_path):
        """Train the persona ranker with multilingual data"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Prepare data
        personas = df['persona'].tolist()
        sections = df['section'].tolist()
        labels = df['relevance_score'].tolist()
        language_labels = df.get('language', ['en'] * len(personas)).tolist()
        
        # Initialize and train
        self.persona_ranker = MultilingualPersonaRanker()
        self.persona_ranker.train(
            personas, sections, labels,
            language_labels=language_labels, epochs=3
        )
    
    def predict_headings(self, texts, language_labels=None):
        """Predict heading levels for texts"""
        if self.heading_classifier is None:
            raise ValueError("Heading classifier not loaded. Train or load the model first.")
        
        return self.heading_classifier.predict(texts, language_labels)
    
    def rank_sections(self, persona, sections, language_labels=None):
        """Rank sections based on persona relevance"""
        if self.persona_ranker is None:
            raise ValueError("Persona ranker not loaded. Train or load the model first.")
        
        # Encode persona and sections
        persona_emb = self.persona_ranker.encode([persona], [language_labels[0] if language_labels else 'en'])
        section_embs = self.persona_ranker.encode(sections, language_labels or ['en'] * len(sections))
        
        # Compute similarities
        similarities = self.persona_ranker.compute_similarity(persona_emb, section_embs)
        
        # Return ranked results
        ranked_indices = np.argsort(similarities)[::-1]
        return [(i, similarities[i]) for i in ranked_indices]

# Global model manager instance
model_manager = MultilingualModelManager() 