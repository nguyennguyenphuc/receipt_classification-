"""Feature extraction utilities - BoW, TF-IDF, Embeddings từ code gốc bill_classify"""

import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.embedding_model = None

    def extract_features(self, X_train: List[str], X_test: List[str]) -> Dict[str, Tuple]:
        """Extract all types of features from code gốc bill_classify"""
        print("🚀 Extracting features...")
        features = {}

        # 1. Bag of Words - từ code gốc
        print("   📝 Bag of Words...")
        features['bow'] = self._extract_bow(X_train, X_test)

        # 2. TF-IDF - từ code gốc
        print("   📊 TF-IDF...")
        features['tfidf'] = self._extract_tfidf(X_train, X_test)

        # 3. Embeddings - từ code gốc
        print("   🤖 Sentence Embeddings...")
        features['embeddings'] = self._extract_embeddings(X_train, X_test)

        print("✅ Feature extraction completed!")
        return features

    def _extract_bow(self, X_train: List[str], X_test: List[str]) -> Tuple:
        """Extract Bag of Words features"""
        self.bow_vectorizer = CountVectorizer(
            max_features=self.config.MAX_FEATURES,
            min_df=self.config.MIN_DF,
            max_df=self.config.MAX_DF,
            ngram_range=self.config.NGRAM_RANGE
        )

        X_train_bow = self.bow_vectorizer.fit_transform(X_train).toarray()
        X_test_bow = self.bow_vectorizer.transform(X_test).toarray()

        print(f"      Shape: {X_train_bow.shape}")
        return (X_train_bow, X_test_bow)

    def _extract_tfidf(self, X_train: List[str], X_test: List[str]) -> Tuple:
        """Extract TF-IDF features"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.MAX_FEATURES,
            min_df=self.config.MIN_DF,
            max_df=self.config.MAX_DF,
            ngram_range=self.config.NGRAM_RANGE
        )

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test).toarray()

        print(f"      Shape: {X_train_tfidf.shape}")
        return (X_train_tfidf, X_test_tfidf)

    def _extract_embeddings(self, X_train: List[str], X_test: List[str]) -> Tuple:
        """Extract sentence embeddings"""
        print("      Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)

        print("      Encoding training data...")
        X_train_embeddings = self.embedding_model.encode(
            X_train, show_progress_bar=True)
        print("      Encoding test data...")
        X_test_embeddings = self.embedding_model.encode(
            X_test, show_progress_bar=True)

        print(f"      Shape: {X_train_embeddings.shape}")
        return (X_train_embeddings, X_test_embeddings)

    def get_vectorizers(self) -> Dict[str, Any]:
        """Get trained vectorizers"""
        return {
            'bow': self.bow_vectorizer,
            'tfidf': self.tfidf_vectorizer,
            'embeddings': self.embedding_model
        }
