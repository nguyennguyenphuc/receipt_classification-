import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple


class FeatureExtractor:
    """Extract BoW, TF-IDF, and embedding features"""

    def __init__(self, config):
        self.config = config
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.embedding_model = None

    def extract_all_features(self, X_train: List[str], X_test: List[str]) -> Dict[str, Tuple]:
        """Extract all feature types"""
        print("\n🔧 Extracting features...")

        features = {}

        # 1. Bag of Words
        print("   📝 Bag of Words...")
        features['bow'] = self._extract_bow(X_train, X_test)

        # 2. TF-IDF
        print("   📊 TF-IDF...")
        features['tfidf'] = self._extract_tfidf(X_train, X_test)

        # 3. Embeddings
        print("   🤖 Sentence Embeddings...")
        features['embeddings'] = self._extract_embeddings(X_train, X_test)

        print("✅ Feature extraction completed!")
        return features

    def _extract_bow(self, X_train: List[str], X_test: List[str]) -> Tuple:
        """Extract Bag of Words features"""
        self.bow_vectorizer = CountVectorizer(
            max_features=self.config.MAX_FEATURES,
            min_df=2,
            max_df=0.8,
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
            min_df=2,
            max_df=0.8,
            ngram_range=self.config.NGRAM_RANGE
        )

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test).toarray()

        print(f"      Shape: {X_train_tfidf.shape}")
        return (X_train_tfidf, X_test_tfidf)

    def _extract_embeddings(self, X_train: List[str], X_test: List[str]) -> Tuple:
        """Extract sentence embeddings"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        X_train_embeddings = self.embedding_model.encode(
            X_train, show_progress_bar=True)
        X_test_embeddings = self.embedding_model.encode(
            X_test, show_progress_bar=True)

        print(f"      Shape: {X_train_embeddings.shape}")
        return (X_train_embeddings, X_test_embeddings)
