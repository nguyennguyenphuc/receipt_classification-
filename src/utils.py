"""Utility functions for prediction and model loading"""

import os
import sys
import joblib
import numpy as np
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Any, Dict

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))


def load_model_for_prediction(model_path: str):
    """Load trained model for prediction"""
    print(f"📥 Loading model from: {model_path}")

    try:
        model_data = joblib.load(model_path)

        model = model_data['model']
        feature_type = model_data['feature_type']
        vectorizers = model_data['vectorizers']
        label_encoder = model_data['label_encoder']

        print(f"✅ Model loaded successfully!")
        print(f"   Feature type: {feature_type}")
        print(f"   Classes: {len(label_encoder.classes_)} classes")
        print(f"   Accuracy: {model_data.get('accuracy', 'N/A'):.4f}")

        return model, feature_type, vectorizers, label_encoder

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def preprocess_text_for_prediction(text: str) -> str:
    """Preprocess text for prediction (same as training)"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def predict_samples(texts: List[str], model, feature_type: str,
                    vectorizers: Dict, label_encoder) -> Tuple[List[str], np.ndarray]:
    """Predict for new text samples"""
    print(
        f"🔮 Predicting {len(texts)} samples using {feature_type} features...")

    # Preprocess texts
    processed_texts = [preprocess_text_for_prediction(text) for text in texts]

    # Transform features
    if feature_type == 'bow':
        X_features = vectorizers['bow'].transform(processed_texts).toarray()
    elif feature_type == 'tfidf':
        X_features = vectorizers['tfidf'].transform(processed_texts).toarray()
    else:  # embeddings
        X_features = vectorizers['embeddings'].encode(processed_texts)

    # Predict
    predictions = model.predict(X_features)
    probabilities = model.predict_proba(X_features)

    # Convert back to labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels, probabilities


def get_latest_model(models_dir: str = "models") -> str:
    """Get path to the latest trained model"""
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_files = list(models_path.glob("*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No model files found in: {models_dir}")

    # Get latest model by modification time
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

    return str(latest_model)


def print_prediction_results(texts: List[str], predictions: List[str],
                             probabilities: np.ndarray, label_encoder):
    """Pretty print prediction results"""
    print("\n🎯 PREDICTION RESULTS")
    print("="*70)

    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities), 1):
        print(f"\n📝 Sample {i}:")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   ➡️  Predicted: {pred}")
        print(f"   🎯 Confidence: {max(prob):.3f}")

        # Top 3 predictions
        top_3_indices = np.argsort(prob)[-3:][::-1]
        print(f"   📊 Top 3 predictions:")
        for rank, idx in enumerate(top_3_indices, 1):
            label = label_encoder.classes_[idx]
            confidence = prob[idx]
            print(f"      {rank}. {label}: {confidence:.3f}")
