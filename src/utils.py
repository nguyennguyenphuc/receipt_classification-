import pickle
import re
import numpy as np
from typing import Dict, Any


def preprocess_text(text: str) -> str:
    """Simple text preprocessing (same as training)"""
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(
        r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_model(model_path: str = "models/receipt_classifier.pkl") -> Dict[str, Any]:
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_category(text: str, model_path: str = "models/receipt_classifier.pkl") -> Dict[str, Any]:
    """Predict category for new receipt text"""
    # Load model
    model_data = load_model(model_path)

    # Preprocess text
    processed_text = preprocess_text(text)

    # Extract features based on best feature type
    feature_type = model_data['best_feature_type']

    if feature_type == 'bow':
        features = model_data['feature_extractor'].bow_vectorizer.transform(
            [processed_text]).toarray()
    elif feature_type == 'tfidf':
        features = model_data['feature_extractor'].tfidf_vectorizer.transform(
            [processed_text]).toarray()
    else:  # embeddings
        features = model_data['feature_extractor'].embedding_model.encode([
                                                                          processed_text])

    # Make prediction
    prediction_id = model_data['model'].predict(features)[0]
    probabilities = model_data['model'].predict_proba(features)[0]

    # Get category name
    category = model_data['label_mapping']['id_to_label'][prediction_id]
    confidence = probabilities.max()

    # Get top 3 predictions
    top_3_ids = np.argsort(probabilities)[-3:][::-1]
    top_3 = [
        (model_data['label_mapping']['id_to_label'][i], probabilities[i])
        for i in top_3_ids
    ]

    return {
        'predicted_category': category,
        'confidence': confidence,
        'top_3_predictions': top_3,
        'processed_text': processed_text
    }


def batch_predict(texts: list, model_path: str = "models/receipt_classifier.pkl") -> list:
    """Predict categories for multiple texts"""
    results = []
    for text in texts:
        try:
            result = predict_category(text, model_path)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e), 'text': text})

    return results
