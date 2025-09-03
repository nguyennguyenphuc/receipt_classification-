#!/usr/bin/env python3
"""
Vietnamese Receipt Classification with GA-Optimized Voting Ensemble
Main entry point for training and inference
"""

from src.utils import load_model_for_prediction, predict_samples, print_prediction_results
from src.trainer import ReceiptClassificationTrainer
from config import Config
import os
import sys
import argparse

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)


def train_model():
    """Train the classification model"""
    print("🚀 Starting Receipt Classification Training...")

    trainer = ReceiptClassificationTrainer(Config)
    best_model, best_feature_type, results = trainer.run_full_pipeline()

    print("✅ Training completed!")
    return best_model, best_feature_type, results


def predict_mode(texts, model_path=None):
    """Prediction mode"""
    print("🔮 Starting prediction mode...")

    if model_path:
        model, feature_type, vectorizers, label_encoder = load_model_for_prediction(
            model_path)
    else:
        # Use latest model
        model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print("❌ No trained model found. Please train first!")
            return

        # Get latest model
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            print("❌ No .pkl model files found!")
            return

        latest_model = max([os.path.join(model_dir, f) for f in model_files],
                           key=os.path.getmtime)
        model, feature_type, vectorizers, label_encoder = load_model_for_prediction(
            latest_model)

    predictions, probabilities = predict_samples(
        texts, model, feature_type, vectorizers, label_encoder)

    print_prediction_results(texts, predictions, probabilities, label_encoder)


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Receipt Classification")
    parser.add_argument("--mode", choices=["train", "predict"], default="train",
                        help="Mode: train or predict")
    parser.add_argument("--texts", nargs="+",
                        help="Texts to predict (for predict mode)")
    parser.add_argument(
        "--model", help="Path to saved model (for predict mode)")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        if not args.texts:
            # Default test samples
            sample_texts = [
                "Hóa đơn thanh toán tại cửa hàng cà phê Feel Coffee với giá 25000 VND",
                "Mua sữa tươi Vinamilk tại siêu thị VinMart với giá 35000 VND",
                "Thanh toán tiền điện hàng tháng EVN 150000 VND",
                "Ăn phở bò tại quán phở Hà Nội giá 45000 VND"
            ]
        else:
            sample_texts = args.texts

        predict_mode(sample_texts, args.model)


if __name__ == "__main__":
    main()
