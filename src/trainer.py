"""Training pipeline module"""

from models import GAVotingClassifier
from feature_extractor import FeatureExtractor
from data_loader import DataLoader
import os
import sys
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))


class ReceiptClassificationTrainer:
    """Main training pipeline"""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.ga_classifier = GAVotingClassifier(config)

        # Create output directories
        Path(config.MODELS_DIR).mkdir(exist_ok=True)
        Path(config.OUTPUTS_DIR).mkdir(exist_ok=True)

    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING RECEIPT CLASSIFICATION PIPELINE")
        print("="*70)

        # 1. Load and prepare data
        print("\n📊 STEP 1: DATA LOADING")
        print("-"*50)
        self.data_loader.load_data()
        X, y = self.data_loader.prepare_data()
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)

        print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")

        # 2. Extract features
        print("\n🔬 STEP 2: FEATURE EXTRACTION")
        print("-"*50)
        features = self.feature_extractor.extract_features(X_train, X_test)

        # 3. GA optimization
        print("\n🧬 STEP 3: GA OPTIMIZATION")
        print("-"*50)
        best_individual, logbook = self.ga_classifier.optimize(
            features, y_train)

        # 4. Train best model
        print("\n🏋️ STEP 4: TRAINING BEST MODEL")
        print("-"*50)
        best_model, best_feature_type = self.ga_classifier.create_best_model(
            features)

        # Get training data for best feature type
        X_train_feat, X_test_feat = features[best_feature_type]

        # Train the model
        print("🏋️ Training final model...")
        best_model.fit(X_train_feat, y_train)

        # 5. Evaluation
        print("\n📊 STEP 5: EVALUATION")
        print("-"*50)
        y_pred = best_model.predict(X_test_feat)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"🎯 Test Accuracy: {accuracy:.4f}")

        # Classification report
        class_names = self.data_loader.label_encoder.classes_
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True)
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # 6. Save model and results
        print("\n💾 STEP 6: SAVING RESULTS")
        print("-"*50)
        results = self._save_results(
            best_model, best_feature_type, accuracy, report, logbook)

        print("="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")

        return best_model, best_feature_type, results

    def _save_results(self, model, feature_type, accuracy, report, logbook):
        """Save model and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model and associated objects
        model_data = {
            'model': model,
            'feature_type': feature_type,
            'vectorizers': self.feature_extractor.get_vectorizers(),
            'label_encoder': self.data_loader.label_encoder,
            'config': self.config,
            'accuracy': accuracy,
            'timestamp': timestamp
        }

        model_path = Path(self.config.MODELS_DIR) / \
            f"receipt_classifier_{timestamp}.pkl"
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved: {model_path}")

        # Save results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_type': feature_type,
            'ga_logbook': logbook,
            'timestamp': timestamp
        }

        results_path = Path(self.config.OUTPUTS_DIR) / \
            f"results_{timestamp}.pkl"
        joblib.dump(results, results_path)
        print(f"📊 Results saved: {results_path}")

        # Save GA evolution plot
        self._plot_ga_evolution(logbook, timestamp)

        return results

    def _plot_ga_evolution(self, logbook, timestamp):
        """Plot GA evolution"""
        try:
            generations = logbook.select("gen")
            avg_fitness = logbook.select("avg")
            max_fitness = logbook.select("max")

            plt.figure(figsize=(10, 6))
            plt.plot(generations, avg_fitness, 'b-',
                     label='Average', linewidth=2)
            plt.plot(generations, max_fitness, 'r-',
                     label='Maximum', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness (Accuracy)')
            plt.title('GA Evolution - Fitness Over Generations')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = Path(self.config.OUTPUTS_DIR) / \
                f"ga_evolution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 GA evolution plot saved: {plot_path}")

        except Exception as e:
            print(f"⚠️  Could not save GA plot: {e}")
