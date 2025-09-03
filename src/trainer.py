import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
from typing import Dict, Any

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .models import GAVotingClassifier


class Trainer:
    """Main training pipeline"""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.results = {}

    def run_training(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        print("🏪 Vietnamese Receipt Classification Training")
        print("=" * 60)

        # 1. Load data
        data = self.data_loader.load_data()

        # 2. Extract features
        features = self.feature_extractor.extract_all_features(
            data['X_train'], data['X_test']
        )

        # 3. Train models for each feature type
        all_results = {}
        for feature_name, (X_train_feat, X_test_feat) in features.items():
            print(f"\n{'='*20} {feature_name.upper()} {'='*20}")

            # Train GA-optimized model
            model = GAVotingClassifier(self.config)
            trained_model = model.train_and_optimize(
                X_train_feat, data['y_train'], feature_name
            )

            # Evaluate
            results = model.evaluate(X_test_feat, data['y_test'])
            results['model'] = trained_model
            results['feature_type'] = feature_name

            all_results[feature_name] = results

            print(f"📊 Test Results:")
            print(f"   Accuracy: {results['test_accuracy']:.4f}")
            print(f"   F1-Score: {results['test_f1_score']:.4f}")

        # 4. Find best model
        best_feature = max(all_results.keys(),
                           key=lambda k: all_results[k]['test_f1_score'])

        # 5. Store results
        self.results = {
            'models': all_results,
            'best_model': best_feature,
            'data_info': {
                'label_mapping': {
                    'label_to_id': data['label_to_id'],
                    'id_to_label': data['id_to_label'],
                    'unique_labels': data['unique_labels']
                }
            },
            'test_data': {
                'y_test': data['y_test'],
                'X_test': data['X_test']
            }
        }

        # 6. Generate outputs
        self._generate_report()
        self._save_model()
        self._create_visualizations()

        return self.results

    def _generate_report(self):
        """Generate final performance report"""
        print(f"\n{'='*60}")
        print("🎯 FINAL RESULTS")
        print(f"{'='*60}")

        print(f"{'Feature':<12} {'CV F1':<8} {'Test Acc':<10} {'Test F1':<8}")
        print("-" * 40)

        for name, results in self.results['models'].items():
            marker = "🏆" if name == self.results['best_model'] else "  "
            print(f"{marker} {name.upper():<10} {results['cv_score']:<8.4f} "
                  f"{results['test_accuracy']:<10.4f} {results['test_f1_score']:<8.4f}")

        best = self.results['models'][self.results['best_model']]
        print(f"\n🏆 BEST MODEL: {self.results['best_model'].upper()}")
        print(f"   F1-Score: {best['test_f1_score']:.4f}")
        print(f"   Accuracy: {best['test_accuracy']:.4f}")

        if 'weights' in best['best_params']:
            w = best['best_params']['weights']
            print(
                f"   Optimal Weights: KNN={w[0]:.3f}, DT={w[1]:.3f}, NB={w[2]:.3f}")

    def _save_model(self):
        """Save best model"""
        os.makedirs("models", exist_ok=True)

        model_data = {
            'model': self.results['models'][self.results['best_model']]['model'],
            'feature_extractor': self.feature_extractor,
            'config': self.config,
            'label_mapping': self.results['data_info']['label_mapping'],
            'best_feature_type': self.results['best_model']
        }

        with open(self.config.MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Model saved: {self.config.MODEL_SAVE_PATH}")

    def _create_visualizations(self):
        """Create visualization plots"""
        os.makedirs(self.config.PLOTS_DIR, exist_ok=True)

        self._plot_performance()
        self._plot_weights()
        self._plot_confusion_matrix()

        print(f"📊 Plots saved to: {self.config.PLOTS_DIR}/")

    def _plot_performance(self):
        """Plot performance comparison"""
        feature_names = list(self.results['models'].keys())
        test_f1 = [self.results['models'][name]['test_f1_score']
                   for name in feature_names]
        cv_f1 = [self.results['models'][name]['cv_score']
                 for name in feature_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Test F1
        bars1 = ax1.bar(feature_names, test_f1, color='lightcoral', alpha=0.8)
        ax1.set_title('Test F1 Scores')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)

        # CV F1
        bars2 = ax2.bar(feature_names, cv_f1, color='skyblue', alpha=0.8)
        ax2.set_title('CV F1 Scores')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                bar.axes.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.config.PLOTS_DIR}/performance.png", dpi=300)
        plt.close()

    def _plot_weights(self):
        """Plot optimized voting weights"""
        fig, ax = plt.subplots(figsize=(10, 6))

        feature_types = []
        knn_weights = []
        dt_weights = []
        nb_weights = []

        for name, results in self.results['models'].items():
            if 'weights' in results['best_params']:
                w = results['best_params']['weights']
                feature_types.append(name.upper())
                knn_weights.append(w[0])
                dt_weights.append(w[1])
                nb_weights.append(w[2])

        x = np.arange(len(feature_types))
        width = 0.25

        ax.bar(x - width, knn_weights, width, label='KNN', alpha=0.8)
        ax.bar(x, dt_weights, width, label='Decision Tree', alpha=0.8)
        ax.bar(x + width, nb_weights, width, label='Naive Bayes', alpha=0.8)

        ax.set_xlabel('Feature Types')
        ax.set_ylabel('Weight')
        ax.set_title('GA-Optimized Voting Weights')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.config.PLOTS_DIR}/weights.png", dpi=300)
        plt.close()

    def _plot_confusion_matrix(self):
        """Plot confusion matrix for best model"""
        best_results = self.results['models'][self.results['best_model']]
        y_test = self.results['test_data']['y_test']
        y_pred = best_results['predictions']
        labels = self.results['data_info']['label_mapping']['unique_labels']

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.results["best_model"].upper()}')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(f"{self.config.PLOTS_DIR}/confusion_matrix.png", dpi=300)
        plt.close()
