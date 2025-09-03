import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer, Categorical
from typing import Dict, Any


class GAVotingClassifier:
    """GA-optimized weighted voting classifier"""

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def create_voting_classifier(self, feature_type: str = 'tfidf') -> VotingClassifier:
        """Create voting classifier based on feature type"""
        estimators = [
            ('knn', KNeighborsClassifier()),
            ('dt', DecisionTreeClassifier(random_state=self.config.RANDOM_STATE))
        ]

        # Choose appropriate Naive Bayes
        if feature_type == 'embeddings':
            estimators.append(('nb', GaussianNB()))
        else:
            estimators.append(('nb', MultinomialNB()))

        return VotingClassifier(estimators=estimators, voting='soft')

    def get_search_space(self) -> Dict[str, Any]:
        """Define GA search space"""
        return {
            # KNN parameters
            'knn__n_neighbors': Integer(3, 15),
            'knn__weights': Categorical(['uniform', 'distance']),

            # Decision Tree parameters
            'dt__max_depth': Integer(5, 25),
            'dt__min_samples_split': Integer(2, 10),
            'dt__criterion': Categorical(['gini', 'entropy']),

            # Naive Bayes parameter
            'nb__alpha': Continuous(0.1, 2.0),

            # Voting weights - KEY OPTIMIZATION!
            'weights': Continuous(0.1, 3.0, shape=(3,))
        }

    def train_and_optimize(self, X_train, y_train, feature_type: str = 'tfidf'):
        """Train model with GA optimization"""
        print(f"\n🧬 GA optimization for {feature_type.upper()}...")

        # Create voting classifier
        voting_clf = self.create_voting_classifier(feature_type)

        # Setup GA search
        ga_search = GASearchCV(
            estimator=voting_clf,
            param_grid=self.get_search_space(),
            cv=StratifiedKFold(
                n_splits=self.config.CV_FOLDS,
                shuffle=True,
                random_state=self.config.RANDOM_STATE
            ),
            scoring='f1_weighted',
            population_size=self.config.POPULATION_SIZE,
            generations=self.config.GENERATIONS,
            mutation_probability=self.config.MUTATION_PROB,
            crossover_probability=self.config.CROSSOVER_PROB,
            tournament_size=3,
            elitism=True,
            keep_top_k=2,
            criteria='max',
            n_jobs=-1,
            verbose=True,
            random_state=self.config.RANDOM_STATE
        )

        # Fit GA search
        ga_search.fit(X_train, y_train)

        # Store results
        self.best_model = ga_search.best_estimator_
        self.best_params = ga_search.best_params_
        self.best_score = ga_search.best_score_

        print(f"✅ GA completed! Best CV F1: {self.best_score:.4f}")

        # Show optimized weights
        if 'weights' in self.best_params:
            w = self.best_params['weights']
            print(
                f"⚖️  Optimized weights: KNN={w[0]:.3f}, DT={w[1]:.3f}, NB={w[2]:.3f}")

        return self.best_model

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model on test set"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')

        return {
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'cv_score': self.best_score,
            'best_params': self.best_params,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
