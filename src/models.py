"""GA-optimized voting classifier module"""

import os
import sys
import random
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools, algorithms
from typing import Dict, Tuple, List, Any
import warnings

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

warnings.filterwarnings('ignore')


class GAVotingClassifier:
    """GA-optimized weighted voting classifier"""

    def __init__(self, config):
        self.config = config
        self.toolbox = None
        self.best_individual = None
        self.best_model = None
        self.ga_features = None
        self.y_train = None

    def setup_ga(self):
        """Setup Genetic Algorithm"""
        # Clear existing classes if they exist
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Gene definitions: [feature_type, knn_k, knn_weights, dt_depth, dt_criterion, nb_alpha, knn_w, dt_w, nb_w]
        # 0=bow, 1=tfidf, 2=embeddings
        self.toolbox.register("feature_type", random.randint, 0, 2)
        self.toolbox.register("knn_k", random.randint, 3, 15)
        self.toolbox.register("knn_weights", random.choice, [
                              'uniform', 'distance'])
        self.toolbox.register("dt_depth", random.randint, 3, 20)
        self.toolbox.register(
            "dt_criterion", random.choice, ['gini', 'entropy'])
        self.toolbox.register("nb_alpha", random.uniform, 0.1, 2.0)
        self.toolbox.register("voting_weight", random.uniform, 0.1, 1.0)

        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.feature_type, self.toolbox.knn_k,
                               self.toolbox.knn_weights, self.toolbox.dt_depth,
                               self.toolbox.dt_criterion, self.toolbox.nb_alpha,
                               self.toolbox.voting_weight, self.toolbox.voting_weight,
                               self.toolbox.voting_weight), n=1)

        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.2)
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.config.TOURNAMENT_SIZE)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _mutate_individual(self, individual, indpb):
        """Custom mutation function"""
        if random.random() < indpb:
            individual[0] = random.randint(0, 2)  # feature_type
        if random.random() < indpb:
            individual[1] = random.randint(3, 15)  # knn_k
        if random.random() < indpb:
            individual[2] = random.choice(
                ['uniform', 'distance'])  # knn_weights
        if random.random() < indpb:
            individual[3] = random.randint(3, 20)  # dt_depth
        if random.random() < indpb:
            individual[4] = random.choice(['gini', 'entropy'])  # dt_criterion
        if random.random() < indpb:
            individual[5] = random.uniform(0.1, 2.0)  # nb_alpha
        if random.random() < indpb:
            individual[6] = random.uniform(0.1, 1.0)  # knn_weight
        if random.random() < indpb:
            individual[7] = random.uniform(0.1, 1.0)  # dt_weight
        if random.random() < indpb:
            individual[8] = random.uniform(0.1, 1.0)  # nb_weight
        return individual,

    def _evaluate_individual(self, individual):
        """Evaluate individual in GA"""
        try:
            (feature_type, knn_k, knn_weights, dt_depth, dt_criterion,
             nb_alpha, knn_w, dt_w, nb_w) = individual

            # Select feature type
            feature_names = ['bow', 'tfidf', 'embeddings']
            selected_feature = feature_names[int(feature_type)]
            X_train_feat, _ = self.ga_features[selected_feature]

            # Create models with hyperparameters
            knn = KNeighborsClassifier(
                n_neighbors=int(knn_k), weights=knn_weights)
            dt = DecisionTreeClassifier(
                max_depth=int(dt_depth),
                criterion=dt_criterion,
                random_state=self.config.RANDOM_STATE
            )

            # Choose appropriate Naive Bayes
            if selected_feature == 'embeddings':
                nb = GaussianNB()
            else:
                nb = MultinomialNB(alpha=nb_alpha)

            # Create voting classifier
            estimators = [('knn', knn), ('dt', dt), ('nb', nb)]

            # Normalize weights
            total_weight = knn_w + dt_w + nb_w
            weights = [knn_w/total_weight, dt_w /
                       total_weight, nb_w/total_weight]

            voting_clf = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )

            # Cross validation score
            scores = cross_val_score(voting_clf, X_train_feat, self.y_train,
                                     cv=self.config.CV_FOLDS, scoring='accuracy', n_jobs=-1)

            return scores.mean(),

        except Exception as e:
            if self.config.VERBOSE:
                print(f"⚠️  Error in evaluation: {e}")
            return 0.0,

    def optimize(self, features: Dict[str, Tuple], y_train: np.ndarray) -> Tuple:
        """Run GA optimization"""
        print("🧬 Running Genetic Algorithm optimization...")

        self.ga_features = features
        self.y_train = y_train

        # Setup GA
        self.setup_ga()

        # Create initial population
        population = self.toolbox.population(n=self.config.POPULATION_SIZE)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame
        hof = tools.HallOfFame(self.config.ELITISM_SIZE)

        # Run algorithm
        print(
            f"🏃 Running GA: {self.config.POPULATION_SIZE} individuals, {self.config.GENERATIONS} generations")
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.config.CROSSOVER_PROB,
            mutpb=self.config.MUTATION_PROB,
            ngen=self.config.GENERATIONS,
            stats=stats,
            halloffame=hof,
            verbose=self.config.VERBOSE
        )

        # Get best individual
        self.best_individual = hof[0]
        print(f"🏆 Best individual: {self.best_individual}")
        print(f"🎯 Best fitness: {self.best_individual.fitness.values[0]:.4f}")

        return self.best_individual, logbook

    def create_best_model(self, features: Dict[str, Tuple]) -> Tuple:
        """Create the best model from GA results"""
        if self.best_individual is None:
            raise ValueError("No optimization run yet. Call optimize() first.")

        (feature_type, knn_k, knn_weights, dt_depth, dt_criterion,
         nb_alpha, knn_w, dt_w, nb_w) = self.best_individual

        # Select features
        feature_names = ['bow', 'tfidf', 'embeddings']
        selected_feature = feature_names[int(feature_type)]

        print(f"🎯 Best configuration:")
        print(f"   Feature type: {selected_feature}")
        print(f"   KNN: k={int(knn_k)}, weights={knn_weights}")
        print(f"   DT: depth={int(dt_depth)}, criterion={dt_criterion}")
        print(f"   NB: alpha={nb_alpha:.3f}")

        # Create models
        knn = KNeighborsClassifier(n_neighbors=int(knn_k), weights=knn_weights)
        dt = DecisionTreeClassifier(
            max_depth=int(dt_depth),
            criterion=dt_criterion,
            random_state=self.config.RANDOM_STATE
        )

        # Choose appropriate Naive Bayes
        if selected_feature == 'embeddings':
            nb = GaussianNB()
        else:
            nb = MultinomialNB(alpha=nb_alpha)

        # Normalize weights
        total_weight = knn_w + dt_w + nb_w
        weights = [knn_w/total_weight, dt_w/total_weight, nb_w/total_weight]

        print(
            f"   Voting weights: KNN={weights[0]:.3f}, DT={weights[1]:.3f}, NB={weights[2]:.3f}")

        # Create voting classifier
        estimators = [('knn', knn), ('dt', dt), ('nb', nb)]

        self.best_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )

        return self.best_model, selected_feature
