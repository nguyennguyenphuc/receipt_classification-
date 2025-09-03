"""Configuration settings for Receipt Classification"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class Config:
    # Data settings
    DATA_FILE = "data/viet_receipt_categorized_label.xlsx"
    TEXT_COLUMN = "description"
    LABEL_COLUMN = "Category_Detailed"

    # Feature extraction settings
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 2
    MAX_DF = 0.8

    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 3

    # GA settings
    POPULATION_SIZE = 30
    GENERATIONS = 15
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.3
    TOURNAMENT_SIZE = 3
    ELITISM_SIZE = 1

    # Embedding model
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    # Output paths
    MODELS_DIR = "models"
    OUTPUTS_DIR = "outputs"

    # Logging
    VERBOSE = True
