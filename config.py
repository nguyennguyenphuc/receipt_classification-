class Config:
    """Configuration settings"""

    # Dataset
    EXCEL_FILE_PATH = "data/viet_receipt_categorized_label.xlsx"
    TEXT_COLUMN = "description"
    TARGET_COLUMN = "Category_Detailed"

    # Data processing
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MIN_SAMPLES_PER_CLASS = 5

    # Feature extraction
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)

    # GA optimization
    POPULATION_SIZE = 20
    GENERATIONS = 25
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.8
    CV_FOLDS = 5

    # Output paths
    MODEL_SAVE_PATH = "models/receipt_classifier.pkl"
    PLOTS_DIR = "outputs/plots"
    REPORTS_DIR = "outputs/reports"
