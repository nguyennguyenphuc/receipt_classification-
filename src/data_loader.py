"""Data loading and preprocessing utilities"""

import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        print("🔄 Loading data from Excel...")

        try:
            self.df = pd.read_excel(self.config.DATA_FILE)
            print(f"✅ Loaded {len(self.df)} samples")
            print(f"📊 Columns: {self.df.columns.tolist()}")

            # Check for missing values
            missing_count = self.df.isnull().sum().sum()
            if missing_count > 0:
                print(f"⚠️  Found {missing_count} missing values, removing...")
                self.df = self.df.dropna()
                print(f"✅ Clean dataset: {len(self.df)} samples")

            # Display label distribution
            print(f"\n📈 Label distribution:")
            label_counts = self.df[self.config.LABEL_COLUMN].value_counts()
            for label, count in label_counts.head(10).items():
                print(f"   {label}: {count}")

            if len(label_counts) > 10:
                print(f"   ... and {len(label_counts) - 10} more classes")

            return self.df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Remove special characters, keep letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def prepare_data(self) -> Tuple[List[str], np.ndarray]:
        """Prepare data for training"""
        print("🔧 Preprocessing data...")

        # Preprocess text
        self.df['processed_text'] = self.df[self.config.TEXT_COLUMN].apply(
            self.preprocess_text)

        # Encode labels
        y = self.label_encoder.fit_transform(self.df[self.config.LABEL_COLUMN])
        X = self.df['processed_text'].tolist()

        print(f"🏷️  Number of classes: {len(self.label_encoder.classes_)}")
        print(f"📝 Classes: {list(self.label_encoder.classes_)}")

        return X, y

    def split_data(self, X: List[str], y: np.ndarray) -> Tuple:
        """Split data into train and test sets"""
        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
