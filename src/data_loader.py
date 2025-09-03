import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess receipt data"""

    def __init__(self, config):
        self.config = config

    def load_data(self) -> Dict:
        """Load data from Excel file"""
        print("📂 Loading Vietnamese receipt dataset...")

        # Load Excel
        df = pd.read_excel(self.config.EXCEL_FILE_PATH)
        print(f"✅ Loaded {len(df)} samples")

        # Show distribution
        print(f"\n📊 Category distribution:")
        category_counts = df[self.config.TARGET_COLUMN].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count}")

        # Clean data
        df_clean = self._clean_data(df)

        # Create train-test split
        return self._create_split(df_clean)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter data"""
        # Remove missing values
        df_clean = df.dropna(
            subset=[self.config.TEXT_COLUMN, self.config.TARGET_COLUMN])

        # Filter categories with enough samples
        category_counts = df_clean[self.config.TARGET_COLUMN].value_counts()
        valid_categories = category_counts[category_counts >=
                                           self.config.MIN_SAMPLES_PER_CLASS].index
        df_filtered = df_clean[df_clean[self.config.TARGET_COLUMN].isin(
            valid_categories)]

        print(
            f"\nAfter filtering (min {self.config.MIN_SAMPLES_PER_CLASS} samples per class):")
        print(f"   Samples: {len(df_filtered)}")
        print(f"   Categories: {len(valid_categories)}")

        # Preprocess text
        df_filtered = df_filtered.copy()
        df_filtered['processed_text'] = df_filtered[self.config.TEXT_COLUMN].apply(
            self._preprocess_text)

        return df_filtered

    def _preprocess_text(self, text: str) -> str:
        """Simple text preprocessing"""
        if pd.isna(text) or not text:
            return ""

        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters (keep Vietnamese chars)
        text = re.sub(
            r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)

        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _create_split(self, df: pd.DataFrame) -> Dict:
        """Create train-test split with label mapping"""
        # Create label mappings
        unique_labels = sorted(df[self.config.TARGET_COLUMN].unique())
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for i, label in enumerate(unique_labels)}

        print(f"\n🏷️ Label mappings:")
        for i, label in enumerate(unique_labels):
            count = len(df[df[self.config.TARGET_COLUMN] == label])
            print(f"   {i}: {label} ({count} samples)")

        # Prepare features and targets
        X = df['processed_text'].tolist()
        y = [label_to_id[label] for label in df[self.config.TARGET_COLUMN]]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        print(f"\n📊 Dataset split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'unique_labels': unique_labels
        }
