# Vietnamese Receipt Classification

A machine learning system that classifies Vietnamese receipts by category using GA-optimized weighted voting ensemble.

## 🎯 Features

- **Simple preprocessing**: Basic text cleaning for Vietnamese receipts
- **Multi-feature approach**: BoW, TF-IDF, and Sentence Embeddings
- **GA optimization**: Genetic Algorithm optimizes voting weights automatically
- **Complete pipeline**: Data loading → Feature extraction → Training → Evaluation
- **Easy prediction**: Simple API for classifying new receipts

## 📊 Dataset

- **File**: `viet_receipt_categorized_label.xlsx`
- **Structure**: `id | description | Category_Detailed`
- **Size**: 2,034 Vietnamese receipts
- **Categories**: 10 categories (Siêu thị, Ăn uống, Sữa & Đồ uống, etc.)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in root directory
# viet_receipt_categorized_label.xlsx

# 3. Run training
python main.py

# 4. Results saved to:
# - models/receipt_classifier.pkl (trained model)
# - outputs/plots/ (performance charts)
```

## 📁 Project Structure

```
receipt_classification/
├── main.py                 # ⭐ Single entry point
├── config.py              # ⚙️ Configuration
├── requirements.txt       # 📦 Dependencies
├── src/                   # 📁 Source code
│   ├── data_loader.py     # Data loading & preprocessing
│   ├── feature_extractor.py # BoW, TF-IDF, Embeddings
│   ├── models.py          # GA-optimized voting
│   ├── trainer.py         # Training pipeline
│   └── utils.py           # Utilities & prediction
├── models/                # 💾 Saved models
├── outputs/               # 📊 Results & plots
└── notebooks/             # 📓 Experiments
```

## 🔮 Usage

### Training
```python
from config import Config
from src.trainer import Trainer

config = Config()
trainer = Trainer(config)
results = trainer.run_training()
```

### Prediction
```python
from src.utils import predict_category

result = predict_category("Hoá đơn VinCommerce sữa 33.100")
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Batch Prediction
```python
from src.utils import batch_predict

texts = [
    "Hoá đơn Feel Coffee",
    "VinCommerce sữa Vinamilk",
    "Thanh toán VinID"
]

results = batch_predict(texts)
for result in results:
    print(f"{result['predicted_category']}: {result['confidence']:.3f}")
```

## 🎯 Expected Performance

- **TF-IDF**: ~0.84-0.87 F1-score (typically best)
- **Embeddings**: ~0.82-0.85 F1-score
- **BoW**: ~0.80-0.83 F1-score

## ⚙️ Configuration

Edit `config.py` to customize:

```python
class Config:
    # Dataset
    EXCEL_FILE_PATH = "your_dataset.xlsx"
    TARGET_COLUMN = "Category_Detailed"
    
    # GA Optimization
    POPULATION_SIZE = 20
    GENERATIONS = 25
    
    # Features
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
```

## 📈 Results

The system automatically generates:

1. **Performance comparison** across feature types
2. **GA-optimized voting weights** visualization  
3. **Confusion matrix** for best model
4. **Detailed classification report**

## 🔧 Advanced Usage

### Custom Preprocessing
```python
from src.data_loader import DataLoader

# Modify _preprocess_text method for domain-specific cleaning
```

### Different Models
```python
from src.models import GAVotingClassifier

# Add more base estimators to voting classifier
```

### Feature Engineering
```python
from src.feature_extractor import FeatureExtractor

# Add custom feature extraction methods
```
"""