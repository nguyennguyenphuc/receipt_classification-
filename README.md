# Vietnamese Receipt Classification Project

## 📁 Project Structure
```
receipt_classification/
├── main.py                           # ⭐ Main entry point
├── config.py                        # ⚙️ Configuration settings  
├── requirements.txt                 # 📦 Python dependencies
├── README.md                        # 📖 This documentation
├── src/                            # 📁 Source code directory
│   ├── __init__.py                 # Package initialization
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── feature_extractor.py        # BoW, TF-IDF, Embeddings (from bill_classify)
│   ├── models.py                   # GA-optimized voting classifier
│   ├── trainer.py                  # Complete training pipeline
│   └── utils.py                    # Prediction utilities
├── models/                         # 💾 Saved models (auto-created)
├── outputs/                        # 📊 Results & plots (auto-created)  
└── viet_receipt_categorized_label.xlsx  # 📋 Your dataset file
```

## 🚀 Quick Start

### 1. Setup Project
```bash
# Create project directory
mkdir receipt_classification
cd receipt_classification

# Copy all files from artifacts to respective locations
# Make sure to create the src/ directory and copy files correctly
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Dataset
- Put your `viet_receipt_categorized_label.xlsx` file in the root directory
- Make sure it has columns: `description` and `Category_Detailed`

### 4. Run Training
```bash
python main.py --mode train
```

### 5. Run Prediction
```bash
# Default samples
python main.py --mode predict

# Custom texts
python main.py --mode predict --texts "Hóa đơn thanh toán tại cửa hàng cà phê Feel Coffee với giá 25000 VND" "Mua sữa tươi Vinamilk tại siêu thị VinMart với giá 35000 VND"

# Use specific model
python main.py --mode predict --model models/receipt_classifier_20231201_143022.pkl
```

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
class Config:
    # Data settings
    DATA_FILE = "viet_receipt_categorized_label.xlsx"
    
    # Feature extraction
    MAX_FEATURES = 5000      # Max features for BoW/TF-IDF
    NGRAM_RANGE = (1, 2)     # Unigram + Bigram
    
    # GA optimization
    POPULATION_SIZE = 30     # Increase for better results (e.g. 50-100)
    GENERATIONS = 15         # Increase for better results (e.g. 30-50) 
    
    # Model settings
    TEST_SIZE = 0.2         # 20% for testing
    RANDOM_STATE = 42       # For reproducibility
```

## 🎯 Key Features

### 🔬 Feature Extraction (from original bill_classify code)
- **Bag of Words (BoW)**: CountVectorizer with n-grams
- **TF-IDF**: TfidfVectorizer with n-grams  
- **Sentence Embeddings**: SentenceTransformer ('all-MiniLM-L6-v2')

### 🤖 Models
- **K-Nearest Neighbors (KNN)**: With optimized k and weights
- **Decision Tree (DT)**: With optimized depth and criterion
- **Naive Bayes (NB)**: MultinomialNB for sparse, GaussianNB for dense features

### 🧬 Genetic Algorithm Optimization
- **Chromosome**: `[feature_type, knn_k, knn_weights, dt_depth, dt_criterion, nb_alpha, knn_w, dt_w, nb_w]`
- **Population Size**: 30 (configurable)
- **Generations**: 15 (configurable)
- **Selection**: Tournament selection
- **Crossover**: Two-point crossover
- **Mutation**: Custom mutation with 20% probability

### 🗳️ Voting Ensemble
- **Soft Voting**: Uses predicted probabilities
- **Weighted**: GA-optimized weights for each classifier
- **Adaptive**: Automatically selects best feature type

## 📊 Expected Output

### Training Process:
```
🚀 STARTING RECEIPT CLASSIFICATION PIPELINE
======================================================================

📊 STEP 1: DATA LOADING
--------------------------------------------------
✅ Loaded 2034 samples
📊 Columns: ['id', 'description', 'Category_Detailed']
📈 Label distribution:
   Siêu thị tổng hợp: 551
   Khác: 441
   Ăn uống ngoài hàng: 290
   Sữa & Đồ uống: 279
   Thực phẩm tươi sống: 213
   Đồ khô & nhu yếu phẩm: 136
   Mua sắm gia đình: 95
   Tài chính - Ngân hàng: 19
   Điện / Nước / Internet: 6
   Y tế / Thuốc men: 4

🔬 STEP 2: FEATURE EXTRACTION
--------------------------------------------------
🚀 Extracting features...
   📝 Bag of Words...
      Shape: (1627, 5000)
   📊 TF-IDF...
      Shape: (1627, 5000) 
   🤖 Sentence Embeddings...
      Shape: (1627, 384)

🧬 STEP 3: GA OPTIMIZATION
--------------------------------------------------
🧬 Running Genetic Algorithm optimization...
🏃 Running GA: 30 individuals, 15 generations
gen     nevals  avg             min             max     
0       30      0.732487        0.574663        0.851261
1       22      0.798953        0.658879        0.883839
2       24      0.822599        0.591876        0.883839
3       21      0.850979        0.68346         0.887524
4       21      0.876528        0.808853        0.887524
5       24      0.882077        0.854948        0.887524
6       21      0.883469        0.820526        0.887524
7       18      0.859496        0.582666        0.887524
8       22      0.87181         0.754163        0.887524
9       22      0.85966         0.582666        0.887524
10      26      0.85886         0.693895        0.887524
11      18      0.861954        0.586949        0.887524
12      25      0.870866        0.722171        0.887524
13      26      0.881337        0.78795         0.887524
14      18      0.881029        0.8402          0.887524
15      29      0.870826        0.582666        0.887524
🏆 Best individual: [0, 14, 'uniform', 20, 'gini', 1.809617648150661, 0.2560473715614762, 0.8577710563211314, 0.2245035040899067]
🎯 Best fitness: 0.8875

🏋️ STEP 4: TRAINING BEST MODEL
--------------------------------------------------
🎯 Best configuration:
   Feature type: bow
   KNN: k=14, weights=uniform
   DT: depth=20, criterion=gini
   NB: alpha=1.810
   Voting weights: KNN=0.191, DT=0.641, NB=0.1687

📊 STEP 5: EVALUATION
--------------------------------------------------
🎯 Test Accuracy: 0.9312

📈 Classification Report:
                       precision    recall  f1-score   support

                  Khác       0.91      0.99      0.95        88
      Mua sắm gia đình       0.95      1.00      0.97        19
     Siêu thị tổng hợp       0.94      0.97      0.96       110
         Sữa & Đồ uống       0.96      0.98      0.97        56
   Thực phẩm tươi sống       0.86      0.70      0.77        43
 Tài chính - Ngân hàng       1.00      0.75      0.86         4
      Y tế / Thuốc men       1.00      1.00      1.00         1
    Ăn uống ngoài hàng       0.98      0.93      0.96        58
Điện / Nước / Internet       0.00      0.00      0.00         1
 Đồ khô & nhu yếu phẩm       0.88      0.85      0.87        27

              accuracy                           0.93       407
             macro avg       0.85      0.82      0.83       407
          weighted avg       0.93      0.93      0.93       407


💾 STEP 6: SAVING RESULTS
--------------------------------------------------
💾 Model saved: models/receipt_classifier_20250903_162237.pkl
📊 Results saved: outputs/results_20250903_162237.pkl
📈 GA evolution plot saved: outputs/ga_evolution_20250903_162237.png

======================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY!
```

### Prediction Results:
```
🎯 PREDICTION RESULTS
======================================================================

📝 Sample 1:
   Text: Hóa đơn thanh toán tại cửa hàng cà phê Feel Coffee với giá 25000 VND
   ➡️  Predicted: Ăn uống ngoài hàng
   🎯 Confidence: 0.850
   📊 Top 3 predictions:
      1. Ăn uống ngoài hàng: 0.850
      2. Siêu thị tổng hợp: 0.096
      3. Sữa & Đồ uống: 0.027

📝 Sample 2:
   Text: Mua sữa tươi Vinamilk tại siêu thị VinMart với giá 35000 VND
   ➡️  Predicted: Sữa & Đồ uống
   🎯 Confidence: 0.781
   📊 Top 3 predictions:
      1. Sữa & Đồ uống: 0.781
      2. Siêu thị tổng hợp: 0.096
      3. Ăn uống ngoài hàng: 0.055
```

## 🔧 Advanced Usage

### Python API:
```python
from config import Config
from src.trainer import ReceiptClassificationTrainer
from src.utils import predict_samples, load_model_for_prediction

# Training
trainer = ReceiptClassificationTrainer(Config)
model, feature_type, results = trainer.run_full_pipeline()

# Prediction
model, feature_type, vectorizers, label_encoder = load_model_for_prediction("models/latest.pkl")
predictions, probs = predict_samples(["sample text"], model, feature_type, vectorizers, label_encoder)
```

### Performance Tuning:
- **Better Results**: Increase `POPULATION_SIZE` (50-100) and `GENERATIONS` (30-50)
- **Faster Training**: Decrease `POPULATION_SIZE` (10-20) and `GENERATIONS` (5-10)  
- **Memory Optimization**: Reduce `MAX_FEATURES` (2000-3000)

## 🔍 Troubleshooting

### Common Issues:
1. **Import Errors**: All files have `os.path` and `sys.path` handling - should work automatically
2. **Missing Dataset**: Make sure `viet_receipt_categorized_label.xlsx` is in root directory
3. **Memory Issues**: Reduce `POPULATION_SIZE` and `MAX_FEATURES` in config
4. **CUDA Issues**: SentenceTransformers will auto-fallback to CPU

### File Structure Check:
```bash
receipt_classification/
├── main.py                    ✅ 
├── config.py                  ✅
├── requirements.txt           ✅
├── viet_receipt_categorized_label.xlsx  ✅
└── src/
    ├── __init__.py           ✅
    ├── data_loader.py        ✅
    ├── feature_extractor.py  ✅
    ├── models.py             ✅
    ├── trainer.py            ✅
    └── utils.py              ✅
```

## 📈 Expected Performance

- **Training Time**: 5-15 minutes (depending on GA settings)
- **Expected Accuracy**: 85-95% (depends on dataset quality)
- **Feature Selection**: Automatically chooses best among BoW/TF-IDF/Embeddings
- **Model Ensemble**: Optimized combination of KNN + Decision Tree + Naive Bayes

## 🎉 Success Indicators

Look for these in the output:
- ✅ **Data Loading**: Clean dataset with good class distribution
- ✅ **Feature Extraction**: All three feature types extracted successfully  
- ✅ **GA Evolution**: Fitness should improve over generations
- ✅ **Final Accuracy**: Should be > 0.85 for good datasets
- ✅ **Model Saving**: Files saved in models/ and outputs/ directories

Happy classifying! 🚀

---
*This project implements a GA-optimized voting ensemble for Vietnamese receipt classification, using features extracted exactly like the original bill_classify code.*
