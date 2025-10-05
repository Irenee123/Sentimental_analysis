# Twitter Airline Sentiment Analysis Project

## Overview

This project implements a sentiment analysis system for airline-related tweets using machine learning techniques. The system analyzes customer sentiment expressed in tweets about various airlines, classifying them into positive, negative, or neutral categories. This analysis can help airlines understand customer satisfaction, identify service issues, and improve customer experience.

## Project Description

The Twitter Airline Sentiment Analysis project processes and analyzes approximately 14,000+ tweets directed at major US airlines. Using advanced natural language processing (NLP) techniques and machine learning algorithms, the system automatically categorizes customer sentiment, providing valuable insights for business intelligence and customer service improvement.

### Key Features

- **Multi-phase Data Pipeline**: Comprehensive data preprocessing, cleaning, and feature engineering
- **Text Processing**: Tokenization, normalization, stopword removal, lemmatization, and stemming
- **Feature Engineering**: TF-IDF vectorization, character n-grams, and statistical features
- **Machine Learning Model**: Optimized Logistic Regression classifier with hyperparameter tuning
- **Performance Metrics**: Detailed evaluation with confusion matrices, classification reports, and cross-validation
- **Scalable Architecture**: Efficient sparse matrix operations for handling large-scale text data

## Dataset

**Source**: <https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>
**Size**: ~14,874 tweets (after preprocessing: 14,456 tweets)
**Airlines Covered**: Virgin America, United, Southwest, Delta, US Airways, American Airlines
**Target Classes**:

- Positive sentiment
- Negative sentiment
- Neutral sentiment

### Data Distribution

- **Negative**: ~62.7% (dominant class)
- **Neutral**: ~21.1%
- **Positive**: ~16.2%

## How to Run the Project

### Prerequisites

1. Ensure Python 3.x is installed on your system
2. Install all required dependencies:

   ```bash
   pip install pandas numpy scikit-learn tensorflow keras nltk matplotlib seaborn scipy
   ```

3. Download NLTK data (required for text preprocessing):

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

### Running the Analysis

1. **Clone or download the repository**:

   ```bash
   git clone https://github.com/Irenee123/Sentimental_analysis.git
   cd Sentimental_analysis
   ```

2. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook Notebooks/Santimental_analysis.ipynb
   ```

3. **Execute the notebook cells sequentially**:

   - The notebook is organized into phases (1-9)
   - Each phase builds on the previous one
   - Run cells in order from top to bottom
4. **For training models**:

   - **Logistic Regression** (Phase 5-8): Executes in minutes on CPU
   - **Deep Learning Models** (Phase 9): May take 20-60 minutes depending on hardware
     - GPU recommended for faster training
     - Models are saved automatically to `Models/` directory
5. **Using pre-trained models**:

   ```python
   import pickle

   # Load Logistic Regression model
   with open('Models/logistic_regression_model.pkl', 'rb') as f:
       lr_model = pickle.load(f)

   # Load Deep Learning model
   from tensorflow.keras.models import load_model
   dl_model = load_model('Models/dl_opt_adam.keras')
   ```

### Expected Outputs

- **Preprocessed Dataset**: `Dataset/tweets_preprocessed.csv`
- **Trained Models**: Saved in `Models/` directory
- **Visualizations**: Confusion matrices, performance charts, sentiment distributions
- **Performance Metrics**: Classification reports, accuracy scores, F1-scores

## Technical Implementation

### Phase 1: Exploratory Data Analysis (EDA)

- Dataset structure and quality assessment
- Sentiment distribution analysis
- Text characteristics and statistics
- Airline-specific sentiment patterns
- Content pattern analysis (mentions, hashtags, URLs)

### Phase 2: Data Cleaning and Preprocessing

- Missing value handling
- Duplicate removal (exact duplicates and text duplicates)
- HTML entity decoding
- Basic text normalization

### Phase 3: Text Preprocessing Pipeline

- Text normalization (lowercase conversion, URL/mention removal)
- Tokenization using NLTK
- Stopword removal (with sentiment-aware filtering)
- Lemmatization and stemming
- Vocabulary analysis and optimization

### Phase 4: Feature Engineering

- **TF-IDF Features**: 10,000 unigram/bigram features with L2 normalization
- **Character N-grams**: 3,000 features capturing misspellings and informal text
- **Statistical Features**: 25 engineered features including:
  - Text length and word count metrics
  - Punctuation and capitalization patterns
  - Sentiment-specific word counts
  - Social media engagement indicators

### Phase 5: Model Training and Evaluation (Classical ML)

- **Algorithm**: Logistic Regression (optimized for high-dimensional sparse data)
- **Feature Space**: 13,025 total features
- **Hyperparameter Tuning**: Grid Search with 5-fold cross-validation
- **Evaluation**: Stratified train/validation/test split (60%/20%/20%)

### Phase 6: Deep Learning Models (RNN/LSTM/GRU)

- **Architecture Types**: SimpleRNN, LSTM, and GRU with bidirectional variants
- **Sequence Processing**:
  - Tokenization with 30,000 vocabulary size
  - Sequence padding to max length of 60 tokens
  - Embedding layer (128 dimensions)
- **Model Configurations**:
  - Embedding dimension: 128
  - RNN units: 128
  - Bidirectional RNN layers
  - Dropout: 0.3 (regularization)
  - Dense layer: 128 units with ReLU activation
- **Optimizer Comparison**: Adam, RMSprop, and SGD with momentum
- **Training Strategy**:
  - Early stopping with patience=4 (monitor validation accuracy)
  - Learning rate reduction on plateau (factor=0.5, patience=2)
  - Batch sizes: 64 and 128
  - Epochs: 10-12 with early stopping
- **Loss Function**: Sparse categorical cross-entropy
- **Evaluation Metrics**: Accuracy, Macro F1-Score, Log Loss, MSE

## Results

### Logistic Regression Performance

- **Test Accuracy**: ~79-82% (varies with hyperparameter settings)
- **Weighted F1-Score**: ~0.79-0.82
- **Cross-validation**: Consistent performance across folds
- **Class-specific Performance**:
  - Negative sentiment: High precision and recall (largest class)
  - Neutral sentiment: Moderate performance
  - Positive sentiment: Good precision, challenging due to class imbalance

### Deep Learning Model Performance

The deep learning models (LSTM/GRU) provide competitive performance with different characteristics:

- **Model Comparison**:
  - **LSTM with Adam**: Best overall performance with balanced metrics
  - **GRU**: Faster training, slightly lower accuracy than LSTM
  - **Optimizer Impact**: Adam outperforms RMSprop and SGD for this task
- **Key Advantages**:
  - Automatic feature learning from raw text
  - Better handling of sequential dependencies
  - No manual feature engineering required
  - Captures long-range contextual information
- **Trade-offs**:
  - Longer training time vs Logistic Regression
  - Requires more computational resources (GPU beneficial)
  - Less interpretable than linear models

### Key Insights

- Negative tweets are most frequent and easiest to classify because of the dataset imbalance.
- Text length and punctuation patterns are strong sentiment indicators.
- Class imbalance handling improved minority class performance.
- LSTM models with Adam optimizer provided the best performance among deep learning approaches.
- Early stopping and learning rate reduction helped prevent overfitting in deep learning models.

## Project Structure

```text
Sentimental_analysis/
├── README.md                           # Project documentation
├── Dataset/
│   ├── Tweets.csv                      # Original dataset
│   └── tweets_preprocessed.csv         # Cleaned and processed data
├── Models/
│   ├── logistic_regression_model.pkl   # Trained Logistic Regression model
│   ├── dl_opt_adam.keras               # Deep Learning model (Adam optimizer)
│   ├── dl_opt_rmsprop.keras            # Deep Learning model (RMSprop optimizer)
│   ├── dl_opt_sgd.keras                # Deep Learning model (SGD optimizer)
│   ├── preprocessing_artifacts.pkl     # Feature engineering artifacts
│   └── dl_preprocessing_artifacts.pkl  # Tokenizer and label encoder for DL
└── Notebooks/
    └── Santimental_analysis.ipynb      # Complete analysis workflow
```
