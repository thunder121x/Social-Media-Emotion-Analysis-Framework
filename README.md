# Social-Media-Emotion-Analysis-Framework
The project classifies mental health-related tweets (depression and anxiety) as positive or negative using sentiment analysis. It employs NLP and machine learning, also exploring topic modeling (LDA) and emoji sentiment for better accuracy.

---

## 📦 Setup Instructions

### 1. (Optional) Create a Virtual Environment

We recommend using a virtual environment to manage dependencies.

**For Windows:**

```bash
python -m venv studysession
.\studysession\Scripts\activate
```

**For macOS/Linux:**

```bash
python -m venv studysession
source studysession/bin/activate
```

### 2. Install Required Packages

Install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing & Tokenize

Run the `preprocessing.ipynb` notebook to clean and preprocess the data before feature extraction.

### 4. Run Preprocessing & Tokenize

Run the `tokenize.ipynb` notebook to tokenization the post text and preprocess the token before training.

### 5. Data Splitting

Use `split.ipynb` to split your dataset into train, validation, and test sets (if not already split).

### 6. Feature Engineering

Use one or more of the following notebooks for feature extraction:

* `featureEn_TFIDFvader.ipynb`: TF-IDF vectorization + VADER sentiment + metadata features
* `featureEn_BowVader.ipynb`: Bag-of-Words + VADER sentiment
* `featureEn_Bert.ipynb`: BERT embeddings
* `featureEn_TokenBestModel.ipynb`: Token-based representations using pre-trained models or word embeddings for deep learning.
* `featureEn_TokenPCA.ipynb`: Word2Vec model to generate embeddings for tokens, followed by dimensionality reduction using PCA for visualization and model input.

## 📁 Project Structure

```text
Social-Media-Emotion-Analysis-Framework/
├── resource/
│   ├── Mental-Health-Twitter-Preprocessed/     # Preprocessed CSVs (train/val/test)
│   ├── Mental-Health-Twitter-Tokenized/        # Tokenized CSVs (train/val/test)
│   ├── Mental-Health-Twitter.csv               # Original raw dataset
│   ├── Mental-Health-Twitter-Preprocessed.csv  # Single cleaned dataset
│   ├── Mental-Health-Twitter-Tokenized.csv     # Single tokenized dataset
│   └── slang.json                              # Slang replacement dictionary
├── script/
│   ├── eda.ipynb                               # Exploratory Data Analysis
│   ├── featureEn_Bert.ipynb                    # Feature engineering with BERT
│   ├── featureEn_BowVader.ipynb                # BoW + VADER sentiment features
│   ├── featureEn_TFIDFvader.ipynb              # TF-IDF + VADER + metadata features
│   ├── featureEn_TokenBestModel.ipynb          # Word embeddings for deep learning
│   ├── featureEn_TokenPCA.ipynb                # Word2Vec model to generate embeddings for tokens
│   ├── preprocessing.ipynb                     # Data cleaning and preparation
│   ├── split.ipynb                             # Train/Val/Test data split
│   └── tokenize.ipynb                          # Word/Token preprocessing
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```
