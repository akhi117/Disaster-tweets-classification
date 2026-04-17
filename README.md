# Disaster Tweets Classification with Imbalance Handling

An Artificial Intelligence project focused on **binary classification of disaster-related tweets**, with special emphasis on solving the **class imbalance problem** using multiple sampling, weighting, and threshold optimization techniques.

## Overview During emergencies, social media sends out a lot of information, but only a small amount of it has useful disaster signals.

This project builds and tests machine learning models that put tweets into two groups: 1) disaster-related and 0) not disaster. The main problem is that there is a big class imbalance (81% non-disaster / 19% disaster). This study looks at how different ways of dealing with the imbalance affect how well the models classify tweets.

--- 

## Features

### Models Implemented
- Logistic Regression
- Random Forest
- Neural Network (TensorFlow/Keras)

### Imbalance Handling Techniques
- Class Weighting
- SMOTE
- Random Oversampling
- Random Undersampling
- Balanced Mini-Batch Training
- Threshold Moving
- SMOTE + Threshold Moving

--- 

## Dataset

**Kaggle Disaster Tweets Dataset**

- 11,370 samples
- 4.38:1 imbalance ratio

Features used:

- Tweet text
- Keyword feature
- TF-IDF Vectorization (1-3 grams)
- 10,000 max features

--- 

## Preprocessing Pipeline

- Lowercasing
- URL removal
- Mention removal
- Hashtag normalization
- Negation preservation
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF feature extraction

--- 

## Evaluation Metrics

This project evaluates:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Macro
- F1 Weighted
- ROC-AUC
- PR-AUC
- Confusion Matrices

--- 

## Best Results

## Best Overall Model

**Random Forest + SMOTE + Threshold Moving**

| Metric | Score |
|--------|-------|
| Accuracy | 0.8826 |
| Recall | 0.7092 |
| F1-Macro | 0.8097 |
| ROC-AUC | 0.9128 |
| PR-AUC | 0.7640 |

--- 

## Key Findings

- Accuracy alone is misleading for imbalanced datasets.
- SMOTE + Threshold Moving produced the strongest overall results.
- Random Undersampling gave highest recall.
- Random Forest outperformed Logistic Regression and Neural Networks overall.

---

## Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- NLTK
- imbalanced-learn
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```bash
Disaster-tweets-classification/
│
├── data/
│   └── train.csv
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── imbalance_methods.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── pr_curves/
│
├── report.pdf
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/akhi117/Disaster-tweets-classification.git
cd Disaster-tweets-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

## Example Use Cases

- Emergency response support
- Disaster monitoring systems
- Social media crisis detection
- Imbalanced text classification research

---

## Future Improvements

- BERT / Transformer models
- Cost-sensitive learning
- Model ensembling
- Real-time tweet streaming deployment

