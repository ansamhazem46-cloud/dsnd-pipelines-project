# Customer Review Recommendation Pipeline

## Project Overview

This project builds a machine learning pipeline to predict whether a customer recommends a product based on their review. The dataset includes numerical, categorical, and text features, and the goal is to classify the recommendation (`Recommended IND`) as either 1 (recommended) or 0 (not recommended).

The pipeline incorporates advanced preprocessing, natural language processing (NLP) for text, feature engineering, and model tuning.

---

## Dataset

- **Source:** Provided dataset (`reviews.csv`)
- **Features:**
  - `Clothing ID`: Categorical integer representing the reviewed product
  - `Age`: Customer's age (numerical)
  - `Title`: Review title (text)
  - `Review Text`: Review body (text)
  - `Positive Feedback Count`: Number of positive votes from other customers
  - `Division Name`: Product division (categorical)
  - `Department Name`: Product department (categorical)
  - `Class Name`: Product class (categorical)
- **Target:**
  - `Recommended IND`: Binary variable (1 = recommended, 0 = not recommended)

---

## Data Exploration

The notebook performs exploratory data analysis (EDA) including:

- Summary statistics of numerical features (`Age`, `Positive Feedback Count`)
- Distribution analysis of categorical features (`Division Name`, `Department Name`, `Class Name`)
- Target variable distribution (`Recommended IND`)
- Histograms for numerical features by recommendation
- Boxplots for positive feedback by recommendation
- Word cloud visualization of review texts (optional)
- Part-of-speech (POS) tagging and named entity recognition (NER) counts for reviews

---

## Pipeline Architecture

The pipeline handles:

1. **Numerical Features**
   - Missing value imputation using mean
   - Standard scaling

2. **Categorical Features**
   - Missing value imputation using most frequent value
   - One-hot encoding

3. **Text Features**
   - Preprocessing (lowercase, punctuation removal, stemming)
   - TF-IDF vectorization

4. **Optional NLP Features**
   - POS tag counts
   - Named entity counts

5. **Model**
   - Random Forest classifier (with hyperparameter tuning)
   - Grid search for optimal parameters

---

## Training & Evaluation

- Split dataset into training (90%) and test (10%) sets
- Evaluate models using:
  - Accuracy
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
  - AUC-ROC score
  - ROC curve visualization
- Hyperparameter tuning using `GridSearchCV` to optimize Random Forest performance

---

## Key Findings

- The dataset has a class imbalance (~82% recommended, 18% not recommended)
- Text features are highly predictive of recommendation
- Categorical features like `Department Name` also provide meaningful signals
- Fine-tuning improves model accuracy and AUC-ROC score
- Both Random Forest and XGBoost pipelines perform well, with Random Forest slightly preferred in our experiments

---

## Usage

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```
---
## Run the notebook:
```bash
jupyter notebook starter.ipynb

```
---
## Train and save the pipeline:
```python
import joblib
joblib.dump(pipeline, 'trained_pipeline.pkl')
```
---
## Load and use the trained pipeline:
```python
import joblib
pipeline = joblib.load('trained_pipeline.pkl')
predictions = pipeline.predict(new_data)

```
---
## Requirements
- Python 3.8+
- Libraries:
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn, xgboost
  - nltk, spacy
  - joblib
  - wordcloud (optional for visualization)
- SpaCy model:
 ```bash
  python -m spacy download en_core_web_sm
```
---
