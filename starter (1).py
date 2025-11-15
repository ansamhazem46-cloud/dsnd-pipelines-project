#!/usr/bin/env python
# coding: utf-8

# # Pipeline Project

# You will be using the provided data to create a machine learning model pipeline.
# 
# You must handle the data appropriately in your pipeline to predict whether an
# item is recommended by a customer based on their review.
# Note the data includes numerical, categorical, and text data.
# 
# You should ensure you properly train and evaluate your model.

# ## The Data

# The dataset has been anonymized and cleaned of missing values.
# 
# There are 8 features for to use to predict whether a customer recommends or does
# not recommend a product.
# The `Recommended IND` column gives whether a customer recommends the product
# where `1` is recommended and a `0` is not recommended.
# This is your model's target/

# The features can be summarized as the following:
# 
# - **Clothing ID**: Integer Categorical variable that refers to the specific piece being reviewed.
# - **Age**: Positive Integer variable of the reviewers age.
# - **Title**: String variable for the title of the review.
# - **Review Text**: String variable for the review body.
# - **Positive Feedback Count**: Positive Integer documenting the number of other customers who found this review positive.
# - **Division Name**: Categorical name of the product high level division.
# - **Department Name**: Categorical name of the product department name.
# - **Class Name**: Categorical name of the product class name.
# 
# The target:
# - **Recommended IND**: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.

# ## Load Data

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import spacy
import joblib
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv(
    'data/reviews.csv',
)

df.info()
df.head()


# ## Preparing features (`X`) & target (`y`)

# In[2]:


data = df

# separate features from labels
X = data.drop('Recommended IND', axis=1)
y = data['Recommended IND'].copy()

print('Labels:', y.unique())
print('Features:')
display(X.head())


# In[3]:


# Split data into train and test sets


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    random_state=27,
)


# In[4]:


import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

X_train['Review Text'] = X_train['Review Text'].apply(preprocess_text)
X_test['Review Text'] = X_test['Review Text'].apply(preprocess_text)


# # Your Work

# 

# ## Data Exploration

# 

# In[5]:


# Basic statistics of numerical features
print("Summary statistics of numerical features:")
display(X.describe())

# Distribution of target variable
print("\nDistribution of target variable:")
print(y.value_counts(normalize=True))

# Check for missing values
print("\nMissing values in features:")
print(X.isnull().sum())

# Visualize the distribution of numerical features
import matplotlib.pyplot as plt

numerical_features = ['Age', 'Positive Feedback Count']
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Visualize the distribution of categorical features
categorical_features = ['Division Name', 'Department Name', 'Class Name']
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=X, y=feature, order=X[feature].value_counts().index)
    plt.title(f'Distribution of {feature}')
    plt.show()


# In[ ]:





# ## Building Pipeline

# 

# In[6]:


# Define preprocessing for numerical, categorical, and text features
numerical_features = ['Age', 'Positive Feedback Count']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values with the mean
    ('scaler', StandardScaler())
])

categorical_features = ['Division Name', 'Department Name', 'Class Name']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_features = 'Review Text'
text_transformer = TfidfVectorizer(max_features=500)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features),
    ],
    remainder='drop'
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=27))
])


# In[7]:


# Modify the function to return counts of POS tags and entities
def extract_pos_ner_counts(text):
    doc = nlp(text)
    pos_counts = {token.pos_: 0 for token in doc}
    entity_counts = {ent.label_: 0 for ent in doc.ents}
    
    for token in doc:
        pos_counts[token.pos_] += 1
    for ent in doc.ents:
        entity_counts[ent.label_] += 1
    
    return {
        "pos_counts": pos_counts,
        "entity_counts": entity_counts
    }


# In[9]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[10]:


import spacy
nlp = spacy.load("en_core_web_sm")

# Function to extract POS tags and named entities
def extract_pos_ner(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]  # Part-of-speech tags
    entities = [ent.label_ for ent in doc.ents]  # Named entities
    return {
        "pos_tags": " ".join(pos_tags),
        "entities": " ".join(entities)
    }

# Apply the function to the 'Review Text' column
df['POS Tags'] = df['Review Text'].apply(lambda x: extract_pos_ner(x)['pos_tags'])
df['Entities'] = df['Review Text'].apply(lambda x: extract_pos_ner(x)['entities'])

# Display the updated DataFrame
print(df[['Review Text', 'POS Tags', 'Entities']].head())


# In[11]:


from sklearn.feature_extraction import DictVectorizer

# Define transformers for POS and entity counts
pos_transformer = Pipeline(steps=[
    ('vectorizer', DictVectorizer(sparse=False))
])

entity_transformer = Pipeline(steps=[
    ('vectorizer', DictVectorizer(sparse=False))
])

# Update the ColumnTransformer to include POS and entity features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features),
        ('pos', pos_transformer, 'POS Counts'),
        ('entity', entity_transformer, 'Entity Counts')
    ],
    remainder='drop'
)


# In[12]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=27))
])


# ## Training Pipeline

# 

# In[13]:


# Compute 'POS Counts' and 'Entity Counts' for X_train and X_test
X_train['POS Counts'] = X_train['Review Text'].apply(lambda x: extract_pos_ner_counts(x)['pos_counts'])
X_train['Entity Counts'] = X_train['Review Text'].apply(lambda x: extract_pos_ner_counts(x)['entity_counts'])

X_test['POS Counts'] = X_test['Review Text'].apply(lambda x: extract_pos_ner_counts(x)['pos_counts'])
X_test['Entity Counts'] = X_test['Review Text'].apply(lambda x: extract_pos_ner_counts(x)['entity_counts'])

# Train the pipeline
pipeline.fit(X_train, y_train)
print("Pipeline trained successfully.")


# ## Fine-Tuning Pipeline

# In[14]:


# Evaluate the pipeline
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc_score:.4f}")



# In[15]:


# Define the parameter grid for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}


# In[16]:


# Set up GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1
)


# In[17]:


# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print(f"Best AUC-ROC Score: {grid_search.best_score_:.4f}")


# In[18]:


# Update the pipeline with the best estimator
best_pipeline = grid_search.best_estimator_


# In[19]:


# Evaluate the fine-tuned pipeline
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("Classification Report (Fine-Tuned):")
print(classification_report(y_test, y_pred))


# In[20]:


auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score (Fine-Tuned): {auc_score:.4f}")


# In[21]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[22]:


# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Not Recommended', 'Recommended'], cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# In[23]:


# Save the trained pipeline to the root directory
joblib.dump(pipeline, '../trained_pipeline.pkl')

print("Pipeline saved successfully in the root directory as 'trained_pipeline.pkl'.")


# In[ ]:




