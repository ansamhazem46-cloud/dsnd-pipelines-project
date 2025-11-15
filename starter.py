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

# ## Import libraries and Requirement things :

# #### Configure a NLP

# In[9]:


get_ipython().system('pip install spacy')


# In[11]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[12]:


import spacy


# In[14]:


nlp = spacy.load("en_core_web_sm")


# #### Import Libraries :

# In[17]:


import matplotlib.pyplot as plt
import spacy
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import train_test_split , HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer , StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer


# In[18]:


# Load data
df = pd.read_csv(
    'data/reviews.csv',
)


df.info()
print("Data shape :" , df.shape)
# Show the first head data in dataset
df.head()


# ## Preparing features (`X`) & target (`y`)

# In[22]:


data = df

# separate features from labels
X = data.drop('Recommended IND', axis=1)
y = data['Recommended IND'].copy()

print('Labels:', y.unique())
print('Features:')
display(X.head())


# In[24]:


# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    random_state=27,
)


# # Your Work

# ## Data Exploration

# In[28]:


# View the columns of the data and datatypes 
df.info()


# In[30]:


# Check the missing values
df.isnull().sum()


# ### Distribution of Recommended IND

# In[33]:


# Bat chart plotting of distribution of recommended ind

df = data.copy()
df.drop('Clothing ID', axis=1, inplace=True)

# Show a data and informations
print(df.info())

print(df.describe())

# Using matplotlib
df['Recommended IND'].value_counts().plot(kind='bar')
plt.title('Distribution of Recommended IND')
plt.show()


# ### Correlation Matrix

# In[36]:


# Plotting Correlation Matrix

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr_matrix, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns.tolist(), rotation=90)
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_yticklabels(corr_matrix.columns.tolist())
plt.title('Correlation Matrix')
plt.show()


# ### Avg Positive Feedbacks & Number of Reviews by Department (Bar charts)

# In[39]:


avg_feedback_by_dept = df.groupby('Department Name')['Positive Feedback Count'].mean()

# Plotting the average of positivies feedbacks per departments
avg_feedback_by_dept.plot(kind= 'bar')
plt.title('Average Positive Feedback per Departments')
plt.xlabel("Department")
plt.ylabel("Average Positive Feedback")
plt.show()


department_counts = df['Department Name'].value_counts()

# Plotting the number of reviews per departments.
department_counts.plot(kind='bar')
plt.title('Number of Reviews by Department')
plt.xlabel('Department Name')
plt.ylabel('Number of Reviews')
plt.show()


# ### Visualization of Age distribution of reviewers

# In[42]:


data = {'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]}
df = pd.DataFrame(data)

density, bins, _ = plt.hist(df['Age'], bins=20, density=True, alpha=0)
cum_density = np.cumsum(density * np.diff(bins))

plt.plot(bins[:-1], cum_density, color="blue")
plt.title('Age Distribution of Reviewers')
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# ## Building Pipeline

# In[45]:


nlp = spacy.load("en_core_web_sm")

class TextLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmatized_texts = [' '.join(token.lemma_ for token in doc if not token.is_stop)
                            for doc in self.nlp.pipe(X)]
        return lemmatized_texts


# In[47]:


nlp = spacy.load("en_core_web_sm")




class TextLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [' '.join([token.lemma_ for token in doc if not token.is_stop]) for doc in self.nlp.pipe(X)]

text_processing_pipeline = Pipeline([
    ('lemmatizer', TextLemmatizer(nlp=nlp)),
    ('tfidf_vectorizer', TfidfVectorizer(stop_words='english'))
])

feature_engineering = ColumnTransformer([
    ('text_data', text_processing_pipeline, 'Review Text'),
    ('categorical_data', OneHotEncoder(handle_unknown='ignore'), ['Division Name', 'Department Name', 'Class Name']),
    ('numeric_data', StandardScaler(), ['Age', 'Positive Feedback Count'])  # Rating removed
], remainder='drop')




# ## Training Pipeline

# In[50]:


# Load the data
df = pd.read_csv('data/reviews.csv')


#  Separate features X and target variable y
X = df.drop('Recommended IND', axis=1)
y = df['Recommended IND']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define a pipeline including a preprocessing and classification .
pipeline = Pipeline([
    ('preprocessor', feature_engineering),
    ('classifier', RandomForestClassifier(random_state=42))

    
])

param_grid ={
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15, 20]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


# ## Training the model :

# In[55]:


# Fit the pipeline on a data
pipeline.fit(X_train, y_train)
baseline_pred = pipeline.predict(X_test)

# Evaluate baseline model performance
print("Baseline accuracy:", accuracy_score(y_test, baseline_pred))

print("Baseline Classification Report:\n", classification_report(y_test, baseline_pred))


# ## Fine-Tuning Pipeline

# In[ ]:


# Run the grid search
grid_search.fit(X_train, y_train)

# Show the best parameters founded by the grid search
print("Best parameters found: ", grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the tuned model performance
print("Accuracy after hyperparameter tuning:", accuracy_score(y_test, y_pred))


print("Classification Report after tuning:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




