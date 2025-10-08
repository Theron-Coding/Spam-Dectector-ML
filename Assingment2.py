
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#importing each model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering

# evaluation for each model
from sklearn.metrics import (
    #classification
    accuracy_score, precision_score, recall_score, f1_score,
    #clustering
    silhouette_score,
    v_measure_score)

# plot data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def load_data(filename):
    #loads the csv fule at filename e.g ("MergedDataCleaned.csv") 
    dataset = pd.read_csv(filename, encoding='latin-1')

    # Our data set has column name of text, spam. Renaming it to make it clearer for later 
    # waiitng for more data features to be added. so changing it helps
    if 'text' in dataset.columns and 'spam' in dataset.columns: # and 'vowel_count' in dataset.columns etc
        dataset = dataset.rename(columns={'spam': 'Spam', 'text': 'Message'})

    print("Dataset Overview:")

    print(f"Dataset shape: {dataset.shape}")
    print(f"Available features: {list(dataset.columns)}")
    print(f"Spam distribution:\n{dataset['Spam'].value_counts().to_string()}")
    print(f"Spam percentage: {dataset['Spam'].mean():.2%}")
    
    print("First 5 rows of dataset:")
    print(dataset.head())
    print("\nLast 5 rows of dataset:") 
    print(dataset.tail(), '\n')
    
    return dataset

def get_train_test_split(data): 
    X = data['Message']
    y = data['Spam']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Classification model pipelines
def create_logistic_regression_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lr', LogisticRegression(solver='liblinear'))
    ])


def create_multinomial_naive_bayes_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('mnb', MultinomialNB())
    ])


# Cluster model Pipeline
def create_kmeans_pipeline():
    return Pipeline([
        ('count', CountVectorizer(stop_words='english')),
        ('km', KMeans(n_clusters=2, random_state=42))
    ])
    

def train_evaluate_lr(X_train, X_test, y_train, y_test):
    #Logistic Regression Pipleine Created
    clf = create_logistic_regression_pipeline()
    
    # Trained data using the Logistic Regression pipleline
    clf.fit(X_train, y_train)

    #predicts on test data 
    y_pred = clf.predict(X_test)

    # Evaluate predictions
    print('Logistic Regression Results')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred), '\n')
    
    # return clf, y_pred #if need to use later 


def train_evaluate_mnb(X_train, X_test, y_train, y_test):
    #Multinomial Naive Beyer Pipleine Created
    clf = create_multinomial_naive_bayes_pipeline()
    
    # Trained data using the Multinomial Naive Beyer pipleline
    clf.fit(X_train, y_train)

    #predicts on test data 
    y_pred = clf.predict(X_test)

    # Evaluate predictions
    print('MultinomialNB Results')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred), '\n')
    
    # return clf, y_pred #if need to use later 

def train_evaluate_km(dataset):

    #K-Means Clustering Pipleine Created
    clf = create_kmeans_pipeline()
        
    # Trained data using the 'Message' column, as it uses unlabeled data for training
    clf.fit(dataset['Message'])

    #cluster label
    cluster_labels = clf.named_steps['km'].labels_ # accesess the cluster labels within K means pipeline
    
    X_vec = clf.named_steps['count'].transform(dataset['Message'])

    # Evaluate predictions
    print('K-Means Clustering Results')
    print("Silhouette Score:", silhouette_score(X_vec, cluster_labels))

    true_labels = dataset['Spam'] # shows the ground truth labels to compare predicted against 
    print("V-Measure Score:", v_measure_score(true_labels, cluster_labels), '\n')
    
    #return clf, dataset


data = load_data('OLDMergedDataCleaned.csv')
X_train, X_test, y_train, y_test = get_train_test_split(data)

train_evaluate_lr(X_train, X_test, y_train, y_test)
train_evaluate_mnb(X_train, X_test, y_train, y_test)
train_evaluate_km(data)

