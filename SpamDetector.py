import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
current_working_directory = os.getcwd()
filename = "spam.csv"
file_path = os.path.join(current_working_directory, filename)
data = pd.read_csv(file_path, encoding="latin-1")

if 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['Category', 'Message']

print("First 5 rows of dataset:")
print(data.head())

data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['Message'], data['Spam'], test_size=0.25, random_state=42
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

emails = [
    "<Message here>"
]

predictions = clf.predict(emails)
print("\nCustom Email Predictions:")
for email, pred in zip(emails, predictions):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f" - \"{email}\" → {label}")

accuracy = clf.score(X_test, y_test)
print(f"\nModel Accuracy on test set: {accuracy:.2%}")
